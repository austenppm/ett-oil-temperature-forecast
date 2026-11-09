"""Main experiment: tuning on development data, then 12-fold rolling-origin evaluation.

Run:  python -m ettpoc.experiment          (writes outputs/predictions.csv.gz and tables)

Protocol (test windows by TARGET time T):
  development : hyper-parameter tuning, validated on targets from 2017-03-01 onwards,
                finished with the labels observed at the first evaluation issue time.
  evaluation  : 2017-07 .. 2018-06, 12 monthly folds, expanding training window. Every
                model is refitted at the first issue time of its fold (first target - 24 h)
                using only labels observed by then, and used for the whole month.
Rows whose target is a known data artefact (stuck / spike / 31st-day fill) are never
trained on or scored; for training this is judged with the data available at the refit.
Rows whose inputs come from a stuck load sensor are never trained on, and at prediction
time fall back to persistence (a realistic run-time rule).
"""

import json
import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .data import (
    bad_targets_known_at,
    clean_ot_causal,
    development,
    load_station,
    quality_flags,
    valid_at,
)
from .evaluation import fold_masks, label_cutoff, monthly_folds
from .features import ABSOLUTE_LEVEL_COLS, apparent_power_sq, build, feature_groups, raw_window
from .models import CQR, GreyBoxThermal, make_gbm, make_mlp, make_ridge
from .progress import done, step, wrote

RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]
WINDOW_ALPHAS = [10, 100, 1000, 10000]
RECENCY_HALFLIFE_DAYS = 180  # drift-robust models weight recent training data more
PEAK_WEIGHT = 3.0  # peak-aware model: weight of targets above the training 90th percentile
GBM_GRID = [
    {"learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 50},
    {"learning_rate": 0.05, "max_leaf_nodes": 31, "min_samples_leaf": 50},
    {"learning_rate": 0.10, "max_leaf_nodes": 15, "min_samples_leaf": 100},
    {"learning_rate": 0.03, "max_leaf_nodes": 31, "min_samples_leaf": 100, "max_iter": 500},
]
CAL_DAYS = 60  # conformal calibration window at the end of each training set
BENCHMARKS = ("Persistence", "MovingAvg24")
NOISE_SEEDS = tuple(range(C.SEED, C.SEED + 5))  # independent plan-error realisations


def mae(a, b) -> float:
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def noise_model_name(model: str, sigma: float, seed: int) -> str:
    """Name of a framing-C model scored on a load plan with relative error `sigma`."""
    return f"{model}@noise{round(sigma * 100)}@seed{seed}"


def tuning_split(df: pd.DataFrame, meta: pd.DataFrame, h: int = C.H) -> tuple[pd.Series, pd.Series]:
    """Development (train, validation) rows for hyper-parameter selection.

    Training labels end at the first validation issue time; validation targets start at
    TUNE_VAL_START and end with the labels observed at the first evaluation issue time.
    Label quality is judged with the data available at each of those two times.
    """
    tt = meta.target_time
    train_end, val_end = label_cutoff(C.TUNE_VAL_START, h), label_cutoff(C.EVAL_START, h)
    ok_train = valid_at(bad_targets_known_at(df, train_end), tt)
    ok_val = valid_at(bad_targets_known_at(df, val_end), tt)
    inputs_ok = ~meta.input_flagged
    tr = inputs_ok & ok_train & (tt <= train_end)
    va = inputs_ok & ok_val & (tt >= C.TUNE_VAL_START) & (tt <= val_end)
    return tr, va


def tune(
    df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, h: int = C.H
) -> tuple[dict, pd.DataFrame]:
    """Pick Ridge alpha and GBM parameters on the development split."""
    tr, va = tuning_split(df, meta, h)
    rows = []
    for alpha in RIDGE_ALPHAS:
        m = make_ridge(alpha).fit(X[tr], y[tr])
        rows.append(
            {"model": "Ridge", "params": {"alpha": alpha}, "val_MAE": mae(m.predict(X[va]), y[va])}
        )
    for params in GBM_GRID:
        m = make_gbm(params).fit(X[tr], y[tr])
        rows.append({"model": "GBM", "params": params, "val_MAE": mae(m.predict(X[va]), y[va])})
    rows.append({"model": "Persistence", "params": {}, "val_MAE": mae(0, y[va])})
    res = pd.DataFrame(rows)
    best = {
        model: res[res.model == model].sort_values("val_MAE").iloc[0].params
        for model in ("Ridge", "GBM")
    }
    return best, res.assign(params=res.params.map(json.dumps))


def tune_window_alpha(df: pd.DataFrame, W: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> float:
    tr, va = tuning_split(df, meta)
    complete = W.notna().all(axis=1)
    tr, va = tr & complete, va & complete
    scores = {a: mae(make_ridge(a).fit(W[tr], y[tr]).predict(W[va]), y[va]) for a in WINDOW_ALPHAS}
    return min(scores, key=scores.get)


def recency_weights(target_times, origin: pd.Timestamp) -> np.ndarray:
    age_days = (origin - pd.DatetimeIndex(target_times)) / pd.Timedelta(days=1)
    return np.power(0.5, np.asarray(age_days) / RECENCY_HALFLIFE_DAYS)


def grouped_permutation_importance(model, X, y, rng, repeats: int = 3) -> dict[str, float]:
    """MAE increase when all features of one group are shuffled together."""
    base = mae(model.predict(X), y)
    out = {}
    for group, cols in feature_groups(X.columns).items():
        increases = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            increases.append(mae(model.predict(Xp), y) - base)
        out[group] = float(np.mean(increases))
    return out


@dataclass
class FramingData:
    """Everything the fold loop needs for one station and framing."""

    framing: str
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    window: pd.DataFrame  # raw lag window for WindowLinear
    greybox: GreyBoxThermal  # design matrices precomputed for all issue times
    params: dict  # tuned hyper-parameters per model
    # (sigma, seed) -> X with a noisy load plan (framing C only)
    noisy_features: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Fold:
    """The test window of one fold; turns model outputs into prediction records."""

    station: str
    framing: str
    start: pd.Timestamp
    meta: pd.DataFrame  # meta rows of the test window, indexed by issue time

    @property
    def label(self) -> str:
        return self.start.strftime("%Y-%m")

    @property
    def fallback(self) -> np.ndarray:
        return self.meta.input_flagged.to_numpy()

    def predictions(self, model: str, delta, lo=None, hi=None) -> pd.DataFrame:
        """Prediction records from a forecast change dOT (and optional interval on dOT).

        Learned models fall back to persistence where an input sensor is stuck.
        """
        benchmark = model in BENCHMARKS
        if not benchmark:
            delta = np.where(self.fallback, 0.0, delta)
        ot_now = self.meta.ot_now.to_numpy()
        rec = pd.DataFrame(
            {
                "station": self.station,
                "framing": "-" if benchmark else self.framing,
                "model": model,
                "fold": self.label,
                "issue_time": self.meta.index,
                "target_time": self.meta.target_time.to_numpy(),
                "ot_now": ot_now,
                "ot_true": self.meta.ot_true.to_numpy(),
                "ot_pred": ot_now + delta,
                "bad_target": self.meta.bad_target.to_numpy(),
                "fallback": self.fallback,
            }
        )
        if lo is not None:
            rec["lo"] = ot_now + np.where(self.fallback, np.nan, lo)
            rec["hi"] = ot_now + np.where(self.fallback, np.nan, hi)
        return rec

    def interval_stats(self, raw: tuple, conformal: tuple, q: float) -> dict:
        """Coverage and width of the raw quantile-regression and the CQR intervals."""
        good = ~self.meta.bad_target.to_numpy() & ~self.fallback
        d = (self.meta.ot_true - self.meta.ot_now).to_numpy()[good]
        out = {"station": self.station, "framing": self.framing, "fold": self.label}
        out["n"] = int(good.sum())
        for name, (lo, hi) in (("raw_qr", raw), ("cqr", conformal)):
            out[f"coverage_{name}"] = float(np.mean((d >= lo[good]) & (d <= hi[good])))
            out[f"width_{name}"] = float(np.mean(hi[good] - lo[good]))
        out["conformal_q"] = q
        return out


@dataclass
class StationResults:
    predictions: list = field(default_factory=list)
    tuning: list = field(default_factory=list)
    greybox: list = field(default_factory=list)
    importance: list = field(default_factory=list)
    intervals: list = field(default_factory=list)
    hyperparameters: dict = field(default_factory=dict)


def prepare(name: str, df: pd.DataFrame, results: StationResults) -> dict[str, FramingData]:
    """Build features, tune hyper-parameters and precompute model inputs for both framings."""
    flags = quality_flags(df)
    ot_clean = clean_ot_causal(df.OT)
    s2_scale = development(apparent_power_sq(df)).mean()
    out = {}
    for fr in C.FRAMINGS:
        X, y, meta = build(df, ot_clean, flags, fr)
        params, table = tune(df, X, y, meta)
        results.tuning.append(table.assign(station=name, framing=fr))
        window = raw_window(df, ot_clean, X.index, fr)
        params["WindowLinear"] = {"alpha": tune_window_alpha(df, window, y, meta)}
        greybox = GreyBoxThermal(fr).prepare(df, ot_clean, X.index, s2_scale)
        out[fr] = FramingData(fr, X, y, meta, window, greybox, params)
        results.hyperparameters[fr] = params
    for sigma in C.NOISE_SIGMAS:
        for seed in NOISE_SEEDS:
            X_noisy, _, _ = build(df, ot_clean, flags, "C", load_sigma=sigma, seed=seed)
            out["C"].noisy_features[sigma, seed] = X_noisy.reindex(out["C"].X.index)
    return out


def evaluate_fold(
    name: str,
    data: FramingData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    bad_known: pd.Series,
    rng: np.random.Generator,
    results: StationResults,
) -> None:
    """Refit every model at the fold's first issue time and predict targets in [start, end).

    `bad_known` holds the label-quality decisions computable at that refit time.
    """
    X, y, meta, W, params = data.X, data.y, data.meta, data.window, data.params
    tt = meta.target_time
    cutoff = label_cutoff(start)
    available, test = fold_masks(tt, start, end)
    label_ok = valid_at(bad_known, tt)
    usable = available & label_ok & ~meta.input_flagged & W.notna().all(axis=1)
    cal_start = cutoff - pd.Timedelta(days=CAL_DAYS)
    train = usable
    train_q = usable & (tt <= cal_start)  # quantile models; the rest calibrates CQR
    cal = usable & (tt > cal_start)
    X_train, y_train, X_test = X[train], y[train], X[test]
    rel = [c for c in X.columns if c not in ABSOLUTE_LEVEL_COLS]  # drift-robust inputs
    fold = Fold(name, data.framing, start, meta[test])
    record = results.predictions.append

    if data.framing == "B":  # benchmarks do not depend on the framing
        record(fold.predictions("Persistence", np.zeros(len(X_test))))
        record(fold.predictions("MovingAvg24", X_test.ot_dev_mean24.to_numpy()))

    ridge = make_ridge(params["Ridge"]["alpha"]).fit(X_train, y_train)
    record(fold.predictions("Ridge", ridge.predict(X_test)))

    gbm = make_gbm(params["GBM"]).fit(X_train, y_train)
    cqr = CQR(params["GBM"]).fit(X[train_q], y[train_q], X[cal], y[cal])
    conformal = cqr.predict(X_test)
    record(fold.predictions("GBM", gbm.predict(X_test), *conformal))
    results.intervals.append(
        fold.interval_stats(cqr.predict(X_test, conformal=False), conformal, cqr.q)
    )

    greybox = data.greybox.fit(X_train.index, meta.ot_true[train].to_numpy())
    greybox_test = greybox.predict_delta(X_test.index)
    record(fold.predictions("GreyBox", greybox_test))
    coefs = {f"coef_{k}": v for k, v in greybox.coefficients().items()}
    results.greybox.append(
        {
            "station": name,
            "framing": data.framing,
            "fold": fold.label,
            "tau_h": greybox.tau,
            **coefs,
        }
    )

    # Drift-robust GBM: no absolute OT level, recent data weighted more.
    w = recency_weights(tt[train], cutoff)
    gbm_robust = make_gbm(params["GBM"]).fit(X_train[rel], y_train, sample_weight=w)
    record(fold.predictions("GBM-robust", gbm_robust.predict(X_test[rel])))

    # Hybrid: physics model plus GBM on its residual.
    resid = y_train - greybox.predict_delta(X_train.index)
    hybrid = make_gbm(params["GBM"]).fit(X_train[rel], resid, sample_weight=w)
    record(fold.predictions("Hybrid", greybox_test + hybrid.predict(X_test[rel])))

    # Peak-aware: up-weight the hottest training targets.
    ot_train = meta.ot_true[train]
    peak_w = w * np.where(ot_train > ot_train.quantile(0.9), PEAK_WEIGHT, 1.0)
    gbm_peak = make_gbm(params["GBM"]).fit(X_train[rel], y_train, sample_weight=peak_w)
    record(fold.predictions("GBM-peak", gbm_peak.predict(X_test[rel])))

    # Neural and raw-window baselines.
    mlp = make_mlp().fit(X_train, y_train)
    record(fold.predictions("MLP", mlp.predict(X_test)))
    window_linear = make_ridge(params["WindowLinear"]["alpha"]).fit(W[train], y_train)
    record(fold.predictions("WindowLinear", window_linear.predict(W[test].fillna(0))))

    scored = ~fold.meta.bad_target.to_numpy()
    importance = grouped_permutation_importance(gbm, X_test[scored], y[test][scored], rng)
    results.importance += [
        {
            "station": name,
            "framing": data.framing,
            "fold": fold.label,
            "group": g,
            "mae_increase": v,
        }
        for g, v in importance.items()
    ]

    # Robustness to an imperfect load plan: same fitted models, noisy plan at test time.
    for (sigma, seed), X_noisy in data.noisy_features.items():
        for model_name, model in (("Ridge", ridge), ("GBM", gbm)):
            delta = model.predict(X_noisy[test])
            record(fold.predictions(noise_model_name(model_name, sigma, seed), delta))


def run_station(name: str) -> StationResults:
    results = StationResults()
    rng = np.random.default_rng(C.SEED)
    df = load_station(name)
    bad_known = {
        start: bad_targets_known_at(df, label_cutoff(start)) for start, _ in monthly_folds()
    }
    step(f"{name}: building features and tuning hyper-parameters on development data")
    prepared = prepare(name, df, results)
    done(f"{name}: features ready, hyper-parameters chosen")
    for fr, data in prepared.items():
        step(f"{name} framing {fr}: refitting all models for each of the 12 months")
        for i, (start, end) in enumerate(monthly_folds(), 1):
            evaluate_fold(name, data, start, end, bad_known[start], rng, results)
            done(f"{name} framing {fr}: month {i:2d}/12 ({start:%Y-%m})")
    return results


def main() -> None:
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    stations = {name: run_station(name) for name in C.STATIONS}

    def frames(attr: str) -> pd.DataFrame:  # fields holding lists of DataFrames
        return pd.concat(
            [f for res in stations.values() for f in getattr(res, attr)], ignore_index=True
        )

    def records(attr: str) -> pd.DataFrame:  # fields holding lists of dict rows
        return pd.DataFrame([r for res in stations.values() for r in getattr(res, attr)])

    predictions = frames("predictions")
    predictions.to_csv(C.OUT_DIR / "predictions.csv.gz", index=False)
    # Identifies this run; the LSTM and the report check they belong to it.
    C.RUN_ID_FILE.write_text(uuid.uuid4().hex)
    frames("tuning").to_csv(C.TAB_DIR / "tuning.csv", index=False)
    records("greybox").to_csv(C.TAB_DIR / "greybox_folds.csv", index=False)
    records("importance").to_csv(C.TAB_DIR / "importance_folds.csv", index=False)
    records("intervals").to_csv(C.TAB_DIR / "intervals_folds.csv", index=False)
    hyperparameters = {name: res.hyperparameters for name, res in stations.items()}
    (C.TAB_DIR / "chosen_hyperparameters.json").write_text(json.dumps(hyperparameters, indent=2))
    done(f"Saved {len(predictions):,} forecasts")
    wrote(C.OUT_DIR / "predictions.csv.gz", C.TAB_DIR)


if __name__ == "__main__":
    main()
