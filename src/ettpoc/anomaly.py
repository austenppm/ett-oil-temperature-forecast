"""Residual-based fault detection ("expected vs measured oil temperature").

Idea used in transformer condition monitoring (e.g. Djamali & Tenbohlen's cooling-
malfunction detection; CIGRE digital-twin guidance): learn what OT *should* be from
the loads, compare with the measurement, and alarm on a persistent deviation.

Expected OT(T) = Ridge model trained on development data from OT history up to T-1
plus the loads measured AT T (features at T are explicitly permitted by the brief).
A 1-hour reference is used because a 24 h-ahead expectation carries 3-4 degC of
weather-driven error (ambient temperature is not measured), which would mask faults.
Detector statistic = 6-hour rolling mean of the residual OT(T) - expected; alarm when
|statistic| exceeds the 99.5th percentile observed on clean calibration windows.

Evaluation:
  1. false alarms on the real evaluation year,
  2. detection rate / delay for synthetic cooling faults of 3 to 20 degC,
  3. how many of the real frozen 31st-day records (OT constant for 24 h) are flagged.

Run:  python -m ettpoc.anomaly
"""

import numpy as np
import pandas as pd

from . import config as C
from .data import load_station, quality_flags
from .features import build
from .models import make_ridge
from .plotting import INK, MUTED, STATION_COLORS, plt, save

WINDOW = 6  # hours averaged by the detector statistic
REF_H = 1  # horizon of the "expected OT" reference model
RIDGE_ALPHA = 1.0
ALARM_QUANTILE = 0.995  # of |statistic| on clean calibration windows
FAULT_SIZES = (3.0, 5.0, 8.0, 12.0, 20.0)
EXAMPLE_FAULT_SIZE = 5.0
RAMP_H, HOLD_H = 12, 36
N_FAULTS = 30
MIN_FAULT_SPACING_DAYS = 7


def expected_ot(df: pd.DataFrame, model=None) -> tuple[pd.Series, object, pd.Series]:
    """Expected OT at each hour T, from raw OT up to T-1 and loads up to T.

    Returns (expected OT, fitted model, whether the row is artefact-free); the model is
    trained on development data unless a fitted one is passed in.
    """
    flags = quality_flags(df)
    X, y, meta = build(df, df.OT, flags, "C", h=REF_H)
    tt = meta.target_time
    if model is None:
        ok = ~meta.bad_target & ~meta.input_flagged
        tr = ok & (tt < C.TUNE_VAL_START)
        model = make_ridge(RIDGE_ALPHA).fit(X[tr], y[tr])
    exp = pd.Series(meta.ot_now.to_numpy() + model.predict(X), index=pd.DatetimeIndex(tt))
    clean = pd.Series(
        (~meta.bad_target & ~meta.input_flagged).to_numpy(), index=pd.DatetimeIndex(tt)
    )
    return exp, model, clean


def statistic(measured: pd.Series, expected: pd.Series) -> pd.Series:
    """Rolling mean of the measured-minus-expected residual."""
    resid = measured.reindex(expected.index) - expected
    return resid.rolling(WINDOW, min_periods=WINDOW).mean()


def inject_faults(
    df: pd.DataFrame, size: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Add a cooling-fault signature to OT: linear ramp to +size over 12 h, hold 36 h.

    Onsets are random hours in the evaluation year, at least a week apart.
    """
    eval_idx = df[C.EVAL_START : C.EVAL_END].index
    candidates = eval_idx[(eval_idx.day != 31) & (eval_idx.hour == 0)]
    onsets = []
    for day in rng.permutation(candidates):
        t = pd.Timestamp(day) + pd.Timedelta(hours=int(rng.integers(0, 24)))
        spaced = all(abs((t - o) / pd.Timedelta(days=1)) > MIN_FAULT_SPACING_DAYS for o in onsets)
        if spaced and t + pd.Timedelta(hours=72) < eval_idx[-1]:
            onsets.append(t)
        if len(onsets) == N_FAULTS:
            break
    faulty = df.copy()
    profile = np.concatenate([np.linspace(0, size, RAMP_H), np.full(HOLD_H, size)])
    for o in onsets:
        idx = faulty.index.get_indexer(pd.date_range(o, periods=len(profile), freq="h"))
        faulty.iloc[idx, faulty.columns.get_loc("OT")] += profile
    return faulty, sorted(onsets)


def run_station(name: str, rng: np.random.Generator) -> tuple[list[dict], tuple]:
    df = load_station(name)
    flags = quality_flags(df)
    exp, model, clean = expected_ot(df)
    stat = statistic(df.OT, exp)
    # Calibrate on development windows whose whole 6 h window is artefact-free.
    window_clean = clean.astype(float).rolling(WINDOW, min_periods=WINDOW).min() == 1
    calib = stat[C.TUNE_VAL_START : C.EVAL_START]
    calib = calib[window_clean.reindex(calib.index).fillna(False).to_numpy()]
    thr = float(np.nanquantile(calib.abs(), ALARM_QUANTILE))

    ev = stat[C.EVAL_START : C.EVAL_END]
    alarm = ev.abs() > thr
    onsets = alarm & ~alarm.shift(fill_value=False)
    months = len(ev) / (24 * 30.44)

    # Real artefacts: 31st-day records frozen on every channel.
    frozen_days = flags.all_channels_flat[C.EVAL_START : C.EVAL_END]
    frozen_days = frozen_days[frozen_days].index.normalize().unique()
    caught_frozen = sum(
        alarm.reindex(pd.date_range(d, periods=36, freq="h")).fillna(False).any()
        for d in frozen_days
    )

    # Alarm episodes on the real data that do NOT overlap a known artefact.
    known_bad = flags.bad_target.reindex(ev.index).fillna(False)
    unexplained = int((onsets & ~known_bad.rolling(24, min_periods=1).max().astype(bool)).sum())

    rows = [
        {
            "station": name,
            "test": "real data",
            "threshold_degC": thr,
            "alarm_episodes_per_month": onsets.sum() / months,
            "unexplained_episodes": unexplained,
            "frozen_31st_days": len(frozen_days),
            "frozen_days_flagged": int(caught_frozen),
        }
    ]

    example = None
    for size in FAULT_SIZES:
        faulty, fault_onsets = inject_faults(df, size, rng)
        exp_f, _, _ = expected_ot(faulty, model)
        stat_f = statistic(faulty.OT, exp_f)
        alarm_f = stat_f.abs() > thr
        delays = []
        for o in fault_onsets:
            win = alarm_f[o : o + pd.Timedelta(hours=RAMP_H + HOLD_H - 1)]
            hit = win[win]
            if len(hit):
                delays.append((hit.index[0] - o) / pd.Timedelta(hours=1))
        rows.append(
            {
                "station": name,
                "test": f"synthetic +{size:.0f} degC cooling fault",
                "threshold_degC": thr,
                "n_faults": len(fault_onsets),
                "detection_rate": len(delays) / len(fault_onsets),
                "median_delay_h": float(np.median(delays)) if delays else np.nan,
            }
        )
        if size == EXAMPLE_FAULT_SIZE:
            example = (faulty, exp_f, stat_f, fault_onsets[len(fault_onsets) // 2], thr)
    return rows, example


def fig_example(name: str, example: tuple) -> None:
    faulty, exp_f, stat_f, onset, thr = example
    lo, hi = onset - pd.Timedelta(days=2), onset + pd.Timedelta(days=3)
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 4.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    axes[0].plot(
        faulty.OT[lo:hi],
        color=INK,
        lw=1.2,
        label=f"Measured OT (with injected +{EXAMPLE_FAULT_SIZE:.0f} °C fault)",
    )
    axes[0].plot(exp_f[lo:hi], color=STATION_COLORS[name], lw=1.2, label="Expected OT from loads")
    axes[0].axvspan(
        onset,
        onset + pd.Timedelta(hours=RAMP_H + HOLD_H),
        color="#f6d5d5",
        lw=0,
        label="Fault active",
    )
    axes[0].set_ylabel("OT")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].set_title(
        f"{name}: residual-based fault detection on a synthetic cooling fault", loc="left"
    )
    s = stat_f[lo:hi]
    axes[1].plot(s, color=MUTED, lw=1.2)
    axes[1].axhline(thr, color="#c53030", ls=":", lw=1)
    axes[1].axhline(-thr, color="#c53030", ls=":", lw=1)
    first = s[(s.abs() > thr) & (s.index >= onset)]
    if len(first):
        axes[1].axvline(first.index[0], color="#c53030", lw=1)
        axes[1].text(
            first.index[0],
            thr,
            f"  alarm after {(first.index[0] - onset) / pd.Timedelta(hours=1):.0f} h",
            fontsize=8,
            color="#c53030",
            va="bottom",
        )
    axes[1].set_ylabel("6 h mean residual")
    save(fig, f"res_anomaly_example_{name}")


def main() -> None:
    rng = np.random.default_rng(C.SEED)
    rows = []
    for name in C.STATIONS:
        r, example = run_station(name, rng)
        rows += r
        fig_example(name, example)
    t = pd.DataFrame(rows)
    t.round(3).to_csv(C.TAB_DIR / "results_anomaly.csv", index=False)
    print(t.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
