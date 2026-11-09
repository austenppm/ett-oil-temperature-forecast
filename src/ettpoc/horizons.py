"""Horizon sweep: how far ahead is forecasting worth it?

Same 12-fold rolling-origin protocol, horizons 1-48 h, for Ridge (B, C) and the
drift-robust GBM (C). Skill is measured against the STRONGER of two benchmarks:
persistence OT(t) and seasonal naive OT(T - 24 h). At horizons that are not multiples
of 24 h, persistence is out of phase with the daily cycle and would flatter the models.

Run:  python -m ettpoc.horizons
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as C
from .data import bad_targets_known_at, clean_ot_causal, load_station, quality_flags, valid_at
from .evaluation import fold_masks, label_cutoff, monthly_folds
from .experiment import recency_weights, tune
from .features import ABSOLUTE_LEVEL_COLS, build
from .models import make_gbm, make_ridge
from .plotting import INK, MODEL_COLORS, STATION_COLORS, plt, save
from .progress import done, step, wrote

HORIZONS = (1, 3, 6, 12, 24, 48)


def run() -> pd.DataFrame:
    rows = []
    for name in C.STATIONS:
        df = load_station(name)
        flags = quality_flags(df)
        ot = clean_ot_causal(df.OT)
        step(f"{name}: tuning and 12 monthly refits at horizons {', '.join(map(str, HORIZONS))} h")
        for h in HORIZONS:
            abs_errors = defaultdict(list)  # model -> absolute errors per fold
            for fr in C.FRAMINGS:
                X, y, meta = build(df, ot, flags, fr, h=h)
                tt = meta.target_time
                params, _ = tune(df, X, y, meta, h)
                rel = [c for c in X.columns if c not in ABSOLUTE_LEVEL_COLS]
                # Seasonal naive OT(T - 24 h) as a change from OT(t); persistence for h >= 24.
                seasonal = (ot.shift(max(24 - h, 0)) - ot).reindex(X.index)
                for start, end in monthly_folds():
                    cutoff = label_cutoff(start, h)
                    available, test = fold_masks(tt, start, end, h)
                    label_ok = valid_at(bad_targets_known_at(df, cutoff), tt)
                    train = available & label_ok & ~meta.input_flagged
                    test = test & ~meta.bad_target
                    fallback = meta.input_flagged[test].to_numpy()
                    y_test = y[test].to_numpy()
                    ridge = make_ridge(params["Ridge"]["alpha"]).fit(X[train], y[train])
                    delta = np.where(fallback, 0, ridge.predict(X[test]))
                    abs_errors[f"Ridge ({fr})"].append(np.abs(delta - y_test))
                    if fr == "C":
                        w = recency_weights(tt[train], cutoff)
                        gbm = make_gbm(params["GBM"]).fit(
                            X.loc[train, rel], y[train], sample_weight=w
                        )
                        delta = np.where(fallback, 0, gbm.predict(X.loc[test, rel]))
                        abs_errors["GBM-robust (C)"].append(np.abs(delta - y_test))
                    else:  # benchmarks: no fallback needed
                        abs_errors["Persistence"].append(np.abs(y_test))
                        abs_errors["Seasonal naive"].append(
                            np.abs(seasonal[test].to_numpy() - y_test)
                        )
            pooled = {m: np.concatenate(e).mean() for m, e in abs_errors.items()}
            benchmark = min(pooled["Persistence"], pooled["Seasonal naive"])
            rows += [
                {
                    "station": name,
                    "horizon_h": h,
                    "model": m,
                    "MAE": v,
                    "skill_vs_best_benchmark": 1 - v / benchmark,
                }
                for m, v in pooled.items()
            ]
            done(f"{name}: {h:2d} h ahead")
    t = pd.DataFrame(rows)
    t.round(4).to_csv(C.TAB_DIR / "results_horizons.csv", index=False)
    return t


def figure(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    styles = {
        "Ridge (B)": ("Ridge", "--", 0.6),
        "Ridge (C)": ("Ridge", "-", 1),
        "GBM-robust (C)": ("GBM", "-", 1),
    }
    for ax, st in zip(axes, C.STATIONS, strict=True):
        d = t[t.station == st]
        for m, (c, ls, a) in styles.items():
            dm = d[d.model == m]
            ax.plot(
                range(len(dm)),
                dm.skill_vs_best_benchmark * 100,
                ls=ls,
                marker="o",
                ms=3,
                color=MODEL_COLORS[c],
                alpha=a,
                label=m,
            )
            if m != "Ridge (C)":
                continue
            for x, v in enumerate(dm.skill_vs_best_benchmark * 100):
                ax.annotate(
                    f"{v:.0f}%",
                    (x, v),
                    textcoords="offset points",
                    xytext=(0, 6),
                    fontsize=7,
                    ha="center",
                    color=MODEL_COLORS[c],
                )
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xticks(range(len(HORIZONS)))
        ax.set_xticklabels([f"{h}h" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("Skill vs best naive benchmark [%]")
        ax.set_title(f"{st}: value of the model by lead time", loc="left", color=STATION_COLORS[st])
    axes[0].legend(fontsize=8, loc="upper left")
    save(fig, "res_horizons")


def main() -> None:
    figure(run())
    wrote(C.TAB_DIR / "results_horizons.csv", C.FIG_DIR / "res_horizons.png")


if __name__ == "__main__":
    main()
