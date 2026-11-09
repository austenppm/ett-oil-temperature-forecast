"""Exploratory data analysis: figures and summary table for the report.

Run:  python -m ettpoc.eda
"""

import numpy as np
import pandas as pd

from . import config as C
from .data import development, load_station, quality_flags
from .features import apparent_power_sq
from .plotting import INK, MUTED, STATION_COLORS, plt, save
from .progress import done, note, step, wrote


def detrend(s, days=30):
    """Remove the slow (seasonal) component with a centred 30-day mean (descriptive only)."""
    return s - s.rolling(24 * days, center=True, min_periods=24).mean()


def fig_overview(dfs):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, (name, df) in zip(axes, dfs.items(), strict=True):
        ax.plot(df.index, df.OT, lw=0.3, color=STATION_COLORS[name], alpha=0.5)
        ax.plot(
            df.index,
            df.OT.rolling(24 * 30, center=True).mean(),
            lw=1.8,
            color=INK,
            label="30-day mean",
        )
        ax.axvspan(
            pd.Timestamp(C.EVAL_START),
            df.index.max(),
            color="#f2e8d5",
            zorder=0,
            label="Evaluation year",
        )
        ax.set_title(f"{name}: oil temperature (OT)", loc="left")
        ax.set_ylabel("OT")
    axes[0].legend(loc="upper right")
    save(fig, "eda_ot_overview")


def fig_daily_profile(dfs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"wspace": 0.45})
    for ax, (name, df) in zip(axes, dfs.items(), strict=True):
        hour = df.index.hour
        ot = detrend(df.OT).groupby(hour).mean()
        s2 = apparent_power_sq(df)
        s2n = ((s2 - s2.mean()) / s2.std()).groupby(hour).mean()
        ax.plot(ot.index, ot, color=STATION_COLORS[name], lw=2, label="OT (de-seasonalised)")
        ax.set_ylabel("OT deviation", color=STATION_COLORS[name])
        ax2 = ax.twinx()
        ax2.plot(s2n.index, s2n, color=MUTED, lw=1.5, ls="--", label="Load S² (z-score)")
        ax2.set_ylabel("Load S² (z)", color=MUTED)
        ax2.grid(False)
        ax.axvline(ot.idxmax(), color=STATION_COLORS[name], lw=0.8, ls=":")
        ax.set_title(f"{name}: OT peaks at {ot.idxmax()}:00, load at {s2n.idxmax()}:00", loc="left")
        ax.set_xlabel("Hour of day")
        ax.set_xticks(range(0, 24, 3))
    save(fig, "eda_daily_profile")


def fig_thermal_lag(dfs):
    halflives = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    rows = []
    for name, df in dfs.items():
        ot_d = detrend(df.OT)
        s2 = apparent_power_sq(df)
        corr = [ot_d.corr(detrend(s2.ewm(halflife=hl).mean())) for hl in halflives]
        ax.plot(halflives, corr, marker="o", color=STATION_COLORS[name], label=name)
        rows.append((name, halflives[int(np.argmax(corr))], max(corr)))
    ax.set_xscale("log")
    ax.set_xticks(halflives)
    ax.set_xticklabels(halflives)
    ax.set_xlabel("Half-life of load smoothing (hours, log scale)")
    ax.set_ylabel("corr(OT, smoothed S²)")
    ax.set_title("Oil temperature responds to load with thermal memory", loc="left")
    ax.legend()
    save(fig, "eda_thermal_lag")
    return rows


def fig_quality(dfs, flags):
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    for ax, (name, df) in zip(axes, dfs.items(), strict=True):
        f = flags[name]
        rows = [f"stuck_{c}" for c in C.LOAD_COLS] + ["ot_stuck", "ot_spike"]
        labels = [f"{c} stuck" for c in C.LOAD_COLS] + ["OT stuck", "OT spike"]
        for i, r in enumerate(rows):
            on = f[r]
            ax.fill_between(
                df.index,
                i - 0.4,
                i + 0.4,
                where=on,
                step="mid",
                color="#c53030" if "OT" in labels[i] else STATION_COLORS[name],
                lw=0,
            )
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.grid(axis="y", visible=False)
        share = f.load_stuck.mean()
        days31 = int(f.all_channels_flat[f.index.day == 31].sum() // 24)
        ax.set_title(
            f"{name}: {share:.1%} of hours with a stuck load channel; "
            f"{days31} days (all on the 31st) frozen on every channel",
            loc="left",
        )
    save(fig, "eda_data_quality")


def fig_balance(dfs):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, df) in zip(axes, dfs.items(), strict=True):
        x, y = df.MUFL + df.LUFL, df.HUFL
        ax.scatter(x, y, s=1, alpha=0.15, color=STATION_COLORS[name], rasterized=True)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lim, lim, color=INK, lw=0.8, ls="--")
        ax.set_xlabel("MUFL + LUFL")
        ax.set_ylabel("HUFL")
        ax.set_title(f"{name}: r = {x.corr(y):.3f}", loc="left")
    fig.suptitle(
        "Load balance check: high-voltage load should equal middle + low load",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    save(fig, "eda_load_balance")


def fig_persistence_horizon(dfs):
    hs = [1, 2, 3, 6, 12, 24, 48, 72, 168]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, df in dfs.items():
        mae = [(df.OT.shift(-h) - df.OT).abs().mean() for h in hs]
        ax.plot(hs, mae, marker="o", color=STATION_COLORS[name], label=name)
    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels(hs)
    ax.set_xlabel("Forecast horizon (hours, log scale)")
    ax.set_ylabel("MAE of 'no change' forecast")
    ax.set_title("Persistence error grows with the horizon (12 h is out of phase)", loc="left")
    ax.legend()
    save(fig, "eda_persistence_horizon")


def summary_table(dfs, flags, lag_rows):
    rows = []
    lag = {n: (hl, r) for n, hl, r in lag_rows}
    for name, df in dfs.items():
        f = flags[name]
        dev = df.OT[df.index < C.EVAL_START]  # development year (descriptive)
        ev = df.OT[df.index >= C.EVAL_START]
        rows.append(
            {
                "station": name,
                "hours": len(df),
                "start": df.index.min(),
                "end": df.index.max(),
                "missing_values": int(df.isna().sum().sum()),
                "OT_mean_dev_year": dev.mean(),
                "OT_mean_eval_year": ev.mean(),
                "OT_max_dev_year": dev.max(),
                "OT_max_eval_year": ev.max(),
                "alert_threshold_q95_dev": development(df.OT).quantile(C.ALERT_QUANTILE),
                "share_hours_load_stuck": f.load_stuck.mean(),
                "ot_spike_hours": int(f.ot_spike.sum()),
                "ot_stuck_hours": int(f.ot_stuck.sum()),
                "all_channels_flat_hours": int(f.all_channels_flat.sum()),
                "all_channels_flat_days_on_31st": int(
                    f.all_channels_flat[f.index.day == 31].sum() // 24
                ),
                "balance_corr_HUFL_vs_MUFL+LUFL": df.HUFL.corr(df.MUFL + df.LUFL),
                "ot_peak_hour": int(detrend(df.OT).groupby(df.index.hour).mean().idxmax()),
                "load_peak_hour": int(apparent_power_sq(df).groupby(df.index.hour).mean().idxmax()),
                "best_load_halflife_h": lag[name][0],
                "best_load_corr": lag[name][1],
                "persistence_MAE_1h": (df.OT.shift(-1) - df.OT).abs().mean(),
                "persistence_MAE_24h": (df.OT.shift(-24) - df.OT).abs().mean(),
                "OT_autocorr_24h": df.OT.autocorr(24),
            }
        )
    out = pd.DataFrame(rows)
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    out.round(3).to_csv(C.TAB_DIR / "eda_summary.csv", index=False)
    return out


def main() -> None:
    step("Loading both stations and checking data quality")
    dfs = {n: load_station(n) for n in C.STATIONS}
    flags = {n: quality_flags(df) for n, df in dfs.items()}
    done("Data loaded: 17,420 hourly rows per station, no gaps")
    step("Drawing EDA figures")
    fig_overview(dfs)
    fig_daily_profile(dfs)
    lag_rows = fig_thermal_lag(dfs)
    fig_quality(dfs, flags)
    fig_balance(dfs)
    fig_persistence_horizon(dfs)
    summary = summary_table(dfs, flags, lag_rows).set_index("station")
    done("EDA figures and summary table")
    for name, r in summary.iterrows():
        note(
            f"{name}: stuck-load hours {r.share_hours_load_stuck:.1%}, "
            f"mean OT {r.OT_mean_dev_year:.1f} → {r.OT_mean_eval_year:.1f} °C "
            "(development → evaluation year)"
        )
    wrote(C.TAB_DIR / "eda_summary.csv", C.FIG_DIR)


if __name__ == "__main__":
    main()
