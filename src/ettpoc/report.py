"""Turn predictions into the tables and figures used in the PoC report.

Run:  python -m ettpoc.report      (after ettpoc.experiment and ettpoc.neural)

Post-processing done here (both strictly causal):
- Ensemble: equal-weight mean of Ridge, GBM-robust and Hybrid. Equal weights are a
  deliberate choice: estimated combination weights rarely beat the simple average
  out of sample (the "forecast combination puzzle", Smith & Wallis 2009).
- Bias correction (+BC): subtract the model's own mean error over the last 7 days,
  using only errors whose target time has already passed at the issue time, and
  excluding readings judged invalid with the data available at that time.

Significance: Diebold-Mariano p-values are Holm-adjusted across all models compared with
persistence on a station (DM_p_holm); the tables keep the unadjusted p-values as well.
"""

import numpy as np
import pandas as pd

from . import config as C
from .data import bad_target_causal, development, load_station, quality_flags, valid_at
from .evaluation import diebold_mariano, evaluate_alerts, holm, point_metrics
from .experiment import NOISE_SEEDS, noise_model_name
from .plotting import INK, MODEL_COLORS, MUTED, STATION_COLORS, plt, save
from .progress import done, note, step, wrote

ENSEMBLE_MEMBERS = ["Ridge", "GBM-robust", "Hybrid"]
BC_MODELS = ["Ridge", "Ensemble"]
LEARNED = [
    "GreyBox",
    "Ridge",
    "GBM",
    "GBM-robust",
    "GBM-peak",
    "Hybrid",
    "MLP",
    "WindowLinear",
    "LSTM",
    "Ensemble",
    "Ridge+BC",
    "Ensemble+BC",
]
MAIN_MODELS = [("-", "Persistence"), ("-", "MovingAvg24")] + [
    (fr, m) for fr in ("B", "C") for m in LEARNED
]
FAMILY = {
    "GreyBox": "Physics",
    "Ridge": "Linear",
    "WindowLinear": "Linear",
    "GBM": "Trees",
    "GBM-robust": "Trees",
    "GBM-peak": "Trees",
    "Hybrid": "Physics+ML",
    "MLP": "Neural",
    "LSTM": "Neural",
    "Ensemble": "Ensemble",
    "Ridge+BC": "Linear",
    "Ensemble+BC": "Ensemble",
    "Persistence": "Benchmark",
    "MovingAvg24": "Benchmark",
}
FAMILY_COLORS = {
    "Benchmark": "#8a8a8a",
    "Physics": "#2f855a",
    "Linear": "#2b6cb0",
    "Trees": "#c05621",
    "Physics+ML": "#6b46c1",
    "Neural": "#b83280",
    "Ensemble": "#1a202c",
}


def label(fr: str, model: str) -> str:
    return model if fr == "-" else f"{model} ({fr})"


# ---------------------------------------------------------------- post-processing
def add_ensemble(p: pd.DataFrame) -> pd.DataFrame:
    """Append the equal-weight ensemble as a model, per station and framing."""
    out = []
    for _, g in p[p.model.isin(ENSEMBLE_MEMBERS)].groupby(["station", "framing"]):
        members = g.pivot_table(index="issue_time", columns="model", values="ot_pred")
        e = g[g.model == ENSEMBLE_MEMBERS[0]].set_index("issue_time")
        e["model"] = "Ensemble"
        e["ot_pred"] = members[ENSEMBLE_MEMBERS].mean(axis=1).reindex(e.index)
        e[["lo", "hi"]] = np.nan
        out.append(e.reset_index())
    return pd.concat([p, *out], ignore_index=True)


def bias_corrected(g: pd.DataFrame, bad_causal: pd.Series, days: int = 7) -> pd.DataFrame:
    """Causal rolling bias correction for one (station, framing, model) series.

    At issue time t, only errors of forecasts whose target time is <= t are known, and
    whether such a reading was invalid must be decided with data up to it (`bad_causal`).
    """
    g = g.sort_values("issue_time").copy()
    invalid = ~valid_at(bad_causal, g.target_time)
    scored = g[~invalid].set_index("target_time")
    err = (scored.ot_pred - scored.ot_true).sort_index()
    bias = err.rolling(f"{days}D").mean().rename("bias").reset_index()
    known = pd.merge_asof(
        g[["issue_time"]].reset_index(),
        bias,
        left_on="issue_time",
        right_on="target_time",
        direction="backward",
        allow_exact_matches=True,
    ).set_index("index")
    g["ot_pred"] = g.ot_pred - known.bias.fillna(0.0).reindex(g.index)
    return g


def add_bias_correction(p: pd.DataFrame) -> pd.DataFrame:
    """Append bias-corrected versions ("<model>+BC") of BC_MODELS."""
    bad_causal = {st: bad_target_causal(load_station(st)) for st in p.station.unique()}
    candidates = p[p.model.isin(BC_MODELS) & (p.framing != "-")]
    out = [
        bias_corrected(g, bad_causal[st]).assign(model=f"{model}+BC", lo=np.nan, hi=np.nan)
        for (st, _, model), g in candidates.groupby(["station", "framing", "model"])
    ]
    return pd.concat([p, *out], ignore_index=True)


def load_neural(p: pd.DataFrame) -> pd.DataFrame | None:
    """LSTM predictions, if present; they must come from the same run as `p`."""
    nn_path = C.OUT_DIR / "predictions_nn.csv.gz"
    if not nn_path.exists():
        return None
    run_id = C.RUN_ID_FILE.read_text()
    nn_run_id = C.NN_RUN_ID_FILE.read_text() if C.NN_RUN_ID_FILE.exists() else None
    if nn_run_id != run_id:
        raise RuntimeError(f"{nn_path} is from another experiment run; re-run `make neural`")
    nn = pd.read_csv(nn_path, parse_dates=["issue_time", "target_time"])
    keys = ["station", "framing", "fold", "issue_time"]
    ridge = p[p.model == "Ridge"][keys].sort_values(keys).reset_index(drop=True)
    if not nn[keys].sort_values(keys).reset_index(drop=True).equals(ridge):
        raise RuntimeError("LSTM predictions do not cover the same forecasts as the experiment")
    return nn


def load_all() -> pd.DataFrame:
    """All predictions (tabular models, LSTM if available, ensemble and +BC variants)."""
    p = pd.read_csv(C.OUT_DIR / "predictions.csv.gz", parse_dates=["issue_time", "target_time"])
    nn = load_neural(p)
    if nn is not None:
        p = pd.concat([p, nn], ignore_index=True)
    return add_bias_correction(add_ensemble(p))


# ---------------------------------------------------------------- tables
def overall_table(p):
    rows = []
    for st in C.STATIONS:
        ps = p[p.station == st]
        bench = ps[ps.model == "Persistence"].set_index("issue_time")
        hot_thr = development(load_station(st).OT).quantile(0.90)
        for fr, m in MAIN_MODELS:
            d = ps[(ps.framing == fr) & (ps.model == m)].set_index("issue_time")
            if d.empty:
                continue
            b = bench.loc[d.index]
            met = point_metrics(d.ot_true, d.ot_pred)
            stat, pval = (
                (np.nan, np.nan)
                if m == "Persistence"
                else diebold_mariano(
                    (d.ot_pred - d.ot_true).to_numpy(), (b.ot_pred - b.ot_true).to_numpy()
                )
            )
            hot = d.ot_true > hot_thr
            rows.append(
                {
                    "station": st,
                    "framing": fr,
                    "model": m,
                    "family": FAMILY[m],
                    **met,
                    "skill_vs_persistence": 1
                    - met["MAE"] / point_metrics(b.ot_true, b.ot_pred)["MAE"],
                    "DM_stat": stat,
                    "DM_p": pval,
                    "MAE_hot": point_metrics(d.ot_true[hot], d.ot_pred[hot])["MAE"]
                    if hot.any()
                    else np.nan,
                    "MAE_hot_persistence": point_metrics(b.ot_true[hot], b.ot_pred[hot])["MAE"]
                    if hot.any()
                    else np.nan,
                    "n_hot": int(hot.sum()),
                    "fallback_share": d.fallback.mean(),
                }
            )
    t = pd.DataFrame(rows)
    t["DM_p_holm"] = np.nan
    for st in C.STATIONS:
        rows_st = t.station == st
        t.loc[rows_st, "DM_p_holm"] = holm(t.loc[rows_st, "DM_p"])
    t.round(4).to_csv(C.TAB_DIR / "results_overall.csv", index=False)
    return t


def monthly_table(p):
    keep = p[p.model.isin(["Persistence", "Ridge", "Ensemble", "Ensemble+BC", "GBM", "GBM-robust"])]
    t = (
        keep.assign(ae=(keep.ot_pred - keep.ot_true).abs())
        .groupby(["station", "framing", "model", "fold"])
        .ae.mean()
        .reset_index(name="MAE")
    )
    t.round(4).to_csv(C.TAB_DIR / "results_monthly.csv", index=False)
    return t


def interval_table():
    t = pd.read_csv(C.TAB_DIR / "intervals_folds.csv")
    agg = (
        t.groupby(["station", "framing"])
        .apply(
            lambda g: pd.Series(
                {
                    "coverage_raw_qr": np.average(g.coverage_raw_qr, weights=g.n),
                    "coverage_cqr": np.average(g.coverage_cqr, weights=g.n),
                    "width_raw_qr": np.average(g.width_raw_qr, weights=g.n),
                    "width_cqr": np.average(g.width_cqr, weights=g.n),
                    "target_coverage": 1 - C.INTERVAL_ALPHA,
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    agg.round(3).to_csv(C.TAB_DIR / "results_intervals.csv", index=False)
    return agg


def _alert_inputs(st):
    """Proxy alarm level (development data) and the truth series for alert scoring.

    The truth starts H hours before the evaluation year so that hot-spell onsets early in
    July can be checked for the required H quiet hours before them; invalid hours are NaN.
    """
    df = load_station(st)
    flags = quality_flags(df)
    thr = development(df.OT).quantile(C.ALERT_QUANTILE)
    first = pd.Timestamp(C.EVAL_START)
    truth = df.OT.where(~flags.bad_target)[first - pd.Timedelta(hours=C.H) : C.EVAL_END]
    return thr, truth, first


def alert_table(p_all):
    rows = []
    for st in C.STATIONS:
        thr, truth, first = _alert_inputs(st)
        ps = p_all[p_all.station == st]
        rows.append(
            {
                "station": st,
                "system": "Threshold alarm on current OT (status quo)",
                "threshold": thr,
                **evaluate_alerts([], [], truth, thr, scored_from=first),
                "recall_early": 0.0,
                "median_lead_h": 0.0,
            }
        )
        for fr, m, use_hi in [
            ("-", "Persistence", False),
            ("B", "Ridge", False),
            ("B", "GBM", True),
            ("C", "Ridge", False),
            ("C", "Ensemble", False),
            ("C", "GBM-peak", False),
            ("C", "GBM", True),
        ]:
            d = ps[(ps.framing == fr) & (ps.model == m)].sort_values("issue_time")
            pred = d.hi.fillna(d.ot_pred) if use_hi else d.ot_pred
            name = label(fr, m) + (" upper 90% bound" if use_hi else " point forecast")
            rows.append(
                {
                    "station": st,
                    "system": f"24 h forecast alarm: {name}",
                    "threshold": thr,
                    **evaluate_alerts(d.issue_time, pred.to_numpy(), truth, thr, scored_from=first),
                }
            )
    t = pd.DataFrame(rows)
    t.round(3).to_csv(C.TAB_DIR / "results_alerts.csv", index=False)
    return t


def alert_tradeoff(p_all, st="ETTh2", model="Ensemble"):
    """Sweep the alert margin: alert when forecast > threshold - margin."""
    thr, truth, first = _alert_inputs(st)
    ps = p_all[p_all.station == st]
    rows = []
    for fr in ("B", "C"):
        d = ps[(ps.framing == fr) & (ps.model == model)].sort_values("issue_time")
        for margin in np.arange(0, 10.5, 1.0):
            pred = d.ot_pred.to_numpy() + margin
            r = evaluate_alerts(d.issue_time, pred, truth, thr, scored_from=first)
            rows.append(
                {"framing": fr, "model": model, "rule": "point + margin", "margin": margin, **r}
            )
        g = ps[(ps.framing == fr) & (ps.model == "GBM")].sort_values("issue_time")
        upper = g.hi.fillna(g.ot_pred).to_numpy()
        r = evaluate_alerts(g.issue_time, upper, truth, thr, scored_from=first)
        rows.append(
            {"framing": fr, "model": "GBM", "rule": "CQR upper bound", "margin": np.nan, **r}
        )
    t = pd.DataFrame(rows)
    t.round(3).to_csv(C.TAB_DIR / "results_alert_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    color = FAMILY_COLORS["Ensemble"]
    for fr, ls, a in (("B", "--", 0.45), ("C", "-", 1.0)):
        d = t[(t.framing == fr) & (t.rule == "point + margin")]
        ax.plot(
            d.false_alarms_per_month,
            d.recall_useful * 100,
            ls=ls,
            marker="o",
            ms=3,
            color=color,
            alpha=a,
            label=f"{model} ({fr}) + margin 0–10 °C",
        )
        if fr == "C":
            for _, r in d[d.margin.isin([2, 4, 5, 6])].iterrows():
                ax.annotate(
                    f"{r.margin:.0f} °C",
                    (r.false_alarms_per_month, r.recall_useful * 100),
                    textcoords="offset points",
                    xytext=(6, -10),
                    fontsize=7,
                    color=color,
                )
        u = t[(t.framing == fr) & (t.rule == "CQR upper bound")].iloc[0]
        ax.scatter(
            u.false_alarms_per_month,
            u.recall_useful * 100,
            marker="*",
            s=110,
            zorder=5,
            color=MODEL_COLORS["GBM"],
            alpha=a,
        )
    u = t[(t.framing == "C") & (t.rule == "CQR upper bound")].iloc[0]
    ax.annotate(
        "★ alarm on the 90 % upper bound (CQR)",
        (u.false_alarms_per_month, u.recall_useful * 100),
        textcoords="offset points",
        xytext=(12, -34),
        fontsize=7,
        color=MODEL_COLORS["GBM"],
        arrowprops={"arrowstyle": "-", "color": MODEL_COLORS["GBM"], "lw": 0.6},
    )
    ax.scatter([0], [0], color=INK, zorder=5)
    ax.annotate(
        "Today's threshold alarm (no advance warning)",
        (0, 0),
        textcoords="offset points",
        xytext=(8, 4),
        fontsize=7,
    )
    ax.set_xlabel("False alarms per month (consecutive alert hours count as one alarm)")
    ax.set_ylabel("Hot spells warned ≥12 h ahead [%]")
    ax.set_title(f"{st}: early-warning trade-off (proxy alarm level {thr:.1f})", loc="left")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "res_alert_tradeoff")
    return t


def noise_table(p):
    rows = []
    for st in C.STATIONS:
        ps = p[p.station == st]
        base = point_metrics(*ps[ps.model == "Persistence"][["ot_true", "ot_pred"]].T.to_numpy())[
            "MAE"
        ]
        for fam in ("Ridge", "GBM"):
            strict = point_metrics(
                *ps[(ps.framing == "B") & (ps.model == fam)][["ot_true", "ot_pred"]].T.to_numpy()
            )["MAE"]
            variants = [(0, [fam])] + [
                (round(s * 100), [noise_model_name(fam, s, seed) for seed in NOISE_SEEDS])
                for s in C.NOISE_SIGMAS
            ]
            for pct, names in variants:
                maes = np.array(
                    [
                        point_metrics(d.ot_true, d.ot_pred)["MAE"]
                        for d in (ps[(ps.framing == "C") & (ps.model == n)] for n in names)
                    ]
                )
                skills = 1 - maes / base
                rows.append(
                    {
                        "station": st,
                        "model": fam,
                        "load_plan_error_pct": pct,
                        "n_seeds": len(names),
                        "MAE": maes.mean(),
                        "skill_vs_persistence": skills.mean(),
                        "skill_min": skills.min(),
                        "skill_max": skills.max(),
                        "skill_strict_B": 1 - strict / base,
                    }
                )
    t = pd.DataFrame(rows)
    t.round(4).to_csv(C.TAB_DIR / "results_load_noise.csv", index=False)
    return t


def importance_table():
    t = pd.read_csv(C.TAB_DIR / "importance_folds.csv")
    agg = t.groupby(["station", "framing", "group"]).mae_increase.mean().reset_index()
    agg.round(4).to_csv(C.TAB_DIR / "results_importance.csv", index=False)
    return agg


def greybox_table():
    t = pd.read_csv(C.TAB_DIR / "greybox_folds.csv")
    agg = (
        t.groupby(["station", "framing"])
        .agg(
            tau_median_h=("tau_h", "median"),
            tau_min_h=("tau_h", "min"),
            tau_max_h=("tau_h", "max"),
            coef_s2=("coef_s2", "median"),
        )
        .reset_index()
    )
    agg.round(3).to_csv(C.TAB_DIR / "results_greybox.csv", index=False)
    return agg


# ---------------------------------------------------------------- figures
def fig_skill(t):
    order = [m for m in LEARNED if m in set(t.model)] + ["MovingAvg24"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, st in zip(axes, C.STATIONS, strict=True):
        d = t[t.station == st]
        for i, m in enumerate(order):
            for fr, mk, fill in (("B", "o", "none"), ("C", "o", None), ("-", "s", None)):
                r = d[(d.model == m) & (d.framing == fr)]
                if r.empty:
                    continue
                v = r.skill_vs_persistence.iloc[0] * 100
                col = FAMILY_COLORS[FAMILY[m]]
                ax.scatter(
                    max(v, -15),
                    i,
                    marker=mk,
                    s=42,
                    facecolors=fill or col,
                    edgecolors=col,
                    zorder=3,
                    linewidths=1.4,
                )
                if fr == "C" or fr == "-":
                    sig = r.DM_p_holm.iloc[0] < 0.05 if fr != "-" else False
                    ax.text(
                        max(v, -15) + 0.4,
                        i,
                        f"{v:+.1f}%{'*' if sig else ''}",
                        va="center",
                        fontsize=7,
                        color=INK,
                    )
            rb, rc = d[(d.model == m) & (d.framing == "B")], d[(d.model == m) & (d.framing == "C")]
            if len(rb) and len(rc):
                ax.plot(
                    [
                        max(rb.skill_vs_persistence.iloc[0] * 100, -15),
                        max(rc.skill_vs_persistence.iloc[0] * 100, -15),
                    ],
                    [i, i],
                    color="#cccccc",
                    lw=1,
                    zorder=1,
                )
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(order)
        ax.set_xlim(-16, 14)
        ax.set_xlabel("MAE reduction vs persistence [%]  (clipped at −15)")
        ax.set_title(f"{st}", loc="left", color=STATION_COLORS[st])
        ax.invert_yaxis()
    axes[0].scatter(
        [], [], marker="o", facecolors="none", edgecolors=INK, label="B: strict (data up to t)"
    )
    axes[0].scatter([], [], marker="o", color=INK, label="C: with day-ahead load plan")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "24 h-ahead forecast skill, 12 monthly rolling folds "
        "(* = Diebold–Mariano p < 0.05 vs persistence, Holm-adjusted)",
        x=0.01,
        ha="left",
        fontweight="bold",
        fontsize=11,
    )
    save(fig, "res_skill")


def fig_error_analysis(p):
    """Where the model helps and where it fails: skill by month, and error by temperature."""
    fig, (ax_m, ax_t) = plt.subplots(
        1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.1, 1]}
    )
    for st in C.STATIONS:
        ps = p[p.station == st]
        mae = (ps.ot_pred - ps.ot_true).abs().groupby([ps.model, ps.framing, ps.fold]).mean()
        skill = 1 - mae["Ridge", "C"] / mae["Persistence", "-"]
        ax_m.plot(
            pd.to_datetime(skill.index),
            skill * 100,
            marker="o",
            ms=3,
            color=STATION_COLORS[st],
            label=f"{st}: Ridge (C)",
        )
    ax_m.axhline(0, color=INK, lw=0.8)
    ax_m.set_ylabel("Error reduction vs persistence [%]")
    ax_m.set_title("By month: gains vary; a few months are worse", loc="left")
    ax_m.tick_params(axis="x", rotation=45)
    ax_m.legend(fontsize=8, loc="upper right")

    st = "ETTh2"
    ps = p[p.station == st]
    hot = development(load_station(st).OT).quantile(0.90)
    edges = [-np.inf, 20, 30, 40, hot, np.inf]
    names = ["<20", "20–30", "30–40", f"40–{hot:.0f}", f">{hot:.0f}\n(hottest 10 %)"]
    models = [("-", "Persistence"), ("C", "Ridge"), ("C", "Ensemble")]
    width = 0.27
    for i, (fr, m) in enumerate(models):
        d = ps[(ps.framing == fr) & (ps.model == m)]
        band = pd.cut(d.ot_true, edges, labels=names)
        mae = (d.ot_pred - d.ot_true).abs().groupby(band, observed=False).mean()
        x = np.arange(len(names)) + (i - 1) * width
        ax_t.bar(x, mae, width, color=FAMILY_COLORS[FAMILY[m]], label=label(fr, m))
    ax_t.set_xticks(range(len(names)))
    ax_t.set_xticklabels(names, fontsize=8)
    ax_t.set_xlabel("Measured oil temperature at the target time [°C]")
    ax_t.set_ylabel("MAE [°C]")
    ax_t.set_title(f"{st} by temperature: worse than persistence when hottest", loc="left")
    ax_t.set_ylim(0, ax_t.get_ylim()[1] * 1.2)
    ax_t.legend(fontsize=8, loc="upper center", ncol=3)
    save(fig, "res_error_analysis")


def fig_noise(t):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for st in C.STATIONS:
        d = t[(t.station == st) & (t.model == "Ridge")]
        ax.plot(
            d.load_plan_error_pct,
            d.skill_vs_persistence * 100,
            marker="o",
            color=STATION_COLORS[st],
            label=f"{st}: Ridge (C), mean of {d.n_seeds.max()} plan-error draws",
        )
        ax.fill_between(
            d.load_plan_error_pct,
            d.skill_min * 100,
            d.skill_max * 100,
            color=STATION_COLORS[st],
            alpha=0.15,
            lw=0,
        )
        b = d.skill_strict_B.iloc[0] * 100
        ax.axhline(b, color=STATION_COLORS[st], ls=":", lw=1)
        # Higher line: label above it at the right; lower line: below it at the left.
        above = b >= t.skill_strict_B.max() * 100
        ax.annotate(
            f"{st} without plan (B)",
            (26, b) if above else (0, b),
            textcoords="offset points",
            xytext=(0, 3) if above else (0, -3),
            va="bottom" if above else "top",
            ha="right" if above else "left",
            fontsize=7,
            color=STATION_COLORS[st],
        )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks([0, 5, 10, 20])
    ax.set_xlim(-1, 26)
    ax.set_xlabel("Error of the day-ahead load plan (σ, % of load)")
    ax.set_ylabel("Skill vs persistence [%]")
    ax.set_title("The load plan's value depends on its accuracy", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "res_load_noise")


def fig_importance(agg):
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), sharex=True)
    for ax, st in zip(axes, C.STATIONS, strict=True):
        d = agg[(agg.station == st) & (agg.framing == "C")].sort_values("mae_increase")
        ax.barh(d.group, d.mae_increase, color=STATION_COLORS[st])
        ax.set_title(f"{st}: what GBM (C) relies on", loc="left")
        ax.set_xlabel("MAE increase when the feature group is shuffled")
    save(fig, "res_importance")


def fig_headline(t):
    """The headline: how much lower the 24 h error is than persistence, with and without plan."""
    bars = [
        ("B", "Ridge", "Ridge (B), no plan", "#90b4dc"),
        ("C", "Ridge", "Ridge (C), with plan", FAMILY_COLORS["Linear"]),
        ("C", "Ensemble", "Ensemble (C), with plan", FAMILY_COLORS["Ensemble"]),
    ]
    width = 0.26
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for i, (fr, m, name, colour) in enumerate(bars):
        for j, st in enumerate(C.STATIONS):
            r = t[(t.station == st) & (t.framing == fr) & (t.model == m)].iloc[0]
            x = j + (i - 1) * width
            value = r.skill_vs_persistence * 100
            ax.bar(x, value, width, color=colour, label=name if j == 0 else None)
            star = "*" if r.DM_p_holm < 0.05 else ""
            ax.annotate(
                f"{value:.1f}%{star}",
                (x, value),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8,
            )
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(range(len(C.STATIONS)))
    ax.set_xticklabels(C.STATIONS)
    ax.set_ylabel("24 h error reduction vs persistence [%]")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.set_title(
        "How much lower the 24 h-ahead error is than persistence (* Holm-adjusted p < 0.05)",
        loc="left",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "res_headline")


def fig_coverage():
    """Does a nominal 90 % interval contain the truth 90 % of the time, month by month?"""
    t = pd.read_csv(C.TAB_DIR / "intervals_folds.csv")
    t = t[t.framing == "C"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for st in C.STATIONS:
        d = t[t.station == st]
        months = pd.to_datetime(d.fold)
        colour = STATION_COLORS[st]
        ax.plot(
            months,
            d.coverage_raw_qr * 100,
            ls="--",
            marker="o",
            ms=3,
            color=colour,
            alpha=0.5,
            label=f"{st}: quantile regression",
        )
        ax.plot(
            months, d.coverage_cqr * 100, marker="o", ms=3, color=colour, label=f"{st}: after CQR"
        )
    ax.axhline(100 * (1 - C.INTERVAL_ALPHA), color=INK, ls=":", lw=1)
    ax.annotate("target 90 %", (months.iloc[-1], 90.5), fontsize=7, color=INK, ha="right")
    ax.set_ylabel("Truths inside the interval [%]")
    ax.set_title("Coverage of the 90 % interval by month (framing C)", loc="left")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    save(fig, "res_coverage")


def fig_example(p):
    """One month of forecasts, chosen by rule: the month with the median Ridge (C) skill."""
    st = "ETTh2"
    ps = p[p.station == st]
    mae = (ps.ot_pred - ps.ot_true).abs().groupby([ps.model, ps.framing, ps.fold]).mean()
    skill = (1 - mae["Ridge", "C"] / mae["Persistence", "-"]).sort_values()
    month = skill.index[(len(skill) - 1) // 2]
    d = (
        p[(p.station == st) & (p.framing == "C") & (p.model == "GBM")]
        .set_index("target_time")
        .sort_index()
    )
    e = (
        p[(p.station == st) & (p.framing == "C") & (p.model == "Ridge")]
        .set_index("target_time")
        .sort_index()
    )
    pers = p[(p.station == st) & (p.model == "Persistence")].set_index("target_time").sort_index()
    start = pd.Timestamp(month)
    sl = slice(start, start + pd.Timedelta(days=20))
    win, e, pers = d[sl], e[sl], pers[sl]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.fill_between(
        win.index,
        win.lo,
        win.hi,
        color=MODEL_COLORS["GBM"],
        alpha=0.16,
        lw=0,
        label="90 % interval (CQR)",
    )
    ax.plot(win.index, win.ot_true, color=INK, lw=1.3, label="Measured OT")
    ax.plot(
        e.index,
        e.ot_pred,
        color=FAMILY_COLORS["Linear"],
        lw=1.2,
        label="Ridge (C), issued 24 h earlier",
    )
    ax.plot(pers.index, pers.ot_pred, color=MUTED, lw=1, ls="--", label="Persistence")
    ax.set_title(
        f"{st}, {start:%B %Y} (median month: {skill[month]:+.0%} vs persistence), "
        "24 h-ahead forecasts at their target time",
        loc="left",
        fontsize=10,
    )
    ax.set_ylabel("OT")
    ax.legend(fontsize=8, ncol=3, loc="lower left")
    save(fig, "res_example_forecast")


def main() -> None:
    step("Loading all forecasts and adding the ensemble and bias-corrected variants")
    p_all = load_all()
    p = p_all[~p_all.bad_target]
    done("Forecasts loaded")
    step("Scoring every model against persistence")
    t = overall_table(p)
    fig_headline(t)
    fig_skill(t)
    monthly_table(p)
    fig_error_analysis(p)
    done("Accuracy tables and figures")
    for st in C.STATIONS:
        r = t[(t.station == st) & (t.framing == "C") & (t.model == "Ridge")].iloc[0]
        note(f"{st}: Ridge (C) 24 h error {r.skill_vs_persistence:.1%} lower than persistence")
    step("Intervals, alerts, load-plan robustness and feature importance")
    intervals = interval_table()
    fig_coverage()
    alert_table(p_all)
    tradeoff = alert_tradeoff(p_all)
    fig_noise(noise_table(p))
    fig_importance(importance_table())
    greybox_table()
    fig_example(p)
    done("All report tables and figures")
    c = intervals[intervals.framing == "C"].set_index("station")
    note(
        f"90% interval coverage after CQR: "
        f"ETTh1 {c.coverage_cqr.ETTh1:.0%}, ETTh2 {c.coverage_cqr.ETTh2:.0%}"
    )
    a = tradeoff[(tradeoff.framing == "C") & (tradeoff.margin == 5.0)].iloc[0]
    note(
        f"ETTh2 alerts at a 5 °C margin: {a.recall_useful:.0%} of hot spells warned ≥12 h ahead, "
        f"{a.false_alarms_per_month:.1f} false alarms per month"
    )
    wrote(C.TAB_DIR, C.FIG_DIR)


if __name__ == "__main__":
    main()
