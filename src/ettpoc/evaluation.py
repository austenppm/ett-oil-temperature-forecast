"""Metrics, significance testing and alert simulation."""

from collections.abc import Iterator

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C


def point_metrics(y, yhat) -> dict:
    """MAE, RMSE, bias (forecast minus truth) and sample size."""
    e = np.asarray(yhat) - np.asarray(y)
    return {
        "MAE": np.mean(np.abs(e)),
        "RMSE": np.sqrt(np.mean(e**2)),
        "Bias": np.mean(e),
        "n": len(e),
    }


def diebold_mariano(e_model, e_bench, h: int = C.H) -> tuple[float, float]:
    """DM test on absolute-error loss differential; Newey-West variance with h-1 lags.

    Overlapping h-step forecasts produce autocorrelated errors, so a plain t-test
    would overstate significance (Diebold & Mariano, 1995; Harvey et al., 1997).
    Returns (statistic, two-sided p-value); negative statistic = model better.
    Identical losses give (0, 1); a constant non-zero differential gives (+-inf, 0).
    """
    if h < 1:
        raise ValueError("horizon h must be >= 1")
    d = np.abs(np.asarray(e_model, dtype=float)) - np.abs(np.asarray(e_bench, dtype=float))
    d = d[~np.isnan(d)]
    n = len(d)
    if n <= 2 * h:
        raise ValueError(f"need more than 2h = {2 * h} loss differentials, got {n}")
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    if not var > 0:
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    stat = d.mean() / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjustment of a family of p-values (NaN entries are left as NaN)."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def monthly_folds() -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of the monthly test windows, by target time, across the evaluation year."""
    for start in pd.date_range(C.EVAL_START, C.EVAL_END, freq="MS"):
        yield start, min(start + pd.offsets.MonthBegin(1), pd.Timestamp(C.EVAL_END))


def label_cutoff(test_start: pd.Timestamp, h: int = C.H) -> pd.Timestamp:
    """Refit time of a test window whose targets start at `test_start`.

    Its first forecast is issued at test_start - h, so only labels (target times) up to
    then may be used for fitting, calibration or label-quality decisions.
    """
    return pd.Timestamp(test_start) - pd.Timedelta(hours=h)


def fold_masks(
    target_time: pd.Series, start: pd.Timestamp, end: pd.Timestamp, h: int = C.H
) -> tuple[pd.Series, pd.Series]:
    """(labels available at the refit time, test rows with targets in [start, end))."""
    return target_time <= label_cutoff(start, h), (target_time >= start) & (target_time < end)


def exceedance_episodes(
    ot: pd.Series, threshold: float, merge_gap: int = 6
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Contiguous periods with OT above threshold; gaps shorter than merge_gap hours are merged."""
    above = ot > threshold
    times = ot.index[above.to_numpy()]
    episodes = []
    for t in times:
        if episodes and (t - episodes[-1][1]) <= pd.Timedelta(hours=merge_gap):
            episodes[-1][1] = t
        else:
            episodes.append([t, t])
    return [(start, end) for start, end in episodes]


def evaluate_alerts(
    issue_times,
    predicted_ot_T,
    ot_true_series: pd.Series,
    threshold: float,
    tolerance: int = 6,
    merge_gap: int = C.H,
    scored_from: pd.Timestamp | None = None,
    useful_lead: int = 12,
) -> dict:
    """Simulate a predictive alarm: at issue time t, alert if predicted OT(t+H) > threshold.

    `ot_true_series` is hourly; NaN marks hours whose true value is unknown (invalid data).

    Episodes are "hot spells": exceedances separated by >= merge_gap hours below the
    threshold. With merge_gap = 24 h, an episode onset can never be anticipated simply
    because the previous afternoon was already hot (daily-cycle recurrence). An onset is
    scored only if all merge_gap hours before it are observed (otherwise it is unknown
    whether it really starts a new hot spell). Only onsets at or after `scored_from` count;
    earlier truth serves as history.

    - Episode recall: share of scored episodes (onset s) preceded by >=1 alert issued
      in [s-H, s-1], i.e. a warning that arrives before the temperature crosses the limit.
    - Lead time: s minus the first such alert time (hours). Leads are typically either about
      a day or only an hour or two, so the share warned >= `useful_lead` hours ahead
      (recall_useful) is the operational measure; the median lead mixes the two groups.
    - An alert is true if an observed OT within +/- tolerance hours of its target time
      exceeds the threshold, false if that whole window is observed and none does, and
      unscorable otherwise. Precision and false alert-hours use the scored alerts only.
    - Alarms: consecutive alert hours form one alarm, as an operator would experience it.
      An alarm is false if all its alert hours are false (none true, none unscorable).
    """
    issue_times = pd.DatetimeIndex(issue_times)
    alerts = issue_times[np.asarray(predicted_ot_T) > threshold]
    observed = ot_true_series.notna()
    above = ot_true_series > threshold
    hour = pd.Timedelta(hours=1)

    def fully_observed(lo: pd.Timestamp, hi: pd.Timestamp) -> bool:
        window = observed[lo:hi]
        return len(window) == round((hi - lo) / hour) + 1 and bool(window.all())

    episodes, unscorable_episodes = [], 0
    for onset, end in exceedance_episodes(ot_true_series, threshold, merge_gap=merge_gap):
        if scored_from is not None and onset < scored_from:
            continue
        if fully_observed(onset - merge_gap * hour, onset - hour):
            episodes.append((onset, end))
        else:
            unscorable_episodes += 1

    leads = []
    for onset, _ in episodes:
        in_window = alerts[(alerts >= onset - C.H * hour) & (alerts <= onset - hour)]
        if len(in_window):
            leads.append((onset - in_window.min()) / hour)

    true_alerts = false_alerts = 0
    hot_times = above[above].index
    outcome = []  # per alert: "true", "false" or "unscorable"
    for t in alerts:
        target = t + C.H * hour
        lo, hi = target - tolerance * hour, target + tolerance * hour
        if ((hot_times >= lo) & (hot_times <= hi)).any():
            true_alerts += 1
            outcome.append("true")
        elif fully_observed(lo, hi):
            false_alerts += 1
            outcome.append("false")
        else:
            outcome.append("unscorable")
    alarm_id = np.cumsum(np.r_[True, np.diff(alerts) != hour]) if len(alerts) else np.array([])
    alarms = pd.Series(outcome, dtype=object).groupby(alarm_id)
    false_alarms = int(alarms.apply(lambda o: (o == "false").all()).sum()) if len(alerts) else 0
    months = (issue_times.max() - issue_times.min()) / pd.Timedelta(days=30.44)
    scored = true_alerts + false_alerts
    return {
        "episodes": len(episodes),
        "episodes_unscorable": unscorable_episodes,
        "recall_early": len(leads) / len(episodes) if episodes else np.nan,
        "recall_useful": sum(lead >= useful_lead for lead in leads) / len(episodes)
        if episodes
        else np.nan,
        "median_lead_h": float(np.median(leads)) if leads else np.nan,
        "alerts": len(alerts),
        "alerts_unscorable": len(alerts) - scored,
        "precision": true_alerts / scored if scored else np.nan,
        "false_alerts_per_month": false_alerts / months if len(issue_times) else np.nan,
        "alarms": alarms.ngroups if len(alerts) else 0,
        "false_alarms_per_month": false_alarms / months if len(issue_times) else np.nan,
    }
