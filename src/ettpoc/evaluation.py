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
    """
    d = np.abs(e_model) - np.abs(e_bench)
    d = d[~np.isnan(d)]
    n = len(d)
    dc = d - d.mean()
    gamma = [np.dot(dc[k:], dc[: n - k]) / n for k in range(h)]
    var = gamma[0] + 2 * sum((1 - k / h) * gamma[k] for k in range(1, h))
    stat = d.mean() / np.sqrt(var / n)
    # Harvey-Leybourne-Newbold small-sample correction
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * stats.t.sf(abs(stat), df=n - 1)
    return stat, p


def monthly_folds() -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of the monthly test windows, by target time, across the evaluation year."""
    for start in pd.date_range(C.EVAL_START, C.EVAL_END, freq="MS"):
        yield start, min(start + pd.offsets.MonthBegin(1), pd.Timestamp(C.EVAL_END))


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
) -> dict:
    """Simulate a predictive alarm: at issue time t, alert if predicted OT(t+H) > threshold.

    Episodes are "hot spells": exceedances separated by >= merge_gap hours below the
    threshold. With merge_gap = 24 h, an episode onset can never be anticipated simply
    because the previous afternoon was already hot (daily-cycle recurrence).

    - Episode recall: share of exceedance episodes (onset s) preceded by >=1 alert issued
      in [s-H, s-1], i.e. a warning that arrives before the temperature crosses the limit.
    - Lead time: s minus the first such alert time (hours).
    - Alert precision: share of alerts for which OT actually exceeds the threshold
      within +/- tolerance hours of the predicted target time.
    """
    issue_times = pd.DatetimeIndex(issue_times)
    alerts = issue_times[np.asarray(predicted_ot_T) > threshold]
    above = ot_true_series > threshold
    episodes = exceedance_episodes(ot_true_series, threshold, merge_gap=merge_gap)

    hour = pd.Timedelta(hours=1)
    leads = []
    for onset, _ in episodes:
        in_window = alerts[(alerts >= onset - C.H * hour) & (alerts <= onset - hour)]
        if len(in_window):
            leads.append((onset - in_window.min()) / hour)

    true_alerts = 0
    hot_times = above[above].index
    for t in alerts:
        target = t + C.H * hour
        near = (hot_times >= target - tolerance * hour) & (hot_times <= target + tolerance * hour)
        true_alerts += near.any()
    months = (issue_times.max() - issue_times.min()) / pd.Timedelta(days=30.44)
    return {
        "episodes": len(episodes),
        "recall_early": len(leads) / len(episodes) if episodes else np.nan,
        "median_lead_h": float(np.median(leads)) if leads else np.nan,
        "alerts": len(alerts),
        "precision": true_alerts / len(alerts) if len(alerts) else np.nan,
        "false_alerts_per_month": (len(alerts) - true_alerts) / months,
    }
