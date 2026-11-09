from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from ettpoc import config as C
from ettpoc.evaluation import (
    diebold_mariano,
    evaluate_alerts,
    exceedance_episodes,
    monthly_folds,
)


def autocorrelated_errors(rng, n, scale):
    """Errors of overlapping 24 h forecasts: a 24-hour moving sum of white noise."""
    return scale * np.convolve(rng.normal(size=n + C.H), np.ones(C.H), "valid")[:n]


def test_diebold_mariano_detects_a_better_model_and_not_an_equal_one():
    rng = np.random.default_rng(0)
    bench = autocorrelated_errors(rng, 5000, 1.0)
    _, p_equal = diebold_mariano(autocorrelated_errors(rng, 5000, 1.0), bench)
    stat, p_better = diebold_mariano(autocorrelated_errors(rng, 5000, 0.7), bench)
    assert p_equal > 0.05
    assert stat < 0 and p_better < 0.01


def test_monthly_folds_tile_the_evaluation_year():
    folds = list(monthly_folds())
    assert len(folds) == 12
    assert folds[0][0] == pd.Timestamp(C.EVAL_START)
    assert folds[-1][1] == pd.Timestamp(C.EVAL_END)
    assert all(end == next_start for (_, end), (next_start, _) in pairwise(folds))


def test_exceedances_closer_than_the_merge_gap_form_one_episode():
    index = pd.date_range("2017-07-01", periods=72, freq="h")
    ot = pd.Series(0.0, index=index)
    ot.iloc[[10, 11, 15, 50]] = 1.0  # 10-15 merge (gap 4 h), 50 is separate
    episodes = exceedance_episodes(ot, threshold=0.5, merge_gap=6)
    assert episodes == [(index[10], index[15]), (index[50], index[50])]


def test_alert_issued_before_onset_counts_with_its_lead_time():
    index = pd.date_range("2017-07-01", periods=24 * 5, freq="h")
    ot = pd.Series(0.0, index=index)
    onset = index[60]
    ot[onset : onset + pd.Timedelta(hours=3)] = 10.0
    issue_times = index[: -C.H]
    predicted = np.zeros(len(issue_times))
    predicted[issue_times.get_loc(onset - pd.Timedelta(hours=C.H))] = 10.0  # predicts the onset
    predicted[issue_times.get_loc(index[0])] = 10.0  # predicts a hot spell that never comes
    result = evaluate_alerts(issue_times, predicted, ot, threshold=5.0)
    assert result["episodes"] == 1
    assert result["recall_early"] == 1.0
    assert result["median_lead_h"] == C.H
    assert result["precision"] == pytest.approx(0.5)
