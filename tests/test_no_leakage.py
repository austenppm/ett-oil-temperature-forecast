"""Causality: nothing computed for issue time t may change when the future changes."""

import numpy as np
import pandas as pd
import pytest

from ettpoc import config as C
from ettpoc.data import clean_ot_causal, quality_flags
from ettpoc.experiment import tuning_split
from ettpoc.features import S2_HALFLIVES, apparent_power_sq, build, build_features


def scramble_after(df: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Copy of df with every value after `cut` replaced by noise."""
    out = df.copy()
    future = out.index > cut
    rng = np.random.default_rng(1)
    out.loc[future] = rng.normal(50, 20, (future.sum(), out.shape[1]))
    return out


@pytest.mark.parametrize("framing", C.FRAMINGS)
def test_features_ignore_data_after_their_information_set(station, framing):
    # Framing B may use data up to t; framing C additionally the load plan up to t + H.
    cut = station.index[24 * 40]
    issue_times = station.index[station.index <= cut - pd.Timedelta(hours=C.H)]
    horizon_visible = C.H if framing == "C" else 0
    future_cut = issue_times[-1] + pd.Timedelta(hours=horizon_visible)

    def features(df):
        X, _, _ = build(df, clean_ot_causal(df.OT), quality_flags(df), framing)
        return X.loc[X.index.isin(issue_times)]

    original = features(station)
    scrambled = features(scramble_after(station, future_cut))
    assert len(original) > 24 * 20
    pd.testing.assert_frame_equal(original, scrambled)


def test_causal_cleaning_and_flags_ignore_the_future(station):
    cut = station.index[24 * 40]
    scrambled = scramble_after(station, cut)
    past = slice(None, cut)
    pd.testing.assert_series_equal(
        clean_ot_causal(station.OT)[past], clean_ot_causal(scrambled.OT)[past]
    )
    pd.testing.assert_series_equal(
        quality_flags(station).load_stuck_causal[past],
        quality_flags(scrambled).load_stuck_causal[past],
    )


def test_target_is_the_future_change(station):
    X, y, _ = build(station, station.OT, quality_flags(station), "B")
    t = X.index[100]
    expected = station.OT[t + pd.Timedelta(hours=C.H)] - station.OT[t]
    assert y[t] == pytest.approx(expected)


def test_noisy_plan_leaves_measured_history_unchanged(station):
    exact = build_features(station, station.OT, "C")
    noisy = build_features(station, station.OT, "C", load_sigma=0.2)
    history = [c for c in exact.columns if c.endswith("_now") or c.startswith("ot_")]
    pd.testing.assert_frame_equal(exact[history], noisy[history])
    # Without plan error the planned EWMA equals the EWMA of the actual load at T.
    s2 = apparent_power_sq(station)
    for hl in S2_HALFLIVES:
        actual_T = s2.ewm(halflife=hl, adjust=False).mean().shift(-C.H).reindex(exact.index)
        np.testing.assert_allclose(exact[f"s2_ewm{hl}_T"], actual_T, rtol=1e-9)


def test_tuning_uses_only_labels_observed_before_evaluation(development_station):
    df = development_station
    _, _, meta = build(df, df.OT, quality_flags(df), "B")
    train, val = tuning_split(df, meta)
    tt = meta.target_time
    assert tt[train].max() <= pd.Timestamp(C.TUNE_VAL_START) - pd.Timedelta(hours=C.H)
    assert tt[val].min() >= pd.Timestamp(C.TUNE_VAL_START)
    assert tt[val].max() <= C.DEV_END
