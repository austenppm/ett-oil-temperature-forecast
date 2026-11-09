import numpy as np
import pandas as pd

from ettpoc import config as C
from ettpoc.data import (
    bad_target_causal,
    bad_targets_known_at,
    clean_ot_causal,
    development,
    quality_flags,
)


def test_single_hour_spike_is_removed_but_a_level_shift_is_kept():
    index = pd.date_range("2017-01-01", periods=40, freq="h")
    ot = pd.Series(20.0, index=index)
    ot.iloc[10] = 40.0  # one-hour glitch
    ot.iloc[25:] = 35.0  # genuine, persistent change
    clean = clean_ot_causal(ot)
    assert clean.iloc[10] == 20.0
    assert clean.iloc[26:].eq(35.0).all()  # accepted after one confirming reading


def test_stuck_load_sensor_is_flagged_but_zero_reactive_load_is_not(station):
    df = station.copy()
    df.loc[df.index[100:140], "HUFL"] = 7.0  # useful channel frozen for 40 h
    df.loc[df.index[300:340], "HULL"] = 0.0  # reactive load fully compensated
    flags = quality_flags(df)
    assert flags.stuck_HUFL.iloc[100:140].all()
    assert not flags.stuck_HULL.any()
    # The causal flag only fires once the run has lasted STUCK_LOAD_HOURS.
    causal = flags.load_stuck_causal.iloc[100:140].to_numpy()
    assert not causal[: C.STUCK_LOAD_HOURS - 1].any() and causal[C.STUCK_LOAD_HOURS - 1 :].all()


def test_day_frozen_on_every_channel_is_a_bad_target(station):
    df = station.copy()
    frozen = df.index[(df.index >= "2016-07-31") & (df.index < "2016-08-01")]
    df.loc[frozen] = np.tile(df.loc[frozen[0]].to_numpy(), (len(frozen), 1))
    flags = quality_flags(df)
    assert flags.all_channels_flat[frozen].all() and flags.bad_target[frozen].all()


def test_label_quality_known_at_a_cutoff_ignores_later_data(station):
    cutoff = station.index[24 * 30]
    later = station.copy()
    # A stuck run that starts before the cutoff only becomes long enough after it.
    stuck = (later.index > cutoff - pd.Timedelta(hours=6)) & (
        later.index <= cutoff + pd.Timedelta(hours=20)
    )
    later.loc[stuck, "OT"] = 33.0
    known = bad_targets_known_at(later, cutoff)
    assert known.index.max() == cutoff
    assert not known.any()  # 6 constant hours at the cutoff are not yet a stuck sensor
    assert quality_flags(later).bad_target[stuck].all()  # the full series says they are


def test_causal_bad_target_ignores_the_future(station):
    cut = station.index[24 * 40]
    scrambled = station.copy()
    scrambled.loc[scrambled.index > cut, "OT"] = 99.0
    pd.testing.assert_series_equal(
        bad_target_causal(station)[:cut], bad_target_causal(scrambled)[:cut]
    )


def test_development_ends_at_the_first_evaluation_issue_time():
    s = pd.Series(1.0, index=pd.date_range("2017-06-29", "2017-07-02", freq="h"))
    dev = development(s)
    assert dev.index.max() == pd.Timestamp(C.EVAL_START) - pd.Timedelta(hours=C.H)
