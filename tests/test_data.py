import numpy as np
import pandas as pd

from ettpoc import config as C
from ettpoc.data import clean_ot_causal, quality_flags


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
