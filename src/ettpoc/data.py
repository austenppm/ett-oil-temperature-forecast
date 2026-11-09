"""Loading the ETT data and flagging data-quality problems.

Two kinds of flags are produced:
- retrospective flags (use the whole series) -> only for EDA and for deciding which
  target values are trustworthy enough to score against;
- causal flags (use data up to the current hour only) -> safe to use as model inputs
  or for run-time fallbacks.
"""

import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

USEFUL_LOADS = ("HUFL", "MUFL", "LUFL")


def ensure_csv(name: str) -> Path:
    """Return the local CSV path, downloading from upstream if missing; verify SHA-256."""
    path = C.DATA_DIR / f"{name}.csv"
    if not path.exists():
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {name} from {C.UPSTREAM_URL.format(name=name)}")
        urllib.request.urlretrieve(C.UPSTREAM_URL.format(name=name), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != C.SHA256[name]:
        raise ValueError(f"{path} has unexpected SHA-256 {digest}; expected {C.SHA256[name]}")
    return path


def load_station(name: str) -> pd.DataFrame:
    """Hourly data of one station, indexed by timestamp; fails on gaps or duplicates."""
    df = pd.read_csv(ensure_csv(name), parse_dates=["date"]).set_index("date").sort_index()
    steps = df.index.to_series().diff().dropna()
    if not df.index.is_unique or (steps != pd.Timedelta(hours=1)).any():
        raise ValueError(f"{name}: expected a gap-free hourly index")
    return df


def run_length(s: pd.Series) -> pd.Series:
    """Number of consecutive identical values ending at each timestamp (causal)."""
    groups = (s != s.shift()).cumsum()
    return s.groupby(groups).cumcount() + 1


def total_run_length(s: pd.Series) -> pd.Series:
    """Length of the whole constant run each timestamp belongs to (retrospective)."""
    groups = (s != s.shift()).cumsum()
    return s.groupby(groups).transform("size")


def implausible_constant(s: pd.Series, run: pd.Series) -> pd.Series:
    """A long constant run is a stuck sensor, except exact zeros on a 'useless'
    (reactive) channel, which can be genuine (fully compensated reactive load)."""
    plausible_zero = (s == 0) & (s.name not in USEFUL_LOADS)
    return (run >= C.STUCK_LOAD_HOURS) & ~plausible_zero


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Boolean data-quality flags per hour (see the module docstring for the two kinds)."""
    f = pd.DataFrame(index=df.index)
    # Retrospective: whole stuck runs, and spike-and-return glitches in OT.
    for c in C.LOAD_COLS:
        f[f"stuck_{c}"] = implausible_constant(df[c], total_run_length(df[c]))
    f["load_stuck"] = f[[f"stuck_{c}" for c in C.LOAD_COLS]].any(axis=1)
    f["ot_stuck"] = total_run_length(df.OT) >= C.STUCK_OT_HOURS
    local_median = df.OT.rolling(5, center=True, min_periods=3).median()
    f["ot_spike"] = (df.OT - local_median).abs() > C.SPIKE_DEG
    # Every 31st day of a month is a single value repeated on ALL channels for 24 h
    # (found in EDA): almost certainly a fill artefact of the published dataset.
    f["all_channels_flat"] = pd.concat(
        [total_run_length(df[c]) >= C.STUCK_OT_HOURS for c in df.columns], axis=1
    ).all(axis=1)
    f["bad_target"] = f.ot_stuck | f.ot_spike | f.all_channels_flat
    # Physical consistency: high-voltage load should equal middle + low load.
    f["balance_resid"] = df.HUFL - (df.MUFL + df.LUFL)
    # Causal: a load channel that has not changed for the last 24 h.
    f["load_stuck_causal"] = pd.concat(
        [implausible_constant(df[c], run_length(df[c])) for c in C.LOAD_COLS], axis=1
    ).any(axis=1)
    return f


def clean_ot_causal(ot: pd.Series) -> pd.Series:
    """Replace single-hour OT glitches using only past data.

    A reading that deviates more than SPIKE_DEG from the median of the previous six
    accepted readings is treated as a glitch and replaced by the last accepted value.
    A genuine level change is accepted once two consecutive raw readings agree
    (i.e. with a one-hour delay), so the filter never looks into the future.
    """
    values = ot.to_numpy(dtype=float)
    accepted = values.copy()
    for i in range(6, len(values)):
        ref = np.median(accepted[i - 6 : i])
        previous_rejected = accepted[i - 1] != values[i - 1]
        confirmed = previous_rejected and abs(values[i] - values[i - 1]) < C.SPIKE_DEG / 2
        if abs(values[i] - ref) > C.SPIKE_DEG and not confirmed:
            accepted[i] = accepted[i - 1]
    return pd.Series(accepted, index=ot.index, name="OT_clean")
