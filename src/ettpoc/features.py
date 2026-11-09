"""Feature construction for the 24 h-ahead oil-temperature forecast.

Every row is one forecast issued at time t for target time T = t + H.

Framing B (strict):   only information measured up to t (plus the calendar of T).
Framing C (what-if):  additionally the load profile over (t, T], interpreted as the
                      operator's day-ahead load plan / load forecast. Athena's brief
                      explicitly allows using features at the target time T.

The model target is the CHANGE in oil temperature, dOT = OT(T) - OT(t). Tree models
cannot extrapolate beyond the target range they were trained on, and OT drifts between
years; the 24 h change is far more stationary than the level.
"""

import numpy as np
import pandas as pd

from . import config as C

S2_HALFLIVES = (3, 12, 48)  # hours; span the thermal time constants seen in EDA


def apparent_power_sq(df: pd.DataFrame) -> pd.Series:
    """Proxy for current^2 on the HV side: useful (active) ^2 + useless (reactive) ^2.

    Resistive winding losses, the main heat source, scale with current squared
    (IEC 60076-7 thermal model), so S^2 is the physically motivated load input.
    """
    return df.HUFL**2 + df.HULL**2


def cyclic(values: np.ndarray, period: float, name: str, harmonic: int = 1) -> dict:
    """Sine/cosine encoding of a periodic quantity (e.g. hour of day, day of year)."""
    angle = 2 * np.pi * harmonic * values / period
    suffix = "" if harmonic == 1 else str(harmonic)
    return {f"{name}{suffix}_sin": np.sin(angle), f"{name}{suffix}_cos": np.cos(angle)}


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Daily and annual cycle plus a weekend flag for each timestamp."""
    return pd.DataFrame(
        {
            **cyclic(index.hour.to_numpy(), 24, "hour"),
            **cyclic(index.dayofyear.to_numpy(), 365.25, "doy"),
            "weekend": (index.dayofweek >= 5).astype(float),
        },
        index=index,
    )


def noisy_loads(df: pd.DataFrame, sigma: float, seed: int = C.SEED) -> pd.DataFrame:
    """Simulate an imperfect day-ahead load plan: daily bias + hourly noise, both ~sigma."""
    if sigma <= 0:
        return df[C.LOAD_COLS]
    rng = np.random.default_rng(seed)
    days = df.index.normalize()
    day_bias = (
        pd.Series(rng.normal(0, sigma, days.nunique()), index=days.unique())
        .reindex(days)
        .to_numpy()
    )
    hourly = rng.normal(0, sigma, (len(df), 1))
    factor = 1 + day_bias[:, None] / np.sqrt(2) + hourly / np.sqrt(2)
    return df[C.LOAD_COLS] * factor


def build(
    df: pd.DataFrame,
    ot_clean: pd.Series,
    flags: pd.DataFrame,
    framing: str,
    load_sigma: float = 0.0,
    h: int = C.H,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Return X (features), y (target dOT), meta (times, truth, flags) aligned on issue time t."""
    ot = ot_clean
    X = pd.DataFrame(index=df.index)

    # --- oil-temperature history (known at t) ---
    X["ot_now"] = ot
    for k in (1, 2, 3, 6, 12, 24, 48, 168):
        X[f"ot_diff_{k}h"] = ot - ot.shift(k)
    X["ot_dev_mean24"] = ot.rolling(24).mean() - ot
    X["ot_dev_mean168"] = ot.rolling(168).mean() - ot
    X["ot_range24"] = ot.rolling(24).max() - ot.rolling(24).min()

    # --- load history (known at t) ---
    s2 = apparent_power_sq(df)
    for c in C.LOAD_COLS:
        X[f"{c}_now"] = df[c]
    X["s2_now"] = s2
    for hl in S2_HALFLIVES:
        X[f"s2_ewm{hl}_now"] = s2.ewm(halflife=hl).mean()
    X["s2_diff_24h"] = s2 - s2.shift(24)

    # --- calendar of the target time (deterministic, always known) ---
    cal_T = calendar(df.index + pd.Timedelta(hours=h))
    cal_T.index = df.index
    X = X.join(cal_T.add_suffix("_T"))

    # --- planned loads over (t, T] (framing C only) ---
    if framing == "C":
        plan = noisy_loads(df, load_sigma)
        s2p = apparent_power_sq(plan)
        for c in C.LOAD_COLS:
            X[f"{c}_T"] = plan[c].shift(-h)
        X["s2_T"] = s2p.shift(-h)
        X["s2_mean_next24"] = s2p.rolling(h).mean().shift(-h)
        X["s2_max_next24"] = s2p.rolling(h).max().shift(-h)
        # Thermal-memory features at T (EWMA of S^2 up to the target time).
        # Approximation for the noise study: the EWMA runs over the perturbed series,
        # so history before t is perturbed too; with halflives <= 48 h the plan
        # window dominates, and with load_sigma == 0 the feature is exact.
        for hl in S2_HALFLIVES:
            X[f"s2_ewm{hl}_T"] = s2p.ewm(halflife=hl).mean().shift(-h)
            X[f"s2_ewm{hl}_change"] = X[f"s2_ewm{hl}_T"] - X[f"s2_ewm{hl}_now"]
    elif framing != "B":
        raise ValueError(f"unknown framing {framing!r}")

    meta = pd.DataFrame(index=df.index)
    meta["target_time"] = df.index + pd.Timedelta(hours=h)
    meta["ot_now"] = ot
    meta["ot_true"] = df.OT.shift(-h)
    meta["bad_target"] = flags.bad_target.shift(-h, fill_value=True)
    # Run-time fallback trigger: a load sensor looks stuck at issue time
    # (or, in framing C, the load "plan" at T comes from a stuck channel).
    stuck = flags.load_stuck_causal
    meta["input_flagged"] = stuck if framing == "B" else (stuck | stuck.shift(-h, fill_value=True))
    y = (meta.ot_true - meta.ot_now).rename("d_ot")

    valid = X.notna().all(axis=1) & y.notna()
    return X[valid], y[valid], meta[valid]


ABSOLUTE_LEVEL_COLS = ["ot_now"]  # dropped by the drift-robust models


def raw_window(
    df: pd.DataFrame,
    ot_clean: pd.Series,
    index: pd.Index,
    framing: str,
    h: int = C.H,
    lookback: int = 168,
) -> pd.DataFrame:
    """Raw lag window for a DLinear-style linear model (Zeng et al., 2023).

    Past `lookback` hours of OT (relative to OT(t)) and of S^2; framing C adds the
    planned S^2 over (t, T]. No hand-crafted features.
    """
    s2 = apparent_power_sq(df)
    s2 = s2 / s2[: C.EVAL_START].mean()
    cols = {f"ot_lag{k}": ot_clean.shift(k) - ot_clean for k in range(1, lookback)}
    cols.update({f"s2_lag{k}": s2.shift(k) for k in range(lookback)})
    if framing == "C":
        cols.update({f"s2_lead{j}": s2.shift(-j) for j in range(1, h + 1)})
    return pd.DataFrame(cols, index=df.index).reindex(index)


GROUP_ORDER = ("OT history", "Past loads", "Planned loads (t,T]", "Calendar")
CALENDAR_PREFIXES = ("hour", "doy", "weekend")


def feature_group(column: str) -> str:
    """Which information source a feature comes from (for grouped permutation importance)."""
    if column.split("_")[0] in CALENDAR_PREFIXES:
        return "Calendar"
    if column.startswith("ot_"):
        return "OT history"
    if column.endswith("_now") or column == "s2_diff_24h":
        return "Past loads"
    return "Planned loads (t,T]"


def feature_groups(columns) -> dict[str, list[str]]:
    """Feature columns keyed by group (in GROUP_ORDER, empty groups omitted)."""
    groups: dict[str, list[str]] = {g: [] for g in GROUP_ORDER}
    for column in columns:
        groups[feature_group(column)].append(column)
    return {g: cols for g, cols in groups.items() if cols}
