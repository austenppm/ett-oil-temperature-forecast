import numpy as np
import pandas as pd
import pytest

from ettpoc import config as C


@pytest.fixture
def station() -> pd.DataFrame:
    """Synthetic hourly station: daily load cycle, OT following load with thermal lag."""
    rng = np.random.default_rng(0)
    index = pd.date_range("2016-07-01", periods=24 * 60, freq="h")
    daily = np.sin(2 * np.pi * index.hour / 24)
    loads = {c: 5 + 2 * daily + rng.normal(0, 0.3, len(index)) for c in C.LOAD_COLS}
    ot = 30 + pd.Series(4 * daily + rng.normal(0, 0.3, len(index))).ewm(halflife=6).mean()
    return pd.DataFrame({**loads, "OT": ot.to_numpy()}, index=index)
