import numpy as np
import pandas as pd
import pytest

from ettpoc import config as C
from ettpoc.features import apparent_power_sq
from ettpoc.models import CQR, GreyBoxThermal


def test_cqr_reaches_nominal_coverage_under_heteroscedastic_noise():
    rng = np.random.default_rng(0)
    n = 6000
    X = pd.DataFrame({"x": rng.uniform(-2, 2, n)})
    y = np.sin(X.x) + rng.normal(0, 0.2 + 0.4 * np.abs(X.x))  # heteroscedastic noise
    tr, cal, te = slice(0, 2000), slice(2000, 4000), slice(4000, n)
    params = {"learning_rate": 0.1, "max_leaf_nodes": 31, "min_samples_leaf": 5}
    cqr = CQR(params).fit(X[tr], y[tr], X[cal], y[cal])
    lo, hi = cqr.predict(X[te])
    coverage = np.mean((y[te] >= lo) & (y[te] <= hi))
    assert coverage == pytest.approx(1 - C.INTERVAL_ALPHA, abs=0.025)


def test_greybox_recovers_the_thermal_time_constant():
    # Simulate the discretised first-order top-oil model with a known tau.
    tau = 12
    rng = np.random.default_rng(0)
    index = pd.date_range("2016-07-01", periods=24 * 120, freq="h")
    useful = 5 + 2 * np.sin(2 * np.pi * index.hour / 24) + rng.normal(0, 1, len(index))
    df = pd.DataFrame({"HUFL": useful, "HULL": 1.0}, index=index)
    s2 = apparent_power_sq(df).to_numpy()
    a = np.exp(-1 / tau)
    theta = np.empty(len(index))
    theta[0] = 30.0
    for k in range(len(index) - 1):
        theta[k + 1] = a * theta[k] + (1 - a) * (20 + 0.5 * s2[k])
    ot = pd.Series(theta, index=index)

    issue = index[200 : -C.H - 1]
    model = GreyBoxThermal("C").prepare(df, ot, issue, s2_scale=1.0)
    model.fit(issue, ot.shift(-C.H)[issue].to_numpy())
    assert model.tau == tau
    assert model.coefficients()["s2"] > 0
