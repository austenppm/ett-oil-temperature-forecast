"""Forecasting models.

All models predict dOT = OT(T) - OT(t); the forecast is OT(t) + dOT_hat.

- Persistence / MovingAverage: the benchmarks every model must beat.
- Ridge:      linear model on physics-motivated features.
- GBM:        histogram gradient boosting (LightGBM-style), captures interactions.
- GreyBox:    discretised first-order top-oil thermal model (IEC 60076-7 structure)
              with parameters learned from data; ambient temperature is not measured,
              so it is represented by calendar harmonics and the weekly mean OT.
- CQR:        conformalized quantile regression (Romano et al., 2019) for 90 % intervals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import apparent_power_sq, cyclic


def make_ridge(alpha: float):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


class ChronoMLP:
    """MLP (64-32) with early stopping on the chronologically last part of the training data.

    scikit-learn's built-in early stopping holds out a random 10 %, which for overlapping
    time-series targets is almost the training data itself. Here the hold-out is the last
    10 % of rows, separated from the fitting rows by a gap of `gap` rows (the horizon), and
    the scaler is fitted on the fitting rows only.
    """

    def __init__(self, validation_fraction=0.1, gap=C.H, max_epochs=200, patience=10):
        self.validation_fraction = validation_fraction
        self.gap = gap
        self.max_epochs = max_epochs
        self.patience = patience

    def fit(self, X, y) -> "ChronoMLP":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n_val = max(1, int(len(X) * self.validation_fraction))
        fit_rows = slice(0, len(X) - n_val - self.gap)
        val_rows = slice(len(X) - n_val, len(X))
        self.scaler = StandardScaler().fit(X[fit_rows])
        Xf, Xv = self.scaler.transform(X[fit_rows]), self.scaler.transform(X[val_rows])
        self.net = MLPRegressor(
            hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, random_state=C.SEED
        )
        best, best_state, stale = np.inf, None, 0
        for _ in range(self.max_epochs):
            self.net.partial_fit(Xf, y[fit_rows])
            val_mae = float(np.mean(np.abs(self.net.predict(Xv) - y[val_rows])))
            if val_mae < best - 1e-4:
                best, stale = val_mae, 0
                best_state = (
                    [w.copy() for w in self.net.coefs_],
                    [b.copy() for b in self.net.intercepts_],
                )
            else:
                stale += 1
                if stale >= self.patience:
                    break
        self.net.coefs_, self.net.intercepts_ = best_state
        return self

    def predict(self, X) -> np.ndarray:
        return self.net.predict(self.scaler.transform(np.asarray(X, dtype=float)))


def make_mlp() -> ChronoMLP:
    return ChronoMLP()


def make_gbm(params: dict, quantile: float | None = None):
    """Gradient boosting with tuned `params`; a quantile model if `quantile` is given."""
    kw = {
        "max_iter": 300,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": C.SEED,
    }
    kw.update(params)
    if quantile is not None:
        return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **kw)
    return HistGradientBoostingRegressor(loss="squared_error", **kw)


class GreyBoxThermal:
    """First-order thermal model integrated over the 24 h horizon.

    Continuous form (IEC 60076-7, top oil):  tau * dtheta/dt = theta_ult(t) - theta(t)
    with theta_ult = ambient + rise(load).  Discretised with a = exp(-1/tau) per hour:

        theta(T) = a^H * theta(t) + sum_{j<H} (1-a) a^(H-1-j) * theta_ult(t+j)

    theta_ult is linear in drivers (constant, S^2, weekly-mean OT as ambient proxy,
    daily and annual harmonics), so for a fixed tau the model is a linear regression.
    tau is chosen on the training data from a grid; this yields an interpretable
    thermal time constant per transformer.

    Framing B holds the load at its value at t (no plan); framing C uses the plan.
    """

    TAUS = (1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72)

    def __init__(self, framing: str):
        self.framing = framing
        self.designs: dict[float, tuple[np.ndarray, float]] = {}

    def prepare(
        self, df: pd.DataFrame, ot_clean: pd.Series, index: pd.Index, s2_scale: float
    ) -> "GreyBoxThermal":
        """Precompute the design matrix for every tau on the issue times `index`."""
        h = C.H
        s2 = apparent_power_sq(df) / s2_scale
        hour = df.index.hour.to_numpy()
        varying = pd.DataFrame(
            {
                **cyclic(hour, 24, "hour"),
                **cyclic(hour, 24, "hour", harmonic=2),
                **cyclic(df.index.dayofyear.to_numpy(), 365.25, "doy"),
            },
            index=df.index,
        )
        # Inputs held at their value at t over the horizon (weekly mean OT ~ ambient proxy).
        held = pd.DataFrame({"const": 1.0, "ot_week": ot_clean.rolling(168).mean()}, index=df.index)
        if self.framing == "C":
            varying["s2"] = s2
        else:
            held["s2"] = s2
        self.ot_now = ot_clean.reindex(index).to_numpy()
        for tau in self.TAUS:
            a = np.exp(-1 / tau)
            w = (1 - a) * a ** (h - 1 - np.arange(h))
            integrated = sum(w[j] * varying.shift(-j) for j in range(h))
            design = pd.concat([held * (1 - a**h), integrated], axis=1)
            self.designs[tau] = (design.reindex(index).to_numpy(), a**h)
        self.columns = list(held.columns) + list(varying.columns)
        self.index = index
        return self

    def _rows(self, idx: pd.Index) -> np.ndarray:
        return self.index.get_indexer(idx)

    def fit(self, idx: pd.Index, ot_true: np.ndarray) -> "GreyBoxThermal":
        """Choose tau by in-sample MSE; for each tau the remaining model is linear."""
        r = self._rows(idx)
        best_err = np.inf
        for tau, (design, decay) in self.designs.items():
            z = ot_true - decay * self.ot_now[r]
            model = Ridge(alpha=1e-3, fit_intercept=False).fit(design[r], z)
            err = np.mean((model.predict(design[r]) - z) ** 2)
            if err < best_err:
                best_err, self.tau, self.model = err, tau, model
        return self

    def predict_delta(self, idx: pd.Index) -> np.ndarray:
        r = self._rows(idx)
        design, decay = self.designs[self.tau]
        ot_hat = decay * self.ot_now[r] + self.model.predict(design[r])
        return ot_hat - self.ot_now[r]

    def coefficients(self) -> dict[str, float]:
        return dict(zip(self.columns, self.model.coef_, strict=True))


def conformal_quantile(scores, alpha: float) -> float:
    """The ceil((n + 1)(1 - alpha))-th smallest score (exact split-conformal order statistic)."""
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        raise ValueError(f"{n} calibration scores cannot give {1 - alpha:.0%} coverage")
    return float(scores[k - 1])


class CQR:
    """Conformalized quantile regression for a two-sided (1 - alpha) interval.

    The finite-sample guarantee of Romano et al. (2019) assumes exchangeable data; for a
    time series the coverage is an empirical property, checked on every evaluation fold.
    """

    def __init__(self, gbm_params: dict, alpha: float = C.INTERVAL_ALPHA):
        self.alpha = alpha
        self.lo = make_gbm(gbm_params, quantile=alpha / 2)
        self.hi = make_gbm(gbm_params, quantile=1 - alpha / 2)

    def fit(self, X_tr, y_tr, X_cal, y_cal) -> "CQR":
        """Fit the quantile models on X_tr, then widen them by the conformal score on X_cal."""
        self.lo.fit(X_tr, y_tr)
        self.hi.fit(X_tr, y_tr)
        scores = np.maximum(self.lo.predict(X_cal) - y_cal, y_cal - self.hi.predict(X_cal))
        self.q = conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X, conformal: bool = True) -> tuple[np.ndarray, np.ndarray]:
        q = self.q if conformal else 0.0
        return self.lo.predict(X) - q, self.hi.predict(X) + q
