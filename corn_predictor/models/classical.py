"""Classical time-series baselines: ARIMA and GARCH-on-returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import warnings

from .base import BasePriceModel


@dataclass
class ARIMABaseline(BasePriceModel):
    """ARIMA on log-prices; forecast is exponentiated back to price level."""

    order: Tuple[int, int, int] = (1, 1, 1)
    name: str = "arima"
    _model_fit: object = field(default=None, init=False)

    def fit(self, prices: np.ndarray, **kwargs) -> "ARIMABaseline":
        from statsmodels.tsa.arima.model import ARIMA

        prices = np.asarray(prices, dtype=float)
        log_p = np.log(prices)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model_fit = ARIMA(log_p, order=self.order).fit()
        return self

    def predict_next(self, history: np.ndarray) -> float:
        from statsmodels.tsa.arima.model import ARIMA

        history = np.asarray(history, dtype=float)
        log_p = np.log(history)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ARIMA(log_p, order=self.order).fit()
            forecast = fit.forecast(1)
            value = float(np.asarray(forecast).ravel()[0])
        return float(np.exp(value))


@dataclass
class GARCHBaseline(BasePriceModel):
    """GARCH(1,1) mean model on log-returns; next price uses conditional mean."""

    name: str = "garch"
    mean_return_: float = 0.0
    last_vol_: float = 0.0

    def fit(self, prices: np.ndarray, **kwargs) -> "GARCHBaseline":
        from arch import arch_model

        prices = np.asarray(prices, dtype=float)
        # arch prefers percent returns for numerical stability
        rets = 100.0 * np.diff(np.log(prices))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            am = arch_model(rets, mean="Constant", vol="Garch", p=1, q=1, rescale=False)
            res = am.fit(disp="off")
        self.mean_return_ = float(res.params.get("mu", 0.0) / 100.0)
        self.last_vol_ = float(np.sqrt(res.conditional_volatility[-1]) / 100.0)
        self._res = res
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 10:
            return float(history[-1])
        # Refit lightly on history for adaptive mean
        try:
            self.fit(history)
        except Exception:
            pass
        return float(history[-1] * np.exp(self.mean_return_))

    def summary(self):
        return {
            "name": self.name,
            "mean_return": self.mean_return_,
            "last_vol": self.last_vol_,
        }


@dataclass
class OHLCGaussianHMM(BasePriceModel):
    """Gaussian HMM on OHLC-derived features: return, range, close-location."""

    n_states: int = 3
    n_iter: int = 200
    random_state: int = 24
    name: str = "ohlc_gaussian_hmm"
    model_: object = field(default=None, init=False)
    _feature_means: Optional[np.ndarray] = field(default=None, init=False)

    def fit_ohlc(self, ohlc: np.ndarray) -> "OHLCGaussianHMM":
        """``ohlc`` array with columns [open, high, low, close]."""
        from hmmlearn.hmm import GaussianHMM

        feats = self._features_from_ohlc(ohlc)
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="diag",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(feats)
        self.model_ = model
        self._feature_means = model.means_.copy()
        self._train_close = np.asarray(ohlc[:, 3], dtype=float)
        return self

    def fit(self, prices: np.ndarray, **kwargs) -> "OHLCGaussianHMM":
        ohlc = kwargs.get("ohlc")
        if ohlc is None:
            # Fall back to synthetic OHLC from close-only series
            p = np.asarray(prices, dtype=float)
            ohlc = np.column_stack([p, p, p, p])
        return self.fit_ohlc(np.asarray(ohlc, dtype=float))

    @staticmethod
    def _features_from_ohlc(ohlc: np.ndarray) -> np.ndarray:
        ohlc = np.asarray(ohlc, dtype=float)
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
        rng = (h - l) / np.maximum(c, 1e-8)
        # Close location value in the bar
        clv = np.where(h > l, (2 * c - h - l) / (h - l), 0.0)
        # Drop the first artificial return row alignment: keep all rows for hmmlearn
        return np.column_stack([log_ret, rng, clv])[1:]

    def predict_next(self, history: np.ndarray) -> float:
        # Without OHLC in history, degrade to return-mean mixture using close-only
        history = np.asarray(history, dtype=float)
        if self.model_ is None or len(history) < 3:
            return float(history[-1])
        # Rebuild pseudo-OHLC from closes
        ohlc = np.column_stack([history, history, history, history])
        feats = self._features_from_ohlc(ohlc)
        posteriors = self.model_.predict_proba(feats)
        next_state = posteriors[-1] @ self.model_.transmat_
        exp_ret = float(next_state @ self.model_.means_[:, 0])
        return float(history[-1] * np.exp(exp_ret))

    def predict_next_ohlc(self, ohlc: np.ndarray) -> float:
        ohlc = np.asarray(ohlc, dtype=float)
        feats = self._features_from_ohlc(ohlc)
        posteriors = self.model_.predict_proba(feats)
        next_state = posteriors[-1] @ self.model_.transmat_
        exp_ret = float(next_state @ self.model_.means_[:, 0])
        return float(ohlc[-1, 3] * np.exp(exp_ret))


@dataclass
class EnsembleForecaster(BasePriceModel):
    """Simple average of several fitted one-step models."""

    models: Tuple[BasePriceModel, ...] = ()
    weights: Optional[Tuple[float, ...]] = None
    name: str = "ensemble"

    def fit(self, prices: np.ndarray, **kwargs) -> "EnsembleForecaster":
        for m in self.models:
            m.fit(prices, **kwargs)
        if self.weights is None:
            self.weights = tuple(1.0 / len(self.models) for _ in self.models)
        return self

    def predict_next(self, history: np.ndarray) -> float:
        preds = np.array([m.predict_next(history) for m in self.models], dtype=float)
        w = np.asarray(self.weights, dtype=float)
        w = w / w.sum()
        return float(np.dot(w, preds))
