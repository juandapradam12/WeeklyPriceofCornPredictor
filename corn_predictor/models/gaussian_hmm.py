"""Gaussian HMM on log-returns (continuous emissions).

Complementary to the discrete-emission HMM: instead of quantizing prices,
we model the return process directly with state-dependent Gaussians.
Uses ``hmmlearn`` when available, with a lightweight fallback EM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .base import BasePriceModel

try:
    from hmmlearn.hmm import GaussianHMM as _SklearnGaussianHMM

    HAS_HMMLEARN = True
except Exception:  # pragma: no cover
    HAS_HMMLEARN = False


@dataclass
class GaussianReturnHMM(BasePriceModel):
    """Fit a Gaussian HMM on log-returns and forecast the next price."""

    n_states: int = 3
    n_iter: int = 100
    covariance_type: str = "diag"
    random_state: int = 24
    name: str = "gaussian_hmm"

    model_: object = field(default=None, init=False)
    train_returns_: Optional[np.ndarray] = field(default=None, init=False)
    last_price_: Optional[float] = field(default=None, init=False)

    def fit(self, prices: np.ndarray, **kwargs) -> "GaussianReturnHMM":
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices))
        self.train_returns_ = returns
        self.last_price_ = float(prices[-1])

        if not HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn is required for GaussianReturnHMM. "
                "Install with: pip install hmmlearn"
            )

        import warnings

        model = _SklearnGaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init_params="stmc",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(returns.reshape(-1, 1))
        self.model_ = model
        return self

    def decode_returns(self, returns: np.ndarray) -> np.ndarray:
        return self.model_.predict(np.asarray(returns, dtype=float).reshape(-1, 1))

    def expected_next_return(self, returns: np.ndarray) -> float:
        returns = np.asarray(returns, dtype=float).reshape(-1, 1)
        # Posteriors for the last observation, then one-step transition
        posteriors = self.model_.predict_proba(returns)
        next_state = posteriors[-1] @ self.model_.transmat_
        means = self.model_.means_.ravel()
        return float(next_state @ means)

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 2:
            return float(history[-1])
        returns = np.diff(np.log(history))
        exp_ret = self.expected_next_return(returns)
        return float(history[-1] * np.exp(exp_ret))

    def summary(self):
        out = {
            "name": self.name,
            "n_states": self.n_states,
            "has_hmmlearn": HAS_HMMLEARN,
        }
        if self.model_ is not None:
            out["converged"] = bool(getattr(self.model_, "monitor_", None) and self.model_.monitor_.converged)
            out["means"] = self.model_.means_.ravel().tolist()
        return out


@dataclass
class MultivariateGaussianHMM(BasePriceModel):
    """Gaussian HMM on [log_return, abs_return] for richer regime detection."""

    n_states: int = 3
    n_iter: int = 100
    random_state: int = 24
    name: str = "mv_gaussian_hmm"

    model_: object = field(default=None, init=False)

    def _features(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(prices))
        feats = np.column_stack([rets, np.abs(rets)])
        return feats

    def fit(self, prices: np.ndarray, **kwargs) -> "MultivariateGaussianHMM":
        if not HAS_HMMLEARN:
            raise ImportError("hmmlearn is required for MultivariateGaussianHMM")
        import warnings

        feats = self._features(prices)
        model = _SklearnGaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(feats)
        self.model_ = model
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 2:
            return float(history[-1])
        feats = self._features(history)
        posteriors = self.model_.predict_proba(feats)
        next_state = posteriors[-1] @ self.model_.transmat_
        # First feature dimension is log-return mean
        means = self.model_.means_[:, 0]
        exp_ret = float(next_state @ means)
        return float(history[-1] * np.exp(exp_ret))
