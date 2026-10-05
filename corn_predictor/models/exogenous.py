"""Models that condition corn returns on exogenous drivers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import warnings

from .base import BasePriceModel


@dataclass
class ExogenousReturnRegression(BasePriceModel):
    """OLS of corn log-return on lagged exogenous returns (+ intercept)."""

    name: str = "exo_regression"
    coef_: Optional[np.ndarray] = field(default=None, init=False)
    feature_names: Tuple[str, ...] = ()

    def fit_xy(self, X: np.ndarray, y: np.ndarray) -> "ExogenousReturnRegression":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        design = np.column_stack([np.ones(len(X)), X])
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        self.coef_ = beta
        return self

    def fit(self, prices: np.ndarray, **kwargs) -> "ExogenousReturnRegression":
        X = kwargs.get("X")
        y = kwargs.get("y")
        if X is None or y is None:
            raise ValueError("ExogenousReturnRegression.fit requires X= and y=")
        self.feature_names = tuple(kwargs.get("feature_names", ()))
        return self.fit_xy(X, y)

    def predict_return(self, x_row: np.ndarray) -> float:
        x_row = np.asarray(x_row, dtype=float).ravel()
        return float(self.coef_[0] + self.coef_[1:] @ x_row)

    def predict_next(self, history: np.ndarray) -> float:
        # Without a fresh exogenous row, fall back to last price (no signal).
        return float(np.asarray(history, dtype=float)[-1])

    def predict_next_with_exo(self, last_price: float, x_row: np.ndarray) -> float:
        return float(last_price * np.exp(self.predict_return(x_row)))


@dataclass
class ExogenousGaussianHMM(BasePriceModel):
    """Gaussian HMM on [corn_return, lagged exo returns...]."""

    n_states: int = 3
    n_iter: int = 200
    random_state: int = 24
    name: str = "exo_gaussian_hmm"
    model_: object = field(default=None, init=False)

    def fit_features(self, feats: np.ndarray) -> "ExogenousGaussianHMM":
        from hmmlearn.hmm import GaussianHMM

        feats = np.asarray(feats, dtype=float)
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
        return self

    def fit(self, prices: np.ndarray, **kwargs) -> "ExogenousGaussianHMM":
        feats = kwargs.get("feats")
        if feats is None:
            raise ValueError("ExogenousGaussianHMM.fit requires feats=")
        return self.fit_features(feats)

    def expected_next_return(self, feats: np.ndarray) -> float:
        feats = np.asarray(feats, dtype=float)
        post = self.model_.predict_proba(feats)
        next_state = post[-1] @ self.model_.transmat_
        # Dimension 0 is corn return
        return float(next_state @ self.model_.means_[:, 0])

    def predict_next(self, history: np.ndarray) -> float:
        return float(np.asarray(history, dtype=float)[-1])

    def predict_next_with_feats(self, last_price: float, feats: np.ndarray) -> float:
        r = self.expected_next_return(feats)
        return float(last_price * np.exp(r))
