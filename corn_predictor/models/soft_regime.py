"""Markov regime-switching AR(1) with soft EM (Forward-Backward responsibilities)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .base import BasePriceModel
from .discrete_hmm import _logsumexp


@dataclass
class SoftRegimeSwitchingAR(BasePriceModel):
    """Two+ regime AR(1) estimated with soft EM instead of hard Viterbi."""

    n_states: int = 2
    n_iter: int = 40
    use_returns: bool = True
    random_state: int = 24
    name: str = "soft_regime_switching_ar"

    intercepts_: Optional[np.ndarray] = field(default=None, init=False)
    phis_: Optional[np.ndarray] = field(default=None, init=False)
    sigmas_: Optional[np.ndarray] = field(default=None, init=False)
    transmat_: Optional[np.ndarray] = field(default=None, init=False)
    startprob_: Optional[np.ndarray] = field(default=None, init=False)
    responsibilities_: Optional[np.ndarray] = field(default=None, init=False)

    def _series(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if self.use_returns:
            return np.diff(np.log(prices))
        return np.log(prices)

    def _emit_ll(self, y: np.ndarray, y_lag: np.ndarray) -> np.ndarray:
        T = len(y)
        K = self.n_states
        ll = np.zeros((T, K))
        for k in range(K):
            mu = self.intercepts_[k] + self.phis_[k] * y_lag
            ll[:, k] = stats.norm.logpdf(y, loc=mu, scale=self.sigmas_[k])
        return ll

    def _forward_backward(
        self, emit_ll: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        T, N = emit_ll.shape
        log_alpha = np.full((T, N), -np.inf)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + emit_ll[0]
        logA = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + logA[:, j]) + emit_ll[t, j]
        log_beta = np.zeros((T, N))
        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = _logsumexp(logA[i] + emit_ll[t + 1] + log_beta[t + 1])
        log_lik = float(_logsumexp(log_alpha[-1]))
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            log_xi = (
                log_alpha[t][:, None]
                + logA
                + emit_ll[t + 1][None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi -= _logsumexp(log_xi)
            xi[t] = np.exp(log_xi)
        return gamma, xi, log_lik

    def _weighted_ols(
        self, y: np.ndarray, y_lag: np.ndarray, w: np.ndarray
    ) -> Tuple[float, float, float]:
        w = np.asarray(w, dtype=float)
        sw = w.sum() + 1e-12
        X = np.column_stack([np.ones(len(y)), y_lag])
        # Weighted least squares
        Ww = np.sqrt(w)[:, None]
        beta, _, _, _ = np.linalg.lstsq(Ww * X, Ww[:, 0] * y, rcond=None)
        resid = y - X @ beta
        sigma = float(np.sqrt(np.sum(w * resid**2) / sw + 1e-12))
        return float(beta[0]), float(beta[1]), max(sigma, 1e-5)

    def fit(self, prices: np.ndarray, **kwargs) -> "SoftRegimeSwitchingAR":
        series = self._series(prices)
        y_lag = series[:-1]
        y = series[1:]
        T = len(y)
        rng = np.random.default_rng(self.random_state)

        self.startprob_ = np.ones(self.n_states) / self.n_states
        self.transmat_ = np.full((self.n_states, self.n_states), 0.1 / max(self.n_states - 1, 1))
        np.fill_diagonal(self.transmat_, 0.9)
        self.intercepts_ = rng.normal(0, 0.001, size=self.n_states)
        self.phis_ = rng.normal(0, 0.1, size=self.n_states)
        self.sigmas_ = np.full(self.n_states, float(np.std(y) + 1e-6))

        # Init responsibilities by |return| magnitude
        if self.use_returns:
            soft = (np.abs(y) - np.abs(y).min()) / (np.ptp(np.abs(y)) + 1e-12)
            gamma = np.column_stack([1 - soft, soft]) if self.n_states == 2 else None
        else:
            gamma = None
        if gamma is None or gamma.shape[1] != self.n_states:
            gamma = rng.dirichlet(np.ones(self.n_states), size=T)

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            for k in range(self.n_states):
                self.intercepts_[k], self.phis_[k], self.sigmas_[k] = self._weighted_ols(
                    y, y_lag, gamma[:, k]
                )
            emit_ll = self._emit_ll(y, y_lag)
            gamma, xi, ll = self._forward_backward(emit_ll)
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi.sum(axis=0)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True) + 1e-300
            if abs(ll - prev_ll) < 1e-4:
                break
            prev_ll = ll

        self.responsibilities_ = gamma
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 3:
            return float(history[-1])
        series = self._series(history)
        y_lag = series[:-1]
        y = series[1:]
        emit_ll = self._emit_ll(y, y_lag)
        gamma, _, _ = self._forward_backward(emit_ll)
        next_state = gamma[-1] @ self.transmat_
        last_y = series[-1]
        pred_y = 0.0
        for k in range(self.n_states):
            pred_y += next_state[k] * (self.intercepts_[k] + self.phis_[k] * last_y)
        if self.use_returns:
            return float(history[-1] * np.exp(pred_y))
        return float(np.exp(pred_y))

    def summary(self):
        return {
            "name": self.name,
            "n_states": self.n_states,
            "intercepts": None if self.intercepts_ is None else self.intercepts_.tolist(),
            "phis": None if self.phis_ is None else self.phis_.tolist(),
            "sigmas": None if self.sigmas_ is None else self.sigmas_.tolist(),
        }
