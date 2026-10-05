"""Markov regime-switching AR(1) as a complementary technique.

Two latent regimes (e.g. calm vs volatile / mean-reverting vs trending),
each with its own AR(1) coefficients. Parameters are estimated with a
simplified EM that alternates Viterbi hard assignment and OLS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy import stats

from .base import BasePriceModel


@dataclass
class RegimeSwitchingAR(BasePriceModel):
    """Two-regime AR(1) on log-prices or log-returns."""

    n_states: int = 2
    n_iter: int = 30
    use_returns: bool = True
    random_state: int = 24
    name: str = "regime_switching_ar"

    # Per-regime AR(1): y_t = c_k + phi_k * y_{t-1} + eps
    intercepts_: Optional[np.ndarray] = field(default=None, init=False)
    phis_: Optional[np.ndarray] = field(default=None, init=False)
    sigmas_: Optional[np.ndarray] = field(default=None, init=False)
    transmat_: Optional[np.ndarray] = field(default=None, init=False)
    startprob_: Optional[np.ndarray] = field(default=None, init=False)
    states_: Optional[np.ndarray] = field(default=None, init=False)

    def _series(self, prices: np.ndarray) -> np.ndarray:
        prices = np.asarray(prices, dtype=float)
        if self.use_returns:
            return np.diff(np.log(prices))
        return np.log(prices)

    def _emit_ll(self, y: np.ndarray, y_lag: np.ndarray) -> np.ndarray:
        """Log-likelihood of each observation under each regime. Shape (T, K)."""
        T = len(y)
        K = self.n_states
        ll = np.zeros((T, K))
        for k in range(K):
            mu = self.intercepts_[k] + self.phis_[k] * y_lag
            ll[:, k] = stats.norm.logpdf(y, loc=mu, scale=self.sigmas_[k])
        return ll

    def _viterbi(self, emit_ll: np.ndarray) -> np.ndarray:
        T, K = emit_ll.shape
        log_delta = np.full((T, K), -np.inf)
        psi = np.zeros((T, K), dtype=int)
        log_delta[0] = np.log(self.startprob_ + 1e-300) + emit_ll[0]
        logA = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            for j in range(K):
                scores = log_delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(scores))
                log_delta[t, j] = scores[psi[t, j]] + emit_ll[t, j]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def _ols(self, y: np.ndarray, y_lag: np.ndarray) -> Tuple[float, float, float]:
        X = np.column_stack([np.ones(len(y)), y_lag])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma = float(np.sqrt(np.mean(resid**2) + 1e-12))
        return float(beta[0]), float(beta[1]), sigma

    def fit(self, prices: np.ndarray, **kwargs) -> "RegimeSwitchingAR":
        series = self._series(prices)
        y_lag = series[:-1]
        y = series[1:]
        T = len(y)
        rng = np.random.default_rng(self.random_state)

        # Initialize regimes by return magnitude quantiles
        if self.use_returns:
            labels = (np.abs(y) > np.median(np.abs(y))).astype(int)
        else:
            labels = (y > np.median(y)).astype(int)
        if labels.sum() == 0 or labels.sum() == T:
            labels = rng.integers(0, self.n_states, size=T)

        self.startprob_ = np.ones(self.n_states) / self.n_states
        self.transmat_ = np.full((self.n_states, self.n_states), 0.1 / (self.n_states - 1))
        np.fill_diagonal(self.transmat_, 0.9)

        self.intercepts_ = np.zeros(self.n_states)
        self.phis_ = np.zeros(self.n_states)
        self.sigmas_ = np.ones(self.n_states)

        for _ in range(self.n_iter):
            for k in range(self.n_states):
                mask = labels == k
                if mask.sum() < 3:
                    self.intercepts_[k], self.phis_[k], self.sigmas_[k] = 0.0, 0.0, float(np.std(y) + 1e-6)
                else:
                    self.intercepts_[k], self.phis_[k], self.sigmas_[k] = self._ols(
                        y[mask], y_lag[mask]
                    )

            # Update transitions from hard path
            counts = np.ones((self.n_states, self.n_states)) * 1e-2
            for t in range(T - 1):
                counts[labels[t], labels[t + 1]] += 1
            self.transmat_ = counts / counts.sum(axis=1, keepdims=True)
            self.startprob_ = np.bincount(labels, minlength=self.n_states).astype(float)
            self.startprob_ /= self.startprob_.sum()

            emit_ll = self._emit_ll(y, y_lag)
            new_labels = self._viterbi(emit_ll)
            if np.array_equal(new_labels, labels):
                labels = new_labels
                break
            labels = new_labels

        self.states_ = labels
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 3:
            return float(history[-1])

        series = self._series(history)
        y_lag_all = series[:-1]
        y_all = series[1:]
        emit_ll = self._emit_ll(y_all, y_lag_all)
        path = self._viterbi(emit_ll)
        last_state = path[-1]
        next_state_dist = self.transmat_[last_state]

        last_y = series[-1]
        # Mixture prediction across regimes
        pred_y = 0.0
        for k in range(self.n_states):
            pred_y += next_state_dist[k] * (
                self.intercepts_[k] + self.phis_[k] * last_y
            )

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
