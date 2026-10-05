"""Discrete-emission Hidden Markov Model with Baum-Welch and Viterbi.

Fits a categorical HMM on discretized price (or return) symbols using a
numerically stable Forward-Backward / Baum-Welch implementation, with
Viterbi decoding for regime paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .base import BasePriceModel


def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    if axis is None:
        a_max = np.max(a)
        return a_max + np.log(np.sum(np.exp(a - a_max)))
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


@dataclass
class DiscreteHMM(BasePriceModel):
    """Categorical HMM trained via Baum-Welch (EM)."""

    n_states: int = 2
    n_symbols: int = 5
    n_iter: int = 50
    tol: float = 1e-4
    random_state: int = 24
    name: str = "discrete_hmm"

    startprob_: Optional[np.ndarray] = field(default=None, init=False)
    transmat_: Optional[np.ndarray] = field(default=None, init=False)
    emissionprob_: Optional[np.ndarray] = field(default=None, init=False)
    log_likelihoods_: List[float] = field(default_factory=list, init=False)
    symbol_centers_: Optional[np.ndarray] = field(default=None, init=False)

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.random_state)

    def _init_params(self) -> None:
        rng = self._rng()
        pi = rng.random(self.n_states) + 0.1
        self.startprob_ = pi / pi.sum()

        A = rng.random((self.n_states, self.n_states)) + 0.1
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

        B = rng.random((self.n_states, self.n_symbols)) + 0.1
        self.emissionprob_ = B / B.sum(axis=1, keepdims=True)

    def _forward(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        T = len(obs)
        N = self.n_states
        log_alpha = np.full((T, N), -np.inf)
        log_alpha[0] = np.log(self.startprob_ + 1e-300) + np.log(
            self.emissionprob_[:, obs[0]] + 1e-300
        )
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = _logsumexp(
                    log_alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-300)
                ) + np.log(self.emissionprob_[j, obs[t]] + 1e-300)
        log_lik = float(_logsumexp(log_alpha[-1]))
        return log_alpha, np.array([log_lik])

    def _backward(self, obs: np.ndarray) -> np.ndarray:
        T = len(obs)
        N = self.n_states
        log_beta = np.zeros((T, N))
        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = _logsumexp(
                    np.log(self.transmat_[i] + 1e-300)
                    + np.log(self.emissionprob_[:, obs[t + 1]] + 1e-300)
                    + log_beta[t + 1]
                )
        return log_beta

    def _e_step(
        self, obs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        log_alpha, ll = self._forward(obs)
        log_beta = self._backward(obs)
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)

        T, N = gamma.shape
        xi = np.zeros((T - 1, N, N))
        for t in range(T - 1):
            log_xi = (
                log_alpha[t][:, None]
                + np.log(self.transmat_ + 1e-300)
                + np.log(self.emissionprob_[:, obs[t + 1]] + 1e-300)[None, :]
                + log_beta[t + 1][None, :]
            )
            log_xi -= _logsumexp(log_xi)
            xi[t] = np.exp(log_xi)
        return gamma, xi, float(ll[0])

    def _m_step(self, obs: np.ndarray, gamma: np.ndarray, xi: np.ndarray) -> None:
        self.startprob_ = gamma[0] / gamma[0].sum()
        self.transmat_ = xi.sum(axis=0)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True) + 1e-300

        B = np.zeros((self.n_states, self.n_symbols))
        for k in range(self.n_symbols):
            mask = obs == k
            if mask.any():
                B[:, k] = gamma[mask].sum(axis=0)
        self.emissionprob_ = B / (B.sum(axis=1, keepdims=True) + 1e-300)

    def fit_symbols(
        self,
        symbols: np.ndarray,
        symbol_centers: Optional[np.ndarray] = None,
    ) -> "DiscreteHMM":
        obs = np.asarray(symbols, dtype=int)
        if obs.ndim != 1:
            raise ValueError("symbols must be 1-D")
        if obs.min() < 0 or obs.max() >= self.n_symbols:
            raise ValueError("symbol values out of range for n_symbols")

        self.symbol_centers_ = (
            np.asarray(symbol_centers, dtype=float)
            if symbol_centers is not None
            else np.arange(self.n_symbols, dtype=float)
        )
        self._init_params()
        self.log_likelihoods_ = []

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            gamma, xi, ll = self._e_step(obs)
            self._m_step(obs, gamma, xi)
            self.log_likelihoods_.append(ll)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return self

    def fit(self, prices: np.ndarray, **kwargs) -> "DiscreteHMM":
        """Fit after discretizing prices with an external Discretizer.

        Prefer ``fit_symbols`` when you already have discrete observations.
        This convenience path expects kwargs ``symbols`` and optional
        ``symbol_centers``.
        """
        symbols = kwargs.get("symbols")
        if symbols is None:
            raise ValueError("DiscreteHMM.fit requires symbols=...")
        return self.fit_symbols(symbols, kwargs.get("symbol_centers"))

    def decode(self, symbols: np.ndarray) -> np.ndarray:
        """Viterbi most-likely state path."""
        obs = np.asarray(symbols, dtype=int)
        T = len(obs)
        N = self.n_states
        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)

        log_delta[0] = np.log(self.startprob_ + 1e-300) + np.log(
            self.emissionprob_[:, obs[0]] + 1e-300
        )
        for t in range(1, T):
            for j in range(N):
                scores = log_delta[t - 1] + np.log(self.transmat_[:, j] + 1e-300)
                psi[t, j] = int(np.argmax(scores))
                log_delta[t, j] = scores[psi[t, j]] + np.log(
                    self.emissionprob_[j, obs[t]] + 1e-300
                )

        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def predict_next_symbol(self, symbols: np.ndarray) -> int:
        """Predict the next discrete observation via filtered state beliefs."""
        obs = np.asarray(symbols, dtype=int)
        log_alpha, _ = self._forward(obs)
        filtered = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        next_state = filtered @ self.transmat_
        next_obs_dist = next_state @ self.emissionprob_
        return int(np.argmax(next_obs_dist))

    def predict_next(self, history: np.ndarray) -> float:
        """Predict next continuous price from discrete history + centers.

        ``history`` here is expected to be discrete symbols when
        ``symbol_centers_`` is set. For rolling evaluation, use the
        dedicated pipeline helpers instead.
        """
        symbols = np.asarray(history, dtype=int)
        nxt = self.predict_next_symbol(symbols)
        return float(self.symbol_centers_[nxt])

    def score(self, symbols: np.ndarray) -> float:
        _, ll = self._forward(np.asarray(symbols, dtype=int))
        return float(ll[0])

    def summary(self):
        return {
            "name": self.name,
            "n_states": self.n_states,
            "n_symbols": self.n_symbols,
            "n_iter_ran": len(self.log_likelihoods_),
            "final_loglik": self.log_likelihoods_[-1] if self.log_likelihoods_ else None,
        }
