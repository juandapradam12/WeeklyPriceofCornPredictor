"""Sticky Hidden Markov Model with Dirichlet priors (MAP / variational-lite).

This is a practical Bayesian-flavoured sticky HMM:
- Dirichlet priors on rows of A and on π
- Sticky self-transition bias (κ) encouraging persistent regimes
- Optional birth/death of weak states via occupancy thresholding
  (a lightweight stand-in for full HDP-HMM)

Full hierarchical Dirichlet process sampling is expensive and brittle on
short weekly series; sticky MAP EM captures the main modelling benefit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from .base import BasePriceModel
from .discrete_hmm import _logsumexp


@dataclass
class StickyGaussianHMM(BasePriceModel):
    """Sticky Gaussian HMM on log-returns with Dirichlet prior regularization."""

    n_states: int = 3
    n_iter: int = 100
    sticky_kappa: float = 10.0  # extra pseudo-counts on self-transitions
    dirichlet_alpha: float = 1.0
    min_occupancy: float = 0.02  # prune nearly unused states after fit
    random_state: int = 24
    name: str = "sticky_gaussian_hmm"

    startprob_: Optional[np.ndarray] = field(default=None, init=False)
    transmat_: Optional[np.ndarray] = field(default=None, init=False)
    means_: Optional[np.ndarray] = field(default=None, init=False)
    vars_: Optional[np.ndarray] = field(default=None, init=False)
    log_likelihoods_: List[float] = field(default_factory=list, init=False)
    active_states_: Optional[np.ndarray] = field(default=None, init=False)

    def _init(self, x: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        n = self.n_states
        # K-means-ish init via quantiles of |x| and sign
        qs = np.quantile(x, np.linspace(0.1, 0.9, n))
        self.means_ = qs.astype(float)
        self.vars_ = np.full(n, float(np.var(x) + 1e-6))
        self.startprob_ = np.ones(n) / n
        A = np.full((n, n), 1.0 / n)
        # sticky init
        A = A * 0.2
        np.fill_diagonal(A, 0.8)
        self.transmat_ = A / A.sum(axis=1, keepdims=True)

    def _emit_ll(self, x: np.ndarray) -> np.ndarray:
        T = len(x)
        ll = np.zeros((T, self.n_states))
        for k in range(self.n_states):
            ll[:, k] = stats.norm.logpdf(
                x, loc=self.means_[k], scale=np.sqrt(self.vars_[k])
            )
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
                log_beta[t, i] = _logsumexp(
                    logA[i] + emit_ll[t + 1] + log_beta[t + 1]
                )
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

    def fit_returns(self, returns: np.ndarray) -> "StickyGaussianHMM":
        x = np.asarray(returns, dtype=float).ravel()
        self._init(x)
        self.log_likelihoods_ = []
        alpha = self.dirichlet_alpha
        kappa = self.sticky_kappa
        prev = -np.inf
        for _ in range(self.n_iter):
            emit_ll = self._emit_ll(x)
            gamma, xi, ll = self._forward_backward(emit_ll)
            self.log_likelihoods_.append(ll)

            # MAP updates with Dirichlet / sticky pseudo-counts
            self.startprob_ = gamma[0] + alpha
            self.startprob_ /= self.startprob_.sum()

            counts = xi.sum(axis=0) + alpha
            np.fill_diagonal(counts, np.diag(counts) + kappa)
            self.transmat_ = counts / counts.sum(axis=1, keepdims=True)

            for k in range(self.n_states):
                w = gamma[:, k]
                sw = w.sum() + 1e-12
                self.means_[k] = float((w @ x) / sw)
                self.vars_[k] = float((w @ (x - self.means_[k]) ** 2) / sw + 1e-8)

            if abs(ll - prev) < 1e-4:
                break
            prev = ll

        occ = gamma.mean(axis=0)
        self.active_states_ = np.where(occ >= self.min_occupancy)[0]
        return self

    def fit(self, prices: np.ndarray, **kwargs) -> "StickyGaussianHMM":
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices))
        return self.fit_returns(returns)

    def decode_returns(self, returns: np.ndarray) -> np.ndarray:
        x = np.asarray(returns, dtype=float).ravel()
        emit_ll = self._emit_ll(x)
        # Viterbi
        T, N = emit_ll.shape
        log_delta = np.full((T, N), -np.inf)
        psi = np.zeros((T, N), dtype=int)
        log_delta[0] = np.log(self.startprob_ + 1e-300) + emit_ll[0]
        logA = np.log(self.transmat_ + 1e-300)
        for t in range(1, T):
            for j in range(N):
                scores = log_delta[t - 1] + logA[:, j]
                psi[t, j] = int(np.argmax(scores))
                log_delta[t, j] = scores[psi[t, j]] + emit_ll[t, j]
        path = np.zeros(T, dtype=int)
        path[-1] = int(np.argmax(log_delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def next_state_probs(self, returns: np.ndarray) -> np.ndarray:
        x = np.asarray(returns, dtype=float).ravel()
        emit_ll = self._emit_ll(x)
        gamma, _, _ = self._forward_backward(emit_ll)
        return gamma[-1] @ self.transmat_

    def predictive_return_moments(self, returns: np.ndarray) -> Tuple[float, float]:
        from corn_predictor.evaluation.calibration import mixture_return_moments

        probs = self.next_state_probs(returns)
        return mixture_return_moments(probs, self.means_, self.vars_)

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 2:
            return float(history[-1])
        rets = np.diff(np.log(history))
        mu, _ = self.predictive_return_moments(rets)
        return float(history[-1] * np.exp(mu))

    def summary(self):
        return {
            "name": self.name,
            "n_states": self.n_states,
            "sticky_kappa": self.sticky_kappa,
            "active_states": None
            if self.active_states_ is None
            else self.active_states_.tolist(),
            "means": None if self.means_ is None else self.means_.tolist(),
            "final_loglik": self.log_likelihoods_[-1] if self.log_likelihoods_ else None,
        }
