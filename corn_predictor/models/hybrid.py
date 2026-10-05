"""Hybrid predictors that combine discrete HMM regimes with continuous forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from corn_predictor.preprocess import Discretizer

from .base import BasePriceModel
from .discrete_hmm import DiscreteHMM, _logsumexp


@dataclass
class DiscreteHMMRegimeDrift(BasePriceModel):
    """Original discrete-price HMM used as a *regime detector*.

    One-step forecasts apply the empirical mean log-return of the predicted
    next regime to the last observed price. This avoids treating coarse
    K-Means centers as price forecasts.
    """

    n_states: int = 2
    n_symbols: int = 5
    discretize_method: str = "kmeans"
    n_iter: int = 50
    random_state: int = 24
    name: str = "discrete_hmm_regime_drift"

    hmm_: Optional[DiscreteHMM] = field(default=None, init=False)
    disc_: Optional[Discretizer] = field(default=None, init=False)
    state_mean_returns_: Optional[np.ndarray] = field(default=None, init=False)

    def fit(self, prices: np.ndarray, **kwargs) -> "DiscreteHMMRegimeDrift":
        prices = np.asarray(prices, dtype=float)
        self.disc_ = Discretizer(
            n_symbols=self.n_symbols,
            method=self.discretize_method,
            random_state=self.random_state,
        )
        symbols = self.disc_.fit_transform(prices)
        self.hmm_ = DiscreteHMM(
            n_states=self.n_states,
            n_symbols=self.n_symbols,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.hmm_.fit_symbols(symbols, symbol_centers=self.disc_.centers_)

        states = self.hmm_.decode(symbols)
        returns = np.diff(np.log(prices))
        # state[t] aligns with price[t]; return[t-1] is price[t-1] -> price[t]
        self.state_mean_returns_ = np.zeros(self.n_states)
        for k in range(self.n_states):
            mask = states[1:] == k
            if mask.any():
                self.state_mean_returns_[k] = float(np.mean(returns[mask]))
            else:
                self.state_mean_returns_[k] = float(np.mean(returns))
        return self

    def _next_state_dist(self, symbols: np.ndarray) -> np.ndarray:
        log_alpha, _ = self.hmm_._forward(np.asarray(symbols, dtype=int))
        filtered = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        return filtered @ self.hmm_.transmat_

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        symbols = self.disc_.transform(history)
        next_state = self._next_state_dist(symbols)
        exp_ret = float(next_state @ self.state_mean_returns_)
        return float(history[-1] * np.exp(exp_ret))

    def decode_prices(self, prices: np.ndarray) -> np.ndarray:
        symbols = self.disc_.transform(prices)
        return self.hmm_.decode(symbols)


@dataclass
class DiscreteReturnHMM(BasePriceModel):
    """Discrete HMM on quantized log-returns (complementary to price quantization)."""

    n_states: int = 2
    n_symbols: int = 5
    discretize_method: str = "quantile"
    n_iter: int = 50
    random_state: int = 24
    name: str = "discrete_return_hmm"

    hmm_: Optional[DiscreteHMM] = field(default=None, init=False)
    disc_: Optional[Discretizer] = field(default=None, init=False)

    def fit(self, prices: np.ndarray, **kwargs) -> "DiscreteReturnHMM":
        prices = np.asarray(prices, dtype=float)
        returns = np.diff(np.log(prices))
        self.disc_ = Discretizer(
            n_symbols=self.n_symbols,
            method=self.discretize_method,
            random_state=self.random_state,
        )
        symbols = self.disc_.fit_transform(returns)
        self.hmm_ = DiscreteHMM(
            n_states=self.n_states,
            n_symbols=self.n_symbols,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.hmm_.fit_symbols(symbols, symbol_centers=self.disc_.centers_)
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        if len(history) < 2:
            return float(history[-1])
        returns = np.diff(np.log(history))
        symbols = self.disc_.transform(returns)
        log_alpha, _ = self.hmm_._forward(np.asarray(symbols, dtype=int))
        filtered = np.exp(log_alpha[-1] - _logsumexp(log_alpha[-1]))
        next_state = filtered @ self.hmm_.transmat_
        next_obs_dist = next_state @ self.hmm_.emissionprob_
        exp_ret = float(next_obs_dist @ self.disc_.centers_)
        return float(history[-1] * np.exp(exp_ret))
