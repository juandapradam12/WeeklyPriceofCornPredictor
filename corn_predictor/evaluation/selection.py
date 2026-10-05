"""Information criteria and latent-state selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from corn_predictor.models.discrete_hmm import DiscreteHMM
from corn_predictor.preprocess import Discretizer


@dataclass
class StateSelectionResult:
    n_states: int
    log_likelihood: float
    n_params: int
    aic: float
    bic: float


def _aic(ll: float, k: int) -> float:
    return float(-2.0 * ll + 2.0 * k)


def _bic(ll: float, k: int, n: int) -> float:
    return float(-2.0 * ll + k * np.log(n))


def discrete_hmm_n_params(n_states: int, n_symbols: int) -> int:
    """Free parameters: startprob (N-1) + trans (N*(N-1)) + emit (N*(M-1))."""
    return (n_states - 1) + n_states * (n_states - 1) + n_states * (n_symbols - 1)


def select_discrete_states(
    prices: np.ndarray,
    state_grid: Sequence[int] = (2, 3, 4),
    n_symbols: int = 5,
    method: str = "kmeans",
    random_state: int = 24,
) -> pd.DataFrame:
    """Fit discrete HMMs over a state grid and rank by BIC."""
    prices = np.asarray(prices, dtype=float)
    disc = Discretizer(n_symbols=n_symbols, method=method, random_state=random_state)
    symbols = disc.fit_transform(prices)
    rows: List[Dict] = []
    for n_states in state_grid:
        hmm = DiscreteHMM(
            n_states=n_states,
            n_symbols=n_symbols,
            random_state=random_state,
        )
        hmm.fit_symbols(symbols, symbol_centers=disc.centers_)
        ll = hmm.score(symbols)
        k = discrete_hmm_n_params(n_states, n_symbols)
        rows.append(
            {
                "n_states": n_states,
                "log_likelihood": ll,
                "n_params": k,
                "aic": _aic(ll, k),
                "bic": _bic(ll, k, len(symbols)),
            }
        )
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)


def select_gaussian_states(
    prices: np.ndarray,
    state_grid: Sequence[int] = (2, 3, 4),
    random_state: int = 24,
    n_iter: int = 200,
) -> pd.DataFrame:
    """Fit Gaussian HMMs on log-returns over a state grid; rank by BIC."""
    from hmmlearn.hmm import GaussianHMM
    import warnings

    prices = np.asarray(prices, dtype=float)
    returns = np.diff(np.log(prices)).reshape(-1, 1)
    n = len(returns)
    rows: List[Dict] = []
    for n_states in state_grid:
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(returns)
        ll = float(model.score(returns))
        # start (N-1) + trans N(N-1) + means N + diag vars N
        k = (n_states - 1) + n_states * (n_states - 1) + 2 * n_states
        rows.append(
            {
                "n_states": n_states,
                "log_likelihood": ll,
                "n_params": k,
                "aic": _aic(ll, k),
                "bic": _bic(ll, k, n),
            }
        )
    return pd.DataFrame(rows).sort_values("bic").reset_index(drop=True)
