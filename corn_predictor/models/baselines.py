"""Simple baselines for fair comparison against HMM-family models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BasePriceModel


@dataclass
class PersistenceBaseline(BasePriceModel):
    """Naive forecast: next price equals last observed price."""

    name: str = "persistence"

    def fit(self, prices: np.ndarray, **kwargs) -> "PersistenceBaseline":
        return self

    def predict_next(self, history: np.ndarray) -> float:
        return float(np.asarray(history, dtype=float)[-1])


@dataclass
class MovingAverageBaseline(BasePriceModel):
    """Next price equals the trailing moving average of the last ``window`` prices."""

    window: int = 4
    name: str = "moving_average"

    def fit(self, prices: np.ndarray, **kwargs) -> "MovingAverageBaseline":
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        w = min(self.window, len(history))
        return float(history[-w:].mean())


@dataclass
class DriftBaseline(BasePriceModel):
    """Random-walk with drift estimated from historical mean log-return."""

    name: str = "drift"
    mean_return_: float = 0.0

    def fit(self, prices: np.ndarray, **kwargs) -> "DriftBaseline":
        prices = np.asarray(prices, dtype=float)
        rets = np.diff(np.log(prices))
        self.mean_return_ = float(np.mean(rets)) if len(rets) else 0.0
        return self

    def predict_next(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)
        return float(history[-1] * np.exp(self.mean_return_))
