"""Shared model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BasePriceModel(ABC):
    """Minimal interface for one-step-ahead price predictors."""

    name: str = "base"

    @abstractmethod
    def fit(self, prices: np.ndarray, **kwargs: Any) -> "BasePriceModel":
        raise NotImplementedError

    @abstractmethod
    def predict_next(self, history: np.ndarray) -> float:
        """Predict the next price given a history ending at the current week."""
        raise NotImplementedError

    def predict_sequence(self, prices: np.ndarray, start: int = 1) -> np.ndarray:
        """Rolling one-step predictions for indices ``start..len(prices)-1``
        using only information available up to ``t-1`` for target ``prices[t]``.
        """
        preds = []
        for t in range(start, len(prices)):
            preds.append(self.predict_next(prices[:t]))
        return np.asarray(preds, dtype=float)

    def summary(self) -> Dict[str, Any]:
        return {"name": self.name}
