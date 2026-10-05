"""Observation discretization for discrete-emission HMMs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer


@dataclass
class Discretizer:
    """Map continuous prices (or returns) to discrete observation symbols."""

    n_symbols: int = 5
    method: str = "kmeans"  # "kmeans" | "quantile" | "uniform"
    random_state: int = 24

    _model: object = None
    centers_: Optional[np.ndarray] = None

    def fit(self, values: np.ndarray) -> "Discretizer":
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        if self.method == "kmeans":
            self._model = KMeans(
                n_clusters=self.n_symbols,
                random_state=self.random_state,
                n_init=10,
            )
            labels = self._model.fit_predict(x)
            # Remap labels so symbols are ordered by increasing center
            order = np.argsort(self._model.cluster_centers_.ravel())
            remap = {old: new for new, old in enumerate(order)}
            self._label_map = remap
            self.centers_ = self._model.cluster_centers_.ravel()[order]
            self._fitted_labels = np.vectorize(remap.get)(labels)
        elif self.method in {"quantile", "uniform"}:
            strategy = "quantile" if self.method == "quantile" else "uniform"
            self._model = KBinsDiscretizer(
                n_bins=self.n_symbols,
                encode="ordinal",
                strategy=strategy,
                subsample=None,
            )
            labels = self._model.fit_transform(x).ravel().astype(int)
            edges = self._model.bin_edges_[0]
            self.centers_ = 0.5 * (edges[:-1] + edges[1:])
            self._label_map = None
            self._fitted_labels = labels
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Discretizer must be fit before transform")
        x = np.asarray(values, dtype=float).reshape(-1, 1)
        if self.method == "kmeans":
            labels = self._model.predict(x)
            return np.vectorize(self._label_map.get)(labels).astype(int)
        labels = self._model.transform(x).ravel().astype(int)
        return np.clip(labels, 0, self.n_symbols - 1)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        return self._fitted_labels.astype(int)

    def decode_centers(self, symbols: np.ndarray) -> np.ndarray:
        """Map discrete symbols back to representative continuous values."""
        if self.centers_ is None:
            raise RuntimeError("Discretizer must be fit before decode")
        return self.centers_[np.asarray(symbols, dtype=int)]
