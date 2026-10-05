"""Diebold-Mariano test and related forecast comparison utilities."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats


def _loss(y_true: np.ndarray, y_pred: np.ndarray, criterion: str = "se") -> np.ndarray:
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if criterion == "se":
        return err**2
    if criterion == "ae":
        return np.abs(err)
    raise ValueError("criterion must be 'se' or 'ae'")


def diebold_mariano(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    criterion: str = "se",
    h: int = 1,
) -> Dict[str, float]:
    """Diebold-Mariano test: H0 that models A and B have equal predictive accuracy.

    Negative DM statistic means A has lower loss than B (A is better) under the
    chosen criterion.
    """
    d = _loss(y_true, pred_a, criterion) - _loss(y_true, pred_b, criterion)
    T = len(d)
    d_bar = float(np.mean(d))
    # Newey-West variance for h-step forecasts
    gamma0 = float(np.mean((d - d_bar) ** 2))
    var = gamma0
    for lag in range(1, h):
        cov = float(np.mean((d[lag:] - d_bar) * (d[:-lag] - d_bar)))
        var += 2.0 * (1.0 - lag / h) * cov
    dm_var = var / T
    if dm_var <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": d_bar}
    dm_stat = d_bar / np.sqrt(dm_var)
    # Harvey et al. small-sample correction
    dm_stat = dm_stat * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    p_value = float(2.0 * stats.t.sf(np.abs(dm_stat), df=T - 1))
    return {
        "dm_stat": float(dm_stat),
        "p_value": p_value,
        "mean_loss_diff": d_bar,
    }


def pairwise_dm_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    baseline: str = "Persistence",
    criterion: str = "se",
) -> Dict[str, Dict[str, float]]:
    """Compare every model against a named baseline with the DM test."""
    if baseline not in predictions:
        raise KeyError(f"baseline '{baseline}' not in predictions")
    out = {}
    for name, pred in predictions.items():
        if name == baseline:
            continue
        out[name] = diebold_mariano(y_true, pred, predictions[baseline], criterion=criterion)
    return out
