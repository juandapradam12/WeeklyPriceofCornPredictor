"""Forecast evaluation metrics and walk-forward helpers."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_prev: np.ndarray) -> float:
    """Fraction of weeks where the sign of the predicted change matches reality."""
    true_dir = np.sign(np.asarray(y_true) - np.asarray(y_prev))
    pred_dir = np.sign(np.asarray(y_pred) - np.asarray(y_prev))
    # Ignore flat weeks in the true series
    mask = true_dir != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(true_dir[mask] == pred_dir[mask]))


def summarize_forecasts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    out = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    if y_prev is not None:
        out["directional_accuracy"] = directional_accuracy(y_true, y_pred, y_prev)
    return out


def walk_forward_predict(
    model_factory: Callable[[], object],
    prices: np.ndarray,
    train_size: int,
    refit_every: int = 1,
) -> np.ndarray:
    """Expanding-window one-step predictions.

    Parameters
    ----------
    model_factory:
        Zero-arg callable that returns a fresh, unfitted model instance.
    prices:
        Full price series.
    train_size:
        Minimum training length before the first prediction.
    refit_every:
        Refit frequency in steps (1 = refit every week).
    """
    prices = np.asarray(prices, dtype=float)
    preds = np.full(len(prices), np.nan)
    model = None
    for t in range(train_size, len(prices)):
        if model is None or (t - train_size) % refit_every == 0:
            model = model_factory()
            model.fit(prices[:t])
        preds[t] = model.predict_next(prices[:t])
    return preds


def metrics_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Build a sorted comparison table from ``{model: metrics}``."""
    df = pd.DataFrame(results).T
    if "rmse" in df.columns:
        df = df.sort_values("rmse")
    return df
