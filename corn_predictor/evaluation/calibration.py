"""Predictive intervals and probability-integral-transform calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass
class PredictiveInterval:
    mean: float
    lower: float
    upper: float
    std: float
    level: float = 0.9


def gaussian_return_interval(
    last_price: float,
    expected_return: float,
    return_std: float,
    level: float = 0.9,
) -> PredictiveInterval:
    """Lognormal price interval from a Gaussian return forecast."""
    z = stats.norm.ppf(0.5 + level / 2.0)
    mean_price = float(last_price * np.exp(expected_return))
    lower = float(last_price * np.exp(expected_return - z * return_std))
    upper = float(last_price * np.exp(expected_return + z * return_std))
    # Approximate std on price scale via delta method
    price_std = float(mean_price * return_std)
    return PredictiveInterval(mean_price, lower, upper, price_std, level)


def mixture_return_moments(
    state_probs: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
) -> Tuple[float, float]:
    """Mean and std of a Gaussian mixture given next-state probabilities."""
    state_probs = np.asarray(state_probs, dtype=float)
    means = np.asarray(means, dtype=float).ravel()
    vars_ = np.asarray(vars_, dtype=float).ravel()
    mu = float(state_probs @ means)
    second = float(state_probs @ (vars_ + means**2))
    var = max(second - mu**2, 1e-12)
    return mu, float(np.sqrt(var))


def hmm_predictive_interval(
    last_price: float,
    state_probs: np.ndarray,
    means: np.ndarray,
    vars_: np.ndarray,
    level: float = 0.9,
) -> PredictiveInterval:
    mu, sd = mixture_return_moments(state_probs, means, vars_)
    return gaussian_return_interval(last_price, mu, sd, level=level)


def pit_values(
    y_true: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
) -> np.ndarray:
    """Probability integral transform values under N(mean, std^2)."""
    y_true = np.asarray(y_true, dtype=float)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    return stats.norm.cdf(y_true, loc=means, scale=np.maximum(stds, 1e-8))


def pit_uniformity_test(pits: np.ndarray) -> Dict[str, float]:
    """Kolmogorov–Smirnov test of PIT ~ Uniform(0,1)."""
    pits = np.asarray(pits, dtype=float)
    pits = pits[np.isfinite(pits)]
    stat, p_value = stats.kstest(pits, "uniform")
    return {"ks_stat": float(stat), "p_value": float(p_value), "n": float(len(pits))}


def coverage_rate(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def interval_summary(
    y_true: np.ndarray,
    means: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    stds: np.ndarray,
    level: float = 0.9,
) -> Dict[str, float]:
    pits = pit_values(y_true, means, stds)
    uni = pit_uniformity_test(pits)
    return {
        "coverage": coverage_rate(y_true, lower, upper),
        "nominal_level": level,
        "mean_width": float(np.mean(upper - lower)),
        "pit_ks_stat": uni["ks_stat"],
        "pit_ks_pvalue": uni["p_value"],
    }
