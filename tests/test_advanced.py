"""Tests for advanced enhancements."""

from __future__ import annotations

import numpy as np
import pandas as pd

from corn_predictor.data import load_prices
from corn_predictor.evaluation.calibration import (
    hmm_predictive_interval,
    interval_summary,
    pit_values,
)
from corn_predictor.evaluation.simulator import (
    choose_long_regimes_by_mean_return,
    simulate_regime_long_flat,
)
from corn_predictor.models import SoftRegimeSwitchingAR, StickyGaussianHMM
from corn_predictor.models.exogenous import ExogenousReturnRegression


def test_sticky_hmm_fit_predict():
    prices = load_prices()["price"].to_numpy()[:120]
    model = StickyGaussianHMM(n_states=3, n_iter=20, random_state=0).fit(prices)
    pred = model.predict_next(prices)
    assert np.isfinite(pred) and pred > 0
    assert len(model.log_likelihoods_) >= 2
    assert model.log_likelihoods_[-1] >= model.log_likelihoods_[0] - 1e-6


def test_soft_regime_switching():
    prices = load_prices()["price"].to_numpy()[:100]
    model = SoftRegimeSwitchingAR(n_states=2, n_iter=15, random_state=0).fit(prices)
    assert model.responsibilities_ is not None
    assert np.isfinite(model.predict_next(prices))


def test_predictive_interval_and_pit():
    last = 4.0
    probs = np.array([0.7, 0.3])
    means = np.array([0.0, 0.01])
    vars_ = np.array([0.0004, 0.001])
    interval = hmm_predictive_interval(last, probs, means, vars_, level=0.9)
    assert interval.lower < interval.mean < interval.upper
    y = np.array([4.01, 3.99, 4.02])
    m = np.array([4.0, 4.0, 4.0])
    s = np.array([0.05, 0.05, 0.05])
    pits = pit_values(y, m, s)
    assert np.all((pits >= 0) & (pits <= 1))
    summary = interval_summary(y, m, m - 0.1, m + 0.1, s)
    assert 0 <= summary["coverage"] <= 1


def test_hedge_simulator():
    prices = np.array([1.0, 1.1, 1.0, 1.2, 1.15, 1.3])
    states = np.array([0, 0, 1, 1, 0, 0])
    sim = simulate_regime_long_flat(prices, states, long_regimes=[0], cost_bps=0.0)
    assert len(sim.equity_curve) == len(prices) - 1
    assert "sharpe" in sim.stats
    chosen = choose_long_regimes_by_mean_return(states, prices)
    assert isinstance(chosen, list) and len(chosen) >= 1


def test_exo_regression_ols():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    y = 0.01 + 0.5 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(scale=0.01, size=80)
    model = ExogenousReturnRegression().fit_xy(X, y)
    pred = model.predict_return(np.array([1.0, -1.0]))
    assert np.isfinite(pred)
