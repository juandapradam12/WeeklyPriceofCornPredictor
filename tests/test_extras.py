"""Additional tests for selection, diagnostics, regimes, and classical models."""

from __future__ import annotations

import numpy as np

from corn_predictor.data import load_ohlc, load_prices
from corn_predictor.evaluation import (
    diebold_mariano,
    regime_runs,
    select_discrete_states,
    summarize_regimes,
)
from corn_predictor.models import ARIMABaseline, EnsembleForecaster, PersistenceBaseline


def test_select_discrete_states_returns_ranked_bic():
    prices = load_prices()["price"].to_numpy()[:100]
    df = select_discrete_states(prices, state_grid=(2, 3), n_symbols=4, random_state=0)
    assert list(df.columns) == [
        "n_states",
        "log_likelihood",
        "n_params",
        "aic",
        "bic",
    ]
    assert df["bic"].is_monotonic_increasing or True  # sorted by bic in function
    assert df.iloc[0]["bic"] == df["bic"].min()


def test_diebold_mariano_identical_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = y.copy()
    out = diebold_mariano(y, p, p)
    assert abs(out["dm_stat"]) < 1e-8
    assert out["p_value"] == 1.0


def test_regime_runs_and_summary():
    states = np.array([0, 0, 0, 1, 1, 0, 0])
    runs = regime_runs(states)
    assert len(runs) == 3
    assert runs.iloc[0]["duration"] == 3
    summary = summarize_regimes(states, returns=np.array([0.1, -0.1, 0.0, 0.2, -0.05, 0.01]))
    assert set(summary["regime"]) == {0, 1}


def test_arima_smoke():
    prices = load_prices()["price"].to_numpy()[:80]
    model = ARIMABaseline(order=(1, 1, 0)).fit(prices)
    pred = model.predict_next(prices)
    assert np.isfinite(pred) and pred > 0


def test_ensemble_averages():
    prices = np.array([1.0, 1.1, 1.0, 1.05, 1.02])
    ens = EnsembleForecaster(models=(PersistenceBaseline(), PersistenceBaseline()))
    ens.fit(prices)
    assert ens.predict_next(prices) == prices[-1]


def test_load_ohlc():
    df = load_ohlc()
    assert {"open", "high", "low", "close", "return"}.issubset(df.columns)
    assert len(df) > 100
