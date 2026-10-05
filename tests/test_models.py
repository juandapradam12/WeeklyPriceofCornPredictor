"""Unit tests for core corn predictor components."""

from __future__ import annotations

import numpy as np
import pytest

from corn_predictor.data import load_prices, train_test_split_time
from corn_predictor.models import DiscreteHMM, PersistenceBaseline, RegimeSwitchingAR
from corn_predictor.preprocess import Discretizer


def test_load_prices_shape():
    df = load_prices("corn2013-2017.txt")
    assert len(df) > 100
    assert list(df.columns) == ["week", "price"]
    assert df["price"].isna().sum() == 0


def test_train_test_split_time_is_chronological():
    df = load_prices("corn2013-2017.txt")
    train, test = train_test_split_time(df, test_ratio=0.2)
    assert train["week"].iloc[-1] < test["week"].iloc[0]
    assert len(train) + len(test) == len(df)


def test_discretizer_orders_centers():
    rng = np.random.default_rng(0)
    values = np.concatenate([rng.normal(3, 0.2, 40), rng.normal(7, 0.2, 40)])
    disc = Discretizer(n_symbols=2, method="kmeans", random_state=0)
    labels = disc.fit_transform(values)
    assert disc.centers_[0] < disc.centers_[1]
    assert set(labels.tolist()) == {0, 1}


def test_discrete_hmm_improves_or_stables_loglik():
    rng = np.random.default_rng(1)
    # Simple 2-state sticky sequence
    symbols = []
    state = 0
    for _ in range(80):
        if rng.random() < 0.1:
            state = 1 - state
        symbols.append(state * 2 + int(rng.random() < 0.3))
    symbols = np.asarray(symbols, dtype=int)
    hmm = DiscreteHMM(n_states=2, n_symbols=4, n_iter=25, random_state=0)
    hmm.fit_symbols(symbols, symbol_centers=np.arange(4))
    ll = hmm.log_likelihoods_
    assert len(ll) >= 2
    # Non-decreasing within numerical tolerance
    assert all(ll[i + 1] + 1e-6 >= ll[i] for i in range(len(ll) - 1))


def test_persistence_predicts_last_value():
    model = PersistenceBaseline().fit(np.array([1.0, 2.0, 3.0]))
    assert model.predict_next(np.array([1.0, 2.0, 3.0])) == 3.0


def test_regime_switching_fit_and_predict():
    df = load_prices("corn2013-2017.txt")
    prices = df["price"].to_numpy()[:120]
    model = RegimeSwitchingAR(n_states=2, n_iter=10, random_state=0)
    model.fit(prices)
    pred = model.predict_next(prices)
    assert np.isfinite(pred)
    assert pred > 0
