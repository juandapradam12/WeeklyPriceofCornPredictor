"""End-to-end experiment pipeline for corn price regime models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from corn_predictor.data import add_features, load_prices, train_test_split_time
from corn_predictor.evaluation import metrics_table, summarize_forecasts
from corn_predictor.models import (
    DiscreteHMM,
    DiscreteHMMRegimeDrift,
    DiscreteReturnHMM,
    DriftBaseline,
    GaussianReturnHMM,
    MovingAverageBaseline,
    MultivariateGaussianHMM,
    PersistenceBaseline,
    RegimeSwitchingAR,
)
from corn_predictor.preprocess import Discretizer
from corn_predictor.visualization import (
    plot_emission_matrix,
    plot_forecast_overlay,
    plot_loglik_convergence,
    plot_metrics_bars,
    plot_price_series,
    plot_regimes_on_price,
    plot_returns_by_regime,
    plot_transition_matrix,
)


@dataclass
class ExperimentConfig:
    data_file: str = "corn2013-2017.txt"
    test_ratio: float = 0.25
    n_symbols: int = 5
    discretize_method: str = "kmeans"
    discrete_states: int = 2
    gaussian_states: int = 3
    regime_states: int = 2
    random_state: int = 24
    figures_dir: Path = field(default_factory=lambda: Path("figures"))


def _holdout_predictions_generic(model, train_prices: np.ndarray, test_prices: np.ndarray) -> np.ndarray:
    """Fit on train, then one-step predict each test point with expanding history."""
    model.fit(train_prices)
    history = list(train_prices)
    preds = []
    for y in test_prices:
        preds.append(model.predict_next(np.asarray(history, dtype=float)))
        history.append(y)
    return np.asarray(preds, dtype=float)


def _holdout_center_decoding(
    train_prices: np.ndarray,
    test_prices: np.ndarray,
    n_symbols: int,
    n_states: int,
    method: str,
    random_state: int,
) -> np.ndarray:
    """Legacy notebook-style forecast: next price ≈ next symbol cluster center."""
    disc = Discretizer(n_symbols=n_symbols, method=method, random_state=random_state)
    train_sym = disc.fit_transform(train_prices)
    hmm = DiscreteHMM(
        n_states=n_states,
        n_symbols=n_symbols,
        random_state=random_state,
    )
    hmm.fit_symbols(train_sym, symbol_centers=disc.centers_)

    history_sym = list(train_sym)
    preds = []
    for y in test_prices:
        nxt_sym = hmm.predict_next_symbol(np.asarray(history_sym, dtype=int))
        preds.append(float(disc.centers_[nxt_sym]))
        history_sym.append(int(disc.transform(np.array([y]))[0]))
    return np.asarray(preds, dtype=float)


def run_experiment(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Train complementary models, evaluate on a holdout, and write figures."""
    cfg = config or ExperimentConfig()
    cfg.figures_dir = Path(cfg.figures_dir)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(load_prices(cfg.data_file))
    train_df, test_df = train_test_split_time(df, test_ratio=cfg.test_ratio)
    train_prices = train_df["price"].to_numpy()
    test_prices = test_df["price"].to_numpy()
    y_prev = np.concatenate([[train_prices[-1]], test_prices[:-1]])

    preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}

    # Enhanced original idea: discrete HMM regimes + drift forecast
    regime_drift = DiscreteHMMRegimeDrift(
        n_states=cfg.discrete_states,
        n_symbols=cfg.n_symbols,
        discretize_method=cfg.discretize_method,
        random_state=cfg.random_state,
    )
    preds["Discrete HMM + Drift"] = _holdout_predictions_generic(
        regime_drift, train_prices, test_prices
    )
    fitted["discrete_hmm_regime_drift"] = regime_drift
    all_states = regime_drift.decode_prices(df["price"].to_numpy())
    d_hmm = regime_drift.hmm_

    # Complementary: discrete HMM on quantized returns
    ret_hmm = DiscreteReturnHMM(
        n_states=cfg.discrete_states,
        n_symbols=cfg.n_symbols,
        discretize_method="quantile",
        random_state=cfg.random_state,
    )
    preds["Discrete Return HMM"] = _holdout_predictions_generic(
        ret_hmm, train_prices, test_prices
    )
    fitted["discrete_return_hmm"] = ret_hmm

    # Legacy center decoding (educational baseline from original notebook idea)
    preds["Discrete HMM (centers)"] = _holdout_center_decoding(
        train_prices,
        test_prices,
        n_symbols=cfg.n_symbols,
        n_states=cfg.discrete_states,
        method=cfg.discretize_method,
        random_state=cfg.random_state,
    )

    # Gaussian HMM on returns
    g_model = GaussianReturnHMM(
        n_states=cfg.gaussian_states, random_state=cfg.random_state, n_iter=200
    )
    preds["Gaussian HMM"] = _holdout_predictions_generic(g_model, train_prices, test_prices)
    fitted["gaussian_hmm"] = g_model

    # Multivariate Gaussian HMM
    mv_model = MultivariateGaussianHMM(
        n_states=cfg.gaussian_states, random_state=cfg.random_state, n_iter=200
    )
    preds["MV Gaussian HMM"] = _holdout_predictions_generic(
        mv_model, train_prices, test_prices
    )
    fitted["mv_gaussian_hmm"] = mv_model

    # Regime-switching AR(1)
    rs_model = RegimeSwitchingAR(
        n_states=cfg.regime_states, random_state=cfg.random_state
    )
    preds["Regime-Switching AR"] = _holdout_predictions_generic(
        rs_model, train_prices, test_prices
    )
    fitted["regime_switching"] = rs_model

    # Baselines
    for model in (
        PersistenceBaseline(),
        MovingAverageBaseline(window=4),
        DriftBaseline(),
    ):
        label = {
            "persistence": "Persistence",
            "moving_average": "Moving Average",
            "drift": "Drift",
        }[model.name]
        preds[label] = _holdout_predictions_generic(model, train_prices, test_prices)
        fitted[model.name] = model

    results = {
        name: summarize_forecasts(test_prices, pred, y_prev)
        for name, pred in preds.items()
    }
    table = metrics_table(results)

    figures: List[Path] = []

    def _save(fig, name: str) -> Path:
        path = cfg.figures_dir / name
        fig.savefig(path, bbox_inches="tight")
        figures.append(path)
        return path

    fig = plot_price_series(df, title="Weekly Corn Close Price (2013–2017)")
    _save(fig, "01_price_series.png")

    fig = plot_regimes_on_price(
        df["week"],
        df["price"].to_numpy(),
        all_states,
        title="Discrete HMM — Inferred Regimes on Corn Prices",
    )
    _save(fig, "02_regimes_on_price.png")

    fig = plot_transition_matrix(
        d_hmm.transmat_, title="Discrete HMM Transition Matrix"
    )
    _save(fig, "03_transition_matrix.png")

    fig = plot_emission_matrix(
        d_hmm.emissionprob_, title="Discrete HMM Emission Matrix"
    )
    _save(fig, "04_emission_matrix.png")

    fig = plot_loglik_convergence(
        d_hmm.log_likelihoods_, title="Discrete HMM Baum-Welch Convergence"
    )
    _save(fig, "05_loglik_convergence.png")

    train_returns = np.diff(np.log(train_prices))
    g_states = g_model.decode_returns(train_returns)
    fig = plot_returns_by_regime(
        train_returns,
        g_states,
        title="Gaussian HMM — Return Distributions by Regime",
    )
    _save(fig, "06_returns_by_regime.png")

    g_price_states = np.concatenate([[g_states[0]], g_states])
    fig = plot_regimes_on_price(
        train_df["week"],
        train_prices,
        g_price_states,
        title="Gaussian HMM — Regimes on Training Prices",
    )
    _save(fig, "07_gaussian_regimes.png")

    fig = plot_forecast_overlay(
        test_df["week"],
        test_prices,
        {
            "Discrete HMM + Drift": preds["Discrete HMM + Drift"],
            "Gaussian HMM": preds["Gaussian HMM"],
            "Regime-Switching AR": preds["Regime-Switching AR"],
            "Persistence": preds["Persistence"],
        },
        title="Holdout One-step Forecasts",
    )
    _save(fig, "08_forecast_overlay.png")

    # Exclude the intentionally weak center-decoding model from the main bar chart
    table_main = table.drop(index=["Discrete HMM (centers)"], errors="ignore")
    fig = plot_metrics_bars(table_main, metric="rmse", title="Holdout RMSE by Model")
    _save(fig, "09_rmse_comparison.png")

    fig = plot_metrics_bars(
        table_main,
        metric="directional_accuracy",
        title="Directional Accuracy by Model",
    )
    _save(fig, "10_directional_accuracy.png")

    plt.close("all")

    return {
        "config": cfg,
        "metrics": table,
        "predictions": preds,
        "fitted": fitted,
        "figures": figures,
        "train_df": train_df,
        "test_df": test_df,
        "all_states": all_states,
    }
