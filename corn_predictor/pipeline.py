"""End-to-end experiment pipeline for corn price regime models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corn_predictor.data import (
    add_features,
    load_ohlc,
    load_prices,
    train_test_split_time,
)
from corn_predictor.evaluation import (
    metrics_table,
    pairwise_dm_table,
    regime_runs,
    select_discrete_states,
    select_gaussian_states,
    summarize_forecasts,
    summarize_regimes,
    walk_forward_predict,
)
from corn_predictor.models import (
    ARIMABaseline,
    DiscreteHMMRegimeDrift,
    DiscreteReturnHMM,
    DriftBaseline,
    EnsembleForecaster,
    GARCHBaseline,
    GaussianReturnHMM,
    MovingAverageBaseline,
    MultivariateGaussianHMM,
    OHLCGaussianHMM,
    PersistenceBaseline,
    RegimeSwitchingAR,
)
from corn_predictor.preprocess import Discretizer
from corn_predictor.models.discrete_hmm import DiscreteHMM
from corn_predictor.visualization import (
    plot_dm_bars,
    plot_emission_matrix,
    plot_forecast_overlay,
    plot_loglik_convergence,
    plot_metrics_bars,
    plot_price_series,
    plot_regime_durations,
    plot_regimes_on_price,
    plot_returns_by_regime,
    plot_state_selection,
    plot_transition_matrix,
    plot_walk_forward_errors,
)


@dataclass
class ExperimentConfig:
    data_file: str = "corn2013-2017.txt"
    ohlc_file: str = "corn_OHLC2013-2017.txt"
    test_ratio: float = 0.25
    n_symbols: int = 5
    discretize_method: str = "kmeans"
    discrete_states: int = 2
    gaussian_states: int = 3
    regime_states: int = 2
    random_state: int = 24
    figures_dir: Path = field(default_factory=lambda: Path("figures"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    run_walk_forward: bool = True
    walk_forward_refit_every: int = 4


def _holdout_predictions_generic(
    model, train_prices: np.ndarray, test_prices: np.ndarray, **fit_kwargs
) -> np.ndarray:
    """Fit on train, then one-step predict each test point with expanding history."""
    model.fit(train_prices, **fit_kwargs)
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


def _holdout_ohlc(
    model: OHLCGaussianHMM,
    train_ohlc: np.ndarray,
    test_ohlc: np.ndarray,
) -> np.ndarray:
    model.fit_ohlc(train_ohlc)
    history = train_ohlc.copy()
    preds = []
    for row in test_ohlc:
        preds.append(model.predict_next_ohlc(history))
        history = np.vstack([history, row])
    return np.asarray(preds, dtype=float)


def run_experiment(config: Optional[ExperimentConfig] = None) -> Dict[str, Any]:
    """Train complementary models, evaluate on a holdout, and write figures/reports."""
    cfg = config or ExperimentConfig()
    cfg.figures_dir = Path(cfg.figures_dir)
    cfg.reports_dir = Path(cfg.reports_dir)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(load_prices(cfg.data_file))
    ohlc_df = load_ohlc(cfg.ohlc_file)
    # Align OHLC to the close series by week
    merged = df.merge(ohlc_df, on="week", how="inner", suffixes=("", "_ohlc"))
    if "close" in merged.columns:
        # prefer explicit close from OHLC when available
        pass

    train_df, test_df = train_test_split_time(df, test_ratio=cfg.test_ratio)
    train_prices = train_df["price"].to_numpy()
    test_prices = test_df["price"].to_numpy()
    y_prev = np.concatenate([[train_prices[-1]], test_prices[:-1]])

    ohlc_train_df, ohlc_test_df = train_test_split_time(ohlc_df, test_ratio=cfg.test_ratio)
    train_ohlc = ohlc_train_df[["open", "high", "low", "close"]].to_numpy()
    test_ohlc = ohlc_test_df[["open", "high", "low", "close"]].to_numpy()

    preds: Dict[str, np.ndarray] = {}
    fitted: Dict[str, Any] = {}

    # --- State selection (BIC) ---
    discrete_sel = select_discrete_states(
        train_prices,
        state_grid=(2, 3, 4),
        n_symbols=cfg.n_symbols,
        method=cfg.discretize_method,
        random_state=cfg.random_state,
    )
    gaussian_sel = select_gaussian_states(
        train_prices,
        state_grid=(2, 3, 4),
        random_state=cfg.random_state,
    )
    best_discrete_states = int(discrete_sel.iloc[0]["n_states"])
    best_gaussian_states = int(gaussian_sel.iloc[0]["n_states"])

    # Enhanced original idea: discrete HMM regimes + drift forecast
    regime_drift = DiscreteHMMRegimeDrift(
        n_states=best_discrete_states,
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

    ret_hmm = DiscreteReturnHMM(
        n_states=best_discrete_states,
        n_symbols=cfg.n_symbols,
        discretize_method="quantile",
        random_state=cfg.random_state,
    )
    preds["Discrete Return HMM"] = _holdout_predictions_generic(
        ret_hmm, train_prices, test_prices
    )
    fitted["discrete_return_hmm"] = ret_hmm

    preds["Discrete HMM (centers)"] = _holdout_center_decoding(
        train_prices,
        test_prices,
        n_symbols=cfg.n_symbols,
        n_states=best_discrete_states,
        method=cfg.discretize_method,
        random_state=cfg.random_state,
    )

    g_model = GaussianReturnHMM(
        n_states=best_gaussian_states, random_state=cfg.random_state, n_iter=200
    )
    preds["Gaussian HMM"] = _holdout_predictions_generic(
        g_model, train_prices, test_prices
    )
    fitted["gaussian_hmm"] = g_model

    mv_model = MultivariateGaussianHMM(
        n_states=best_gaussian_states, random_state=cfg.random_state, n_iter=200
    )
    preds["MV Gaussian HMM"] = _holdout_predictions_generic(
        mv_model, train_prices, test_prices
    )
    fitted["mv_gaussian_hmm"] = mv_model

    rs_model = RegimeSwitchingAR(
        n_states=cfg.regime_states, random_state=cfg.random_state
    )
    preds["Regime-Switching AR"] = _holdout_predictions_generic(
        rs_model, train_prices, test_prices
    )
    fitted["regime_switching"] = rs_model

    # Classical + OHLC
    preds["ARIMA"] = _holdout_predictions_generic(
        ARIMABaseline(order=(1, 1, 1)), train_prices, test_prices
    )
    preds["GARCH"] = _holdout_predictions_generic(
        GARCHBaseline(), train_prices, test_prices
    )

    ohlc_model = OHLCGaussianHMM(
        n_states=best_gaussian_states, random_state=cfg.random_state
    )
    # Align lengths: use min test length between price and OHLC splits
    n_test = min(len(test_prices), len(test_ohlc))
    ohlc_preds = _holdout_ohlc(ohlc_model, train_ohlc, test_ohlc[:n_test])
    if n_test < len(test_prices):
        # pad should not happen if same calendar; truncate preds dict targets later if needed
        pass
    preds["OHLC Gaussian HMM"] = ohlc_preds
    fitted["ohlc_gaussian_hmm"] = ohlc_model

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

    # Ensemble of complementary strong models (fit on train, expand)
    ensemble = EnsembleForecaster(
        models=(
            GaussianReturnHMM(
                n_states=best_gaussian_states, random_state=cfg.random_state
            ),
            DiscreteHMMRegimeDrift(
                n_states=best_discrete_states,
                n_symbols=cfg.n_symbols,
                random_state=cfg.random_state,
            ),
            DriftBaseline(),
        )
    )
    preds["Ensemble"] = _holdout_predictions_generic(ensemble, train_prices, test_prices)
    fitted["ensemble"] = ensemble

    # Ensure equal prediction lengths (OHLC alignment)
    min_len = min(len(v) for v in preds.values())
    if min_len < len(test_prices):
        test_prices = test_prices[:min_len]
        y_prev = y_prev[:min_len]
        test_df = test_df.iloc[:min_len]
        preds = {k: v[:min_len] for k, v in preds.items()}

    results = {
        name: summarize_forecasts(test_prices, pred, y_prev)
        for name, pred in preds.items()
    }
    table = metrics_table(results)

    # Diebold-Mariano vs persistence
    dm = pairwise_dm_table(test_prices, preds, baseline="Persistence", criterion="se")
    dm_df = pd.DataFrame(dm).T.sort_values("p_value")

    # Regime analytics
    returns_full = np.diff(np.log(df["price"].to_numpy()))
    regime_summary = summarize_regimes(
        all_states, prices=df["price"].to_numpy(), returns=returns_full
    )
    runs = regime_runs(all_states)

    # Walk-forward on a subset of models (expanding window)
    wf_errors: Dict[str, np.ndarray] = {}
    wf_metrics: Dict[str, Dict[str, float]] = {}
    if cfg.run_walk_forward:
        train_size = len(train_prices)
        all_prices = df["price"].to_numpy()
        factories = {
            "Gaussian HMM": lambda: GaussianReturnHMM(
                n_states=best_gaussian_states, random_state=cfg.random_state
            ),
            "Discrete HMM + Drift": lambda: DiscreteHMMRegimeDrift(
                n_states=best_discrete_states,
                n_symbols=cfg.n_symbols,
                random_state=cfg.random_state,
            ),
            "Persistence": PersistenceBaseline,
            "ARIMA": lambda: ARIMABaseline(order=(1, 1, 1)),
        }
        for name, factory in factories.items():
            wf_preds = walk_forward_predict(
                factory,
                all_prices,
                train_size=train_size,
                refit_every=cfg.walk_forward_refit_every,
            )
            mask = ~np.isnan(wf_preds)
            y_t = all_prices[mask]
            y_hat = wf_preds[mask]
            y_p = all_prices[np.where(mask)[0] - 1]
            wf_metrics[name] = summarize_forecasts(y_t, y_hat, y_p)
            # absolute errors aligned to test weeks only
            test_mask = np.zeros(len(all_prices), dtype=bool)
            test_mask[train_size : train_size + min_len] = True
            wf_errors[name] = np.abs(all_prices[test_mask] - wf_preds[test_mask])

    # --- Reports ---
    table.to_csv(cfg.reports_dir / "holdout_metrics.csv")
    dm_df.to_csv(cfg.reports_dir / "diebold_mariano.csv")
    discrete_sel.to_csv(cfg.reports_dir / "discrete_state_selection.csv", index=False)
    gaussian_sel.to_csv(cfg.reports_dir / "gaussian_state_selection.csv", index=False)
    regime_summary.to_csv(cfg.reports_dir / "regime_summary.csv", index=False)
    runs.to_csv(cfg.reports_dir / "regime_runs.csv", index=False)
    if wf_metrics:
        pd.DataFrame(wf_metrics).T.to_csv(cfg.reports_dir / "walk_forward_metrics.csv")

    forecast_frame = test_df[["week"]].copy()
    forecast_frame["actual"] = test_prices
    for name, pred in preds.items():
        forecast_frame[name] = pred
    forecast_frame.to_csv(cfg.reports_dir / "holdout_forecasts.csv", index=False)

    # --- Visualizations ---
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

    fig = plot_transition_matrix(d_hmm.transmat_, title="Discrete HMM Transition Matrix")
    _save(fig, "03_transition_matrix.png")

    fig = plot_emission_matrix(d_hmm.emissionprob_, title="Discrete HMM Emission Matrix")
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
            "Gaussian HMM": preds["Gaussian HMM"],
            "Ensemble": preds["Ensemble"],
            "ARIMA": preds["ARIMA"],
            "Persistence": preds["Persistence"],
        },
        title="Holdout One-step Forecasts",
    )
    _save(fig, "08_forecast_overlay.png")

    table_main = table.drop(index=["Discrete HMM (centers)"], errors="ignore")
    fig = plot_metrics_bars(table_main, metric="rmse", title="Holdout RMSE by Model")
    _save(fig, "09_rmse_comparison.png")

    fig = plot_metrics_bars(
        table_main,
        metric="directional_accuracy",
        title="Directional Accuracy by Model",
    )
    _save(fig, "10_directional_accuracy.png")

    fig = plot_state_selection(
        discrete_sel, title="Discrete HMM — BIC State Selection"
    )
    _save(fig, "11_discrete_state_selection.png")

    fig = plot_state_selection(
        gaussian_sel, title="Gaussian HMM — BIC State Selection"
    )
    _save(fig, "12_gaussian_state_selection.png")

    fig = plot_regime_durations(runs, title="Discrete HMM Regime Durations")
    _save(fig, "13_regime_durations.png")

    # Exclude legacy center model from DM chart
    dm_plot = {k: v for k, v in dm.items() if k != "Discrete HMM (centers)"}
    fig = plot_dm_bars(dm_plot, title="Diebold–Mariano vs Persistence (SE loss)")
    _save(fig, "14_diebold_mariano.png")

    if wf_errors:
        fig = plot_walk_forward_errors(
            test_df["week"].to_numpy(),
            wf_errors,
            title="Walk-forward Absolute Errors (holdout window)",
        )
        _save(fig, "15_walk_forward_errors.png")

    plt.close("all")

    return {
        "config": cfg,
        "metrics": table,
        "dm": dm_df,
        "discrete_selection": discrete_sel,
        "gaussian_selection": gaussian_sel,
        "regime_summary": regime_summary,
        "predictions": preds,
        "fitted": fitted,
        "figures": figures,
        "train_df": train_df,
        "test_df": test_df,
        "all_states": all_states,
        "walk_forward_metrics": wf_metrics,
        "best_discrete_states": best_discrete_states,
        "best_gaussian_states": best_gaussian_states,
    }
