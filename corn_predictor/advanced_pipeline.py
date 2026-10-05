"""Advanced experiment: exogenous models, sticky HMM, calibration, hedge sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from corn_predictor.data import add_features, load_prices, train_test_split_time
from corn_predictor.data.live import (
    build_exogenous_features,
    fetch_exogenous_weekly,
    make_design_matrix,
)
from corn_predictor.evaluation import metrics_table, summarize_forecasts
from corn_predictor.evaluation.calibration import (
    coverage_rate,
    hmm_predictive_interval,
    interval_summary,
    pit_values,
)
from corn_predictor.evaluation.regimes import summarize_regimes
from corn_predictor.evaluation.simulator import (
    choose_long_regimes_by_mean_return,
    simulate_regime_long_flat,
)
from corn_predictor.models import (
    ExogenousGaussianHMM,
    ExogenousReturnRegression,
    SoftRegimeSwitchingAR,
    StickyGaussianHMM,
)
from corn_predictor.visualization.plots import (
    PALETTE,
    _style,
    plot_metrics_bars,
    plot_regimes_on_price,
)


@dataclass
class AdvancedConfig:
    data_file: str = "corn2013-2017.txt"
    test_ratio: float = 0.25
    n_states: int = 3
    random_state: int = 24
    figures_dir: Path = field(default_factory=lambda: Path("figures"))
    reports_dir: Path = field(default_factory=lambda: Path("reports"))
    interval_level: float = 0.9
    fetch_exogenous: bool = True


def _plot_intervals(weeks, actual, mean, lower, upper, title, path):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weeks, actual, color=PALETTE["ink"], lw=2, label="Actual")
    ax.plot(weeks, mean, color=PALETTE["accent"], lw=1.6, label="Predictive mean")
    ax.fill_between(weeks, lower, upper, color=PALETTE["accent"], alpha=0.2, label="90% interval")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_pit(pits, path, title="PIT histogram"):
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(pits, bins=10, range=(0, 1), color=PALETTE["accent"], edgecolor="white", density=True)
    ax.axhline(1.0, color=PALETTE["accent2"], lw=1.5, ls="--", label="Uniform density")
    ax.set_title(title)
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_equity(sim, path, title="Regime long/flat vs buy & hold"):
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(sim.equity_curve.index, sim.equity_curve.values, color=PALETTE["accent"], lw=2, label="Strategy")
    ax.plot(sim.buy_hold.index, sim.buy_hold.values, color=PALETTE["muted"], lw=1.6, label="Buy & hold")
    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_advanced_experiment(config: Optional[AdvancedConfig] = None) -> Dict[str, Any]:
    cfg = config or AdvancedConfig()
    cfg.figures_dir = Path(cfg.figures_dir)
    cfg.reports_dir = Path(cfg.reports_dir)
    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)

    df = add_features(load_prices(cfg.data_file))
    train_df, test_df = train_test_split_time(df, test_ratio=cfg.test_ratio)
    train_prices = train_df["price"].to_numpy()
    test_prices = test_df["price"].to_numpy()
    all_prices = df["price"].to_numpy()
    y_prev = np.concatenate([[train_prices[-1]], test_prices[:-1]])

    figures: List[Path] = []
    preds: Dict[str, np.ndarray] = {}

    # --- Sticky Gaussian HMM ---
    sticky = StickyGaussianHMM(
        n_states=cfg.n_states, random_state=cfg.random_state
    ).fit(train_prices)
    history = list(train_prices)
    sticky_preds, means, lowers, uppers, stds = [], [], [], [], []
    for y in test_prices:
        hist = np.asarray(history, dtype=float)
        rets = np.diff(np.log(hist))
        probs = sticky.next_state_probs(rets)
        interval = hmm_predictive_interval(
            hist[-1], probs, sticky.means_, sticky.vars_, level=cfg.interval_level
        )
        sticky_preds.append(interval.mean)
        means.append(interval.mean)
        lowers.append(interval.lower)
        uppers.append(interval.upper)
        stds.append(interval.std)
        history.append(y)
    preds["Sticky Gaussian HMM"] = np.asarray(sticky_preds)
    means = np.asarray(means)
    lowers = np.asarray(lowers)
    uppers = np.asarray(uppers)
    stds = np.asarray(stds)
    cal = interval_summary(test_prices, means, lowers, uppers, stds, level=cfg.interval_level)
    pits = pit_values(test_prices, means, stds)

    # --- Soft regime-switching AR ---
    soft = SoftRegimeSwitchingAR(
        n_states=2, random_state=cfg.random_state
    )
    history = list(train_prices)
    soft.fit(train_prices)
    soft_preds = []
    for y in test_prices:
        soft_preds.append(soft.predict_next(np.asarray(history, dtype=float)))
        history.append(y)
        # light refit every step is expensive; keep train-fit for speed
    preds["Soft Regime-Switching AR"] = np.asarray(soft_preds)

    # --- Exogenous models ---
    exo_metrics_note = "exogenous unavailable"
    try:
        exo = fetch_exogenous_weekly(use_cache=True) if cfg.fetch_exogenous else None
        if exo is None:
            raise RuntimeError("fetch disabled")
        featured = build_exogenous_features(df, exo)
        feature_cols = [
            c
            for c in ("ret_oil_lag1", "ret_dxy_lag1", "ret_wheat_lag1", "ret_soy_lag1")
            if c in featured.columns
        ]
        X, y_ret, weeks = make_design_matrix(featured, include=feature_cols)
        # align to price index via weeks
        feat_frame = featured.dropna(subset=["ret_corn"] + feature_cols).reset_index(drop=True)
        split = int(len(feat_frame) * (1.0 - cfg.test_ratio))
        train_f = feat_frame.iloc[:split]
        test_f = feat_frame.iloc[split:]

        reg = ExogenousReturnRegression().fit_xy(
            train_f[feature_cols].to_numpy(dtype=float),
            train_f["ret_corn"].to_numpy(dtype=float),
        )

        # Causal one-step forecasts: features at t → price[t+1]
        exo_reg_preds = []
        exo_hmm_preds = []
        bases = []
        actuals = []
        feat_cols_hmm = ["ret_corn"] + feature_cols
        train_feats = train_f[feat_cols_hmm].to_numpy()
        exo_hmm = ExogenousGaussianHMM(
            n_states=cfg.n_states, random_state=cfg.random_state
        ).fit_features(train_feats)
        hist_feats = train_feats.copy()

        for i in range(len(test_f) - 1):
            row = test_f.iloc[i]
            nxt = test_f.iloc[i + 1]
            x_row = row[feature_cols].to_numpy(dtype=float)
            exo_reg_preds.append(
                reg.predict_next_with_exo(float(row["price"]), x_row)
            )
            cur = row[feat_cols_hmm].to_numpy(dtype=float).reshape(1, -1)
            hist_feats = np.vstack([hist_feats, cur])
            exo_hmm_preds.append(
                exo_hmm.predict_next_with_feats(float(row["price"]), hist_feats)
            )
            bases.append(float(row["price"]))
            actuals.append(float(nxt["price"]))

        preds_exo_reg = np.asarray(exo_reg_preds)
        preds_exo_hmm = np.asarray(exo_hmm_preds)
        actuals_exo = np.asarray(actuals)

        exo_results = {
            "Exo Regression": summarize_forecasts(
                actuals_exo, preds_exo_reg, np.asarray(bases)
            ),
            "Exo Gaussian HMM": summarize_forecasts(
                actuals_exo, preds_exo_hmm, np.asarray(bases)
            ),
        }
        exo_table = metrics_table(exo_results)
        exo_table.to_csv(cfg.reports_dir / "exogenous_metrics.csv")
        exo_metrics_note = "ok"
    except Exception as exc:  # network / ticker failures
        exo_table = pd.DataFrame()
        exo_metrics_note = f"skipped: {exc}"
        featured = None

    # --- Metrics for sticky / soft ---
    results = {
        name: summarize_forecasts(test_prices, pred, y_prev)
        for name, pred in preds.items()
    }
    table = metrics_table(results)
    table.to_csv(cfg.reports_dir / "advanced_holdout_metrics.csv")
    pd.DataFrame([cal]).to_csv(cfg.reports_dir / "calibration_summary.csv", index=False)

    # --- Regime decode + hedge sim on full sample using sticky fit on all train ---
    sticky_full = StickyGaussianHMM(
        n_states=cfg.n_states, random_state=cfg.random_state
    ).fit(all_prices)
    rets_all = np.diff(np.log(all_prices))
    states = sticky_full.decode_returns(rets_all)
    states_price = np.concatenate([[states[0]], states])
    regime_summary = summarize_regimes(states_price, prices=all_prices, returns=rets_all)
    regime_summary.to_csv(cfg.reports_dir / "sticky_regime_summary.csv", index=False)

    long_regs = choose_long_regimes_by_mean_return(states_price, all_prices)
    # Use train-only regime means to choose long regimes (avoid look-ahead in selection)
    train_states = states_price[: len(train_prices)]
    long_regs = choose_long_regimes_by_mean_return(train_states, train_prices)
    sim = simulate_regime_long_flat(
        all_prices,
        states_price,
        long_regs,
        weeks=pd.to_datetime(df["week"]),
        cost_bps=1.0,
    )
    pd.DataFrame([sim.stats]).to_csv(cfg.reports_dir / "hedge_sim_stats.csv", index=False)
    sim.equity_curve.to_csv(cfg.reports_dir / "hedge_equity_curve.csv", header=True)

    # --- Figures ---
    path = cfg.figures_dir / "16_sticky_regimes.png"
    fig = plot_regimes_on_price(
        df["week"], all_prices, states_price, title="Sticky Gaussian HMM Regimes"
    )
    fig.savefig(path, bbox_inches="tight")
    figures.append(path)
    plt.close(fig)

    path = cfg.figures_dir / "17_predictive_intervals.png"
    _plot_intervals(
        test_df["week"],
        test_prices,
        means,
        lowers,
        uppers,
        "Sticky HMM 90% predictive intervals",
        path,
    )
    figures.append(path)

    path = cfg.figures_dir / "18_pit_histogram.png"
    _plot_pit(pits, path, title="PIT calibration (Sticky HMM)")
    figures.append(path)

    path = cfg.figures_dir / "19_hedge_equity.png"
    _plot_equity(sim, path)
    figures.append(path)

    path = cfg.figures_dir / "20_advanced_rmse.png"
    fig = plot_metrics_bars(table, metric="rmse", title="Advanced models — Holdout RMSE")
    fig.savefig(path, bbox_inches="tight")
    figures.append(path)
    plt.close(fig)

    if not exo_table.empty:
        path = cfg.figures_dir / "21_exogenous_rmse.png"
        fig = plot_metrics_bars(
            exo_table, metric="rmse", title="Exogenous models — Holdout RMSE"
        )
        fig.savefig(path, bbox_inches="tight")
        figures.append(path)
        plt.close(fig)

    return {
        "metrics": table,
        "calibration": cal,
        "exo_metrics": exo_table,
        "exo_status": exo_metrics_note,
        "regime_summary": regime_summary,
        "hedge_stats": sim.stats,
        "long_regimes": long_regs,
        "figures": figures,
        "sticky": sticky,
        "soft": soft,
    }
