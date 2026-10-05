"""Visualization helpers for corn price HMM analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Visual identity: grain / harvest atmosphere (not purple/cream AI defaults)
PALETTE = {
    "bg": "#f7f3eb",
    "panel": "#fffdf8",
    "ink": "#1f2a24",
    "muted": "#5c6b63",
    "accent": "#2f6b4f",
    "accent2": "#c45c26",
    "line": "#3d4f45",
    "grid": "#d9e0d8",
    "regimes": ["#2f6b4f", "#c45c26", "#3a6ea5", "#b08900", "#6b4c7a"],
}


def _style():
    sns.set_theme(style="whitegrid", font="DejaVu Sans")
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["grid"],
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["grid"],
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 140,
        }
    )


def plot_price_series(
    df: pd.DataFrame,
    price_col: str = "price",
    title: str = "Weekly Corn Close Price",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(df["week"], df[price_col], color=PALETTE["accent"], lw=1.8)
    ax.fill_between(df["week"], df[price_col], alpha=0.12, color=PALETTE["accent"])
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("USD / bushel")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_regimes_on_price(
    weeks: Sequence,
    prices: np.ndarray,
    states: np.ndarray,
    title: str = "Inferred Market Regimes",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    prices = np.asarray(prices, dtype=float)
    states = np.asarray(states, dtype=int)
    ax.plot(weeks, prices, color=PALETTE["line"], lw=1.2, alpha=0.85, zorder=1)

    unique = sorted(np.unique(states))
    for i, s in enumerate(unique):
        mask = states == s
        ax.scatter(
            np.asarray(weeks)[mask],
            prices[mask],
            s=18,
            color=PALETTE["regimes"][i % len(PALETTE["regimes"])],
            label=f"Regime {s}",
            zorder=2,
        )
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_transition_matrix(
    transmat: np.ndarray,
    title: str = "State Transition Matrix",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    sns.heatmap(
        transmat,
        annot=True,
        fmt=".2f",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_emission_matrix(
    emissionprob: np.ndarray,
    title: str = "Emission Probabilities",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.heatmap(
        emissionprob,
        annot=True,
        fmt=".2f",
        cmap="Oranges",
        vmin=0,
        vmax=1,
        ax=ax,
    )
    ax.set_xlabel("Observation symbol")
    ax.set_ylabel("Hidden state")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_loglik_convergence(
    logliks: Sequence[float],
    title: str = "Baum-Welch Log-Likelihood",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(range(1, len(logliks) + 1), logliks, color=PALETTE["accent2"], lw=2, marker="o", ms=3)
    ax.set_title(title)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("Log-likelihood")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_forecast_overlay(
    weeks: Sequence,
    actual: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "One-step Ahead Forecasts",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(weeks, actual, color=PALETTE["ink"], lw=2.0, label="Actual", zorder=3)
    colors = [PALETTE["accent"], PALETTE["accent2"], "#3a6ea5", "#b08900", "#6b4c7a"]
    for i, (name, pred) in enumerate(predictions.items()):
        ax.plot(
            weeks,
            pred,
            lw=1.5,
            alpha=0.9,
            color=colors[i % len(colors)],
            label=name,
        )
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("Price")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_metrics_bars(
    metrics_df: pd.DataFrame,
    metric: str = "rmse",
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    order = metrics_df.sort_values(metric)
    colors = [PALETTE["accent"] if i == 0 else PALETTE["muted"] for i in range(len(order))]
    ax.barh(order.index.astype(str), order[metric], color=colors)
    ax.set_xlabel(metric.upper())
    ax.set_title(title or f"Model comparison ({metric.upper()})")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_returns_by_regime(
    returns: np.ndarray,
    states: np.ndarray,
    title: str = "Return Distributions by Regime",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(8, 4.0))
    returns = np.asarray(returns, dtype=float)
    states = np.asarray(states, dtype=int)
    data = []
    labels = []
    for s in sorted(np.unique(states)):
        data.append(returns[states == s])
        labels.append(f"Regime {s}")
    parts = ax.violinplot(data, showmeans=True, showmedians=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(PALETTE["regimes"][i % len(PALETTE["regimes"])])
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Log-return")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_state_selection(
    selection_df: pd.DataFrame,
    criterion: str = "bic",
    title: str = "Latent-state selection",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(
        selection_df["n_states"],
        selection_df[criterion],
        color=PALETTE["accent"],
        lw=2,
        marker="o",
    )
    best = selection_df.sort_values(criterion).iloc[0]
    ax.scatter(
        [best["n_states"]], [best[criterion]], color=PALETTE["accent2"], s=60, zorder=3
    )
    ax.set_xlabel("Number of hidden states")
    ax.set_ylabel(criterion.upper())
    ax.set_title(title)
    ax.set_xticks(list(selection_df["n_states"]))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_regime_durations(
    runs_df: pd.DataFrame,
    title: str = "Regime Duration Distribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    regimes = sorted(runs_df["regime"].unique())
    data = [runs_df.loc[runs_df["regime"] == r, "duration"].to_numpy() for r in regimes]
    bp = ax.boxplot(
        data, patch_artist=True, tick_labels=[f"Regime {r}" for r in regimes]
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE["regimes"][i % len(PALETTE["regimes"])])
        patch.set_alpha(0.75)
    ax.set_ylabel("Duration (weeks)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_walk_forward_errors(
    weeks: Sequence,
    errors: Dict[str, np.ndarray],
    title: str = "Walk-forward absolute errors",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    colors = [PALETTE["accent"], PALETTE["accent2"], "#3a6ea5", "#b08900"]
    for i, (name, err) in enumerate(errors.items()):
        ax.plot(
            weeks, err, lw=1.3, alpha=0.85, color=colors[i % len(colors)], label=name
        )
    ax.set_title(title)
    ax.set_xlabel("Week")
    ax.set_ylabel("|error|")
    ax.legend(frameon=False)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig


def plot_dm_bars(
    dm_results: Dict[str, Dict[str, float]],
    title: str = "Diebold–Mariano vs Persistence",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    _style()
    names = list(dm_results.keys())
    stats_ = [dm_results[n]["dm_stat"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [PALETTE["accent"] if s < 0 else PALETTE["accent2"] for s in stats_]
    ax.barh(names, stats_, color=colors)
    ax.axvline(0, color=PALETTE["ink"], lw=1)
    ax.set_xlabel("DM statistic (negative ⇒ better than baseline)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
