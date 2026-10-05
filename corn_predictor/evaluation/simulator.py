"""Simple regime-conditioned long/flat research simulator.

Not investment advice. Maps decoded regimes to a binary position and reports
research performance statistics versus buy-and-hold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class HedgeSimResult:
    equity_curve: pd.Series
    buy_hold: pd.Series
    positions: pd.Series
    stats: Dict[str, float]


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())


def simulate_regime_long_flat(
    prices: np.ndarray,
    states: np.ndarray,
    long_regimes: Sequence[int],
    weeks: Optional[pd.DatetimeIndex] = None,
    cost_bps: float = 1.0,
) -> HedgeSimResult:
    """Long the asset only when the *previous* decoded regime is in ``long_regimes``.

    Position at t uses state[t-1] (no look-ahead). Transaction cost applied on
    position changes in basis points of notional.
    """
    prices = np.asarray(prices, dtype=float)
    states = np.asarray(states, dtype=int)
    if len(prices) != len(states):
        raise ValueError("prices and states must have the same length")

    rets = np.diff(prices) / prices[:-1]
    long_set = set(int(x) for x in long_regimes)
    # position for return from t->t+1 decided by state at t
    pos = np.array([1.0 if s in long_set else 0.0 for s in states[:-1]], dtype=float)
    turnover = np.abs(np.diff(pos, prepend=pos[0]))
    costs = turnover * (cost_bps / 1e4)
    strat_rets = pos * rets - costs

    equity = np.cumprod(1.0 + strat_rets)
    bh = prices[1:] / prices[0]

    idx = weeks[1:] if weeks is not None else pd.RangeIndex(len(equity))
    eq = pd.Series(equity, index=idx, name="strategy")
    bh_s = pd.Series(bh, index=idx, name="buy_hold")
    pos_s = pd.Series(pos, index=idx, name="position")

    # annualize assuming 52 weeks
    ann = 52.0
    strat_vol = float(np.std(strat_rets) * np.sqrt(ann))
    strat_mean = float(np.mean(strat_rets) * ann)
    sharpe = strat_mean / strat_vol if strat_vol > 1e-12 else 0.0
    stats = {
        "total_return": float(equity[-1] - 1.0),
        "buy_hold_return": float(bh[-1] - 1.0),
        "ann_vol": strat_vol,
        "ann_return": strat_mean,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(equity),
        "avg_exposure": float(pos.mean()),
        "n_trades": float(turnover.sum()),
        "cost_bps": float(cost_bps),
    }
    return HedgeSimResult(eq, bh_s, pos_s, stats)


def choose_long_regimes_by_mean_return(
    states: np.ndarray,
    prices: np.ndarray,
) -> list[int]:
    """Pick regimes whose in-sample mean return is >= overall mean return."""
    states = np.asarray(states, dtype=int)
    prices = np.asarray(prices, dtype=float)
    rets = np.diff(np.log(prices))
    # align return t-1->t with state at t
    overall = float(np.mean(rets))
    chosen = []
    for k in sorted(np.unique(states[1:])):
        mask = states[1:] == k
        if mask.any() and float(np.mean(rets[mask])) >= overall:
            chosen.append(int(k))
    if not chosen:
        # fall back to the single best mean-return regime
        means = {
            int(k): float(np.mean(rets[states[1:] == k]))
            for k in np.unique(states[1:])
            if (states[1:] == k).any()
        }
        chosen = [max(means, key=means.get)]
    return chosen
