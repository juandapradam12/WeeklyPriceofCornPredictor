"""Regime occupancy, duration, and summary analytics."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def regime_runs(states: np.ndarray) -> pd.DataFrame:
    """Collapse a state path into contiguous regime runs."""
    states = np.asarray(states, dtype=int)
    if len(states) == 0:
        return pd.DataFrame(columns=["regime", "start", "end", "duration"])
    runs: List[Dict] = []
    start = 0
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            runs.append(
                {
                    "regime": int(states[start]),
                    "start": start,
                    "end": i - 1,
                    "duration": i - start,
                }
            )
            start = i
    runs.append(
        {
            "regime": int(states[start]),
            "start": start,
            "end": len(states) - 1,
            "duration": len(states) - start,
        }
    )
    return pd.DataFrame(runs)


def summarize_regimes(
    states: np.ndarray,
    prices: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Per-regime occupancy, mean duration, and optional return/price stats."""
    states = np.asarray(states, dtype=int)
    runs = regime_runs(states)
    rows = []
    for k in sorted(np.unique(states)):
        mask = states == k
        row = {
            "regime": int(k),
            "n_weeks": int(mask.sum()),
            "occupancy": float(mask.mean()),
            "n_runs": int((runs["regime"] == k).sum()),
            "mean_duration": float(runs.loc[runs["regime"] == k, "duration"].mean())
            if (runs["regime"] == k).any()
            else 0.0,
            "max_duration": int(runs.loc[runs["regime"] == k, "duration"].max())
            if (runs["regime"] == k).any()
            else 0,
        }
        if prices is not None:
            p = np.asarray(prices, dtype=float)
            row["mean_price"] = float(p[mask].mean())
            row["std_price"] = float(p[mask].std())
        if returns is not None:
            # align returns with states[1:] if lengths differ by 1
            r = np.asarray(returns, dtype=float)
            if len(r) == len(states) - 1:
                r_mask = mask[1:]
            else:
                r_mask = mask[: len(r)]
            row["mean_return"] = float(r[r_mask].mean()) if r_mask.any() else float("nan")
            row["std_return"] = float(r[r_mask].std()) if r_mask.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def transition_counts(states: np.ndarray, n_states: Optional[int] = None) -> np.ndarray:
    states = np.asarray(states, dtype=int)
    n = int(n_states if n_states is not None else states.max() + 1)
    counts = np.zeros((n, n), dtype=float)
    for a, b in zip(states[:-1], states[1:]):
        counts[a, b] += 1
    return counts
