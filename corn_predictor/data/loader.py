"""Data loading utilities for weekly corn price series."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_prices(
    filename: str = "corn2013-2017.txt",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load a weekly price series with columns ``week`` and ``price``."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / filename
    df = pd.read_csv(path, names=["week", "price"])
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)
    return df


def load_ohlc(
    filename: str = "corn_OHLC2013-2017.txt",
    data_dir: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Load weekly OHLC corn futures with derived mid/range features."""
    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    path = data_dir / filename
    df = pd.read_csv(path, names=["week", "open", "high", "low", "close"])
    df["week"] = pd.to_datetime(df["week"])
    df = df.sort_values("week").reset_index(drop=True)
    df["mid"] = (df["high"] + df["low"]) / 2.0
    df["range"] = df["high"] - df["low"]
    df["return"] = np.log(df["close"]).diff()
    return df


def add_features(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """Add log-returns and simple rolling features used by complementary models."""
    out = df.copy()
    out["log_price"] = np.log(out[price_col])
    out["log_return"] = out["log_price"].diff()
    out["abs_return"] = out["log_return"].abs()
    out["roll_mean_4"] = out[price_col].rolling(4, min_periods=1).mean()
    out["roll_std_4"] = out["log_return"].rolling(4, min_periods=1).std()
    return out


def train_test_split_time(
    df: pd.DataFrame,
    test_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological train/test split (no shuffle)."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    n = len(df)
    split = int(n * (1.0 - test_ratio))
    return df.iloc[:split].copy(), df.iloc[split:].copy()
