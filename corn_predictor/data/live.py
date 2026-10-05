"""Live market data and exogenous feature construction.

Uses Yahoo Finance (via ``yfinance``) for corn futures and common macro
proxies. Falls back gracefully when offline so unit tests can use local CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"

# Proxies (not official USDA prints): liquid instruments that co-move with ag markets.
DEFAULT_EXOGENOUS = {
    "oil": "CL=F",       # WTI crude
    "dxy": "DX-Y.NYB",   # US Dollar Index
    "wheat": "WEAT",     # wheat equity proxy
    "soy": "SOYB",       # soybean equity proxy
}

CORN_TICKER = "ZC=F"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def download_ticker(
    ticker: str,
    start: str = "2013-01-01",
    end: Optional[str] = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download a daily OHLCV series and return a clean single-index frame."""
    import yfinance as yf

    df = yf.download(
        ticker,
        start=start,
        end=end,
        progress=False,
        auto_adjust=auto_adjust,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for ticker {ticker}")
    df = _flatten_columns(df)
    df = df.rename(columns=str.lower)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


def to_weekly_last(daily: pd.DataFrame, price_col: str = "close") -> pd.Series:
    """Resample daily closes to weekly (Friday) last observation."""
    s = daily[price_col].astype(float)
    weekly = s.resample("W-FRI").last().dropna()
    weekly.name = price_col
    return weekly


def fetch_corn_weekly(
    start: str = "2013-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch weekly corn futures closes (ZC=F), optionally caching to disk."""
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "corn_weekly_live.csv"

    if use_cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["week"])
        df["week"] = pd.to_datetime(df["week"]).astype("datetime64[ns]")
        return df

    daily = download_ticker(CORN_TICKER, start=start, end=end)
    weekly = to_weekly_last(daily, "close").rename("price").reset_index()
    weekly = weekly.rename(columns={"date": "week"})
    if use_cache:
        weekly.to_csv(cache_path, index=False)
    return weekly


def fetch_exogenous_weekly(
    tickers: Optional[Dict[str, str]] = None,
    start: str = "2013-01-01",
    end: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch weekly exogenous proxies and return a wide dataframe on ``week``."""
    tickers = tickers or DEFAULT_EXOGENOUS
    cache_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "exogenous_weekly.csv"

    if use_cache and cache_path.exists():
        out = pd.read_csv(cache_path, parse_dates=["week"])
        out["week"] = pd.to_datetime(out["week"]).astype("datetime64[ns]")
        return out

    frames = []
    for name, ticker in tickers.items():
        daily = download_ticker(ticker, start=start, end=end)
        w = to_weekly_last(daily, "close").rename(name)
        frames.append(w)
    exo = pd.concat(frames, axis=1).dropna(how="all").reset_index().rename(
        columns={"date": "week"}
    )
    if use_cache:
        exo.to_csv(cache_path, index=False)
    return exo


def build_exogenous_features(
    price_df: pd.DataFrame,
    exo_df: pd.DataFrame,
    price_col: str = "price",
) -> pd.DataFrame:
    """Align corn prices with exogenous series and engineer return features.

    Adds columns:
    - ``ret_<name>``: weekly log-return of each exogenous series
    - ``ret_corn``: corn log-return
    - lagged exogenous returns (1 week) for causal forecasting
    """
    left = price_df[["week", price_col]].copy()
    left["week"] = pd.to_datetime(left["week"]).astype("datetime64[ns]")
    right = exo_df.copy()
    right["week"] = pd.to_datetime(right["week"]).astype("datetime64[ns]")

    merged = pd.merge_asof(
        left.sort_values("week"),
        right.sort_values("week"),
        on="week",
        direction="backward",
    )
    merged["ret_corn"] = np.log(merged[price_col]).diff()
    exo_cols = [c for c in right.columns if c != "week"]
    for c in exo_cols:
        merged[f"ret_{c}"] = np.log(merged[c]).diff()
        merged[f"ret_{c}_lag1"] = merged[f"ret_{c}"].shift(1)
    return merged


def make_design_matrix(
    featured: pd.DataFrame,
    include: Sequence[str] = ("ret_oil_lag1", "ret_dxy_lag1", "ret_wheat_lag1"),
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (X, y_returns, weeks) with NaNs dropped for supervised models."""
    cols = list(include)
    frame = featured.dropna(subset=["ret_corn"] + cols).copy()
    X = frame[cols].to_numpy(dtype=float)
    y = frame["ret_corn"].to_numpy(dtype=float)
    weeks = pd.to_datetime(frame["week"])
    return X, y, weeks
