#!/usr/bin/env python3
"""Refresh live corn + exogenous weekly caches from Yahoo Finance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_predictor.data.live import fetch_corn_weekly, fetch_exogenous_weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--no-cache-read", action="store_true", help="Force re-download")
    args = parser.parse_args()

    corn = fetch_corn_weekly(
        start=args.start, end=args.end, use_cache=not args.no_cache_read
    )
    # Always rewrite cache after fetch path; force download:
    corn = fetch_corn_weekly(start=args.start, end=args.end, use_cache=False)
    exo = fetch_exogenous_weekly(start=args.start, end=args.end, use_cache=False)
    print(f"Corn weekly rows: {len(corn)}  [{corn['week'].min()} → {corn['week'].max()}]")
    print(f"Exogenous rows:   {len(exo)}  columns={list(exo.columns)}")
    print("Cached under data/cache/")


if __name__ == "__main__":
    main()
