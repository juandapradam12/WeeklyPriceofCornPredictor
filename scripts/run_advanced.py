#!/usr/bin/env python3
"""Run advanced enhancements: sticky HMM, soft RS-AR, exogenous, calibration, hedge sim."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_predictor.advanced_pipeline import AdvancedConfig, run_advanced_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="corn2013-2017.txt")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--skip-exogenous", action="store_true")
    args = parser.parse_args()

    cfg = AdvancedConfig(
        data_file=args.data_file,
        test_ratio=args.test_ratio,
        n_states=args.n_states,
        figures_dir=Path(args.figures_dir),
        reports_dir=Path(args.reports_dir),
        fetch_exogenous=not args.skip_exogenous,
    )
    result = run_advanced_experiment(cfg)

    print("\nAdvanced holdout metrics:\n")
    print(result["metrics"].round(4).to_string())
    print("\nCalibration (Sticky HMM):\n")
    print(pd_series := __import__("pandas").Series(result["calibration"]).round(4))
    print(f"\nExogenous status: {result['exo_status']}")
    if result["exo_metrics"] is not None and len(result["exo_metrics"]):
        print(result["exo_metrics"].round(4).to_string())
    print("\nHedge simulator stats:\n")
    print(__import__("pandas").Series(result["hedge_stats"]).round(4).to_string())
    print(f"Long regimes: {result['long_regimes']}")
    print(f"\nWrote {len(result['figures'])} figures")


if __name__ == "__main__":
    main()
