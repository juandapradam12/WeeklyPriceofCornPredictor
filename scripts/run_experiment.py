#!/usr/bin/env python3
"""Run the corn price regime-model comparison and print a metrics table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_predictor.pipeline import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="corn2013-2017.txt")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--n-symbols", type=int, default=5)
    parser.add_argument("--figures-dir", default="figures")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--skip-walk-forward", action="store_true")
    parser.add_argument(
        "--metrics-json",
        default="figures/metrics.json",
        help="Optional path to write metrics as JSON",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig(
        data_file=args.data_file,
        test_ratio=args.test_ratio,
        n_symbols=args.n_symbols,
        figures_dir=Path(args.figures_dir),
        reports_dir=Path(args.reports_dir),
        run_walk_forward=not args.skip_walk_forward,
    )
    result = run_experiment(cfg)
    table = result["metrics"]
    print("\nBIC-selected states:")
    print(f"  discrete HMM: {result['best_discrete_states']}")
    print(f"  gaussian HMM: {result['best_gaussian_states']}")
    print("\nHoldout metrics (sorted by RMSE):\n")
    print(table.round(4).to_string())
    print("\nDiebold–Mariano vs Persistence (sorted by p-value):\n")
    print(result["dm"].round(4).to_string())
    print("\nRegime summary:\n")
    print(result["regime_summary"].round(4).to_string(index=False))
    if result["walk_forward_metrics"]:
        print("\nWalk-forward metrics:\n")
        import pandas as pd

        print(pd.DataFrame(result["walk_forward_metrics"]).T.round(4).to_string())
    print(f"\nWrote {len(result['figures'])} figures to {cfg.figures_dir.resolve()}")
    print(f"Reports: {cfg.reports_dir.resolve()}")

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(table.round(6).to_json(orient="index", indent=2))
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
