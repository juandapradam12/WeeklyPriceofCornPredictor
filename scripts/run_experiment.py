#!/usr/bin/env python3
"""Run the corn price regime-model comparison and print a metrics table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running without installation
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from corn_predictor.pipeline import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default="corn2013-2017.txt")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--n-symbols", type=int, default=5)
    parser.add_argument("--discrete-states", type=int, default=2)
    parser.add_argument("--gaussian-states", type=int, default=3)
    parser.add_argument("--figures-dir", default="figures")
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
        discrete_states=args.discrete_states,
        gaussian_states=args.gaussian_states,
        figures_dir=Path(args.figures_dir),
    )
    result = run_experiment(cfg)
    table = result["metrics"]
    print("\nHoldout metrics (sorted by RMSE):\n")
    print(table.round(4).to_string())
    print(f"\nWrote {len(result['figures'])} figures to {cfg.figures_dir.resolve()}")

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(table.round(6).to_json(orient="index", indent=2))
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
