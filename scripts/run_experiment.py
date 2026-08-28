"""CLI entry point for the temporal-disruption experiment harness.

Runs the full pipeline (real features -> missingness -> synthetic labels ->
temporal split -> XGBoost baseline -> evaluation) and prints a JSON summary.

USAGE:
    python scripts/run_experiment.py [--road-sample N] [--n-estimators N]

WARNING:
    Without real source-supported hazard labels, the harness uses SYNTHETIC
    labels and does NOT produce a real performance claim. See ml/experiment.py.
"""

import argparse
import json
import sys

sys.path.insert(0, "")

from ml.experiment import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--road-sample", type=int, default=20000)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--missingness", type=str, default="xgb_native_flag")
    args = parser.parse_args()

    config = ExperimentConfig(
        road_sample=args.road_sample,
        n_estimators=args.n_estimators,
        missingness_strategy=args.missingness,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
