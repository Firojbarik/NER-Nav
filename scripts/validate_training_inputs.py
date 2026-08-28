#!/usr/bin/env python3
"""Validate provenance and labels before any real-data training step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.data.training_gate import (  # noqa: E402
    TrainingDataGap,
    validate_negative_observation_file,
    validate_event_registry,
    validate_negative_sources,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path,
                        default=ROOT / "data/processed/ml/real_temporal_risk_dataset.parquet")
    args = parser.parse_args()
    registry = validate_event_registry()
    negative_file = validate_negative_observation_file()
    negative = None
    if args.dataset.exists():
        negative = validate_negative_sources(pd.read_parquet(args.dataset))
    result = {"registry": registry, "negative_file": negative_file,
              "negative_labels": negative}
    print(json.dumps(result, indent=2, default=str))
    failures = list(registry["errors"])
    failures.extend(negative_file["errors"])
    if negative:
        failures.extend(negative["errors"])
    if failures:
        print("\nDATA GAP: training inputs are not eligible", file=sys.stderr)
        return 2
    print("\nPASS: training input provenance and labels are eligible")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, TrainingDataGap) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(2)
