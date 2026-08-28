#!/usr/bin/env python3
"""
Full rebuild pipeline after adding new confirmed events.

Chains:
  1. Download new CHIRPS dates (if needed).
  2. Extract rainfall features for new dates.
  3. Rebuild the real temporal risk dataset.
  4. Retrain the production model (with honest gate).
  5. Run tests.

Usage:
    python scripts/rebuild_pipeline.py [--skip-download] [--skip-tests]

Exits non-zero on any step failure (stops early, does not mask errors).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PYTHON = sys.executable  # use the same venv python


def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}")
    print(f"STEP: {label}")
    print(f"  command: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(REPO))
    if result.returncode != 0:
        print(f"\nFAILED: {label} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> int:
    skip_download = "--skip-download" in sys.argv
    skip_tests = "--skip-tests" in sys.argv

    # Run the fail-closed provenance/label gate before downloads or generated
    # outputs.  A failed gate is a data gap and must not leave partial rebuild
    # artifacts behind.
    run([PYTHON, str(REPO / "scripts" / "validate_training_inputs.py")],
        "Validate source provenance and training labels")

    if not skip_download:
        run([PYTHON, str(REPO / "scripts" / "download_chirps_dates.py")],
            "Download CHIRPS rasters for all needed dates")

    # Compute prediction dates from temporal_design.py and pass to feature extractor.
    # Without explicit dates, process_chirps_features.py processes ALL available
    # CHIRPS dates (500+), which is extremely slow.
    result = subprocess.run(
        [PYTHON, "-c",
         "import sys; sys.path.insert(0,'ml/features'); "
         "from temporal_design import needed_prediction_dates; "
         "print(' '.join(str(d) for d in sorted(needed_prediction_dates())))"],
        capture_output=True, text=True, cwd=str(REPO))
    dates_str = result.stdout.strip()
    dates = dates_str.split() if dates_str else []
    if not dates:
        print("WARNING: no prediction dates found; skipping feature extraction",
              file=sys.stderr)
    else:
        run([PYTHON, str(REPO / "scripts" / "process_chirps_features.py"),
             "--dates"] + dates +
            ["--output",
             str(REPO / "data" / "processed" / "weather" /
                 "road_rainfall_features_temporal.parquet")],
            f"Extract rainfall features for {len(dates)} prediction dates")

    run([PYTHON, str(REPO / "scripts" / "create_ml_input_manifest.py")],
        "Create deterministic ML input checksum manifest")

    run([PYTHON, str(REPO / "scripts" / "build_real_temporal_dataset.py")],
        "Build real temporal risk dataset")

    run([PYTHON, str(REPO / "scripts" / "train_production_risk_model.py")],
        "Train production model (with honest gate)")

    if not skip_tests:
        run([PYTHON, "-m", "unittest", "discover", "-s", "tests"],
            "Run full test suite")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print("Check data/models/prod_report_*.json for the production model status.")
    print("If status=GATED_LOW_CONFIDENCE, add more real events and re-run.")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
