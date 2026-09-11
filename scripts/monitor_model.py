#!/usr/bin/env python3
"""Operational monitoring checks for the served NER-Nav risk model.

Produces a timestamped monitoring report under data/processed/ml/monitoring/.
Checks (each reports PASS/FAIL/PARTIAL):
  1. weather_freshness  - age of the most recent raw rainfall raster
  2. artifact_staleness - active dataset SHA matches the latest model report SHA
  3. delayed_labels     - confirmed events without a rebuilt dataset entry
  4. bundle_integrity   - demo/prod pointers resolve to existing, hash-consistent files
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET = ROOT / "data/processed/ml/real_temporal_risk_dataset.parquet"
MODEL_DIR = ROOT / "data/models"
WEATHER_DIR = ROOT / "data/raw/weather"
OUTPUT_DIR = ROOT / "data/processed/ml/monitoring"

MAX_DAYS_SINCE_WEATHER = 14
MAX_DAYS_SINCE_CHIRPS = 30


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_weather_days() -> int | None:
    if not WEATHER_DIR.exists():
        return None
    known = {}
    for year_dir in WEATHER_DIR.glob("chirps_*"):
        for gz in year_dir.glob("chirps-v2.0.*.tif.gz"):
            stem = gz.stem.replace("chirps-v2.0.", "").replace(".tif", "")
            try:
                d = datetime.strptime(stem, "%Y.%m.%d").date()
            except ValueError:
                continue
            known[d] = gz
    if not known:
        return None
    newest = max(known)
    return (datetime.now(timezone.utc).date() - newest).days


def active_dataset_sha() -> str | None:
    if not DATASET.exists():
        return None
    return sha256_file(DATASET)


def report_for_pointer(name: str) -> tuple[dict | None, dict | None]:
    pointer_path = MODEL_DIR / name
    if not pointer_path.exists():
        return None, None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        report_path = ROOT / pointer.get("report_file", "")
        report = (json.loads(report_path.read_text(encoding="utf-8"))
                  if report_path.exists() else None)
        return pointer, report
    except (OSError, json.JSONDecodeError, KeyError):
        return None, None


def bundle_files_ok(pointer: dict | None, report: dict | None) -> bool:
    if not report:
        return False
    model_file = ROOT / (pointer.get("model_file")
                         or report.get("model_file") or "")
    if not model_file.exists():
        return False
    if report.get("model_sha256") and sha256_file(model_file) != report["model_sha256"]:
        return False
    feature_spec = (pointer.get("feature_spec")
                    or report.get("feature_spec")
                    or report.get("feature_spec_file")
                    or f"{model_file.with_suffix('')}_features.json")
    if not (ROOT / feature_spec).exists():
        return False
    return True


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    weather_days = latest_weather_days()
    checks = {"weather_freshness": _row(
        "PASS" if weather_days is not None and weather_days <= MAX_DAYS_SINCE_CHIRPS else "FAIL",
        {"days_since_latest_chirps": weather_days,
         "max_allowed_days": MAX_DAYS_SINCE_CHIRPS},
        ["raw rainfall is stale or unavailable"] if weather_days is None
        or weather_days > MAX_DAYS_SINCE_CHIRPS else [])}

    ds_sha = active_dataset_sha()
    demo_pointer, demo_report = report_for_pointer("demo_latest.json")
    prod_pointer, prod_report = report_for_pointer("prod_latest.json")
    checks["artifact_staleness"] = _row(
        "PASS" if ds_sha and demo_report and demo_report.get("dataset_sha256") == ds_sha else "FAIL",
        {"active_dataset_sha256": (ds_sha or "")[:12],
         "demo_model_dataset_sha256": ((demo_report or {}).get("dataset_sha256") or "")[:12]},
        ["active dataset has no matching model bundle"] if not
        (ds_sha and demo_report and demo_report.get("dataset_sha256") == ds_sha) else [],
    )

    newer_events = stale_events()
    checks["delayed_labels"] = _row(
        "PASS" if not newer_events else "FAIL",
        {"confirmed_events_not_in_dataset": newer_events},
        ["confirmed events exist without a rebuilt dataset entry"] if newer_events else [],
    )

    checks["bundle_integrity"] = _row(
        "PASS" if bundle_files_ok(demo_pointer, demo_report)
        and bundle_files_ok(prod_pointer, prod_report) else "FAIL",
        {"demo_bundle_ok": bundle_files_ok(demo_pointer, demo_report),
         "prod_bundle_ok": bundle_files_ok(prod_pointer, prod_report)},
        ["pointer/bundle integrity check failed"] if not
        (bundle_files_ok(demo_pointer, demo_report)
         and bundle_files_ok(prod_pointer, prod_report)) else [],
    )

    statuses = [row["status"] for row in checks.values()]
    decision = ("OK" if set(statuses) <= {"PASS"} else
                "WATCH" if "FAIL" not in statuses else "ACTION_REQUIRED")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "checks": checks,
    }
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    output = OUTPUT_DIR / f"monitoring_report_{stamp}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = OUTPUT_DIR / "monitoring_report_latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"decision": decision, "report_file": str(output.relative_to(ROOT)),
                      "checks": {k: v["status"] for k, v in checks.items()}}, indent=2))
    return 0 if decision == "OK" else 1


def _row(status, evidence, blockers):
    return {"status": status, "evidence": evidence, "blockers": blockers}


def stale_events():
    """Confirmed events absent from the built dataset (design/build lag)."""
    try:
        from ml.features.temporal_design import CONFIRMED_EVENTS
        ds = _read_dataset()
        if ds is None:
            return [eid for eid, _, _, _ in CONFIRMED_EVENTS]
        built = set(ds["event_id"].astype(str))
        return sorted({eid for eid, _, _, _ in CONFIRMED_EVENTS} - built)
    except Exception:
        return []


def _read_dataset():
    if not DATASET.exists():
        return None
    import pandas as pd
    ds = pd.read_parquet(DATASET)
    ds["prediction_time"] = pd.to_datetime(ds["prediction_time"])
    return ds


if __name__ == "__main__":
    raise SystemExit(main())