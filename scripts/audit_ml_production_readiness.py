#!/usr/bin/env python3
"""Audit the active real-data ML system against production acceptance criteria."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features.temporal_design import HORIZON_DAYS, event_date  # noqa: E402

DATASET = ROOT / "data/processed/ml/real_temporal_risk_dataset.parquet"
DATASET_QA = ROOT / "data/processed/ml/real_temporal_risk_dataset_qa.json"
MODEL_DIR = ROOT / "data/models"
OUTPUT_DIR = ROOT / "data/processed/ml/readiness_audits"

REQUIRED_COLUMNS = {
    "event_id", "osm_id", "prediction_time", "label", "sample_kind",
    "label_source", "state", "district", "ref",
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day", "elevation_m", "slope_degrees",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check(status, evidence, blockers=None):
    return {
        "status": status,
        "evidence": evidence,
        "blockers": blockers or [],
    }


def audit_dataset(ds):
    """Return objective data, label, spatial, and temporal findings."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(ds.columns))
    duplicate_keys = int(ds.duplicated(
        ["event_id", "osm_id", "prediction_time"]).sum())
    invalid_rainfall = int(sum(
        (pd.to_numeric(ds[column], errors="coerce") < 0).sum()
        for column in ["rainfall_1day", "rainfall_3day", "rainfall_7day",
                       "rainfall_14day", "rainfall_30day"]
    ))
    invalid_slope = int(((ds["slope_degrees"] < 0)
                         | (ds["slope_degrees"] > 90)).sum())
    rainfall_coverage = float(ds["rainfall_30day"].notna().mean())
    terrain_coverage = float(ds["slope_degrees"].notna().mean())

    temporal_violations = []
    for row in ds.itertuples():
        try:
            disruption_date = pd.Timestamp(event_date(row.event_id))
        except ValueError:
            temporal_violations.append(
                {"event_id": row.event_id, "reason": "event date unavailable"})
            continue
        prediction_time = pd.Timestamp(row.prediction_time)
        if row.sample_kind == "positive":
            valid = (prediction_time < disruption_date
                     <= prediction_time + pd.Timedelta(days=HORIZON_DAYS))
            if not valid:
                temporal_violations.append({
                    "event_id": row.event_id,
                    "reason": "positive event outside future horizon",
                })
        elif row.sample_kind == "same_road_control":
            if disruption_date <= prediction_time + pd.Timedelta(days=HORIZON_DAYS):
                temporal_violations.append({
                    "event_id": row.event_id,
                    "reason": "known event falls inside negative horizon",
                })

    assumed_negatives = int(
        ((ds["label"] == 0)
         & (ds["label_source"] == "assumed_unaffected_real_pool")).sum())
    all_negatives = int((ds["label"] == 0).sum())
    confirmed_negative_count = int(
        ((ds["label"] == 0)
         & ds["label_source"].astype(str).str.contains(
             "confirmed_unaffected", case=False, na=False)).sum())

    return {
        "counts": {
            "samples": int(len(ds)),
            "events": int(ds["event_id"].nunique()),
            "roads": int(ds["osm_id"].nunique()),
            "positives": int(ds["label"].sum()),
            "negatives": all_negatives,
            "states": int(ds["state"].nunique()),
            "districts": int(ds["district"].nunique()),
            "corridors": int(ds["ref"].nunique()),
        },
        "quality": {
            "missing_required_columns": missing_columns,
            "duplicate_sample_keys": duplicate_keys,
            "invalid_rainfall_values": invalid_rainfall,
            "invalid_slope_values": invalid_slope,
            "rainfall_coverage": rainfall_coverage,
            "terrain_coverage": terrain_coverage,
            "missingness_by_column": {
                name: float(value)
                for name, value in ds.isna().mean().sort_values(
                    ascending=False).items()
                if value > 0
            },
        },
        "labels": {
            "assumed_negative_count": assumed_negatives,
            "confirmed_negative_count": confirmed_negative_count,
            "assumed_negative_fraction": (
                float(assumed_negatives / all_negatives) if all_negatives else None
            ),
            "positive_label_sources": ds.loc[
                ds["label"] == 1, "label_source"].value_counts().to_dict(),
            "negative_label_sources": ds.loc[
                ds["label"] == 0, "label_source"].value_counts().to_dict(),
        },
        "temporal": {
            "prediction_start": pd.Timestamp(ds["prediction_time"].min()).isoformat(),
            "prediction_end": pd.Timestamp(ds["prediction_time"].max()).isoformat(),
            "violations": temporal_violations,
        },
        "geography": {
            "states": ds.groupby("state")["event_id"].nunique().to_dict(),
            "corridors": ds.groupby("ref")["event_id"].nunique().to_dict(),
            "identity_field": "osm_id",
            "stable_internal_road_segment_id_present": (
                "road_segment_id" in ds.columns),
        },
    }


def load_latest_model():
    pointer_path = MODEL_DIR / "prod_latest.json"
    if not pointer_path.exists():
        return None, ["prod_latest.json is missing"]
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    report_path = ROOT / pointer["report_file"]
    if not report_path.exists():
        return None, [f"latest report is missing: {report_path}"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    errors = []
    model_path = ROOT / report["model_file"]
    if not model_path.exists():
        errors.append(f"model file is missing: {model_path}")
    elif report.get("model_sha256") != sha256_file(model_path):
        errors.append("model SHA256 mismatch")
    return report, errors


def build_acceptance(dataset_audit, model_report, lineage_errors, qa_matches):
    counts = dataset_audit["counts"]
    quality = dataset_audit["quality"]
    labels = dataset_audit["labels"]
    temporal = dataset_audit["temporal"]
    geography = dataset_audit["geography"]
    model_metrics = (model_report or {}).get("test_metrics", {})
    split = (model_report or {}).get("split", {})

    acceptance = {}
    acceptance["DATA"] = _check(
        "PASS" if quality["rainfall_coverage"] >= 0.95
        and quality["terrain_coverage"] >= 0.95 else "FAIL",
        {"rainfall_coverage": quality["rainfall_coverage"],
         "terrain_coverage": quality["terrain_coverage"]},
        ["Feature coverage is below the 95% production floor"]
        if min(quality["rainfall_coverage"], quality["terrain_coverage"]) < 0.95 else [],
    )
    acceptance["LABELS"] = _check(
        "FAIL",
        labels,
        ["No source-confirmed unaffected negatives exist",
         "Absence of a recorded event is not verified non-occurrence"],
    )
    acceptance["GEOSPATIAL_PIPELINE"] = _check(
        "FAIL" if not geography["stable_internal_road_segment_id_present"] else "PASS",
        geography,
        ["OSM way IDs are source identifiers, not a versioned stable segment identity"]
        if not geography["stable_internal_road_segment_id_present"] else [],
    )
    acceptance["TEMPORAL_PIPELINE"] = _check(
        "PASS" if not temporal["violations"] else "FAIL", temporal,
        ["Temporal label violations detected"] if temporal["violations"] else [],
    )
    acceptance["LEAKAGE"] = _check(
        "PASS" if not temporal["violations"] else "FAIL",
        {"future_feature_rule": "rainfall windows use dates before prediction time",
         "event_split": "event-disjoint chronological split",
         "verified_violations": len(temporal["violations"])},
    )
    test_events = int(split.get("n_test_events", 0))
    acceptance["VALIDATION"] = _check(
        "PASS" if test_events >= 30 else "FAIL",
        {"future_test_events": test_events,
         "required_future_test_events": 30,
         "test_period": split.get("test_period")},
        [f"Only {test_events} future test events; at least 30 required"],
    )
    production_gate = (model_report or {}).get("gates", {}).get("production", {})
    acceptance["MODEL"] = _check(
        "PASS" if production_gate.get("passed") else "FAIL",
        {"metrics": model_metrics, "production_gate": production_gate},
        production_gate.get("reasons", ["No current model report"]),
    )
    calibration_ok = (
        model_report is not None
        and model_metrics.get("brier_score") is not None
        and int(split.get("n_val_events", 0)) >= 10
    )
    acceptance["CALIBRATION"] = _check(
        "PASS" if calibration_ok else "FAIL",
        {"method": (model_report or {}).get("calibration"),
         "validation_events": split.get("n_val_events"),
         "brier_score": model_metrics.get("brier_score")},
        ["Calibration has too few independent validation events"]
        if not calibration_ok else [],
    )
    acceptance["ROBUSTNESS"] = _check(
        "PARTIAL", {"input_validation": True, "OOD_flagging": True,
                    "live_dependency_failure_tests": False},
        ["External feed outage and stale-data integration tests are not implemented"],
    )
    lineage_ok = model_report is not None and not lineage_errors and qa_matches
    acceptance["INFERENCE"] = _check(
        "PASS" if lineage_ok else "FAIL",
        {"artifact_integrity_errors": lineage_errors,
         "dataset_qa_matches_active_dataset": qa_matches},
        lineage_errors + ([] if qa_matches else ["Dataset QA counts do not match"]),
    )
    acceptance["REPRODUCIBILITY"] = _check(
        "FAIL", {"immutable_model_version": bool(model_report),
                 "raw_source_manifest_complete": False},
        ["Not all weather, terrain, roads, boundaries, and incident inputs have a "
         "single complete checksum manifest",
         "Confirmed events remain maintained in a code list"],
    )
    acceptance["TRACEABILITY"] = _check(
        "PASS" if lineage_ok else "FAIL",
        {"dataset_sha256": (model_report or {}).get("dataset_sha256"),
         "model_sha256": (model_report or {}).get("model_sha256"),
         "feature_version": (model_report or {}).get("feature_version")},
        lineage_errors,
    )
    acceptance["MONITORING"] = _check(
        "FAIL", {"strategy_documented": True, "automated_monitoring_job": False},
        ["Monitoring thresholds are documented but not operationally implemented"],
    )
    documentation_files = [
        ROOT / "docs/ML_DATA_CATALOG.md",
        ROOT / "docs/ML_PRODUCTION_READINESS_REPORT.md",
        ROOT / "docs/MONITORING_STRATEGY.md",
        ROOT / "docs/API_CONTRACT.json",
    ]
    missing_documentation = [str(path.relative_to(ROOT))
                             for path in documentation_files if not path.exists()]
    acceptance["DOCUMENTATION"] = _check(
        "PASS" if not missing_documentation else "FAIL",
        {"required_files": [str(path.relative_to(ROOT))
                            for path in documentation_files]},
        [f"Missing documentation: {name}" for name in missing_documentation],
    )
    alert_rate = None
    if model_metrics.get("confusion_matrix"):
        matrix = np.asarray(model_metrics["confusion_matrix"])
        alert_rate = float((matrix[:, 1].sum()) / matrix.sum())
    acceptance["OPERATIONAL_USEFULNESS"] = _check(
        "PASS" if alert_rate is not None and alert_rate <= 0.30 else "FAIL",
        {"test_alert_rate": alert_rate, "maximum_reviewable_alert_rate": 0.30},
        ["Recall-oriented threshold produces an unreviewable false-alert load"]
        if alert_rate is not None and alert_rate > 0.30 else [],
    )
    return acceptance


def main():
    ds = pd.read_parquet(DATASET)
    ds["prediction_time"] = pd.to_datetime(ds["prediction_time"])
    dataset_sha = sha256_file(DATASET)
    dataset_audit = audit_dataset(ds)

    qa = json.loads(DATASET_QA.read_text(encoding="utf-8"))
    qa_matches = (
        qa.get("n_positive_labels") == dataset_audit["counts"]["positives"]
        and qa.get("n_negative_labels") == dataset_audit["counts"]["negatives"]
    )
    model_report, lineage_errors = load_latest_model()
    if model_report and model_report.get("dataset_sha256") != dataset_sha:
        lineage_errors.append(
            "latest model was not trained on the active dataset SHA256")

    acceptance = build_acceptance(
        dataset_audit, model_report, lineage_errors, qa_matches)
    data_blockers = (
        acceptance["DATA"]["status"] == "FAIL"
        or acceptance["LABELS"]["status"] == "FAIL"
    )
    decision = "BLOCKED BY DATA GAP" if data_blockers else "NOT PRODUCTION READY"

    generated_at = datetime.now(timezone.utc)
    report = {
        "generated_at": generated_at.isoformat(),
        "decision": decision,
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)),
            "sha256": dataset_sha,
            **dataset_audit,
        },
        "model": {
            "model_version": (model_report or {}).get("model_version"),
            "status": (model_report or {}).get("status"),
            "lineage_errors": lineage_errors,
        },
        "acceptance": acceptance,
        "priority": {
            "P0": [
                "Replace assumed-unaffected negatives with observation-backed negatives",
                "Establish a versioned stable road_segment_id mapping",
                "Make all raw-to-model inputs checksum-manifested and reproducible",
            ],
            "P1": [
                "Expand to at least 30 independent future test events",
                "Raise rainfall and terrain coverage to at least 95%",
                "Add independent events across underrepresented states, seasons, and roads",
            ],
            "P2": [
                "Validate calibration with at least 10 independent validation events",
                "Operationalize freshness, drift, and delayed-label monitoring",
                "Reduce false-alert load while preserving safety-oriented recall",
            ],
            "P3": ["Evaluate additional algorithms only after P0/P1 are resolved"],
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_{dataset_sha[:8]}"
    output = OUTPUT_DIR / f"production_readiness_audit_{stem}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = ROOT / "data/processed/ml/production_readiness_audit_latest.json"
    latest.write_text(json.dumps({
        "audit_file": str(output.relative_to(ROOT)),
        "decision": decision,
        "generated_at": generated_at.isoformat(),
    }, indent=2), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "audit_file": str(output.relative_to(ROOT)),
        "failed": [name for name, value in acceptance.items()
                   if value["status"] == "FAIL"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
