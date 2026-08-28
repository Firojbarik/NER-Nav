#!/usr/bin/env python3
"""
Prediction traceability module for NER-Nav hazard classifier.

Every prediction is logged with full traceability:
- prediction_id: unique identifier
- road_segment_id: OSM way ID
- prediction_timestamp: when prediction was made
- model_version: tag of model used
- feature_version: feature spec hash/version
- input_features: all feature values used
- raw_probability: uncalibrated model output
- calibrated_probability: after isotonic calibration
- risk_level: HIGH/MEDIUM/LOW
- top_contributing_factors: explainability output
- inference_timestamp: when inference completed
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACE_DIR = PROJECT_ROOT / "data" / "predictions" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# The legacy files are retained as an immutable audit archive. New writes are
# stream-separated so unit/demo fixtures cannot contaminate production traces.
TRACE_FILE = TRACE_DIR / "prediction_traces.parquet"
TRACE_FILE_JSONL = TRACE_DIR / "prediction_traces.jsonl"
STREAM_FILES = {
    "production": (TRACE_DIR / "production_prediction_traces.parquet",
                    TRACE_DIR / "production_prediction_traces.jsonl"),
    "test": (TRACE_DIR / "test_prediction_traces.parquet",
             TRACE_DIR / "test_prediction_traces.jsonl"),
}


def _feature_version_hash(features: dict[str, Any]) -> str:
    """Compute the same feature-schema version used by model training."""
    import hashlib
    schema = json.dumps(list(features.keys()), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(schema.encode("ascii")).hexdigest()


def log_prediction(
    road_segment_id: str,
    prediction_timestamp: str,
    model_tag: str,
    features: dict[str, Any],
    raw_probability: float,
    calibrated_probability: float,
    risk_level: str,
    top_contributing_factors: list[dict[str, Any]],
    threshold: float,
    ref: str | None = None,
    state: str | None = None,
    district: str | None = None,
    dataset_version: str | None = None,
    risk_policy_version: str | None = None,
    operating_decision: str | None = None,
    data_quality: dict[str, Any] | None = None,
    trace_stream: str = "test",
    input_references: dict[str, Any] | None = None,
) -> str:
    """
    Log a prediction with full traceability.
    
    Returns:
        prediction_id: unique identifier for this prediction
    """
    if trace_stream not in STREAM_FILES:
        raise ValueError(f"unknown trace_stream: {trace_stream}")
    prediction_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    feat_version = _feature_version_hash(features)

    trace = {
        "prediction_id": prediction_id,
        "record_type": "PREDICTION",
        "trace_stream": trace_stream,
        "road_segment_id": road_segment_id,
        "ref": ref,
        "state": state,
        "district": district,
        "prediction_timestamp": prediction_timestamp,
        "prediction_horizon_days": 7,
        "model_version": model_tag,
        "dataset_version": dataset_version,
        "feature_version": feat_version,
        "input_features": json.dumps(features, sort_keys=True),
        "raw_probability": raw_probability,
        "calibrated_probability": calibrated_probability,
        "disruption_probability": calibrated_probability,
        "risk_level": risk_level,
        "risk_policy_version": risk_policy_version,
        "operating_decision": operating_decision,
        "data_quality": json.dumps(data_quality or {}, sort_keys=True),
        "input_references": json.dumps(input_references or {}, sort_keys=True),
        "threshold": threshold,
        "top_contributing_factors": json.dumps(top_contributing_factors, sort_keys=True),
        "inference_timestamp": now,
    }

    # Append to JSONL (always works, append-only)
    trace_jsonl = {k: v for k, v in trace.items()}
    trace_file, trace_file_jsonl = STREAM_FILES[trace_stream]
    with open(trace_file_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace_jsonl, ensure_ascii=False) + "\n")

    # Also append to Parquet (periodic batch)
    _append_to_parquet(trace, trace_file)

    return prediction_id


def _append_to_parquet(trace: dict[str, Any], trace_file: Path) -> None:
    """Append trace to Parquet file (batch writes for efficiency)."""
    df = pd.DataFrame([trace])
    try:
        if trace_file.exists():
            existing = pd.read_parquet(trace_file)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df
        combined.to_parquet(trace_file, index=False)
    except Exception:
        # Parquet append failed; JSONL is the source of truth
        pass


def get_prediction_traces(
    start_date: str | None = None,
    end_date: str | None = None,
    model_version: str | None = None,
    risk_level: str | None = None,
    trace_stream: str | None = None,
) -> pd.DataFrame:
    """Query prediction traces with optional filters."""
    streams = [trace_stream] if trace_stream else list(STREAM_FILES)
    frames = []
    for stream in streams:
        if stream not in STREAM_FILES:
            raise ValueError(f"unknown trace_stream: {stream}")
        parquet, jsonl = STREAM_FILES[stream]
        if parquet.exists():
            frames.append(pd.read_parquet(parquet))
        elif jsonl.exists():
            frames.append(pd.read_json(jsonl, lines=True))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    if start_date:
        df = df[df["inference_timestamp"] >= start_date]
    if end_date:
        df = df[df["inference_timestamp"] <= end_date]
    if model_version:
        df = df[df["model_version"] == model_version]
    if risk_level:
        df = df[df["risk_level"] == risk_level]

    return df


def get_trace_summary() -> dict[str, Any]:
    """Get summary statistics of prediction traces."""
    df = get_prediction_traces()
    if df.empty:
        return {"total_predictions": 0, "by_risk_level": {}, "by_model": {}}

    return {
        "total_predictions": len(df),
        "by_risk_level": df["risk_level"].value_counts().to_dict(),
        "by_model": df["model_version"].value_counts().to_dict(),
        "date_range": {
            "first": df["inference_timestamp"].min(),
            "last": df["inference_timestamp"].max(),
        },
        "avg_probability": float(df["calibrated_probability"].mean()),
    }


if __name__ == "__main__":
    # Demo
    pred_id = log_prediction(
        road_segment_id="44884968",
        prediction_timestamp="2025-07-15T10:00:00Z",
        model_tag="2026-08-28",
        features={"rainfall_7day": 200, "slope_degrees": 25, "elevation_m": 1200},
        raw_probability=0.78,
        calibrated_probability=1.0,
        risk_level="HIGH",
        top_contributing_factors=[
            {"feature": "slope_x_rain7", "value": 5000, "contribution": 0.56}
        ],
        threshold=0.48,
        ref="NH37",
        state="MANIPUR",
        trace_stream="test",
    )
    print(f"Logged prediction: {pred_id}")
    print(f"Summary: {get_trace_summary()}")
