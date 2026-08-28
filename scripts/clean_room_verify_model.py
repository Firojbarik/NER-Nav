#!/usr/bin/env python3
"""Run frozen inference on one real, unseen road observation without retraining."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.predict_real_temporal import (  # noqa: E402
    load_model_bundle, predict, risk_level, validate_and_prepare,
)

TRAINING_DATA = ROOT / "data/processed/ml/real_temporal_risk_dataset.parquet"
RAINFALL_DATA = ROOT / "data/processed/weather/road_rainfall_features_temporal.parquet"
TERRAIN_DATA = ROOT / "data/processed/terrain/road_terrain_features.parquet"
OUTPUT_DIR = ROOT / "data/models/verification_runs"


def main():
    load_started = time.perf_counter()
    bundle = load_model_bundle("latest")
    model_load_ms = (time.perf_counter() - load_started) * 1000

    training = pd.read_parquet(
        TRAINING_DATA, columns=["osm_id", "prediction_time"])
    anchor = pd.Timestamp(training["prediction_time"].max()).date()
    training_ids = set(training["osm_id"].astype("int64"))

    rainfall = pd.read_parquet(
        RAINFALL_DATA,
        filters=[("feature_date", "==", anchor.isoformat())],
    )
    terrain = pd.read_parquet(TERRAIN_DATA)
    real_input = rainfall.merge(terrain, on="osm_id", how="left")
    real_input = real_input[
        ~real_input["osm_id"].astype("int64").isin(training_ids)
        & real_input[[
            "rainfall_1day", "rainfall_3day", "rainfall_7day",
            "rainfall_14day", "rainfall_30day",
        ]].notna().all(axis=1)
    ].sort_values("osm_id")
    if real_input.empty:
        raise RuntimeError("no real unseen observation satisfies the feature contract")

    row = real_input.iloc[0]
    base_values = {
        name: (None if pd.isna(row[name]) else float(row[name]))
        for name in [
            "rainfall_1day", "rainfall_3day", "rainfall_7day",
            "rainfall_14day", "rainfall_30day", "elevation_m", "slope_degrees",
        ]
    }
    prediction_timestamp = f"{anchor.isoformat()}T00:00:00+00:00"
    prepared, quality = validate_and_prepare(
        base_values, bundle["features"], prediction_timestamp,
        weather_observed_at=None, feature_profile=bundle["feature_profile"])

    started = time.perf_counter()
    result = predict(bundle, prepared)
    single_inference_ms = (time.perf_counter() - started) * 1000
    repeat_started = time.perf_counter()
    for _ in range(100):
        predict(bundle, prepared)
    mean_warm_inference_ms = (time.perf_counter() - repeat_started) * 10

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verification_type": "real_unseen_segment_inference_no_label",
        "performance_claim": False,
        "model_version": bundle["tag"],
        "model_readiness_status": bundle["report"]["status"],
        "road_source_identifier": {"type": "osm_way_id", "value": int(row["osm_id"])},
        "was_model_sample": False,
        "prediction_timestamp": prediction_timestamp,
        "input_features": base_values,
        "output": {
            **result,
            "risk_level": risk_level(
                result["calibrated_probability"], bundle["risk_policy"]),
        },
        "data_quality": quality,
        "performance": {
            "model_load_ms": round(model_load_ms, 3),
            "single_inference_ms": round(single_inference_ms, 3),
            "mean_warm_inference_ms_100_runs": round(mean_warm_inference_ms, 3),
        },
        "limitations": [
            "The observation has no future disruption label and cannot measure accuracy.",
            "Weather freshness metadata is unavailable in the processed feature store.",
            "OSM way ID is not yet a versioned internal road_segment_id.",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / (
        f"clean_room_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_"
        f"{bundle['tag']}.json")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite verification: {output_path}")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"verification_file": str(output_path.relative_to(ROOT)),
                      **output["performance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
