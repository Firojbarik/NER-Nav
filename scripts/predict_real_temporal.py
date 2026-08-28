#!/usr/bin/env python3
"""Load a frozen model bundle and score one validated road observation.

This command never trains. It verifies artifact hashes, computes derived
features from observed base values, distinguishes data freshness from model
risk, and logs an append-only prediction trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import interpolate

from ml.features.seasonal import (  # noqa: E402
    days_into_monsoon,
    monsoon_active,
    month_feature,
    seasonal_cos,
    seasonal_sin,
)

MODEL_DIR = Path("data/models")
REQUIRED_RAINFALL = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day",
]

# These are input-boundary checks, not learned model constraints.  The upper
# rainfall bound is deliberately generous for cumulative regional rainfall,
# while still rejecting values that are plainly malformed rather than merely
# out of distribution.
MAX_RAINFALL_MM = 10_000.0
SUPPORTED_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "tertiary_link", "secondary_link", "primary_link", "residential",
    "living_street", "service", "unclassified", "road", "track",
}
OPTIONAL_METADATA = {
    "osm_id", "road_segment_id", "latitude", "longitude", "highway",
    "road_type", "category",
}
DERIVED_FEATURES = {
    "terrain_known", "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
    "rain_concentration_3_in_7", "rain_trend_1_vs_3",
    "rain_cumul_ratio_7_vs_30", "slope_x_rain7", "slope_x_rain30",
    "elev_x_slope", "month", "seasonal_sin", "seasonal_cos",
    "monsoon_active", "days_into_monsoon",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_tag(tag):
    if tag != "latest":
        return tag
    pointer = MODEL_DIR / "prod_latest.json"
    if not pointer.exists():
        raise FileNotFoundError("latest model pointer is missing")
    return json.loads(pointer.read_text(encoding="utf-8"))["model_version"]


def validate_model_tag(tag):
    """Reject path separators before using a model tag in artifact paths."""
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", tag):
        raise ValueError("model tag has an invalid format")
    return tag


def load_model_bundle(tag: str):
    """Load and integrity-check a frozen model bundle."""
    tag = validate_model_tag(_resolve_tag(tag))
    report_path = MODEL_DIR / f"prod_report_{tag}.json"
    feature_path = MODEL_DIR / f"prod_real_temporal_{tag}_features.json"
    model_path = MODEL_DIR / f"prod_real_temporal_{tag}.ubj"
    for path in (report_path, feature_path, model_path):
        if not path.exists():
            raise FileNotFoundError(f"required model artifact is missing: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    spec = json.loads(feature_path.read_text(encoding="utf-8"))
    if not report.get("input_manifest_sha256"):
        raise ValueError(
            "legacy model artifact lacks the verified ML input manifest; "
            "historical/demo artifacts with assumed negatives are not eligible"
        )
    if report.get("model_sha256") and sha256_file(model_path) != report["model_sha256"]:
        raise ValueError("model artifact SHA256 does not match its report")
    if (report.get("feature_spec_sha256")
            and sha256_file(feature_path) != report["feature_spec_sha256"]):
        raise ValueError("feature specification SHA256 does not match its report")
    if spec.get("model_version") and spec["model_version"] != tag:
        raise ValueError("feature specification model version mismatch")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    calibration = None
    calibration_path = MODEL_DIR / f"prod_real_temporal_{tag}_calib.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        expected = report.get("calibration", {}).get("calibrator_sha256")
        if expected and sha256_file(calibration_path) != expected:
            raise ValueError("calibration artifact SHA256 does not match its report")

    policy_path = MODEL_DIR / f"risk_policy_{tag}.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"risk policy is missing: {policy_path}")
    risk_policy = json.loads(policy_path.read_text(encoding="utf-8"))

    return {
        "tag": tag,
        "model": model,
        "features": spec["features"],
        "numeric": spec.get("numeric", spec["features"]),
        "feature_profile": spec.get("training_feature_profile", {}),
        "calibration": calibration,
        "threshold": float(report.get("test_threshold", 0.5)),
        "risk_policy": risk_policy,
        "report": report,
    }


def _parse_timestamp(value, field):
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed


def validate_road_segment_id(value):
    """Return a safe scalar road identifier or reject malformed identity."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("road_segment_id must be a non-empty scalar string or integer")
    normalized = str(value).strip()
    if (not normalized or normalized.lower() == "unknown"
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", normalized)):
        raise ValueError("road_segment_id has an invalid format")
    return normalized


def _coerce_numeric(value):
    """Coerce malformed scalar input to NaN for a controlled error path."""
    try:
        return pd.to_numeric(value, errors="coerce")
    except (TypeError, ValueError):
        return np.nan


def validate_and_prepare(values, features, prediction_timestamp,
                         weather_observed_at=None, feature_profile=None):
    """Validate real observations and derive the frozen model feature vector."""
    if not isinstance(values, dict):
        raise ValueError("features must be a JSON object")
    errors = []
    prepared = dict(values)

    allowed = (set(REQUIRED_RAINFALL) | {"elevation_m", "slope_degrees"}
               | OPTIONAL_METADATA | DERIVED_FEATURES)
    unknown_fields = sorted(set(prepared) - allowed)
    if unknown_fields:
        errors.append(f"unknown input fields: {unknown_fields}")

    for name in ("latitude", "longitude"):
        if name in prepared:
            value = _coerce_numeric(prepared[name])
            if np.ndim(value) != 0 or not np.isfinite(value):
                errors.append(f"{name} must be a finite scalar")
            elif name == "latitude" and not -90 <= float(value) <= 90:
                errors.append("latitude must be between -90 and 90")
            elif name == "longitude" and not -180 <= float(value) <= 180:
                errors.append("longitude must be between -180 and 180")

    for name in ("highway", "road_type", "category"):
        if name in prepared and prepared[name] is not None:
            category = str(prepared[name]).strip().lower()
            if category not in SUPPORTED_HIGHWAY_TYPES:
                errors.append(f"{name} is not a supported road category")

    for name in REQUIRED_RAINFALL:
        value = _coerce_numeric(prepared.get(name))
        if np.ndim(value) != 0 or not np.isfinite(value):
            errors.append(f"{name} is required and must be finite")
        elif value < 0:
            errors.append(f"{name} cannot be negative")
        elif value > MAX_RAINFALL_MM:
            errors.append(f"{name} exceeds the supported physical maximum of {MAX_RAINFALL_MM:g} mm")
        else:
            prepared[name] = float(value)

    for name in ("elevation_m", "slope_degrees"):
        raw = prepared.get(name)
        if raw is None:
            prepared[name] = None
            continue
        value = _coerce_numeric(raw)
        if np.ndim(value) != 0 or not np.isfinite(value):
            errors.append(f"{name} must be finite or null")
        else:
            prepared[name] = float(value)

    slope = prepared.get("slope_degrees")
    elevation = prepared.get("elevation_m")
    if slope is not None and not 0 <= slope <= 90:
        errors.append("slope_degrees must be between 0 and 90")
    if elevation is not None and not -500 <= elevation <= 9000:
        errors.append("elevation_m is outside the supported physical range")
    if errors:
        raise ValueError("; ".join(errors))

    eps = 0.01
    r1, r3, r7 = prepared["rainfall_1day"], prepared["rainfall_3day"], prepared["rainfall_7day"]
    r14, r30 = prepared["rainfall_14day"], prepared["rainfall_30day"]
    slope_value = slope or 0.0
    elevation_value = elevation or 0.0
    prepared.update({
        "rain_intensity_1_vs_7": r1 / max(r7, eps),
        "rain_intensity_3_vs_14": r3 / max(r14, eps),
        "rain_concentration_3_in_7": r3 / max(r7, eps),
        "rain_trend_1_vs_3": r1 / max(r3, eps),
        "rain_cumul_ratio_7_vs_30": r7 / max(r30, eps),
        "terrain_known": int(slope is not None and elevation is not None),
        "slope_x_rain7": slope_value * r7,
        "slope_x_rain30": slope_value * r30,
        "elev_x_slope": elevation_value * slope_value,
    })

    prediction_time = _parse_timestamp(prediction_timestamp, "prediction_timestamp")
    now = pd.Timestamp.now(tz="UTC")
    if prediction_time > now + pd.Timedelta(minutes=5):
        raise ValueError("prediction_timestamp cannot be in the future")

    # Leakage-safe seasonal / monsoon features derived ONLY from the (already
    # validated, non-future) prediction timestamp - identical to training build.
    _anchor_date = prediction_time.date()
    prepared.update({
        "month": float(month_feature(_anchor_date)),
        "seasonal_sin": seasonal_sin(_anchor_date),
        "seasonal_cos": seasonal_cos(_anchor_date),
        "monsoon_active": float(monsoon_active(_anchor_date)),
        "days_into_monsoon": float(days_into_monsoon(_anchor_date)),
    })

    missing_frozen = [name for name in features if name not in prepared]
    if missing_frozen:
        raise ValueError(f"cannot derive frozen features: {missing_frozen}")

    freshness = {"status": "MISSING_METADATA", "age_hours": None}
    flags = []
    if weather_observed_at:
        observed_time = _parse_timestamp(weather_observed_at, "weather_observed_at")
        if observed_time > prediction_time:
            raise ValueError("weather_observed_at cannot be after prediction_timestamp")
        age_hours = float((prediction_time - observed_time).total_seconds() / 3600)
        status = "FRESH" if age_hours <= 48 else "STALE" if age_hours <= 168 else "EXPIRED"
        freshness = {"status": status, "age_hours": round(age_hours, 3)}
        if status != "FRESH":
            flags.append(f"WEATHER_{status}")
    else:
        flags.append("WEATHER_FRESHNESS_UNKNOWN")

    ood_features = []
    for name, bounds in (feature_profile or {}).items():
        value = prepared.get(name)
        if value is None or not np.isfinite(value):
            continue
        low, high = bounds.get("p01"), bounds.get("p99")
        if low is not None and high is not None and not low <= value <= high:
            ood_features.append(name)
    if ood_features:
        flags.append("OUT_OF_DISTRIBUTION")
    if prepared["terrain_known"] == 0:
        flags.append("MISSING_TERRAIN")

    quality = {
        "weather_freshness": freshness,
        "ood_features": sorted(ood_features),
        "confidence_flags": flags,
        "low_confidence": bool(flags),
    }
    return prepared, quality


def _calibrate(raw_probability, calibration):
    if calibration is None:
        return raw_probability
    if calibration.get("format") != "isotonic_thresholds_v1":
        raise ValueError("unsupported calibration artifact format")
    # Reconstruct sklearn's scipy.interp1d + float32 cast semantics.  Using
    # np.interp here changes a few borderline values by ~6e-8 and can change
    # tied ranks in AUC/AP, so inference must match training exactly.
    x = np.asarray(calibration["x_thresholds"], dtype=np.float32)
    y = np.asarray(calibration["y_thresholds"], dtype=np.float32)
    if len(x) == 1:
        return float(y[0])
    value = np.asarray([raw_probability], dtype=x.dtype)
    value = np.clip(value, x[0], x[-1])
    fn = interpolate.interp1d(
        x, y, kind="linear", bounds_error=False,
        fill_value=(y[0], y[-1]),
    )
    return float(fn(value).astype(value.dtype)[0])


def predict(bundle, values):
    """Score one prepared feature vector and produce non-causal contributors."""
    features = bundle["features"]
    X = pd.DataFrame([{name: values.get(name) for name in features}])[features]
    for name in bundle["numeric"]:
        X[name] = pd.to_numeric(X[name], errors="coerce")

    raw_probability = float(bundle["model"].predict_proba(X)[:, 1][0])
    calibrated_probability = _calibrate(raw_probability, bundle["calibration"])
    contributions = bundle["model"].get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[0]

    factors = []
    for index, name in enumerate(features):
        value = X.iloc[0][name]
        factors.append({
            "feature": name,
            "value": None if pd.isna(value) else round(float(value), 6),
            "contribution": round(float(contributions[index]), 6),
        })
    factors.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "raw_probability": round(raw_probability, 6),
        "calibrated_probability": round(calibrated_probability, 6),
        "disruption_probability": round(calibrated_probability, 6),
        "top_contributing_factors": factors[:5],
        "explanation_disclaimer": (
            "Feature contributions describe this model output and are not causal proof."
        ),
    }


def risk_level(probability, policy):
    if probability >= policy["high_at_or_above"]:
        return "HIGH"
    if probability >= policy["low_below"]:
        return "MEDIUM"
    return "LOW"


def main():
    from scripts.prediction_trace import log_prediction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="latest")
    parser.add_argument("--features")
    parser.add_argument("--road-json")
    parser.add_argument("--road-segment-id")
    parser.add_argument("--prediction-timestamp")
    parser.add_argument("--weather-observed-at")
    parser.add_argument("--ref")
    parser.add_argument("--state")
    parser.add_argument("--district")
    args = parser.parse_args()

    try:
        bundle = load_model_bundle(args.tag)
        if args.features:
            values = json.loads(Path(args.features).read_text(encoding="utf-8"))
        elif args.road_json:
            values = json.loads(args.road_json)
        else:
            raise ValueError("provide --features or --road-json")

        prediction_timestamp = (
            args.prediction_timestamp or datetime.now(timezone.utc).isoformat())
        road_segment_id = args.road_segment_id or values.get("road_segment_id", values.get("osm_id"))
        if road_segment_id is None:
            raise ValueError("road_segment_id is required")
        road_segment_id = validate_road_segment_id(road_segment_id)

        prepared, quality = validate_and_prepare(
            values, bundle["features"], prediction_timestamp,
            args.weather_observed_at, bundle["feature_profile"])
        result = predict(bundle, prepared)
        level = risk_level(result["calibrated_probability"], bundle["risk_policy"])
        operating_decision = (
            "ALERT" if result["calibrated_probability"] >= bundle["threshold"]
            else "NO_ALERT")

        prediction_id = log_prediction(
            road_segment_id=str(road_segment_id),
            prediction_timestamp=prediction_timestamp,
            model_tag=bundle["tag"],
            features={name: prepared.get(name) for name in bundle["features"]},
            raw_probability=result["raw_probability"],
            calibrated_probability=result["calibrated_probability"],
            risk_level=level,
            top_contributing_factors=result["top_contributing_factors"],
            threshold=bundle["threshold"],
            ref=args.ref, state=args.state, district=args.district,
            dataset_version=bundle["report"].get("dataset_version"),
            risk_policy_version=bundle["risk_policy"]["version"],
            operating_decision=operating_decision,
            data_quality=quality,
            trace_stream="production",
        )
        output = {
            **result,
            "road_segment_id": str(road_segment_id),
            "prediction_timestamp": prediction_timestamp,
            "prediction_horizon_days": 7,
            "risk_level": level,
            "risk_policy_version": bundle["risk_policy"]["version"],
            "operating_decision": operating_decision,
            "operating_threshold": bundle["threshold"],
            "model_version": bundle["tag"],
            "dataset_version": bundle["report"].get("dataset_version"),
            "feature_version": bundle["report"].get("feature_version"),
            "model_readiness_status": bundle["report"]["status"],
            "data_quality": quality,
            "prediction_id": prediction_id,
        }
        print(json.dumps(output, indent=2, allow_nan=False))
        return 0
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
