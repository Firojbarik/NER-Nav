"""Demo-tier risk model loading and prediction for the NER-Nav API.

This module deliberately loads only models whose card status is
``HACKATHON_DEMO_READY_LIMITED_CONFIDENCE`` (or an explicitly requested
GATED model) so the platform can serve the hackathon demo without
pretending the model has production evidence.

The set of allowed tags is explicit: it lives in ``DEMO_MODEL_TAGS`` and
can be overridden by the ``NERNAV_DEMO_MODEL`` environment variable.  A
GATED model is never served unless it is explicitly authorised.
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import interpolate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# Tags whose card status is "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE".
DEFAULT_DEMO_TAGS = ["2026-08-28_151030_b7486dea"]

ALLOWED_STATUSES = ("PRODUCTION_READY_EVIDENCE_SUPPORTED",
                    "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE",
                    "GATED_LOW_CONFIDENCE")

REQUIRED_RAINFALL = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day",
]

MAX_RAINFALL_MM = 10_000.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def demo_authorised_tags() -> list[str]:
    override = os.getenv("NERNAV_DEMO_MODEL")
    if override:
        return [tag.strip() for tag in override.split(",") if tag.strip()]
    pointer = MODEL_DIR / "demo_latest.json"
    if pointer.exists():
        try:
            tag = json.loads(pointer.read_text(encoding="utf-8"))["model_version"]
            if tag:
                return [tag]
        except (json.JSONDecodeError, KeyError, OSError):
            pass
    return list(DEFAULT_DEMO_TAGS)


def load_bundle(tag: str) -> dict:
    """Load and integrity-check a model bundle, returning a frozen dict.

    Raises ValueError for a mismatched hash, a missing artifact, or an
    unauthorised (GATED) tag that was not explicitly allowed.
    """
    if not tag or any(sep in tag for sep in ("/", "\\", "..")):
        raise ValueError("invalid model tag")
    allowed = demo_authorised_tags()
    if tag not in allowed:
        raise ValueError(
            f"model tag {tag!r} is not authorised for the demo service")

    report_path = MODEL_DIR / f"prod_report_{tag}.json"
    feature_path = MODEL_DIR / f"prod_real_temporal_{tag}_features.json"
    model_path = MODEL_DIR / f"prod_real_temporal_{tag}.ubj"
    policy_path = MODEL_DIR / f"risk_policy_{tag}.json"

    missing = [p.name for p in (report_path, feature_path, model_path, policy_path)
               if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing model artifacts: {missing}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = report.get("status")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"unknown model status: {status!r}")

    if report.get("model_sha256") and _sha256_file(model_path) != report["model_sha256"]:
        raise ValueError("model artifact SHA256 does not match its report")
    features = json.loads(feature_path.read_text(encoding="utf-8"))
    if report.get("feature_spec_sha256") and _sha256_file(feature_path) != report["feature_spec_sha256"]:
        raise ValueError("feature specification SHA256 does not match its report")

    model = xgb.XGBClassifier()
    model.load_model(model_path)

    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    calibration = None
    calibration_path = MODEL_DIR / f"prod_real_temporal_{tag}_calib.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        expected = report.get("calibration", {}).get("calibrator_sha256")
        if expected and _sha256_file(calibration_path) != expected:
            raise ValueError("calibration artifact SHA256 does not match its report")

    return {
        "tag": tag,
        "model": model,
        "features": features["features"],
        "numeric": features.get("numeric", features["features"]),
        "feature_profile": features.get("training_feature_profile", {}),
        "calibration": calibration,
        "threshold": float(report.get("test_threshold", 0.5)),
        "risk_policy": policy,
        "status": status,
        "dataset_version": report.get("dataset_version"),
        "feature_version": report.get("feature_version"),
        "caveat": report.get("caveat", ""),
        "not_for": report.get("not_for"),
        "report": report,
    }


@lru_cache(maxsize=4)
def cached_bundle(tag: str) -> dict:
    return load_bundle(tag)


def _calibrate(raw_probability: float, calibration: dict | None) -> float:
    """Reproduce the training-time isotonic interpolation exactly."""
    if calibration is None:
        return float(raw_probability)
    if calibration.get("format") != "isotonic_thresholds_v1":
        raise ValueError("unsupported calibration artifact format")
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


def predict(bundle: dict, features: dict) -> dict:
    """Score one prepared feature vector and return non-causal contributors."""
    frozen = bundle["features"]
    X = pd.DataFrame([{name: features.get(name) for name in frozen}])[frozen]
    for name in bundle["numeric"]:
        X[name] = pd.to_numeric(X[name], errors="coerce")

    raw_probability = float(bundle["model"].predict_proba(X)[:, 1][0])
    calibrated_probability = _calibrate(raw_probability, bundle["calibration"])
    contributions = bundle["model"].get_booster().predict(
        xgb.DMatrix(X), pred_contribs=True)[0]

    factors = []
    for index, name in enumerate(frozen):
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
    }


def risk_level(probability: float, policy: dict) -> str:
    if probability >= policy.get("high_at_or_above", 0.7):
        return "HIGH"
    if probability >= policy.get("low_below", 0.4):
        return "MEDIUM"
    return "LOW"