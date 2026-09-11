#!/usr/bin/env python3
"""
Train and SAVE the real temporal risk model as a durable artifact.

This turns the exploratory evaluation into a real, saved, loadable model:

* Trains XGBoost on the REAL temporal dataset. Counts are read at runtime.
* Saves a versioned artifact tree under data/models/ in the shape the FastAPI
  demo service loads (demo_authorised_tags reads demo_latest.json):
    - prod_real_temporal_<tag>.ubj            XGBoost binary model (primary)
    - prod_real_temporal_<tag>.json           XGBoost JSON copy (portable)
    - prod_real_temporal_<tag>_features.json  frozen features + label mapping
                                              + training feature profile (OOD)
    - prod_real_temporal_<tag>_card.json      model card (provenance + caveat)
    - prod_report_<tag>.json                  serving report (status, SHAs)
    - risk_policy_<tag>.json                  business risk bands
    - demo_latest.json                        demo pointer consumed by the API

TRANSPARENCY / HONESTY
----------------------
This is a real-data model but it remains limited-confidence, with
assumed-unaffected negatives. The saved card records current counts. Do not
present it as a production/deployable hazard classifier.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.features.temporal_design import CONFIRMED_EVENTS  # noqa: E402

DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
CV_RESULTS_LATEST = Path("data/processed/ml/real_temporal_risk_model_results_latest.json")
MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RAINFALL_FEATURES = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day", "rainfall_days_available",
]
STATIC_FEATURES = ["elevation_m", "slope_degrees", "highway_prior", "bridge_flag"]
FEATURES = RAINFALL_FEATURES + STATIC_FEATURES

# frozen, documented hyperparameters (match the exploratory CV faithfully)
HYPERPARAMS = {
    "n_estimators": 100,
    "max_depth": 2,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 2026,
}

TRAINED_ON = "2026-09-11"

STATUS = "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE"
INTENDED_USE = "Hackathon demonstration and offline research only"
NOT_FOR = "Production routing, emergency response, or public safety decisions"

RISK_POLICY = {
    "version": "1.0.0",
    "low_below": 0.40,
    "high_at_or_above": 0.70,
    "note": "Business risk bands are separate from the model operating threshold.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_schema_version(features):
    payload = json.dumps(features, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def training_feature_profile(X):
    """Training-only reference ranges used by inference for OOD checks."""
    profile = {}
    frame = pd.DataFrame(X, columns=FEATURES)
    for column in FEATURES:
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values[np.isfinite(values)]
        profile[column] = {
            "missing_rate": float(values.isna().mean()),
            "min": float(finite.min()) if len(finite) else None,
            "p01": float(finite.quantile(0.01)) if len(finite) else None,
            "p99": float(finite.quantile(0.99)) if len(finite) else None,
            "max": float(finite.max()) if len(finite) else None,
        }
    return profile


def main() -> None:
    print("=" * 70)
    print("NER-Nav - Train & Save Real Temporal Risk Model")
    print("=" * 70)

    ds = pd.read_parquet(DATASET)
    ds = ds.sort_values("prediction_time").reset_index(drop=True)
    y = ds["label"].to_numpy()
    X = ds[FEATURES].copy()
    for c in RAINFALL_FEATURES + ["elevation_m", "slope_degrees"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print(f"Samples: {len(ds)}  Positives: {int(y.sum())}  Negatives: {int((y == 0).sum())}")
    print(f"Feature columns ({len(FEATURES)}): {FEATURES}")

    model = xgb.XGBClassifier(**HYPERPARAMS)
    model.fit(X, y)
    score = model.score(X, y)
    print(f"Training accuracy (in-sample, informative only): {score:.3f}")

    n_events = int(ds["event_id"].nunique())
    n_corridors = int(ds["ref"].nunique())
    n_positives = int(y.sum())
    n_negatives = int((y == 0).sum())
    n_confirmed_events = len(CONFIRMED_EVENTS)
    n_observation_negatives = int(
        (ds["label_source"] == "observed_open_real").sum()
        if "label_source" in ds.columns else 0)

    dataset_sha256 = sha256_file(DATASET)
    dataset_version = f"real_temporal_risk_dataset:{dataset_sha256[:12]}"
    feature_version = feature_schema_version(FEATURES)
    profile = training_feature_profile(X)

    tag = f"{TRAINED_ON}_n{n_events}_p{n_positives}"
    ubj = MODEL_DIR / f"prod_real_temporal_{tag}.ubj"
    jsn = MODEL_DIR / f"prod_real_temporal_{tag}.json"
    feat_file = MODEL_DIR / f"prod_real_temporal_{tag}_features.json"
    card_file = MODEL_DIR / f"prod_real_temporal_{tag}_card.json"
    report_file = MODEL_DIR / f"prod_report_{tag}.json"
    policy_file = MODEL_DIR / f"risk_policy_{tag}.json"
    pointer_file = MODEL_DIR / "demo_latest.json"

    model.save_model(ubj)
    model.save_model(jsn)

    # load CV metrics to embed the honest out-of-sample estimate
    cv = {}
    if CV_RESULTS_LATEST.exists():
        latest = json.loads(CV_RESULTS_LATEST.read_text(encoding="utf-8"))
        result_path = Path(latest["result_file"])
        if result_path.exists():
            cv = json.loads(result_path.read_text(encoding="utf-8"))

    features_meta = {
        "version": tag,
        "model_file": str(ubj),
        "features": FEATURES,
        "numeric": FEATURES,
        "training_feature_profile": profile,
        "label_mapping": {"0": "no_disruption_within_7d", "1": "disruption_within_7d"},
        "label_column": "label",
        "horizon_days": 7,
        "missing_value_policy": "XGBoost native NaN handling (do not impute)",
        "trained_on": TRAINED_ON,
        "n_train_samples": int(len(ds)),
        "n_train_positives": n_positives,
        "n_train_negatives": n_negatives,
    }
    feat_file.write_text(json.dumps(features_meta, indent=2), encoding="utf-8")

    cv_metrics = cv.get("metrics", {})
    cv_summary = ", ".join(
        f"{key}={value:.3f}" for key, value in
        sorted(cv_metrics.items())
        if isinstance(value, (int, float)) and key not in
        {"confusion_matrix", "threshold"})
    caveat_text = (
        f"{n_confirmed_events} confirmed disruption events plus "
        f"{n_observation_negatives} observation-backed negative road-status records "
        f"({len(ds)} samples: {n_positives} positives, {n_negatives} negatives) across "
        f"{n_corridors} NH corridors; negatives include assumed-unaffected roads. "
        "Grouped leave-one-corridor-out CV: "
        f"{cv_summary}. This is a real-data model but NOT a production/deployable "
        "hazard classifier; metrics are exploratory and high-variance. Not production-safe."
    )

    report = {
        "model_version": tag,
        "dataset": str(DATASET),
        "dataset_sha256": dataset_sha256,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "model_sha256": sha256_file(ubj),
        "feature_spec_sha256": sha256_file(feat_file),
        "status": STATUS,
        "production_ready": False,
        "test_threshold": 0.5,
        "threshold_selection": "fixed default 0.5 (exploratory CV used 0.5)",
        "n_samples": int(len(ds)),
        "n_confirmed_events": n_confirmed_events,
        "n_observation_negatives": n_observation_negatives,
        "n_events": n_events,
        "n_corridors": n_corridors,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "out_of_sample_cv_metrics": cv_metrics,
        "cv_note": "grouped leave-one-corridor-out; same evaluator as exploratory",
        "caveat": caveat_text,
        "not_for": NOT_FOR,
        "trained_on": TRAINED_ON,
        "created_utc": datetime.now().astimezone().isoformat(),
    }
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    policy_file.write_text(json.dumps(RISK_POLICY, indent=2), encoding="utf-8")

    card = {
        "name": "ner_real_temporal_risk",
        "version": tag,
        "framework": "xgboost",
        "task": "binary disruption prediction (disruption within 7 days)",
        "created_utc": datetime.now().astimezone().isoformat(),
        "training_data": str(DATASET),
        "dataset_sha256": dataset_sha256,
        "feature_version": feature_version,
        "n_samples": int(len(ds)),
        "n_events": n_events,
        "n_confirmed_events": n_confirmed_events,
        "n_observation_negatives": n_observation_negatives,
        "n_corridors": n_corridors,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "features": FEATURES,
        "hyperparameters": HYPERPARAMS,
        "out_of_sample_cv_metrics": cv_metrics,
        "cv_note": "grouped leave-one-corridor-out; same evaluator as exploratory",
        "confidence": "LIMITED",
        "status": STATUS,
        "caveat": caveat_text,
        "model_file": str(ubj),
        "json_copy": str(jsn),
        "feature_spec": str(feat_file),
        "report_file": str(report_file),
    }
    card_file.write_text(json.dumps(card, indent=2), encoding="utf-8")

    pointer = {
        "model_version": tag,
        "model_file": str(ubj),
        "report_file": str(report_file),
        "status": STATUS,
        "production_ready": False,
        "intended_use": INTENDED_USE,
        "not_for": NOT_FOR,
        "caveat": caveat_text,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    pointer_file.write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    print(f"\nSaved model  -> {ubj}")
    print(f"Saved json   -> {jsn}")
    print(f"Saved feats  -> {feat_file}")
    print(f"Saved card   -> {card_file}")
    print(f"Saved report -> {report_file}")
    print(f"Saved policy -> {policy_file}")
    print(f"Demo pointer -> {pointer_file} (status={STATUS})")
    print("DONE")


if __name__ == "__main__":
    main()
