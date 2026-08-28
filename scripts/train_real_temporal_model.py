#!/usr/bin/env python3
"""
Train and SAVE the real temporal risk model as a durable artifact.

This turns the exploratory evaluation into a real, saved, loadable model:

* Trains XGBoost on the REAL temporal dataset. Counts are read at runtime.
* Saves a versioned artifact tree under data/models/:
    - real_temporal_<tag>.ubj            XGBoost binary model (primary)
    - real_temporal_<tag>.json           XGBoost JSON copy (portable)
    - real_temporal_<tag>_features.json  frozen feature list + label mapping
    - real_temporal_<tag>_card.json      model card (provenance + honest caveat)

TRANSPARENCY / HONESTY
----------------------
This is a real-data model but it remains limited-confidence, with
assumed-unaffected negatives. The saved card records current counts. Do not
present it as a production/deployable hazard classifier.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

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

TRAINED_ON = "2026-08-28"


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
    tag = f"{TRAINED_ON}_n{n_events}_p{n_positives}"
    ubj = MODEL_DIR / f"real_temporal_{tag}.ubj"
    jsn = MODEL_DIR / f"real_temporal_{tag}.json"
    feat_file = MODEL_DIR / f"real_temporal_{tag}_features.json"
    card_file = MODEL_DIR / f"real_temporal_{tag}_card.json"

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

    card = {
        "name": "ner_real_temporal_risk",
        "version": tag,
        "framework": "xgboost",
        "task": "binary disruption prediction (disruption within 7 days)",
        "created_utc": datetime.now().astimezone().isoformat(),
        "training_data": str(DATASET),
        "n_samples": int(len(ds)),
        "n_events": n_events,
        "n_corridors": n_corridors,
        "n_positives": n_positives,
        "n_negatives": n_negatives,
        "features": FEATURES,
        "hyperparameters": HYPERPARAMS,
        "out_of_sample_cv_metrics": cv.get("metrics", {}),
        "cv_note": "grouped leave-one-corridor-out; same evaluator as exploratory",
        "confidence": "LIMITED",
        "status": "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE",
        "caveat": (
            f"Trained on {n_events} events across {n_corridors} NH corridors "
            f"({len(ds)} samples: {n_positives} positives and {n_negatives} negatives); "
            "negatives include assumed-unaffected roads. This is a "
            "real-data model but NOT a production/deployable hazard classifier. "
            "Metrics are exploratory and high-variance; do not use for operational "
            "decisions without further real-label expansion and a final temporal "
            "split validation."
        ),
        "model_file": str(ubj),
        "json_copy": str(jsn),
        "feature_spec": str(feat_file),
    }
    card_file.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"\nSaved model -> {ubj}")
    print(f"Saved json  -> {jsn}")
    print(f"Saved feats -> {feat_file}")
    print(f"Saved card  -> {card_file}")
    print("DONE")


if __name__ == "__main__":
    main()
