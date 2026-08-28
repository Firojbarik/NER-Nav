"""Evaluate baseline and XGBoost on the real (static-feature) risk dataset.

PROVENANCE / CAVEATS
--------------------
The dataset has 12 real confirmed positives and 48 ASSUMED-unaffected
negatives (no source confirmation for negatives). This is therefore an
EXPLORATORY evaluation; any metric is indicative, not a production hazard
accuracy claim.

Evaluation uses GROUPED leave-one-corridor-out CV: each fold holds out one NH
`ref` corridor group (its confirmed positive segment(s) AND every negative of
that ref), trains on all other groups, and predicts the held-out group. Because
groups are disjoint by `ref`, no road segment (and no same-corridor negative)
appears in both train and test -> no contamination between the positive label
and the test negatives of its own corridor. With static-only features there is
no temporal lookahead.

Two models:
  1. BASELINE: highway-prior threshold-only (no learning) — the established
     deterministic prior from build_risk_training_dataset.py.
  2. XGBoost: trained on static features (terrain + highway prior + bridge).

OUTPUTS
-------
data/processed/ml/real_risk_model_results.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

DATASET = Path("data/processed/ml/real_risk_dataset.parquet")
OUTPUT = Path("data/processed/ml/real_risk_model_results.json")

FEATURES = [
    "elevation_m", "slope_degrees",
    "highway_prior", "bridge_flag",
]

HIGHWAY_SCORES = {
    "motorway": 0.05, "trunk": 0.10, "primary": 0.15, "secondary": 0.20,
    "tertiary": 0.25, "tertiary_link": 0.25, "secondary_link": 0.20,
    "primary_link": 0.15, "residential": 0.30, "living_street": 0.35,
    "service": 0.40, "unclassified": 0.45, "road": 0.45, "track": 0.60,
}


def highway_prior(hw):
    return HIGHWAY_SCORES.get(str(hw).lower(), 0.40)


def main() -> None:
    print("=" * 70)
    print("NER-Nav - Real Risk Model Evaluation (exploratory)")
    print("=" * 70)

    ds = pd.read_parquet(DATASET)
    ds["highway_prior"] = ds["highway"].map(highway_prior)

    y = ds["is_affected"].to_numpy()
    groups = ds["ref"].to_numpy()
    unique_groups = sorted(set(groups))

    X = ds[FEATURES].copy()
    # terrain NaN -> sentinel so XGBoost can use its native missing-path
    for c in ["elevation_m", "slope_degrees"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # baseline highway-prior scores
    prior_scores = ds["highway_prior"].to_numpy()

    print(f"Samples: {len(ds)}  Positives: {int(y.sum())}  Corridor groups: {len(unique_groups)}")

    # ------------------------------------------------------------------ #
    # Grouped leave-one-corridor-out evaluation
    # ------------------------------------------------------------------ #
    baseline_proba = []
    xgb_proba = []
    test_y = []
    per_group = []

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=2026,
    )

    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        y_train = y[train_mask]
        y_test = y[test_mask]

        # baseline prior: class predicted if prior >= 0.25 (moderate+)
        prior_test = prior_scores[test_mask]
        baseline_proba.extend(prior_test.tolist())

        # XGBoost
        if (y_train == 1).sum() >= 1:
            xgb_model.fit(X[train_mask], y_train)
            proba = xgb_model.predict_proba(X[test_mask])[:, 1].tolist()
        else:
            proba = [0.5] * test_mask.sum()
        xgb_proba.extend(proba)
        test_y.extend(y_test.tolist())

        per_group.append({
            "group": g,
            "n_test": int(test_mask.sum()),
            "n_test_pos": int(y_test.sum()),
        })

    test_y = np.asarray(test_y)
    baseline_proba = np.asarray(baseline_proba)
    xgb_proba = np.asarray(xgb_proba)

    # ------------------------------------------------------------------ #
    # Metrics
    # ------------------------------------------------------------------ #
    def summarize(name, proba):
        thr = 0.5
        pred = (proba >= thr).astype(int)
        n_pos = int(test_y.sum())
        n_neg = int((test_y == 0).sum())
        acc = accuracy_score(test_y, pred)
        if len(set(test_y)) > 1:
            auc = roc_auc_score(test_y, proba)
            ap = average_precision_score(test_y, proba)
        else:
            auc = float("nan"); ap = float("nan")
        f1 = f1_score(test_y, pred, zero_division=0)
        prec, rec, _, _ = precision_recall_fscore_support(
            test_y, pred, average="binary", zero_division=0
        )
        print(f"\n[{name}]")
        print(f"  accuracy={acc:.3f}  auc={auc:.3f}  avg_precision={ap:.3f}")
        print(f"  precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}")
        print(f"  positives={n_pos}  negatives={n_neg}")
        return {
            "n_positives": n_pos,
            "n_negatives": n_neg,
            "accuracy": round(acc, 4),
            "roc_auc": (None if np.isnan(auc) else round(auc, 4)),
            "avg_precision": (None if np.isnan(ap) else round(ap, 4)),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    baseline_metrics = summarize("BASELINE (highway-prior threshold)", baseline_proba)
    xgb_metrics = summarize("XGBOOST (static features, grouped LOO-CV)", xgb_proba)

    # ------------------------------------------------------------------ #
    # Feature importance (fit on full data, documented as exploratory)
    # ------------------------------------------------------------------ #
    full_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=2026,
    )
    full_model.fit(X, y)
    importance = dict(sorted(
        zip(FEATURES, full_model.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))

    results = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "provenance_caveat": (
            "Exploratory only. 12 real confirmed positives + 48 ASSUMED-unaffected "
            "negatives (no source confirmation for negatives). Not a production "
            "hazard accuracy claim."
        ),
        "evaluation": "grouped leave-one-corridor-out CV (hold out one NH ref group incl. its negatives)",
        "n_samples": int(len(ds)),
        "n_positives": int(y.sum()),
        "n_negatives": int((y == 0).sum()),
        "feature_columns": FEATURES,
        "baseline_highway_prior": baseline_metrics,
        "xgboost_static": xgb_metrics,
        "feature_importance_xgb_fullfit": importance,
        "per_group": per_group,
        "data_file": str(DATASET),
    }

    OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
