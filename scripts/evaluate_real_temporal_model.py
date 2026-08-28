"""Evaluate XGBoost on the real temporal risk dataset (exploratory).

PROVENANCE / CAVEATS
--------------------
Real-data only. Dataset composition is read from the current dataframe and
written dynamically to the report. Assumed-unaffected negatives are not
source-confirmed unaffected, so this result remains exploratory.

Grouped leave-one-corridor-out CV (hold out one NH `ref` corridor group incl.
its positives and negatives; train on the rest; predict the held-out group).
Groups are disjoint by `ref`, so no segment/corridor appears in both train and
test. All training anchors are chronologically before their own test positives,
so no sample relies on rainfall from after its prediction time (the 30-day
lookback is strictly hindsight by construction in ml/features/rainfall.py).

Because rainfall is time-varying, this is the first NON-degenerate real model:
the highway prior is constant (all trunk), so any signal above chance must come
from the real rainfall/terrain features.

OUTPUTS
-------
data/processed/ml/evaluation_runs/real_temporal_risk_model_results_<run>.json
data/processed/ml/real_temporal_risk_model_results_latest.json
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
OUTPUT_DIR = Path("data/processed/ml/evaluation_runs")
LATEST_OUTPUT = Path("data/processed/ml/real_temporal_risk_model_results_latest.json")

RAINFALL_FEATURES = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day", "rainfall_days_available",
]
STATIC_FEATURES = ["elevation_m", "slope_degrees", "highway_prior", "bridge_flag"]
FEATURES = RAINFALL_FEATURES + STATIC_FEATURES

HIGHWAY_SCORES = {
    "motorway": 0.05, "trunk": 0.10, "primary": 0.15, "secondary": 0.20,
    "tertiary": 0.25, "tertiary_link": 0.25, "secondary_link": 0.20,
    "primary_link": 0.15, "residential": 0.30, "living_street": 0.35,
    "service": 0.40, "unclassified": 0.45, "road": 0.45, "track": 0.60,
}


def summarize_metrics(y, scores, threshold=0.5):
    """Summarize ranking, classification, calibration, and error metrics."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision, recall, _, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y, scores)), 4),
        "avg_precision": round(float(average_precision_score(y, scores)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "brier_score": round(float(brier_score_loss(y, np.clip(scores, 0, 1))), 4),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "false_negative_rate": round(float(fn / (fn + tp)), 4) if fn + tp else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(threshold),
    }


def main() -> None:
    print("=" * 70)
    print("NER-Nav - Real Temporal Risk Model Evaluation (exploratory)")
    print("=" * 70)

    ds = pd.read_parquet(DATASET)
    ds["prediction_time"] = pd.to_datetime(ds["prediction_time"])
    ds = ds.sort_values("prediction_time").reset_index(drop=True)

    y = ds["label"].to_numpy()
    groups = ds["ref"].to_numpy()
    unique_groups = sorted(set(groups))

    X = ds[FEATURES].copy()
    for c in RAINFALL_FEATURES + ["elevation_m", "slope_degrees"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print(f"Samples: {len(ds)}  Positives: {int(y.sum())}  "
          f"Corridor groups: {len(unique_groups)}")
    print(f"Prediction range: {ds['prediction_time'].min().date()} -> "
          f"{ds['prediction_time'].max().date()}")

    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=2026,
    )

    xgb_proba = []
    logistic_proba = []
    rain7_proba = []
    rain30_proba = []
    test_y = []
    per_group = []
    for g in unique_groups:
        test_mask = groups == g
        train_mask = ~test_mask
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue
        y_train = y[train_mask]
        y_test = y[test_mask]
        if (y_train == 1).sum() >= 1:
            # refit a fresh model per fold (prevent cross-fold reuse)
            m = xgb.XGBClassifier(
                n_estimators=100, max_depth=2, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="logloss", random_state=2026,
            )
            m.fit(X[train_mask], y_train)
            proba = m.predict_proba(X[test_mask])[:, 1].tolist()

            logistic = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=2026),
            )
            logistic.fit(X[train_mask], y_train)
            logistic_fold = logistic.predict_proba(X[test_mask])[:, 1].tolist()
        else:
            proba = [0.5] * test_mask.sum()
            logistic_fold = [0.5] * test_mask.sum()

        def rainfall_percentiles(column):
            train_values = pd.to_numeric(X.loc[train_mask, column], errors="coerce").to_numpy()
            test_values = pd.to_numeric(X.loc[test_mask, column], errors="coerce").to_numpy()
            observed = np.sort(train_values[np.isfinite(train_values)])
            scores = np.searchsorted(observed, test_values, side="right") / max(len(observed), 1)
            scores[~np.isfinite(test_values)] = 0.0
            return scores.tolist()

        xgb_proba.extend(proba)
        logistic_proba.extend(logistic_fold)
        rain7_proba.extend(rainfall_percentiles("rainfall_7day"))
        rain30_proba.extend(rainfall_percentiles("rainfall_30day"))
        test_y.extend(y_test.tolist())
        per_group.append({
            "group": g,
            "n_test": int(test_mask.sum()),
            "n_test_pos": int(y_test.sum()),
        })

    test_y = np.asarray(test_y)
    xgb_proba = np.asarray(xgb_proba)

    metrics = summarize_metrics(test_y, xgb_proba)
    baselines = {
        "majority_no_disruption": summarize_metrics(test_y, np.zeros(len(test_y))),
        "rainfall_7day_threshold": summarize_metrics(test_y, rain7_proba),
        "rainfall_30day_threshold": summarize_metrics(test_y, rain30_proba),
        "logistic_regression": summarize_metrics(test_y, logistic_proba),
    }
    comparable = [name for name, values in baselines.items()
                  if metrics["roc_auc"] > values["roc_auc"]
                  and metrics["avg_precision"] > values["avg_precision"]]
    beats_all = len(comparable) == len(baselines)

    print("\n[XGBOOST (rainfall + static, grouped LOO-CV)]")
    print(f"  accuracy={metrics['accuracy']:.3f}  auc={metrics['roc_auc']:.3f}  "
          f"avg_precision={metrics['avg_precision']:.3f}")
    print(f"  precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
          f"f1={metrics['f1']:.3f}")
    print(f"  positives={int(test_y.sum())}  negatives={int((test_y == 0).sum())}")

    # leave-one-out feature importance (exploratory)
    full = xgb.XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=2026,
    )
    full.fit(X, y)
    importance = dict(sorted(
        zip(FEATURES, full.feature_importances_.tolist()),
        key=lambda kv: kv[1], reverse=True,
    ))

    kind_counts = {str(k): int(v) for k, v in ds["sample_kind"].value_counts().items()}
    n_events = int(ds["event_id"].nunique())
    caveat = (
        f"Exploratory only. Current dataset has {len(ds)} samples across {n_events} "
        f"events: {int(y.sum())} positives and {int((y == 0).sum())} negatives "
        f"(sample kinds: {kind_counts}). Rainfall is real CHIRPS lookback data; "
        "assumed-unaffected negatives are not source-confirmed. Not a production claim."
    )
    dataset_sha256 = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    generated_at = datetime.now(timezone.utc)
    run_id = f"{generated_at.strftime('%Y%m%dT%H%M%SZ')}_{dataset_sha256[:8]}"
    results = {
        "run_id": run_id,
        "dataset_sha256": dataset_sha256,
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "provenance_caveat": caveat,
        "evaluation": "grouped leave-one-corridor-out CV (hold out one NH ref group incl. its negatives)",
        "horizon_days": 7,
        "n_samples": int(len(ds)),
        "n_positives": int(y.sum()),
        "n_negatives": int((y == 0).sum()),
        "n_events": n_events,
        "sample_kind_counts": kind_counts,
        "feature_columns": FEATURES,
        "metrics": metrics,
        "baselines": baselines,
        "baseline_comparison": {
            "beats_all_baselines_on_roc_auc_and_avg_precision": beats_all,
            "baselines_beaten_on_both": comparable,
            "statement": ("XGBoost beats all baselines on both ranking metrics."
                          if beats_all else
                          "XGBoost does not beat all baselines on both ranking metrics; "
                          "model advantage is unproven."),
        },
        "feature_importance_xgb_fullfit": importance,
        "per_group": per_group,
        "data_file": str(DATASET),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / f"real_temporal_risk_model_results_{run_id}.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment result: {output}")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LATEST_OUTPUT.write_text(json.dumps({
        "run_id": run_id,
        "result_file": str(output),
        "dataset_sha256": dataset_sha256,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote immutable result {output}")


if __name__ == "__main__":
    main()
