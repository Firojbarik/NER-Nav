#!/usr/bin/env python3
"""Train the real temporal road-disruption model with honest evidence gates.

Events are split chronologically into train, validation, and future test sets.
Calibration, classifier thresholds, and rainfall baseline thresholds are chosen
on validation data only. The held-out test set is used once for final metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml.data.training_gate import enforce_training_input_gate  # noqa: E402

DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
MODEL_DIR = Path("data/models")
INPUT_MANIFEST = Path("data/processed/ml/ml_input_manifest.json")

FEATURES = ["rainfall_1day", "rainfall_3day", "rainfall_7day", "rainfall_14day",
            "rainfall_30day", "elevation_m", "slope_degrees",
            "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
            "rain_concentration_3_in_7", "rain_trend_1_vs_3",
            "rain_cumul_ratio_7_vs_30", "terrain_known",
            "slope_x_rain7", "slope_x_rain30", "elev_x_slope",
            # Leakage-safe seasonal / monsoon features derived from the real
            # prediction timestamp at both training build and inference time.
            "month", "seasonal_sin", "seasonal_cos", "monsoon_active",
            "days_into_monsoon"]
NUMERIC = [c for c in FEATURES if c not in ("terrain_known",)]

# Keep the learner deliberately small for the current evidence volume. The
# validation slice also controls early stopping; a larger tree ensemble can
# memorize the 3 anchor rows per event without learning a transferable rule.
HYPERPARAMS = dict(n_estimators=300, max_depth=1, learning_rate=0.03,
                   min_child_weight=5, reg_alpha=0.5, reg_lambda=10.0,
                   gamma=0.1, subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", random_state=2026)
EARLY_STOPPING_ROUNDS = 20

MAX_GENERALIZATION_GAP = 0.20

TARGET_RECALL = 0.70
MIN_PRECISION_FLOOR = 0.30
MAX_REVIEWABLE_ALERT_RATE = 0.30

DEMO_MIN_TEST_EVENTS = 3
DEMO_MIN_ROC_AUC = 0.65
DEMO_MIN_AVG_PRECISION = 0.50

PRODUCTION_MIN_TEST_EVENTS = 30
PRODUCTION_MIN_RECALL = 0.70
PRODUCTION_MIN_ROC_AUC = 0.75
PRODUCTION_MIN_AVG_PRECISION = 0.60

# The previous three-event holdout contained only the three newest 2026
# positive events. Keep a larger, predeclared chronological holdout so the
# current dataset's test set contains both source-backed negatives and future
# confirmed events. These defaults also satisfy the project's minimum evidence
# split (10 validation events and 30 future test events), subject to full
# metric and provenance gates.
DEFAULT_VALIDATION_EVENTS = 10
DEFAULT_TEST_EVENTS = 30

# Compatibility aliases for code that previously inspected the demo-level gate.
MIN_TEST_EVENTS = DEMO_MIN_TEST_EVENTS
MIN_ROC_AUC = DEMO_MIN_ROC_AUC
MIN_AVG_PRECISION = DEMO_MIN_AVG_PRECISION

RISK_POLICY = {
    "version": "1.0.0",
    "low_below": 0.40,
    "high_at_or_above": 0.70,
    "note": "Business risk bands are separate from the model operating threshold.",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_manifest_hash(path):
    """Return the manifest's canonical declared hash, not its file hash."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = payload.get("manifest_sha256")
    content = dict(payload)
    content.pop("manifest_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    calculated = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if not isinstance(declared, str) or declared != calculated:
        raise RuntimeError("ML input manifest SHA256 does not match its contents")
    return declared


def code_version():
    """Return Git provenance without requiring a clean working tree."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True,
            stderr=subprocess.DEVNULL).strip())
        return {"git_commit": commit, "working_tree_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "working_tree_dirty": None}


def feature_schema_version(features):
    payload = json.dumps(features, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def training_feature_profile(X):
    """Create training-only reference ranges for inference OOD checks."""
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


def fit_xgb_model(X_train, y_train, X_validation, y_validation):
    """Fit the bounded learner with validation-only early stopping."""
    model = xgb.XGBClassifier(
        **HYPERPARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(X_train, y_train,
              eval_set=[(X_validation, y_validation)], verbose=False)
    return model


def overfit_diagnostics(y_train, train_scores, y_validation, validation_scores):
    """Measure train/validation ranking separation without touching test data."""
    train_auc = _safe_auc(y_train, train_scores)
    validation_auc = _safe_auc(y_validation, validation_scores)
    gap = (train_auc - validation_auc
           if train_auc is not None and validation_auc is not None else None)
    return {
        "train_roc_auc": train_auc,
        "validation_roc_auc": validation_auc,
        "generalization_gap": gap,
        "max_allowed_gap": MAX_GENERALIZATION_GAP,
        "overfit_warning": gap is not None and gap > MAX_GENERALIZATION_GAP,
    }


def split_by_events(ds, n_val_events=DEFAULT_VALIDATION_EVENTS,
                    n_test_events=DEFAULT_TEST_EVENTS):
    """Return chronological, event-disjoint train/validation/test frames.
    
    Events are grouped by their minimum prediction_time to ensure events
    with tied dates stay together in the same chronological split.
    Split boundaries are adjusted to respect group boundaries.
    """
    event_min_time = ds.groupby("event_id")["prediction_time"].min()
    
    # Group events by their minimum prediction time
    time_groups = event_min_time.groupby(event_min_time).apply(lambda x: x.index.tolist())
    # Sort groups by time
    sorted_times = sorted(time_groups.index)
    sorted_groups = [time_groups[t] for t in sorted_times]
    
    # Compute cumulative event counts per group to find split points
    group_sizes = [len(g) for g in sorted_groups]
    cum_sizes = np.cumsum(group_sizes)
    n = len(event_order := [eid for g in sorted_groups for eid in g])
    
    # Find split indices that respect group boundaries
    # Test: last n_test_events events, but extend to include full group
    test_start_idx = n - n_test_events
    test_start_idx = next((i for i, c in enumerate(cum_sizes) if c > test_start_idx), 0)
    test_start_idx = cum_sizes[test_start_idx - 1] if test_start_idx > 0 else 0
    
    # Val: n_val_events before test, extended to full group
    val_end_idx = test_start_idx
    val_start_idx = val_end_idx - n_val_events
    val_start_idx = next((i for i, c in enumerate(cum_sizes) if c > val_start_idx), 0)
    val_start_idx = cum_sizes[val_start_idx - 1] if val_start_idx > 0 else 0
    
    train_ids = event_order[:val_start_idx]
    val_ids = event_order[val_start_idx:test_start_idx]
    test_ids = event_order[test_start_idx:]

    def grab(ids):
        return ds[ds["event_id"].isin(ids)]

    return grab(train_ids), grab(val_ids), grab(test_ids)


def require_two_class_test_split(test):
    """Fail closed when a requested test split cannot support ROC-AUC."""
    labels = test["label"].dropna().astype(int)
    if labels.nunique() < 2:
        counts = labels.value_counts().to_dict()
        raise ValueError(
            "chronological test split is single-class; increase "
            "--n-test-events or add source-backed observations before "
            f"evaluating ROC-AUC (label_counts={counts})"
        )


def _event_groups_chronological(ds):
    """Return event groups (lists of event_ids) ordered by min prediction_time.

    Events sharing the same minimum prediction_time are kept in one group so a
    chronological split never separates tied-date events across folds.
    """
    event_min_time = ds.groupby("event_id")["prediction_time"].min()
    time_groups = event_min_time.groupby(event_min_time).apply(
        lambda x: x.index.tolist())
    return [time_groups[t] for t in sorted(time_groups.index)]


def _cv_mat(df):
    X = df[FEATURES].copy()
    for column in NUMERIC:
        X[column] = pd.to_numeric(X[column], errors="coerce")
    return X.values, df["label"].to_numpy()


def grouped_cv_evaluate(ds, n_val_blocks=1, seed=2026):
    """Chronological expanding-window CV that pools out-of-fold predictions.

    Unlike the single hold-out test set (3-4 events, very noisy), this uses
    every event group as a future test block at least once and aggregates the
    out-of-fold predicted scores. It reports a stable, less volatile estimate
    of ranking generalization plus the per-fold spread (robustness).

    Fold layout (expanding window over event groups, oldest -> newest):
        train  -> earliest groups
        val    -> next `n_val_blocks` groups (threshold/logit-free; scores only)
        test   -> next block (variable size, last held groups)
        skip   -> remaining newest groups stay fresh for later folds

    Returns a JSON-serializable dict with pooled ROC-AUC/AP, per-fold metrics,
    and min/max/std across folds.
    """
    groups = _event_groups_chronological(ds)
    all_ids = set(ds["event_id"].unique())
    folds = []
    pooled_labels = []
    pooled_scores = []

    # Expanding window: train grows, test block slides forward.
    # Ensure every event group (except the very first block) gets tested at
    # least once; small n means we walk a horizon of 1 test block at a time.
    n_groups = len(groups)
    # Minimum sizes to make a viable (train, val, test) fold.
    min_train = 4
    min_val = 1
    min_test = 1

    start = min_train
    while start + min_val + min_test <= n_groups:
        train_ids = [eid for g in groups[:start] for eid in g]
        val_ids = [eid for g in groups[start:start + n_val_blocks] for eid in g]
        test_ids = [eid for g in groups[start + n_val_blocks:
                                        start + n_val_blocks + 1] for eid in g]
        train_df = ds[ds["event_id"].isin(train_ids)]
        val_df = ds[ds["event_id"].isin(val_ids)]
        test_df = ds[ds["event_id"].isin(test_ids)]
        if len(test_df) < 1:
            start += 1
            continue
        Xtr, ytr = _cv_mat(train_df)
        Xva, yva = _cv_mat(val_df)
        Xte, yte = _cv_mat(test_df)
        if len(np.unique(ytr)) < 2:
            start += 1
            continue
        model = fit_xgb_model(Xtr, ytr, Xva, yva)
        val_scores = model.predict_proba(Xva)[:, 1]
        test_scores = model.predict_proba(Xte)[:, 1]
        te_auc = _safe_auc(yte, test_scores)
        te_ap = (float(average_precision_score(yte, test_scores))
                 if len(yte) and np.unique(yte).size == 2 else None)
        pooled_labels.extend(yte.tolist())
        pooled_scores.extend(test_scores.tolist())
        folds.append({
            "test_events": test_df["event_id"].unique().tolist(),
            "n_test_positives": int(yte.sum()),
            "test_roc_auc": te_auc,
            "test_avg_precision": te_ap,
        })
        start += n_val_blocks

    if not folds:
        return {
            "method": "chronological_expanding_window_grouped_cv",
            "n_folds": 0,
            "note": "not enough events/labels for grouped CV",
            "pooled_roc_auc": None,
            "pooled_avg_precision": None,
        }

    aucs = [f["test_roc_auc"] for f in folds if f["test_roc_auc"] is not None]
    aps = [f["test_avg_precision"] for f in folds
           if f["test_avg_precision"] is not None]
    pooled_auc = _safe_auc(np.asarray(pooled_labels), np.asarray(pooled_scores))
    pooled_ap = (float(average_precision_score(pooled_labels, pooled_scores))
                 if np.unique(pooled_labels).size == 2 else None)
    return {
        "method": "chronological_expanding_window_grouped_cv",
        "n_folds": len(folds),
        "pooled_roc_auc": pooled_auc,
        "pooled_avg_precision": pooled_ap,
        "pooled_roc_auc_min": float(np.min(aucs)) if aucs else None,
        "pooled_roc_auc_max": float(np.max(aucs)) if aucs else None,
        "pooled_roc_auc_std": float(np.std(aucs)) if len(aucs) > 1 else None,
        "pooled_avg_precision_min": float(np.min(aps)) if aps else None,
        "pooled_avg_precision_max": float(np.max(aps)) if aps else None,
        "pooled_avg_precision_std": float(np.std(aps)) if len(aps) > 1 else None,
        "pooled_n_samples": len(pooled_labels),
        "pooled_n_positives": int(np.sum(pooled_labels)),
        "folds": folds,
    }


def select_top_k_threshold(scores, max_alert_rate=MAX_REVIEWABLE_ALERT_RATE):
    """Select a validation score cutoff for at most ``max_alert_rate`` alerts.

    This label-free top-k policy does not optimize against the tiny validation
    label set. Ties are moved above the tied score so the alert budget remains
    a hard upper bound.
    """
    scores = np.asarray(scores, dtype=float)
    finite = scores[np.isfinite(scores)]
    if not 0 <= max_alert_rate <= 1:
        raise ValueError("max_alert_rate must be between 0 and 1")
    if len(finite) == 0:
        return 0.5, {"strategy": "top_k_validation_only_no_scores",
                      "validation_alert_rate": 0.0,
                      "validation_selected": 0,
                      "max_alert_rate": max_alert_rate}
    k = int(np.floor(len(scores) * max_alert_rate))
    if k <= 0:
        threshold = float(np.nextafter(finite.max(), np.inf))
    else:
        ordered = np.sort(finite)[::-1]
        threshold = float(ordered[min(k, len(ordered)) - 1])
        while np.mean(scores >= threshold) > max_alert_rate:
            threshold = float(np.nextafter(threshold, np.inf))
    selected = int(np.sum(scores >= threshold))
    return threshold, {
        "strategy": "top_k_validation_only",
        "validation_alert_rate": float(selected / len(scores)) if len(scores) else 0.0,
        "validation_selected": selected,
        "max_alert_rate": max_alert_rate,
    }


def select_recall_threshold(y, scores, target_recall=TARGET_RECALL,
                            precision_floor=MIN_PRECISION_FLOOR,
                            max_alert_rate=None):
    """Choose a validation-only operating point for recall-sensitive routing.

    When ``max_alert_rate`` is provided, candidates above that review budget
    are excluded before optimizing recall/precision. The untouched test set is
    never used to choose the threshold.
    """
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.sum() == 0:
        return 0.5, {"strategy": "default_no_validation_positives",
                     "target_recall": target_recall,
                     "precision_floor": precision_floor}

    candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    rows = []
    for threshold in candidates:
        pred = (scores >= threshold).astype(int)
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "alert_rate": float(pred.mean()),
        })

    budget_rows = [r for r in rows
                   if max_alert_rate is None
                   or r["alert_rate"] <= max_alert_rate]
    if not budget_rows:
        budget_rows = rows

    target_rows = [r for r in budget_rows if r["recall"] >= target_recall]
    if target_rows:
        chosen = max(target_rows,
                     key=lambda r: (r["precision"], r["f1"], r["threshold"]))
        strategy = ("target_recall_met_max_precision_with_alert_budget"
                    if max_alert_rate is not None
                    else "target_recall_met_max_precision")
    else:
        floor_rows = [r for r in budget_rows if r["precision"] >= precision_floor]
        pool = floor_rows or budget_rows
        chosen = max(pool, key=lambda r: (r["recall"], r["precision"],
                                          r["f1"], r["threshold"]))
        strategy = ("max_recall_with_precision_floor" if floor_rows
                    else "max_recall_precision_floor_unavailable")
        if max_alert_rate is not None:
            strategy += "_with_alert_budget"

    return chosen["threshold"], {
        "strategy": strategy,
        "target_recall": target_recall,
        "precision_floor": precision_floor,
        "validation_precision": chosen["precision"],
        "validation_recall": chosen["recall"],
        "validation_f1": chosen["f1"],
        "validation_alert_rate": chosen["alert_rate"],
        "max_alert_rate": max_alert_rate,
    }


def _safe_auc(y, p):
    return (float(roc_auc_score(y, p))
            if len(y) and np.unique(y).size == 2 else None)


def eval_metrics(y, p, thresh=0.5, include_calibration=True):
    """Return ranking, classification, calibration, and error-count metrics."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    yp = (p >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    result = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "roc_auc": _safe_auc(y, p),
        "avg_precision": (float(average_precision_score(y, p))
                          if len(y) else None),
        "precision": float(precision_score(y, yp, zero_division=0)),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "f1": float(f1_score(y, yp, zero_division=0)),
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "negative_rate": float((tn + fp) / len(y)) if len(y) else None,
        "balanced_accuracy": float(
            ((tp / (fn + tp)) if fn + tp else 0.0)
            + ((tn / (tn + fp)) if tn + fp else 0.0)
        ) / 2.0,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "threshold": float(thresh),
    }
    if include_calibration:
        result["brier_score"] = float(brier_score_loss(y, np.clip(p, 0, 1)))
    return result


def recall_at_top_fraction(y, scores, fractions=(0.10, 0.25)):
    """Measure positive coverage in the highest-scored sample buckets."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    output = {}
    for fraction in fractions:
        k = max(1, int(np.ceil(len(y) * fraction))) if len(y) else 0
        found = int(y[np.argsort(-scores)[:k]].sum()) if k else 0
        output[f"top_{int(fraction * 100)}pct"] = {
            "n_selected": k,
            "positives_found": found,
            "recall": float(found / positives) if positives else None,
        }
    return output


def assess_gates(n_test_events, metrics, chronological=True):
    """Assess independent demo and production evidence gates."""
    definitions = {
        "demo": {
            "minimum_test_events": DEMO_MIN_TEST_EVENTS,
            "minimum_roc_auc": DEMO_MIN_ROC_AUC,
            "minimum_avg_precision": DEMO_MIN_AVG_PRECISION,
        },
        "production": {
            "minimum_test_events": PRODUCTION_MIN_TEST_EVENTS,
            "minimum_recall": PRODUCTION_MIN_RECALL,
            "minimum_roc_auc": PRODUCTION_MIN_ROC_AUC,
            "minimum_avg_precision": PRODUCTION_MIN_AVG_PRECISION,
        },
    }

    def evaluate(requirements):
        reasons = []
        if not chronological:
            reasons.append("test split is not chronological-future")
        checks = [
            (n_test_events, requirements["minimum_test_events"], "test events"),
            (metrics.get("roc_auc") or 0, requirements["minimum_roc_auc"], "test ROC-AUC"),
            (metrics.get("avg_precision") or 0, requirements["minimum_avg_precision"],
             "test avg_precision"),
        ]
        if "minimum_recall" in requirements:
            checks.append((metrics.get("recall") or 0,
                           requirements["minimum_recall"], "test recall"))
        for value, minimum, label in checks:
            if value < minimum:
                reasons.append(f"{label} {value} < {minimum}")
        return {"passed": not reasons, "requirements": requirements,
                "reasons": reasons}

    gates = {name: evaluate(requirements)
             for name, requirements in definitions.items()}
    gates["demo"]["status"] = (
        "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE"
        if gates["demo"]["passed"] else "DEMO_GATE_FAILED")
    gates["production"]["status"] = (
        "PRODUCTION_READY_EVIDENCE_SUPPORTED"
        if gates["production"]["passed"] else "NOT_PRODUCTION_READY")
    return gates


def _rainfall_baseline(train_values, val_values, test_values, y_val,
                       target_recall, precision_floor):
    """Scale rainfall to train empirical percentiles, then tune on validation."""
    train_values = np.asarray(train_values, dtype=float)
    observed = np.sort(train_values[np.isfinite(train_values)])

    def percentile_score(values):
        values = np.asarray(values, dtype=float)
        score = np.searchsorted(observed, values, side="right") / max(len(observed), 1)
        score[~np.isfinite(values)] = 0.0
        return score

    val_score = percentile_score(val_values)
    test_score = percentile_score(test_values)
    threshold, selection = select_recall_threshold(
        y_val, val_score, target_recall, precision_floor)
    return test_score, threshold, selection


def evaluate_baselines(Xtr, ytr, Xva, yva, Xte, yte, train, val, test,
                       target_recall, precision_floor):
    """Fit leakage-safe simple baselines and score the untouched test split."""
    baselines = {
        "majority_no_disruption": eval_metrics(
            yte, np.zeros(len(yte), dtype=float), 0.5)
    }

    for column in ("rainfall_7day", "rainfall_30day"):
        scores, threshold, selection = _rainfall_baseline(
            pd.to_numeric(train[column], errors="coerce").to_numpy(),
            pd.to_numeric(val[column], errors="coerce").to_numpy(),
            pd.to_numeric(test[column], errors="coerce").to_numpy(),
            yva, target_recall, precision_floor)
        metrics = eval_metrics(yte, scores, threshold)
        metrics["threshold_selection"] = selection
        baselines[f"{column}_threshold"] = metrics

    logistic = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026),
    )
    logistic.fit(Xtr, ytr)
    val_scores = logistic.predict_proba(Xva)[:, 1]
    test_scores = logistic.predict_proba(Xte)[:, 1]
    threshold, selection = select_recall_threshold(
        yva, val_scores, target_recall, precision_floor)
    metrics = eval_metrics(yte, test_scores, threshold)
    metrics["threshold_selection"] = selection
    baselines["logistic_regression"] = metrics
    return baselines


def baseline_comparison(model_metrics, baselines):
    """State whether XGBoost improves both ranking metrics over every baseline."""
    comparable = {name: metrics for name, metrics in baselines.items()
                  if metrics.get("roc_auc") is not None
                  and metrics.get("avg_precision") is not None}
    beaten = [name for name, metrics in comparable.items()
              if model_metrics["roc_auc"] > metrics["roc_auc"]
              and model_metrics["avg_precision"] > metrics["avg_precision"]]
    beats_all = bool(comparable) and len(beaten) == len(comparable)
    return {
        "beats_all_comparable_baselines_on_roc_auc_and_avg_precision": beats_all,
        "baselines_beaten_on_both": beaten,
        "statement": ("XGBoost beats every comparable baseline on both ROC-AUC and "
                      "average precision." if beats_all else
                      "XGBoost does not beat every comparable baseline on both "
                      "ROC-AUC and average precision; model advantage is unproven."),
    }


def build_dataset_caveat(ds, n_test_events):
    """Build limitation text from the current dataframe, never stale constants."""
    total_events = int(ds["event_id"].nunique())
    n_pos = int(ds["label"].sum())
    n_neg = int((ds["label"] == 0).sum())
    return (
        f"Limited-confidence demo model trained/evaluated from {len(ds)} samples "
        f"across {total_events} events ({n_pos} positives, {n_neg} negatives). "
        f"Only {n_test_events} future events are in the held-out test split; "
        "negative labels are source-backed road-status observations, not a substitute "
        "for dense segment-level monitoring. This is not production-safe."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-val-events", type=int, default=DEFAULT_VALIDATION_EVENTS)
    ap.add_argument("--n-test-events", type=int, default=DEFAULT_TEST_EVENTS)
    ap.add_argument("--calibrate", action="store_true",
                    help="opt in to isotonic calibration; disabled by default")
    ap.add_argument("--no-calibrate", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--target-recall", type=float, default=TARGET_RECALL)
    ap.add_argument("--precision-floor", type=float, default=MIN_PRECISION_FLOOR)
    ap.add_argument("--max-validation-alert-rate", type=float,
                    default=MAX_REVIEWABLE_ALERT_RATE)
    args = ap.parse_args()

    ds = pd.read_parquet(DATASET)
    ds["prediction_time"] = pd.to_datetime(ds["prediction_time"])
    # Training the existing artifact is intentionally no longer possible from
    # an assumed-negative or incompletely sourced dataset.  The already-built
    # artifact remains available for audit only; a replacement must clear this
    # gate first.
    enforce_training_input_gate(ds)
    if not INPUT_MANIFEST.exists():
        raise RuntimeError(f"missing ML input manifest: {INPUT_MANIFEST}")
    input_manifest_sha256 = verified_manifest_hash(INPUT_MANIFEST)
    train, val, test = split_by_events(ds, args.n_val_events, args.n_test_events)
    require_two_class_test_split(test)

    def mat(df):
        X = df[FEATURES].copy()
        for column in NUMERIC:
            X[column] = pd.to_numeric(X[column], errors="coerce")
        return X.values, df["label"].to_numpy()

    Xtr, ytr = mat(train)
    Xva, yva = mat(val)
    Xte, yte = mat(test)

    model = fit_xgb_model(Xtr, ytr, Xva, yva)
    val_scores = model.predict_proba(Xva)[:, 1]
    test_scores = model.predict_proba(Xte)[:, 1]

    fit_diagnostics = overfit_diagnostics(
        ytr, model.predict_proba(Xtr)[:, 1], yva, val_scores)

    calibrator = None
    if yva.sum() > 0 and (yva == 0).sum() > 0 and args.calibrate and not args.no_calibrate:
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(val_scores, yva)
        val_scores = calibrator.predict(val_scores)
        test_scores = calibrator.predict(test_scores)

    threshold, threshold_selection = select_top_k_threshold(
        val_scores, args.max_validation_alert_rate)
    test_metrics = eval_metrics(yte, test_scores, threshold)
    test_metrics["recall_at_top_k"] = recall_at_top_fraction(yte, test_scores)

    chronological = bool(test["prediction_time"].min() >= val["prediction_time"].max()
                         >= train["prediction_time"].max())
    n_test_events = int(test["event_id"].nunique())
    gates = assess_gates(n_test_events, test_metrics, chronological)
    if fit_diagnostics["overfit_warning"]:
        gates["production"]["passed"] = False
        gates["production"]["reasons"].append(
            "train/validation ROC-AUC gap "
            f"{fit_diagnostics['generalization_gap']:.3f} > "
            f"{MAX_GENERALIZATION_GAP:.3f}")
        gates["production"]["status"] = "NOT_PRODUCTION_READY"
    production_ready = gates["production"]["passed"]
    demo_ready = gates["demo"]["passed"]
    status = ("PRODUCTION_READY_EVIDENCE_SUPPORTED" if production_ready else
              "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE" if demo_ready else
              "GATED_LOW_CONFIDENCE")

    baselines = evaluate_baselines(
        Xtr, ytr, Xva, yva, Xte, yte, train, val, test,
        args.target_recall, args.precision_floor)
    comparison = baseline_comparison(test_metrics, baselines)

    cv = grouped_cv_evaluate(ds)

    trained_at = datetime.now().astimezone()
    dataset_sha256 = sha256_file(DATASET)
    dataset_version = f"real_temporal_risk_dataset:{dataset_sha256[:12]}"
    feature_version = feature_schema_version(FEATURES)
    tag = (f"{trained_at.strftime('%Y-%m-%d_%H%M%S')}_"
           f"{dataset_sha256[:8]}")
    # Capture source-tree provenance BEFORE writing any artifacts. Writing the
    # model files below creates new untracked files under data/models/ (a
    # tracked directory), so evaluating `git status` AFTER those writes would
    # always report a dirty tree. The version must reflect the tree state at
    # train start, not the state after this run's own artifacts are created.
    code_version_record = code_version()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ubj_path = MODEL_DIR / f"prod_real_temporal_{tag}.ubj"
    json_path = MODEL_DIR / f"prod_real_temporal_{tag}.json"
    model.save_model(ubj_path)
    model.save_model(json_path)

    calibration = {
        "method": "isotonic_on_validation" if calibrator else "identity_no_validation_calibration",
        "optional": True,
        "calibrator_fitted": calibrator is not None,
        "note": ("Calibration is optional and fit only on validation data; "
                 "the current validation set is very small."),
    }
    calib_path = MODEL_DIR / f"prod_real_temporal_{tag}_calib.json"
    calibration_x = (calibrator.X_thresholds_.tolist()
                     if calibrator is not None else [0.0, 1.0])
    calibration_y = (calibrator.y_thresholds_.tolist()
                     if calibrator is not None else [0.0, 1.0])
    # scipy.interp1d is undefined for a one-point isotonic mapping. Encode a
    # constant mapping over the valid probability domain instead, preserving
    # the fitted value while keeping the inference artifact well-defined.
    if len(calibration_x) == 1:
        calibration_x = [0.0, 1.0]
        calibration_y = [calibration_y[0], calibration_y[0]]
    calib_payload = {
        "format": "isotonic_thresholds_v1",
        "x_thresholds": calibration_x,
        "y_thresholds": calibration_y,
        "out_of_bounds": "clip",
    }
    # Persist an identity mapping when the validation slice is one-class. It
    # is not presented as fitted calibration; it simply makes the bundle's
    # inference contract explicit and keeps the undefined calibration case
    # from becoming a missing-artifact failure.
    calib_path.write_text(json.dumps(calib_payload, indent=2), encoding="utf-8")
    calibration["calibrator_file"] = str(calib_path)
    calibration["calibrator_sha256"] = sha256_file(calib_path)

    feat_path = MODEL_DIR / f"prod_real_temporal_{tag}_features.json"
    feature_spec = {
        "model_version": tag,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "input_manifest": str(INPUT_MANIFEST),
        "input_manifest_sha256": input_manifest_sha256,
        "feature_version": feature_version,
        "features": FEATURES,
        "numeric": NUMERIC,
        "training_feature_profile": training_feature_profile(Xtr),
    }
    feat_path.write_text(json.dumps(feature_spec, indent=2), encoding="utf-8")

    risk_policy_path = MODEL_DIR / f"risk_policy_{tag}.json"
    risk_policy_path.write_text(json.dumps(RISK_POLICY, indent=2), encoding="utf-8")

    caveat = build_dataset_caveat(ds, n_test_events)

    report = {
        "status": status,
        "model_version": tag,
        "dataset_version": dataset_version,
        "dataset_sha256": dataset_sha256,
        "input_manifest": str(INPUT_MANIFEST),
        "input_manifest_sha256": input_manifest_sha256,
        "feature_version": feature_version,
        "code_version": code_version_record,
        "training_timestamp": trained_at.isoformat(),
        "demo_ready": demo_ready,
        "production_ready": production_ready,
        "gated": not demo_ready,
        "gate_reasons": gates["demo"]["reasons"],
        "gates": gates,
        "caveat": caveat,
        "split": {
            "train_events": train["event_id"].unique().tolist(),
            "val_events": val["event_id"].unique().tolist(),
            "test_events": test["event_id"].unique().tolist(),
            "n_train_events": int(train["event_id"].nunique()),
            "n_val_events": int(val["event_id"].nunique()),
            "n_test_events": n_test_events,
            "n_test_positives": int(yte.sum()),
            "chronological": chronological,
            "split_by": "event (chronological; no sample leakage)",
            "training_period": {
                "start": train["prediction_time"].min().isoformat(),
                "end": train["prediction_time"].max().isoformat(),
            },
            "validation_period": {
                "start": val["prediction_time"].min().isoformat(),
                "end": val["prediction_time"].max().isoformat(),
            },
            "test_period": {
                "start": test["prediction_time"].min().isoformat(),
                "end": test["prediction_time"].max().isoformat(),
            },
        },
        "threshold_selection": threshold_selection,
        "calibration": calibration,
        "test_threshold": float(threshold),
        "test_metrics": test_metrics,
        "baselines": baselines,
        "baseline_comparison": comparison,
        "grouped_cv": cv,
        "fit_diagnostics": fit_diagnostics,
        "robustness": {
            "note": (
                "Single chronological test set (3-4 events) is noisy; grouped "
                "expanding-window CV aggregates out-of-fold predictions across "
                "many future test blocks for a stabler generalization estimate."),
            # ROC-AUC is undefined when a chronological holdout contains only
            # one class. Preserve that fact in the report instead of crashing
            # and silently preventing the gated artifact from being recorded.
            "single_split_roc_auc": test_metrics["roc_auc"],
            "single_split_avg_precision": test_metrics["avg_precision"],
            "cv_pooled_roc_auc": cv.get("pooled_roc_auc"),
            "cv_pooled_avg_precision": cv.get("pooled_avg_precision"),
            "cv_roc_auc_min": cv.get("pooled_roc_auc_min"),
            "cv_roc_auc_max": cv.get("pooled_roc_auc_max"),
            "cv_roc_auc_std": cv.get("pooled_roc_auc_std"),
            "cv_avg_precision_min": cv.get("pooled_avg_precision_min"),
            "cv_avg_precision_max": cv.get("pooled_avg_precision_max"),
            "cv_avg_precision_std": cv.get("pooled_avg_precision_std"),
            "cv_n_folds": cv.get("n_folds"),
        },
        "hyperparameters": {**HYPERPARAMS,
                             "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        "features": FEATURES,
        "model_file": str(ubj_path),
        "model_sha256": sha256_file(ubj_path),
        "feature_spec_file": str(feat_path),
        "feature_spec_sha256": sha256_file(feat_path),
        "risk_policy_file": str(risk_policy_path),
        "risk_policy_sha256": sha256_file(risk_policy_path),
    }
    out_path = MODEL_DIR / f"prod_report_{tag}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    card = {
        "name": "ner_real_temporal_road_disruption",
        "version": tag,
        "dataset_version": dataset_version,
        "feature_version": feature_version,
        "input_manifest": str(INPUT_MANIFEST),
        "input_manifest_sha256": input_manifest_sha256,
        "status": status,
        "intended_use": "Hackathon demonstration and offline research only",
        "not_for": "Production routing, emergency response, or public safety decisions",
        "caveat": caveat,
        "demo_gate": gates["demo"],
        "production_gate": gates["production"],
        "test_metrics": test_metrics,
        "baseline_comparison": comparison,
        "calibration": calibration,
    }
    (MODEL_DIR / f"prod_real_temporal_{tag}_card.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8")

    latest = {
        "model_version": tag,
        "report_file": str(out_path),
        "status": status,
        "production_ready": production_ready,
        "updated_at": trained_at.isoformat(),
    }
    (MODEL_DIR / "prod_latest.json").write_text(
        json.dumps(latest, indent=2), encoding="utf-8")

    print(f"SPLIT: train={train['event_id'].nunique()} events, "
          f"val={val['event_id'].nunique()}, test={n_test_events} "
          f"(pos={int(yte.sum())})")
    print(f"Chronological: {chronological}")
    print(f"Threshold: {threshold:.6f} ({threshold_selection['strategy']})")
    print(f"Test metrics: {json.dumps(test_metrics)}")
    print(f"Demo gate: {'PASS' if demo_ready else 'FAIL'}")
    print(f"Production gate: {'PASS' if production_ready else 'FAIL'}")
    print(f"STATUS => {status}")
    print(comparison["statement"])
    if cv.get("n_folds"):
        fmt = lambda value: f"{value:.3f}" if value is not None else "n/a"
        print("Grouped CV: "
              f"n_folds={cv['n_folds']}, "
              f"pooled ROC-AUC={fmt(cv['pooled_roc_auc'])} "
              f"[min {fmt(cv['pooled_roc_auc_min'])} - max {fmt(cv['pooled_roc_auc_max'])}], "
              f"pooled AP={fmt(cv['pooled_avg_precision'])} "
              f"[min {fmt(cv['pooled_avg_precision_min'])} - max {fmt(cv['pooled_avg_precision_max'])}]")
    print("Artifacts written to data/models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
