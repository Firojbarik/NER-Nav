"""End-to-end training experiment harness for NER-Nav disruption prediction.

Wires together the validated components:
    compose (static+latest-temporal) -> missingness -> labels -> temporal split
    -> XGBoost baseline -> evaluation (incl. lead-time-aware).

ABSOLUTE RULE / PROVENANCE:
----------------------------
   The harness can train on REAL features but the LABELS it uses are only as
   honest as the event data passed in. This repository currently has NO real
   source-supported hazard labels, so `run_experiment` by default generates
   SYNTHETIC labels purely to exercise the machinery. Those runs are tagged
   `label_source='synthetic_test_only'` and MUST NOT be presented as real
   performance. Only labels produced from real verified event data
   (`label_source='real_verified'`) may ever be used for a real accuracy claim.

All computation is pure (no hidden I/O) except the feature/artifact loading done
in `run_experiment` via explicit paths.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml.features.compose import compose_feature_matrix
from ml.features.missingness import (
    MissingnessStrategy,
    apply_missingness_strategy,
)
from ml.labels.generate import LabelingConfig
from ml.splits.temporal import (
    TEST,
    TRAIN,
    VALIDATION,
    assert_features_not_future,
    temporal_split,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TERRAIN_COLS = ["elevation_m", "slope_degrees", "aspect_degrees"]
RAINFALL_COLS = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day",
]
# non-feature columns the harness must keep for provenance/splitting
KEEP_COLS = ["osm_id", "state", "district", "prediction_time"]


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment run."""

    horizon: pd.Timedelta = pd.Timedelta("7 days")
    prediction_dates: Sequence[str] = (
        "2015-04-16", "2015-04-17", "2015-04-18",
        "2015-04-19", "2015-04-20", "2015-04-21",
    )
    missingness_strategy: str = "xgb_native_flag"
    add_flag: bool = True
    train_until: str = "2015-04-18"
    validation_until: str = "2015-04-20"
    random_state: int = 42
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    road_sample: Optional[int] = 20000  # cap roads for harness speed

    terrain_path: str = os.path.join(
        REPO, "data", "processed", "terrain", "road_terrain_features.parquet")
    rainfall_path: str = os.path.join(
        REPO, "data", "processed", "weather", "road_rainfall_features.parquet")
    base_path: str = os.path.join(
        REPO, "data", "processed", "ml", "risk_training_dataset.parquet")


def _resolve_strategy(raw: str) -> MissingnessStrategy:
    if raw == "xgb_native_flag":
        return MissingnessStrategy.XGB_NATIVE
    if raw == "xgb_native":
        return MissingnessStrategy.XGB_NATIVE
    if raw == "flag":
        return MissingnessStrategy.FLAG
    if raw == "drop_any":
        return MissingnessStrategy.DROP_ANY
    if raw == "impute_in_group":
        return MissingnessStrategy.IMPUTE_IN_GROUP
    raise ValueError(f"unknown missingness strategy: {raw}")


def build_design_matrix(
    config: ExperimentConfig,
) -> pd.DataFrame:
    """Load real artifacts and compose the design matrix + provenance columns.

    Returns a frame with feature_columns + KEEP_COLS + feature_latest_ts.
    """
    terrain = pd.read_parquet(config.terrain_path)
    rainfall = pd.read_parquet(config.rainfall_path)
    base = pd.read_parquet(config.base_path)[["osm_id", "state", "district"]].drop_duplicates()

    # candidate roads present in the (temporal) rainfall feature set
    rain_osms = rainfall["osm_id"].unique()
    if config.road_sample is not None:
        rng = np.random.default_rng(config.random_state)
        rain_osms = rng.choice(rain_osms, size=min(config.road_sample, len(rain_osms)),
                               replace=False)

    samples = pd.DataFrame(
        [(o, t) for o in rain_osms for t in config.prediction_dates],
        columns=["osm_id", "prediction_time"],
    )
    samples["prediction_time"] = pd.to_datetime(samples["prediction_time"])

    out = compose_feature_matrix(
        samples,
        prediction_col="prediction_time",
        static=terrain,
        static_key="osm_id",
        static_cols=TERRAIN_COLS,
        temporal=rainfall,
        temporal_key="osm_id",
        temporal_time="feature_date",
        temporal_cols=RAINFALL_COLS,
        leak_check=True,
    )
    out = out.merge(base, on="osm_id", how="left")
    return out


def apply_missingness(config: ExperimentConfig, df: pd.DataFrame) -> pd.DataFrame:
    feats = TERRAIN_COLS + RAINFALL_COLS
    strat = _resolve_strategy(config.missingness_strategy)
    out = apply_missingness_strategy(df, feats, strat)
    if config.add_flag and strat in (
        MissingnessStrategy.XGB_NATIVE, MissingnessStrategy.FLAG
    ):
        out = apply_missingness_strategy(out, feats, MissingnessStrategy.FLAG)
    return out.reset_index(drop=True)


def make_synthetic_labels(
    matrix: pd.DataFrame,
    config: ExperimentConfig,
) -> pd.DataFrame:
    """SYNTHETIC / TEST ONLY labels to exercise the harness.

    Deterministic AND transparent: an event is synthesised on a road with a
    probability that rises when its real rainfall_7day is above the median, so
    the evaluation metrics are non-degenerate. This NEVER represents real risk -
    it exists solely to prove the train/evaluate machinery. Tagged
    label_source='synthetic_test_only'.
    """
    rng = np.random.default_rng(config.random_state)
    frame = matrix.copy()

    thr = frame["rainfall_7day"].median()
    p = 0.08 + 0.2 * (frame["rainfall_7day"] > thr).astype(float)
    p = p.clip(0, 1)
    event = rng.random(len(frame)) < p

    # prediction windows per row (use the prediction_time directly)
    lead = pd.Series(
        rng.integers(1, int(config.horizon / pd.Timedelta("1 days")) + 1, len(frame)),
        index=frame.index,
    )
    frame["label"] = event.astype(int)
    frame["lead_time"] = pd.to_timedelta(lead, unit="D").where(event)
    frame["label_source"] = "synthetic_test_only"
    return frame


def train_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: Sequence[str],
    config: ExperimentConfig,
) -> Dict:
    """Train an XGBoost baseline and return evaluation metrics (incl. lead time)."""
    import xgboost as xgb
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model = xgb.XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        eval_metric="logloss",
    )
    warn = os.environ.get("XGB_NATIVE_WARNING", "1") == "1"
    if warn and np.any(np.isnan(X_train)):
        warnings.warn(
            "Training matrix contains NaN - relying on XGBoost native NaN handling."
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {}
    if len(np.unique(y_test)) > 1 and len(np.unique(proba)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
        metrics["pr_auc"] = float(average_precision_score(y_test, proba))
        metrics["log_loss"] = float(log_loss(y_test, proba))
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan
        metrics["log_loss"] = np.nan

    metrics["precision"] = float(precision_score(y_test, pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_test, pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_test, pred, zero_division=0))
    metrics["n_test"] = int(len(y_test))
    metrics["n_test_positive"] = int(y_test.sum())
    metrics["feature_importance"] = dict(zip(feature_names, model.feature_importances_.tolist()))
    metrics["model"] = model

    # lead time only meaningful against real positives with a lead_time column;
    # the harness cannot compute it from the 0/1 label alone. Caller attaches.
    return metrics


def run_experiment(config: ExperimentConfig) -> Dict:
    """Full harness run. Returns a JSON-serialisable summary dict."""
    print("=" * 70)
    print("SYNTHETIC LABELS - TEST ONLY - NOT A REAL PERFORMANCE CLAIM")
    print("(no real source-supported hazard labels exist yet in this repo)")
    print("=" * 70)

    matrix = build_design_matrix(config)
    matrix = apply_missingness(config, matrix)

    label_matrix = make_synthetic_labels(matrix, config)

    feats = TERRAIN_COLS + RAINFALL_COLS
    if config.add_flag:
        feats = feats + [f"is_missing_{c}" for c in TERRAIN_COLS + RAINFALL_COLS]

    # temporal split
    times = label_matrix["prediction_time"].to_numpy()
    membership = temporal_split(
        times,
        train_until=config.train_until,
        validation_until=config.validation_until,
    )
    label_matrix["split"] = membership

    train = label_matrix[membership == TRAIN]
    valid = label_matrix[membership == VALIDATION]
    test = label_matrix[membership == TEST]

    X_train = train[feats].to_numpy(dtype=float)
    y_train = train["label"].to_numpy()
    X_valid = valid[feats].to_numpy(dtype=float)
    y_valid = valid["label"].to_numpy()
    X_test = test[feats].to_numpy(dtype=float)
    y_test = test["label"].to_numpy()

    # final safety: no feature may be future for any split
    assert_features_not_future(
        test["prediction_time"].to_numpy(),
        test["feature_latest_ts"].to_numpy(dtype="datetime64[ns]"),
        feats,
    )

    metrics = train_evaluate(X_train, y_train, X_valid, y_valid, X_test, y_test, feats, config)

    # lead-time-aware metric: median warning lead time of TRUE POSITIVES that
    # the model correctly flags at the 0.5 threshold (harness uses synthetic
    # lead_time; real runs would use getattr(events, 'lead_time')).
    proba = metrics["model"].predict_proba(X_test)[:, 1]
    pred_pos = proba >= 0.5
    tp = pred_pos & (y_test == 1)
    test_lead = label_matrix.loc[membership == TEST, "lead_time"]
    median_lead = None
    if tp.any() and test_lead[tp].notna().any():
        median_lead = float(test_lead[tp].dt.total_seconds().median() / 86400.0)

    summary = {
        "config": {
            "horizon": str(config.horizon),
            "prediction_dates": list(config.prediction_dates),
            "missingness_strategy": config.missingness_strategy,
            "label_source": "synthetic_test_only",
        },
        "data": {
            "n_samples": int(len(label_matrix)),
            "split_counts": {k: int(v) for k, v in
                             label_matrix["split"].value_counts().items()},
            "testable_states_train": sorted(train["state"].dropna().unique().tolist()),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "model"},
    }
    if median_lead is not None:
        summary["metrics"]["median_warning_lead_time_days"] = median_lead
    return summary
