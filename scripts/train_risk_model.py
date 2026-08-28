
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

DATASET = Path(
    "data/processed/ml/risk_training_dataset.parquet"
)

OUTPUT_DIR = Path("data/processed/ml")

MODEL_FILE = OUTPUT_DIR / "risk_model.json"
PREPROCESSOR_FILE = (
    OUTPUT_DIR / "risk_model_preprocessor.joblib"
)
METRICS_FILE = (
    OUTPUT_DIR / "risk_model_metrics.json"
)
IMPORTANCE_FILE = (
    OUTPUT_DIR / "risk_feature_importance.csv"
)
METADATA_FILE = (
    OUTPUT_DIR / "risk_model_metadata.json"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RANDOM_STATE = 42

TARGET = "risk_class"

# IMPORTANT:
# These columns must NOT be used as predictors.
#
# risk_label:
#     directly represents the synthetic risk class.
#
# risk_score:
#     is part of the synthetic label-generation logic.
#
# road_risk_prior:
#     is also derived from the same synthetic road-risk prior.
#
# Using them caused target leakage and produced the previous
# unrealistic 1.0000 validation/test scores.
LEAKAGE_COLUMNS = {
    "risk_label",
    "risk_score",
    "road_risk_prior",
}

# Dataset identity / bookkeeping columns.
NON_FEATURE_COLUMNS = {
    TARGET,
    "osm_id",
    "label_source",
}

EXCLUDED_COLUMNS = (
    NON_FEATURE_COLUMNS
    | LEAKAGE_COLUMNS
)


# ---------------------------------------------------------------------
# Expected feature groups
# ---------------------------------------------------------------------

NUMERIC_FEATURES = [
    "longitude",
    "latitude",
    "start_longitude",
    "start_latitude",
    "end_longitude",
    "end_latitude",
    "intersection_length_m",
    "coverage_ratio",
    "bridge_flag",
    "tunnel_flag",
    "lit_flag",
    "service_flag",
    "low_coverage_flag",
]

CATEGORICAL_FEATURES = [
    "district",
    "state",
    "highway",
    "surface",
    "maxspeed",
    "lanes",
    "access",
    "bridge",
    "tunnel",
    "lit",
    "smoothness",
    "tracktype",
    "service",
    "oneway",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the training dataset before model construction."""

    print()
    print("Dataset validation...")
    print("-" * 70)

    required = {
        TARGET,
        "osm_id",
        "label_source",
    }

    missing_required = required - set(df.columns)

    if missing_required:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing_required)}"
        )

    classes = sorted(
        pd.Series(df[TARGET])
        .dropna()
        .unique()
        .tolist()
    )

    print(f"Target classes: {classes}")

    if classes != [0, 1, 2, 3]:
        raise ValueError(
            "Expected risk classes [0, 1, 2, 3], "
            f"found {classes}"
        )

    label_sources = (
        df["label_source"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(f"Label source: {label_sources}")

    print()
    print("Leakage protection...")
    print("-" * 70)

    for column in sorted(LEAKAGE_COLUMNS):
        if column in df.columns:
            print(f"EXCLUDED: {column}")
        else:
            print(f"NOT PRESENT: {column}")

    print()
    print(
        "Synthetic-label warning: "
        "the current target is still synthetic/proxy data."
    )


def build_feature_lists(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Build final feature lists after leakage removal."""

    numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
        and column not in EXCLUDED_COLUMNS
    ]

    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
        and column not in EXCLUDED_COLUMNS
    ]

    missing_numeric = [
        column
        for column in NUMERIC_FEATURES
        if column not in df.columns
    ]

    missing_categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column not in df.columns
    ]

    if missing_numeric:
        print()
        print(
            "WARNING: numeric columns not present:"
        )
        for column in missing_numeric:
            print(f"  - {column}")

    if missing_categorical:
        print()
        print(
            "WARNING: categorical columns not present:"
        )
        for column in missing_categorical:
            print(f"  - {column}")

    if not numeric and not categorical:
        raise ValueError(
            "No usable model features remain."
        )

    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def evaluate_model(
    model: xgb.XGBClassifier,
    X,
    y,
    split_name: str,
) -> dict:
    """Evaluate classifier and return metrics."""

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    macro_f1 = f1_score(
        y,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "low",
            "moderate",
            "high",
            "severe",
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=[0, 1, 2, 3],
    )

    print()
    print(f"{split_name}:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Macro F1:    {macro_f1:.4f}")
    print(f"  Weighted F1: {weighted_f1:.4f}")

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    print_header(
        "NER-Nav — Leakage-Safe Risk Model Training"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DATASET.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {DATASET}"
        )

    print()
    print(f"Dataset: {DATASET}")
    print(f"Model:   {MODEL_FILE}")

    # ---------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------

    print()
    print("Loading training dataset...")

    start = time.perf_counter()

    df = pd.read_parquet(DATASET)

    load_time = time.perf_counter() - start

    print(f"Rows:       {len(df):,}")
    print(f"Columns:    {len(df.columns)}")
    print(f"Load time:  {load_time:.2f} seconds")

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------

    validate_dataset(df)

    # ---------------------------------------------------------------
    # Features
    # ---------------------------------------------------------------

    numeric_features, categorical_features = (
        build_feature_lists(df)
    )

    feature_columns = (
        numeric_features
        + categorical_features
    )

    print()
    print("Final model feature set...")
    print("-" * 70)

    print(
        f"Excluded columns: "
        f"{sorted(EXCLUDED_COLUMNS)}"
    )

    print(
        f"Feature columns:  {len(feature_columns)}"
    )

    print()
    print("Numeric features:")

    for column in numeric_features:
        print(f"  - {column}")

    print()
    print("Categorical features:")

    for column in categorical_features:
        print(f"  - {column}")

    # Explicit leakage assertion.
    leaked_features = (
        set(feature_columns)
        & LEAKAGE_COLUMNS
    )

    if leaked_features:
        raise RuntimeError(
            "TARGET LEAKAGE DETECTED. "
            f"These columns remain in X: "
            f"{sorted(leaked_features)}"
        )

    # ---------------------------------------------------------------
    # X / y
    # ---------------------------------------------------------------

    X = df[feature_columns].copy()

    y = (
        df[TARGET]
        .astype(int)
        .copy()
    )

    # ---------------------------------------------------------------
    # Train / validation / test split
    # ---------------------------------------------------------------

    print()
    print(
        "Creating stratified "
        "train/validation/test split..."
    )
    print("-" * 70)

    X_train, X_temp, y_train, y_temp = (
        train_test_split(
            X,
            y,
            test_size=0.30,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    )

    X_validation, X_test, y_validation, y_test = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=2 / 3,
            stratify=y_temp,
            random_state=RANDOM_STATE,
        )
    )

    print(
        f"Train:       {len(X_train):,}"
    )
    print(
        f"Validation:  {len(X_validation):,}"
    )
    print(
        f"Test:        {len(X_test):,}"
    )

    print()
    print("Class distribution:")

    for name, values in [
        ("Train", y_train),
        ("Validation", y_validation),
        ("Test", y_test),
    ]:
        print(f"  {name}:")

        counts = (
            values
            .value_counts()
            .sort_index()
        )

        for cls in [0, 1, 2, 3]:
            print(
                f"    {cls}: "
                f"{int(counts.get(cls, 0)):,}"
            )

    # ---------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------

    print()
    print("Building preprocessing pipeline...")

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
    )

    print("Fitting preprocessing pipeline...")

    start = time.perf_counter()

    X_train_processed = (
        preprocessor.fit_transform(X_train)
    )

    X_validation_processed = (
        preprocessor.transform(X_validation)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    preprocessing_time = (
        time.perf_counter() - start
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    print(
        f"Encoded feature count: "
        f"{len(feature_names)}"
    )

    print(
        f"Preprocessing time: "
        f"{preprocessing_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------

    print()
    print("Training XGBoost classifier...")
    print("-" * 70)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss",
    )

    print("Configuration:")
    print("  n_estimators:     300")
    print("  max_depth:        8")
    print("  learning_rate:    0.08")
    print("  subsample:        0.85")
    print("  colsample_bytree: 0.85")
    print("  random_state:     42")
    print("  n_jobs:           -1")
    print("  tree_method:      hist")

    start = time.perf_counter()

    model.fit(
        X_train_processed,
        y_train,
    )

    training_time = (
        time.perf_counter() - start
    )

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    print()
    print("Evaluating model...")

    validation_metrics = evaluate_model(
        model,
        X_validation_processed,
        y_validation,
        "Validation",
    )

    test_metrics = evaluate_model(
        model,
        X_test_processed,
        y_test,
        "Test",
    )

    # ---------------------------------------------------------------
    # Detailed test metrics
    # ---------------------------------------------------------------

    print_header("MODEL EVALUATION")

    print("Validation:")
    print(
        f"  Accuracy:    "
        f"{validation_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{validation_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{validation_metrics['weighted_f1']:.4f}"
    )

    print()
    print("Test:")
    print(
        f"  Accuracy:    "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"  Macro F1:    "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"  Weighted F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print()
    print("Per-class test metrics:")

    report = test_metrics[
        "classification_report"
    ]

    for label, display_name in [
        ("low", "low"),
        ("moderate", "moderate"),
        ("high", "high"),
        ("severe", "severe"),
    ]:
        print(
            f"  {display_name:<9} "
            f"precision={report[label]['precision']:.4f} "
            f"recall={report[label]['recall']:.4f} "
            f"f1={report[label]['f1-score']:.4f}"
        )

    print()
    print("Confusion matrix:")
    print("Rows = actual, columns = predicted")

    print(
        np.array(
            test_metrics["confusion_matrix"]
        )
    )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    print()
    print("Calculating feature importance...")

    importance = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    print()
    print("Top 20 features:")

    for _, row in importance_df.head(20).iterrows():
        print(
            f"  {row['feature']:<55} "
            f"{row['importance']:.6f}"
        )

    # ---------------------------------------------------------------
    # Leakage verification
    # ---------------------------------------------------------------

    print()
    print("Final leakage verification...")

    importance_text = " ".join(
        importance_df["feature"]
        .astype(str)
        .tolist()
    )

    forbidden_importance_tokens = [
        "risk_label",
        "risk_score",
        "road_risk_prior",
    ]

    leakage_importance_found = [
        token
        for token in forbidden_importance_tokens
        if token in importance_text
    ]

    if leakage_importance_found:
        raise RuntimeError(
            "Leakage verification failed. "
            f"Forbidden features detected: "
            f"{leakage_importance_found}"
        )

    print("Target-leakage feature check: PASS")

    # ---------------------------------------------------------------
    # Save model artifacts
    # ---------------------------------------------------------------

    print()
    print("Saving model artifacts...")

    model.save_model(
        MODEL_FILE
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_FILE,
    )

    metrics = {
        "dataset": str(DATASET),
        "target": TARGET,
        "random_state": RANDOM_STATE,
        "training_rows": int(len(X_train)),
        "validation_rows": int(
            len(X_validation)
        ),
        "test_rows": int(len(X_test)),
        "feature_count_before_encoding": int(
            len(feature_columns)
        ),
        "encoded_feature_count": int(
            len(feature_names)
        ),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_columns": sorted(
            EXCLUDED_COLUMNS
        ),
        "leakage_columns": sorted(
            LEAKAGE_COLUMNS
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "training_time_seconds": float(
            training_time
        ),
        "preprocessing_time_seconds": float(
            preprocessing_time
        ),
        "synthetic_label_warning": True,
        "label_source": sorted(
            df["label_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
    }

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False,
    )

    metadata = {
        "model": "XGBoost multi-class classifier",
        "xgboost_version": xgb.__version__,
        "target": TARGET,
        "classes": {
            "0": "low",
            "1": "moderate",
            "2": "high",
            "3": "severe",
        },
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_columns": sorted(
            EXCLUDED_COLUMNS
        ),
        "leakage_columns": sorted(
            LEAKAGE_COLUMNS
        ),
        "preprocessing": (
            "median imputation for numeric; "
            "most-frequent imputation + "
            "one-hot encoding for categorical"
        ),
        "model_parameters": {
            "n_estimators": 300,
            "max_depth": 8,
            "learning_rate": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "tree_method": "hist",
        },
        "synthetic_label_warning": (
            "risk_class is currently synthetic/proxy. "
            "Real incident/hazard data must replace "
            "these labels before production claims."
        ),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # ---------------------------------------------------------------
    # Final status
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("LEAKAGE-SAFE RISK MODEL TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Training rows:       {len(X_train):,}"
    )
    print(
        f"Validation rows:     {len(X_validation):,}"
    )
    print(
        f"Test rows:           {len(X_test):,}"
    )
    print(
        f"Original features:   {len(feature_columns):,}"
    )
    print(
        f"Encoded features:    {len(feature_names):,}"
    )

    print()
    print(
        f"Test accuracy:       "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"Test macro F1:       "
        f"{test_metrics['macro_f1']:.4f}"
    )
    print(
        f"Test weighted F1:    "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print()
    print(f"Model:               {MODEL_FILE}")
    print(
        f"Preprocessor:        "
        f"{PREPROCESSOR_FILE}"
    )
    print(
        f"Metrics:             "
        f"{METRICS_FILE}"
    )
    print(
        f"Importance:          "
        f"{IMPORTANCE_FILE}"
    )
    print(
        f"Metadata:            "
        f"{METADATA_FILE}"
    )

    print()
    print(
        "IMPORTANT: This remains a "
        "synthetic-label baseline model."
    )
    print(
        "Real hazard/incident data is required "
        "before production claims."
    )

    print(
        "Target leakage check: PASS"
    )

    print("STATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
