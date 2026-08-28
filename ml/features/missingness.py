"""Feature-missingness handling and coverage gating.

The real terrain artifact has SYSTEMATIC (spatially-correlated) no-data, not
random: whole states have 100% elevation coverage and other whole states have
0%. This is a structural geographic-coverage gap, not a noise problem.

Grounded principles:
* NEVER impute across a group boundary from another group's data (that is
  fabrication, disallowed). Any imputation is intra-group only.
* Never fabricate a value where the group has no observation at all.
* Missingness must be both reported (rates) and gated (which groups are
  learnable/testable for a given feature set).

Strategies (pure/decision functions, no model fitting):
  DROP_ANY   - drop samples missing ANY feature (strict; loses whole states).
  DROP_ALL   - drop samples missing ALL features (retains partially-covered).
  XGB_NATIVE - keep NaN and let a NaN-tolerant model (e.g. XGBoost) use them;
               requires a downstream missingness-aware learner.
  FLAG       - keep NaN and add a per-feature is-missing indicator column.
  IMPUTE_IN_GROUP - intra-group imputation only; leaves entries NaN where the
               group has no observation at all (never cross-boundary).

Coverage gate: a group is "testable" for a feature set only if it has observed
(non-missing) values for it; otherwise claims about that group are unsupported
by data and must be excluded from evaluation. Reported, not silently imputed.
"""

from __future__ import annotations

import enum
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class MissingnessStrategy(str, enum.Enum):
    DROP_ANY = "drop_any"
    DROP_ALL = "drop_all"
    XGB_NATIVE = "xgb_native"
    FLAG = "flag"
    IMPUTE_IN_GROUP = "impute_in_group"


def assess_missingness(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> pd.DataFrame:
    """Return per-column missing-rate summary for a features frame."""
    m = df[list(feature_cols)].isna().mean()
    return pd.DataFrame(
        {"feature": m.index, "missing_rate": m.values},
    ).reset_index(drop=True)


def coverage_by_group(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    """Per-group observed (non-missing) rate for each feature.

    A rate of 0 means that group has NO observation for that feature -> any
    prediction for it using that feature is unsupported by data.
    """
    rows = []
    for g, sub in df.groupby(group_col):
        denom = len(sub)
        for c in feature_cols:
            obs = int(sub[c].notna().sum())
            rows.append({"group": g, "feature": c, "observed_rate": obs / denom,
                         "n_obs": obs, "n_group": denom})
    return pd.DataFrame(rows)


def testable_groups(
    df: pd.DataFrame,
    group_col: str,
    feature_cols: Sequence[str],
) -> Tuple[list, pd.DataFrame]:
    """Groups that have at least one observed value for EVERY requested feature.

    Returns (testable_group_list, full_report_df). Groups missing any requested
    feature entirely are excluded from "testable".
    """
    rep = coverage_by_group(df, group_col, feature_cols)
    pivot = rep.pivot(index="group", columns="feature", values="observed_rate")
    testable = pivot[(pivot > 0).all(axis=1)].index.tolist()
    return list(testable), rep


def apply_missingness_strategy(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    strategy: MissingnessStrategy,
    group_col: Optional[str] = None,
    add_flag_prefix: str = "is_missing",
) -> pd.DataFrame:
    """Apply a missingness strategy to a features frame.

    df : features frame (features may contain NaN).
    strategy : one of MissingnessStrategy.
    group_col : required ONLY for IMPUTE_IN_GROUP.

    Returns a new DataFrame (rows dropped/added columns as per strategy). For
    XGB_NATIVE the features are returned as-is (NaN retained for the model).
    """
    out = df.copy()
    if strategy == MissingnessStrategy.DROP_ANY:
        return out.dropna(subset=list(feature_cols)).reset_index(drop=True)

    if strategy == MissingnessStrategy.DROP_ALL:
        return out.dropna(how="all", subset=list(feature_cols)).reset_index(drop=True)

    if strategy == MissingnessStrategy.XGB_NATIVE:
        return out

    if strategy == MissingnessStrategy.FLAG:
        for c in feature_cols:
            out[f"{add_flag_prefix}_{c}"] = out[c].isna().astype(int)
        return out

    if strategy == MissingnessStrategy.IMPUTE_IN_GROUP:
        if group_col is None or group_col not in out.columns:
            raise ValueError("IMPUTE_IN_GROUP requires group_col")
        for c in feature_cols:
            out[c] = (
                out.groupby(group_col)[c]
                .transform(lambda s: s.fillna(s.mean()))
            )
        return out

    raise ValueError(f"unknown strategy: {strategy}")


def report_retention(
    original_n: int, result: pd.DataFrame
) -> dict:
    """Summary dict of retention vs original row count."""
    return {
        "original_rows": int(original_n),
        "retained_rows": int(len(result)),
        "retention_rate": float(len(result) / original_n) if original_n else 0.0,
    }
