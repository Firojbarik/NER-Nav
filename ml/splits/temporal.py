"""Temporal-leakage-safe data splits.

Two supported strategies (never a random split for time-dependent data):

* temporal_split - chronologically ordered TRAIN / VALIDATION / TEST by
  prediction_timestamp cut points. Optionally inserts a "gap" period between
  validation and test that is used for NEITHER training nor testing, so that
  lookback/lag features spanning the boundary cannot leak.
* geographic_split - TRAIN on some groups (states/districts), TEST on held-out
  groups, to assess spatial generalization. It is orthogonal to time and must be
  combined with a temporal split for time-series data.

Every splitter is pure (no I/O). Tests use synthetic timestamps
(SYNTHETIC / TEST ONLY).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"
GAP = "gap"


def temporal_split(
    times: np.ndarray,
    train_until: str,
    validation_until: Optional[str] = None,
    gap_after_validation: Optional[str] = None,
    gap_after_test: Optional[str] = None,
) -> np.ndarray:
    """Assign TRAIN/VALIDATION/TEST(/GAP) membership by prediction timestamp.

    Parameters
    ----------
    times : array of datetime64 timestamps (prediction times).
    train_until : last timestamp (exclusive upper bound) of the training period.
    validation_until : last timestamp (exclusive) of the validation period.
    gap_after_validation : optional start of the TEST period; the slice
        [validation_until, gap_after_validation) is labelled GAP and used for
        neither training nor test.
    gap_after_test : optional timestamp; the slice [gap_after_test, end) is
        labelled GAP (future data withheld).

    The final TEST period is whichever slice follows the validation/gap region.
    The final test period is NEVER used during model selection.
    """
    t = np.asarray(times) if not isinstance(times, np.ndarray) else times
    if t.dtype.kind != "M":
        raise TypeError("times must be numpy datetime64 array")

    def as_dt(s: Optional[str]) -> Optional[np.datetime64]:
        return None if s is None else np.datetime64(s)

    train_until = as_dt(train_until)
    validation_until = as_dt(validation_until)
    gap_after_validation = as_dt(gap_after_validation)
    gap_after_test = as_dt(gap_after_test)

    if train_until is None:
        raise ValueError("train_until is required")

    membership = np.full(t.shape, GAP, dtype=object)

    # train: times < train_until
    membership[t < train_until] = TRAIN
    if validation_until is not None:
        membership[(t >= train_until) & (t < validation_until)] = VALIDATION
    if gap_after_validation is not None:
        membership[(t >= validation_until) & (t < gap_after_validation)] = GAP
        membership[t >= gap_after_validation] = TEST
    elif validation_until is not None:
        membership[t >= validation_until] = TEST
    else:
        # no validation period -> rest is test
        membership[t >= train_until] = TEST

    if gap_after_test is not None:
        membership[t >= gap_after_test] = GAP

    return membership


def geographic_split(
    groups: np.ndarray,
    train_groups: Sequence,
    test_groups: Sequence,
    holdout: Optional[str] = None,
) -> np.ndarray:
    """Assign TRAIN/TEST(/HOLDOUT) membership by group for spatial generalization.

    groups : array of group labels (states/districts) per row.
    train_groups : groups allowed in training.
    test_groups : groups reserved for testing.
    holdout : optional group label classified OUT (excluded from both).

    Any group NOT listed in train_groups/test_groups is treated as HOLDOUT
    (excluded), preventing accidental testing on groups that were never
    intentionally nominated. Conservative by design.
    """
    g = np.asarray(groups)
    train_set = set(train_groups)
    test_set = set(test_groups)
    if train_set & test_set:
        raise ValueError("a group cannot be both train and test")
    membership = np.full(g.shape, "holdout", dtype=object)
    membership[np.isin(g, list(train_set))] = TRAIN
    membership[np.isin(g, list(test_set))] = TEST
    if holdout is not None:
        membership[np.isin(g, [holdout])] = "holdout"
    return membership


def validate_temporal_monotonic(
    times: np.ndarray,
    membership: np.ndarray,
    order: Tuple[str, ...] = (TRAIN, VALIDATION, TEST),
) -> bool:
    """Assert membership respects chronological order.

    Verifies that every row labelled with a later-ordered split occurs at a time
    at least as large as every earlier-ordered split (monotone bounds).
    Returns True; raises ValueError if violated.
    """
    t = np.asarray(times)
    valid = set(order) | {GAP, "holdout"}
    unknown = set(np.unique(membership)) - valid
    if unknown:
        raise ValueError(f"unexpected membership labels: {sorted(unknown)}")

    relevant = [s for s in order if np.any(membership == s)]
    for i in range(len(relevant) - 1):
        lo = t[membership == relevant[i]]
        hi = t[membership == relevant[i + 1]]
        if lo.size and hi.size and hi.min() < lo.max():
            raise ValueError(
                f"temporal ordering violated: {relevant[i+1]} before {relevant[i]}"
            )
    return True


def assert_features_not_future(
    prediction_times: np.ndarray,
    feature_timestamps: np.ndarray,
    feature_names: Sequence[str],
) -> bool:
    """Guard: no feature may carry a timestamp after its sample's prediction time.

    feature_timestamps : (n_samples) the latest timestamp each sample's features
        rely on. Must satisfy feature_timestamp <= prediction_time for every row,
        otherwise the sample leaks future information.
    Raises ValueError on violation. Returns True.
    """
    p = np.asarray(prediction_times)
    f = np.asarray(feature_timestamps)
    if p.shape != f.shape:
        raise ValueError("prediction_times and feature_timestamps must be same length")
    if np.any(f > p):
        bad = np.flatnonzero(f > p)
        raise ValueError(
            f"TEMPORAL LEAKAGE: {len(bad)} row(s) use future feature data "
            f"(first offending row index {bad[0]})"
        )
    return True
