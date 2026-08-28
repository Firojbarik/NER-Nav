"""Temporal-safe feature composition.

Assembles a design matrix for (road_segment, prediction_time) samples from:

* static per-segment features (e.g. terrain) - time-invariant, safe at any
  prediction time;
* temporal per-(segment, time) features (e.g. daily rainfall) - joined using the
  MOST RECENT observation with time <= prediction_time (never a future value).

Every composed matrix can be checked with ml.splits.temporal.assert_features_not_future
to fail-closed on temporal leakage. Pure computation (no I/O); integration tests use
REAL feature parquets but SYNTHETIC samples/labels (SYNTHETIC / TEST ONLY).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ml.splits.temporal import assert_features_not_future


def compose_feature_matrix(
    samples: pd.DataFrame,
    prediction_col: str,
    static: Optional[pd.DataFrame] = None,
    static_key: Optional[str] = None,
    static_cols: Optional[Sequence[str]] = None,
    temporal: Optional[pd.DataFrame] = None,
    temporal_key: Optional[str] = None,
    temporal_time: Optional[str] = None,
    temporal_cols: Optional[Sequence[str]] = None,
    leak_check: bool = True,
) -> pd.DataFrame:
    """Compose static + latest-temporal features for each sample.

    Parameters
    ----------
    samples : frame with the sample key (must match static_key/temporal_key) and
        prediction_col.
    static : per-key static features frame (dense, one row per key).
    temporal : per-(key, time) temporal features frame (sparse over time).

    Non-None feature frames are merged; rows with no matching features are kept
    with NaN. `leak_check` verifies each sample's features never use a timestamp
    after its prediction time (static treated as non-future).

    Returns
    -------
    samples + composed feature columns, plus a `feature_latest_ts` column giving
    the newest observation timestamp used per row (for the leakage guard).
    """
    s = samples.copy()

    sample_key = static_key or temporal_key
    if sample_key is None:
        raise ValueError("static_key or temporal_key required")
    if sample_key not in s.columns:
        raise ValueError(f"sample lacks key column {sample_key!r}")

    pred = pd.to_datetime(s[prediction_col]).to_numpy()
    out = s.copy()
    feature_latest_ts = np.full(len(s), np.datetime64("NaT", "ns"))

    # --- static features: time-invariant (latest ts = NaT = non-future) ---
    if static is not None:
        if static_key not in static.columns or static_cols is None:
            raise ValueError("static frame needs static_key and static_cols")
        merged = s.merge(
            static[[static_key, *static_cols]], on=static_key, how="left"
        )
        out[list(static_cols)] = merged[list(static_cols)]

    # --- temporal features: most recent prior observation ---
    if temporal is not None:
        if temporal_key not in temporal.columns or temporal_time is None:
            raise ValueError("temporal frame needs temporal_key and temporal_time")
        tcols = list(temporal_cols) if temporal_cols is not None else [
            c for c in temporal.columns if c not in (temporal_key, temporal_time)
        ]
        t = temporal[[temporal_key, temporal_time, *tcols]].copy()
        t["_t"] = pd.to_datetime(t[temporal_time], errors="coerce")
        t = t.dropna(subset=["_t"])

        cand = s.merge(t, on=temporal_key, how="inner")
        cand = cand.sort_values("_t", kind="stable") if not cand.empty else cand
        cand["_p"] = cand[prediction_col].map(pd.to_datetime)
        cand = cand[cand["_t"] <= cand["_p"]]
        if cand.empty:
            for c in tcols:
                out[c] = np.nan
        else:
            keep = cand.drop_duplicates(subset=[sample_key, prediction_col], keep="last")
            merged_ts = s.merge(
                keep[[sample_key, prediction_col, "_t", *tcols]],
                on=[sample_key, prediction_col],
                how="left",
            )
            out[list(tcols)] = merged_ts[list(tcols)]
            feature_latest_ts = merged_ts["_t"].to_numpy(dtype="datetime64[ns]")

    out["feature_latest_ts"] = feature_latest_ts

    if leak_check:
        # static features are time-invariant -> their latest ts (NaT) is <= pred
        guard_ts = np.where(np.isnat(feature_latest_ts), pred, feature_latest_ts)
        assert_features_not_future(pred, guard_ts, [])

    return out
