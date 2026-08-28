"""Date-anchored, nodata-aware rainfall aggregation.

Pure computation over numpy arrays only (no I/O), so the logic is unit-testable
with synthetic arrays. Synthetic arrays here are SYNTHETIC / TEST ONLY - they are
used to verify algorithm mechanics, never for training.

Semantics
---------
* Raw CHIRPS raster samples use negative values (commonly -9999) to indicate
  "no data". These MUST become NaN, not 0 : a missing pixel is "unknown", not
  "no rain". A valid pixel may legitimately be 0 mm.
* Aggregates are anchored to calendar day-offsets relative to a prediction
  timestamp, so gaps never misalign (a 3-day window always means the 3 calendar
  days immediately before the anchor).
* By default a window aggregate is NaN unless EVERY constituent day is present
  and valid - we must not fabricate "low rain" from missing observations. The
  caller may opt into partial-window sums.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

# CHIRPS fills no-data with a large negative sentinel. Any strictly negative
# value in the precip band is treated as no-data.
CHIRPS_NODATA_THRESHOLD = 0.0


def clean_daily_samples(samples: np.ndarray) -> np.ndarray:
    """Convert raw raster samples: negative -> NaN, keep valid values (>=0).

    Parameters
    ----------
    samples : np.ndarray
        Raw (n_roads,) or (n,) samples from a CHIRPS precipitation band.

    Returns
    -------
    np.ndarray
        Same shape; NaN where the pixel was no-data/outside coverage.
    """
    arr = np.asarray(samples, dtype=np.float64).copy()
    arr[arr < CHIRPS_NODATA_THRESHOLD] = np.nan
    return arr


def window_offsets(anchor_lookback_days: int) -> np.ndarray:
    """Calendar day-offsets belonging to a trailing window.

    window_offsets(3) -> [-3, -2, -1] (ascending). Negative means "that many
    days before the anchor"; day 0 (the prediction timestamp itself) is never
    included so a feature can never leak the current/future day's rainfall.
    """
    return -np.arange(anchor_lookback_days, 0, -1, dtype=np.int64)


def aggregate_window(
    daily: np.ndarray,
    day_offsets: np.ndarray,
    window_days: int,
    allow_partial: bool = False,
) -> np.ndarray:
    """Sum rainfall over the trailing `window_days` calendar days.

    Parameters
    ----------
    daily : (n_roads, n_days) float array. NaN = missing/unknown that day.
    day_offsets : (n_days,) int array. Negative days before anchor, e.g.
        [-30, -29, ..., -1]. Must be sorted ascending.
    window_days : int. Size of the trailing window (e.g. 1, 3, 7, 14, 30).
    allow_partial : bool
        If False (default): the window is NaN unless every day in it is valid.
        If True: sum the valid days and return NaN only if all days are missing.
        Partial sums are NOT comparable across rows because the support differs;
        use with care.

    Returns
    -------
    (n_roads,) float array of rolling window sums.
    """
    daily = np.asarray(daily, dtype=np.float64)
    day_offsets = np.asarray(day_offsets, dtype=np.int64)

    if daily.ndim != 2 or day_offsets.ndim != 1:
        raise ValueError("daily must be 2-D and day_offsets must be 1-D")
    if daily.shape[1] != day_offsets.shape[0]:
        raise ValueError(
            "day_offsets length must equal the number of daily columns: "
            f"{day_offsets.shape[0]} != {daily.shape[1]}"
        )

    # Offsets in [-window, -1]: strictly before the anchor, never day 0/future.
    within = (day_offsets >= -window_days) & (day_offsets < 0)
    cols = np.flatnonzero(within)
    if cols.size == 0:
        return np.full(daily.shape[0], np.nan)

    block = daily[:, cols]
    valid = ~np.isnan(block)

    if allow_partial:
        sums = np.nansum(block, axis=1)
        # all-missing rows stay NaN
        sums[np.all(~valid, axis=1)] = np.nan
        return sums

    return np.where(np.all(valid, axis=1), np.sum(block, axis=1), np.nan)


def aggregate_windows(
    daily: np.ndarray,
    day_offsets: np.ndarray,
    windows: List[int],
    allow_partial: bool = False,
) -> Dict[int, np.ndarray]:
    """Aggregate several trailing windows at once.

    Returns a dict mapping each window size to its (n_roads,) aggregate array.
    """
    return {
        w: aggregate_window(daily, day_offsets, w, allow_partial=allow_partial)
        for w in windows
    }


def available_day_counts(daily: np.ndarray) -> np.ndarray:
    """Number of valid (non-NaN) daily observations per row across all days."""
    return np.sum(~np.isnan(np.asarray(daily, dtype=np.float64)), axis=1)
