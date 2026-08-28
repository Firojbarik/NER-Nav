"""Temporal label generation for disruption prediction.

Defines how (road_segment, prediction_timestamp) samples map to a binary
disruption label using REAL historical incident events. Pure/computation only
(no I/O), so it is unit-testable. Synthetic event frames used in tests are
SYNTHETIC / TEST ONLY - never real training labels.

Definition (documented, defensible):
  sample      = (road_segment_id, prediction_time, future_horizon)
  event_time  = timestamp of a qualifying real disruption event
  label       = 1 when a qualifying event affects this road_segment at a time
                in (prediction_time, prediction_time + horizon]
              = 0 otherwise

Temporal rule:
  An event counts only if  prediction_time < event_time <= prediction_time + horizon.
  The anchor day/time itself is NOT inside the future window (an event at
  exactly prediction_time is not a future disruption).

Spatial rule (explicit by default, matching the project's no-distance-only-label
policy): an event affects a road_segment only through an EXPLICIT association
(event.affected_segment_id == segment_id). Buffer-based matching is optional and
must be enabled explicitly via `buffer_m`; it may only be used where the project
policy permits (it is currently disallowed for real labels).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set

import numpy as np
import pandas as pd

REQUIRED_EVENT_COLS = {"event_id", "event_time"}
# Columns needed for buffer (spatial) matching.
SPATIAL_COLS = {"latitude", "longitude"}


@dataclass(frozen=True)
class LabelingConfig:
    """Configuration for label generation.

    horizon            : future prediction window length.
    include_event_types: optional set of event types considered qualifying.
                         If None, all events with a valid time are qualifying.
    spatial_mode       : "explicit" (default) or "buffer".
    buffer_m           : buffer radius (metres) used only when spatial_mode="buffer".
    """

    horizon: pd.Timedelta
    include_event_types: Optional[Set[str]] = None
    spatial_mode: Literal["explicit", "buffer"] = "explicit"
    buffer_m: Optional[float] = None

    def __post_init__(self):
        if self.horizon <= pd.Timedelta(0):
            raise ValueError("horizon must be strictly positive")
        if self.spatial_mode == "buffer" and (self.buffer_m is None or self.buffer_m <= 0):
            raise ValueError("buffer_m is required and must be > 0 for buffer mode")


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Normalise and validate an events frame.

    Expected columns: event_id, event_time, [event_type],
                      [affected_segment_id], [latitude, longitude].
    Returns a copy with event_time as timezone-naive datetime64[ns].
    """
    missing = REQUIRED_EVENT_COLS - set(events.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")

    ev = events.copy()
    ev["event_time"] = pd.to_datetime(ev["event_time"], errors="coerce")
    ev = ev.dropna(subset=["event_time"])
    if ev.empty:
        raise ValueError("no events with valid event_time")

    if "event_type" not in ev.columns:
        ev["event_type"] = np.nan
    for col in ("affected_segment_id", "latitude", "longitude"):
        if col not in ev.columns:
            ev[col] = np.nan
    return ev.reset_index(drop=True)


def _haversine_m(lon1, lat1, lon2, lat2) -> float:
    """Great-circle distance in metres between two points (degrees)."""
    r_earth = 6371000.0
    p1 = np.radians(np.array([lat1, lon1]))
    p2 = np.radians(np.array([lat2, lon2]))
    dphi = p2[0] - p1[0]
    dlmb = p2[1] - p1[1]
    a = np.sin(dphi / 2) ** 2 + np.cos(p1[0]) * np.cos(p2[0]) * np.sin(dlmb / 2) ** 2
    return 2 * r_earth * np.arcsin(np.sqrt(a))


def _coalesce_types(ev: pd.DataFrame, include_event_types: Optional[Set[str]]):
    if include_event_types is None:
        return ev
    return ev[ev["event_type"].isin(include_event_types)]


def apply_labels(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    config: LabelingConfig,
) -> pd.DataFrame:
    """Apply binary disruption labels to samples given real events.

    Parameters
    ----------
    samples : DataFrame with columns road_segment_id and prediction_time.
    events  : DataFrame with event_id, event_time and [event_type],
              [affected_segment_id], [latitude, longitude].
    config  : LabelingConfig.

    Returns
    -------
    A copy of `samples` with columns:
        label                 : int 0/1
        matching_event_count  : number of qualifying events
        earliest_event_time   : first qualifying event time (NaN if none)
        lead_time             : earliest_event_time - prediction_time (NaN if none)
    """
    required = {"road_segment_id", "prediction_time"}
    missing = required - set(samples.columns)
    if missing:
        raise ValueError(f"samples missing required columns: {sorted(missing)}")

    smp = samples.copy()
    smp["prediction_time"] = pd.to_datetime(smp["prediction_time"], errors="coerce")
    if smp["prediction_time"].isna().any():
        raise ValueError("samples contain unparsable prediction_time")

    ev = validate_events(events)
    ev = _coalesce_types(ev, config.include_event_types)
    if ev.empty:
        raise ValueError("no qualifying events after filtering")

    horizon = config.horizon

    # --- temporal matching: vectorised via per-segment sorted event search ---
    smp = smp.sort_values("prediction_time").reset_index(drop=True)

    # Determine per-sample candidate events (segment + time window).
    if config.spatial_mode == "buffer":
        if not SPATIAL_COLS.issubset(ev.columns):
            raise ValueError("buffer mode requires latitude/longitude on events")
        ev = ev.dropna(subset=["latitude", "longitude"])
        # Per sample, buffer match against all events (code below handles time)
    else:
        # explicit mode: merge events onto samples by segment
        ev_cand = ev.dropna(subset=["affected_segment_id"]).rename(
            columns={"affected_segment_id": "road_segment_id"}
        )
        if ev_cand.empty:
            raise ValueError(
                "explicit mode but no events carry an affected_segment_id"
            )
        merged = smp.merge(ev_cand, on="road_segment_id", how="inner")

        # time window filter: prediction_time < event_time <= prediction_time+horizon
        ev_t = merged["event_time"].to_numpy()
        pred_t = merged["prediction_time"].to_numpy()
        in_window = (
            (ev_t > pred_t)
            & (ev_t <= pred_t + np.timedelta64(horizon))
        )
        matches = merged.loc[in_window]

    # NOTE: buffer-mode vectorised implementation is not yet wired for full
    # scale; raise rather than silently mislabel.
    if config.spatial_mode == "buffer":
        raise NotImplementedError(
            "buffer spatial_mode is intentionally not implemented here; "
            "explicit (source-supported) matching is the sanctioned path."
        )

    # Aggregate matches -> label, counts, earliest time, lead time.
    # Labels are (road_segment_id, prediction_time)-specific: two samples for
    # the same segment at different prediction times evaluate DIFFERENT future
    # windows, so aggregation must key on the full sample identity, not on the
    # segment alone (otherwise one segment's window result would be broadcast
    # onto every prediction_time of that segment).
    if matches.empty:
        smp["label"] = 0
        smp["matching_event_count"] = 0
        smp["earliest_event_time"] = pd.NaT
        smp["lead_time"] = pd.NaT
    else:
        g = matches.groupby(["road_segment_id", "prediction_time"]).agg(
            matching_event_count=("event_id", "size"),
            earliest_event_time=("event_time", "min"),
        ).reset_index()
        out = smp.merge(g, on=["road_segment_id", "prediction_time"], how="left")
        out["matching_event_count"] = out["matching_event_count"].fillna(0).astype(int)
        out["label"] = (out["matching_event_count"] > 0).astype(int)
        out["earliest_event_time"] = pd.to_datetime(out["earliest_event_time"])
        out["lead_time"] = (
            out["earliest_event_time"] - out["prediction_time"]
        ).where(out["label"] == 1)
        smp = out

    return smp[["road_segment_id", "prediction_time",
                "label", "matching_event_count",
                "earliest_event_time", "lead_time"]]
