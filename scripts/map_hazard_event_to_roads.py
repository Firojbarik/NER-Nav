"""
NER-Nav — Hazard Event → Road Segment Mapping

Purpose
-------
Map a verified real-world hazard event point to nearby road segments.

IMPORTANT
---------
This script ONLY creates spatial candidates.

It MUST NOT:
    - automatically declare roads affected
    - fabricate training labels
    - infer affected roads solely from proximity
    - overwrite source evidence

A later human/source-supported review step must confirm affected segments.

Outputs
-------
data/processed/hazards/nrsc_sikkim_mantam_2016_road_mapping.parquet
data/processed/hazards/nrsc_sikkim_mantam_2016_road_mapping.json
data/processed/hazards/nrsc_sikkim_mantam_2016_road_mapping_qa.json
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_event.json"
)

ROAD_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "roads"
    / "ner_roads_districts.gpkg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
)

MAPPING_PARQUET = (
    OUTPUT_DIR
    / "nrsc_sikkim_mantam_2016_road_mapping.parquet"
)

MAPPING_JSON = (
    OUTPUT_DIR
    / "nrsc_sikkim_mantam_2016_road_mapping.json"
)

MAPPING_QA_JSON = (
    OUTPUT_DIR
    / "nrsc_sikkim_mantam_2016_road_mapping_qa.json"
)

# Candidate search radius.
SEARCH_RADIUS_M = 2000.0

# Maximum number of candidate roads to retain.
MAX_CANDIDATES = 50

# Expected geographic CRS of the road source.
SOURCE_CRS = "EPSG:4326"

# Metric CRS for Sikkim / Mantam area.
# UTM Zone 45N is appropriate for longitude ~88.5E.
METRIC_CRS = "EPSG:32645"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def fail(message: str) -> None:
    raise RuntimeError(message)


def safe_str(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def require_columns(
    gdf: gpd.GeoDataFrame,
    columns: list[str],
    context: str,
) -> None:
    missing = [c for c in columns if c not in gdf.columns]

    if missing:
        fail(
            f"{context} is missing required columns: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------
# LOAD EVENT
# ---------------------------------------------------------------------

def load_event() -> dict[str, Any]:
    print("Loading verified event...")
    print("-" * 70)

    if not EVENT_PATH.exists():
        fail(f"Event file not found: {EVENT_PATH}")

    with EVENT_PATH.open("r", encoding="utf-8") as f:
        event = json.load(f)

    required = [
        "event_id",
        "event_date",
        "latitude",
        "longitude",
        "spatial_status",
    ]

    missing = [x for x in required if x not in event]

    if missing:
        fail(
            "Event JSON is missing required fields: "
            + ", ".join(missing)
        )

    latitude = event["latitude"]
    longitude = event["longitude"]

    if latitude is None or longitude is None:
        fail("Event does not contain verified coordinates.")

    try:
        latitude = float(latitude)
        longitude = float(longitude)
    except (TypeError, ValueError):
        fail("Event coordinates are not numeric.")

    if not (-90 <= latitude <= 90):
        fail(f"Invalid latitude: {latitude}")

    if not (-180 <= longitude <= 180):
        fail(f"Invalid longitude: {longitude}")

    if event["spatial_status"] != "POINT_GEOMETRY_VERIFIED":
        fail(
            "Event spatial status is not "
            "'POINT_GEOMETRY_VERIFIED'."
        )

    print(f"Event ID:       {event['event_id']}")
    print(f"Event date:     {event['event_date']}")
    print(f"Latitude:       {latitude}")
    print(f"Longitude:      {longitude}")
    print(f"Spatial status: {event['spatial_status']}")
    print()

    return event


# ---------------------------------------------------------------------
# LOAD ROADS
# ---------------------------------------------------------------------

def load_roads() -> gpd.GeoDataFrame:
    print("Loading road network...")
    print("-" * 70)

    if not ROAD_PATH.exists():
        fail(f"Road network not found: {ROAD_PATH}")

    roads = gpd.read_file(ROAD_PATH)

    print(f"Total road records: {len(roads):,}")
    print(f"Road CRS:           {roads.crs}")
    print()

    required = [
        "osm_id",
        "highway",
        "name",
        "district",
        "state",
        "geometry",
    ]

    require_columns(
        roads,
        required,
        "Road network",
    )

    if roads.crs is None:
        fail("Road network has no CRS.")

    # Normalize to WGS84 before state filtering / event creation.
    if roads.crs.to_string() != SOURCE_CRS:
        print(
            f"Reprojecting road network from "
            f"{roads.crs} to {SOURCE_CRS}..."
        )

        roads = roads.to_crs(SOURCE_CRS)

    return roads


# ---------------------------------------------------------------------
# ROAD GEOMETRY QA
# ---------------------------------------------------------------------

def validate_geometry(
    roads: gpd.GeoDataFrame,
) -> None:
    print("Road geometry QA...")
    print("-" * 70)

    invalid_count = int((~roads.geometry.is_valid).sum())
    empty_count = int(roads.geometry.is_empty.sum())
    null_count = int(roads.geometry.isna().sum())

    print(f"Invalid geometries: {invalid_count:,}")
    print(f"Empty geometries:   {empty_count:,}")
    print(f"Null geometries:    {null_count:,}")
    print()

    if invalid_count > 0:
        fail(
            "Road network contains invalid geometries. "
            "Fix the source network before hazard mapping."
        )

    if empty_count > 0:
        fail(
            "Road network contains empty geometries."
        )

    if null_count > 0:
        fail(
            "Road network contains null geometries."
        )


# ---------------------------------------------------------------------
# FILTER STATE
# ---------------------------------------------------------------------

def filter_state(
    roads: gpd.GeoDataFrame,
    state: str,
) -> gpd.GeoDataFrame:

    print("State filter...")
    print("-" * 70)

    state_upper = safe_str(state).upper()

    state_mask = (
        roads["state"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq(state_upper)
    )

    filtered = roads.loc[state_mask].copy()

    print(f"{state_upper} road records: {len(filtered):,}")
    print()

    if filtered.empty:
        fail(
            f"No road records found for state: {state}"
        )

    return filtered


# ---------------------------------------------------------------------
# DISTANCE CALCULATION
# ---------------------------------------------------------------------

def calculate_distances(
    roads: gpd.GeoDataFrame,
    latitude: float,
    longitude: float,
) -> gpd.GeoDataFrame:

    print("Calculating event-to-road distances...")
    print("-" * 70)

    print(f"Metric CRS:       {METRIC_CRS}")
    print(f"Search radius:    {SEARCH_RADIUS_M:,.0f} m")
    print()

    # Event point in WGS84.
    event_point_wgs84 = gpd.GeoSeries(
        [
            Point(
                float(longitude),
                float(latitude),
            )
        ],
        crs=SOURCE_CRS,
    )

    # Project both roads and event point to metric CRS.
    roads_metric = roads.to_crs(METRIC_CRS).copy()

    event_point_metric = event_point_wgs84.to_crs(
        METRIC_CRS
    ).iloc[0]

    # Distance from event point to each road geometry.
    distances = roads_metric.geometry.distance(
        event_point_metric
    )

    roads_metric["distance_m"] = distances.astype(float)

    # Compatibility / explicit semantic field.
    roads_metric["distance_to_event_m"] = (
        roads_metric["distance_m"]
    )

    # Explicit persisted search-radius flag.
    roads_metric["within_search_radius"] = (
        roads_metric["distance_m"] <= SEARCH_RADIUS_M
    )

    # Keep the previous project terminology too.
    roads_metric["spatial_candidate"] = (
        roads_metric["within_search_radius"]
    )

    # Sort nearest first.
    roads_metric = roads_metric.sort_values(
        by="distance_m",
        ascending=True,
    )

    # Retain only closest N candidates.
    roads_metric = roads_metric.head(
        MAX_CANDIDATES
    ).copy()

    return roads_metric


# ---------------------------------------------------------------------
# ROAD NAME MATCH
# ---------------------------------------------------------------------

def add_source_road_name_match(
    roads: gpd.GeoDataFrame,
    event: dict[str, Any],
) -> gpd.GeoDataFrame:

    nearby_road = safe_str(
        event.get("nearby_road")
    )

    if not nearby_road:
        roads["source_road_name_match"] = False
        return roads

    # Normalize road names.
    source_name = (
        nearby_road
        .lower()
        .replace("-", " ")
        .replace("_", " ")
    )

    def name_match(value: Any) -> bool:
        name = safe_str(value)

        if not name:
            return False

        normalized = (
            name
            .lower()
            .replace("-", " ")
            .replace("_", " ")
        )

        # Conservative matching.
        #
        # We do NOT use fuzzy matching here because a false
        # positive could become a false training label.
        return (
            source_name in normalized
            or normalized in source_name
        )

    roads["source_road_name_match"] = (
        roads["name"]
        .apply(name_match)
    )

    return roads


# ---------------------------------------------------------------------
# STATUS FIELDS
# ---------------------------------------------------------------------

def add_mapping_status(
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:

    # These are candidates only.
    roads["mapping_status"] = "SPATIAL_CANDIDATE"

    # NEVER automatically confirm affected roads.
    roads["confirmed_affected"] = False

    # NEVER automatically create training labels.
    roads["label_ready_for_training"] = False

    return roads


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------

def save_outputs(
    mapping: gpd.GeoDataFrame,
    event: dict[str, Any],
    source_sha256: str,
) -> None:

    print("Saving road mapping...")
    print("-" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # IMPORTANT CONTRACT FIELDS
    # -----------------------------------------------------------------
    #
    # The review script expects:
    #
    #   distance_m
    #   within_search_radius
    #
    # We explicitly guarantee them here.
    # -----------------------------------------------------------------

    required_output_columns = [
        "distance_m",
        "distance_to_event_m",
        "within_search_radius",
        "spatial_candidate",
        "source_road_name_match",
        "mapping_status",
        "confirmed_affected",
        "label_ready_for_training",
        "geometry",
    ]

    require_columns(
        mapping,
        required_output_columns,
        "Final mapping output",
    )

    # -----------------------------------------------------------------
    # PARQUET
    # -----------------------------------------------------------------

    mapping.to_parquet(
        MAPPING_PARQUET,
        index=False,
    )

    # -----------------------------------------------------------------
    # JSON RECORDS
    # -----------------------------------------------------------------

    json_records = []

    for _, row in mapping.iterrows():

        record: dict[str, Any] = {}

        for column in mapping.columns:

            if column == "geometry":
                continue

            value = row[column]

            if pd.isna(value):
                value = None
            elif isinstance(
                value,
                (bool, int, str),
            ):
                pass
            elif isinstance(
                value,
                (float,),
            ):
                value = float(value)
            else:
                try:
                    value = value.item()
                except Exception:
                    value = str(value)

            record[column] = value

        json_records.append(record)

    metadata = {
        "generated_at": utc_now(),
        "event_id": event["event_id"],
        "event_date": event["event_date"],
        "event_latitude": float(event["latitude"]),
        "event_longitude": float(event["longitude"]),
        "source_event_sha256": source_sha256,
        "source_road_file": str(
            ROAD_PATH.relative_to(PROJECT_ROOT)
        ),
        "source_road_crs": SOURCE_CRS,
        "metric_crs": METRIC_CRS,
        "search_radius_m": SEARCH_RADIUS_M,
        "max_candidates": MAX_CANDIDATES,
        "candidate_count": len(mapping),
        "within_search_radius_count": int(
            mapping["within_search_radius"].sum()
        ),
        "source_road_name_match_count": int(
            mapping["source_road_name_match"].sum()
        ),
        "confirmed_affected_count": int(
            mapping["confirmed_affected"].sum()
        ),
        "training_ready_label_count": int(
            mapping["label_ready_for_training"].sum()
        ),
        "mapping_policy": {
            "automatic_affected_labels": False,
            "automatic_training_labels": False,
            "manual_source_confirmation_required": True,
        },
        "records": json_records,
    }

    with MAPPING_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------------------
    # QA
    # -----------------------------------------------------------------

    qa = {
        "generated_at": utc_now(),
        "event_id": event["event_id"],
        "event_date": event["event_date"],
        "event_coordinates": {
            "latitude": float(event["latitude"]),
            "longitude": float(event["longitude"]),
            "crs": SOURCE_CRS,
        },
        "mapping": {
            "total_candidates": len(mapping),
            "within_search_radius": int(
                mapping["within_search_radius"].sum()
            ),
            "outside_search_radius": int(
                (~mapping["within_search_radius"]).sum()
            ),
            "road_name_matches": int(
                mapping["source_road_name_match"].sum()
            ),
            "confirmed_affected": int(
                mapping["confirmed_affected"].sum()
            ),
            "training_ready_labels": int(
                mapping["label_ready_for_training"].sum()
            ),
        },
        "required_contract_columns": {
            "distance_m": "PASS",
            "distance_to_event_m": "PASS",
            "within_search_radius": "PASS",
            "spatial_candidate": "PASS",
            "source_road_name_match": "PASS",
            "mapping_status": "PASS",
            "confirmed_affected": "PASS",
            "label_ready_for_training": "PASS",
        },
        "safety_checks": {
            "automatic_affected_labels": False,
            "automatic_training_labels": False,
            "synthetic_values_added": False,
            "manual_source_confirmation_required": True,
        },
        "status": "PASS",
    }

    with MAPPING_QA_JSON.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            qa,
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("NER-Nav — Hazard Event → Road Segment Mapping")
    print("=" * 70)
    print()

    # ---------------------------------------------------------------
    # EVENT
    # ---------------------------------------------------------------

    event = load_event()

    # ---------------------------------------------------------------
    # SOURCE HASH
    # ---------------------------------------------------------------

    source_sha256 = sha256_file(EVENT_PATH)

    # ---------------------------------------------------------------
    # ROADS
    # ---------------------------------------------------------------

    roads = load_roads()

    validate_geometry(roads)

    # ---------------------------------------------------------------
    # STATE FILTER
    # ---------------------------------------------------------------

    roads_state = filter_state(
        roads,
        event.get("state", ""),
    )

    # ---------------------------------------------------------------
    # DISTANCES
    # ---------------------------------------------------------------

    mapping_metric = calculate_distances(
        roads_state,
        float(event["latitude"]),
        float(event["longitude"]),
    )

    # ---------------------------------------------------------------
    # CONVERT BACK TO WGS84
    # ---------------------------------------------------------------

    mapping = mapping_metric.to_crs(
        SOURCE_CRS
    )

    # ---------------------------------------------------------------
    # ROAD NAME MATCH
    # ---------------------------------------------------------------

    mapping = add_source_road_name_match(
        mapping,
        event,
    )

    # ---------------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------------

    mapping = add_mapping_status(
        mapping
    )

    # ---------------------------------------------------------------
    # SORT
    # ---------------------------------------------------------------

    mapping = mapping.sort_values(
        by="distance_m",
        ascending=True,
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------------

    save_outputs(
        mapping,
        event,
        source_sha256,
    )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------

    candidate_count = len(mapping)

    within_count = int(
        mapping["within_search_radius"].sum()
    )

    name_match_count = int(
        mapping["source_road_name_match"].sum()
    )

    confirmed_count = int(
        mapping["confirmed_affected"].sum()
    )

    label_ready_count = int(
        mapping["label_ready_for_training"].sum()
    )

    print()
    print("=" * 70)
    print("HAZARD → ROAD MAPPING COMPLETE")
    print("=" * 70)
    print()

    print(f"Candidate roads:          {candidate_count}")
    print(f"Within search radius:     {within_count}")
    print(f"Road-name matches:        {name_match_count}")
    print(f"Confirmed affected:       {confirmed_count}")
    print(f"Training-ready labels:    {label_ready_count}")
    print()

    if candidate_count > 0:

        nearest = mapping.iloc[0]

        print("Nearest road:")
        print(
            f"  OSM ID:       "
            f"{safe_str(nearest.get('osm_id'))}"
        )
        print(
            f"  Name:         "
            f"{safe_str(nearest.get('name')) or 'nan'}"
        )
        print(
            f"  Highway:      "
            f"{safe_str(nearest.get('highway'))}"
        )
        print(
            f"  District:     "
            f"{safe_str(nearest.get('district'))}"
        )
        print(
            f"  Distance:     "
            f"{float(nearest['distance_m']):.2f} m"
        )

    print()
    print(f"Mapping: {MAPPING_PARQUET}")
    print(f"Details: {MAPPING_JSON}")
    print(f"QA:      {MAPPING_QA_JSON}")
    print()

    print("IMPORTANT:")
    print(
        "No road segment has been declared affected automatically."
    )
    print(
        "No training label has been created."
    )
    print(
        "Manual/source-supported spatial confirmation is still required."
    )
    print()

    print("STATUS: PASS")


if __name__ == "__main__":
    main()