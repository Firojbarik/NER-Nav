from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ============================================================
# NER-Nav — Evidence-Based Hazard → Road Confirmation
#
# IMPORTANT:
#   - No database values are modified directly.
#   - No synthetic road labels are created.
#   - No road is automatically declared affected from distance
#     alone.
#   - Every decision is written to a reproducible JSON artifact.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

EVENT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_event.json"
)

CANDIDATE_PATH = (
    ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_road_candidates.parquet"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_road_confirmation.json"
)

CONFIRMED_OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_confirmed_road.parquet"
)


# ------------------------------------------------------------
# Published evidence
# ------------------------------------------------------------

PUBLISHED_LANDSLIDE_LAT = 27.5397
PUBLISHED_LANDSLIDE_LON = 88.50068611111111

# Historical bridge coordinate reported for Mantam/Namprik.
#
# This is an external evidence anchor, NOT a synthetic event
# coordinate and NOT a database edit.
BRIDGE_LAT = 27.536725
BRIDGE_LON = 88.492531

BRIDGE_NAME = "Historical Mantam vehicular suspension bridge"

BRIDGE_SOURCE = (
    "Bridgemeister historical bridge inventory, Mantam (Namprik), "
    "Talung River; coordinates 27.536725 N, 88.492531 E; "
    "status destroyed August 13, 2016."
)

LANDSLIDE_SOURCE = (
    "Martha, Roy & Kumar (2017), Current Science 113(7), "
    "Assessment of the valley-blocking 'So Bhir' landslide near "
    "Mantam village, North Sikkim, India, using satellite images."
)

ROAD_SOURCE = (
    "NRSC/ISRO Sikkim landslide report: approximately 300 m of "
    "the Passingdang-Mantam Road was damaged."
)


SEARCH_RADIUS_M = 2000.0

# A candidate this close to the historical bridge is potentially
# relevant, but this DOES NOT by itself prove that it was the
# damaged road segment.
BRIDGE_REVIEW_RADIUS_M = 250.0

# A candidate whose geometry passes close to both the landslide
# centre and historical bridge receives stronger spatial evidence.
LANDSLIDE_REVIEW_RADIUS_M = 1000.0


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def fail(message: str) -> None:
    raise RuntimeError(message)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        fail(f"Required JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_string(value: Any) -> str:
    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def distance_m(point_a, point_b, metric_crs: str = "EPSG:32645") -> float:
    """
    Calculate metric distance between two WGS84 shapely geometries.
    """
    a = gpd.GeoSeries([point_a], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    b = gpd.GeoSeries([point_b], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    return float(a.distance(b))


def geometry_distance_to_point(
    geometry,
    point,
    metric_crs: str = "EPSG:32645",
) -> float:
    """
    Calculate distance from a road geometry to a WGS84 point.
    """
    if geometry is None or geometry.is_empty:
        return math.inf

    road = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(metric_crs).iloc[0]
    p = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(metric_crs).iloc[0]

    return float(road.distance(p))


def require_columns(gdf: gpd.GeoDataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in gdf.columns]

    if missing:
        fail(
            "Candidate mapping is missing required columns: "
            + ", ".join(missing)
        )


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("NER-Nav — Evidence-Based Hazard → Road Confirmation")
    print("=" * 70)

    # --------------------------------------------------------
    # Load event
    # --------------------------------------------------------

    print("\nLoading verified event...")
    print("-" * 70)

    event = load_json(EVENT_PATH)

    event_id = event.get("event_id")
    event_date = event.get("event_date")
    latitude = event.get("latitude")
    longitude = event.get("longitude")
    spatial_status = event.get("spatial_status")

    if latitude is None or longitude is None:
        fail("Event does not contain verified coordinates.")

    if spatial_status != "POINT_GEOMETRY_VERIFIED":
        fail(
            "Event spatial status is not POINT_GEOMETRY_VERIFIED: "
            f"{spatial_status}"
        )

    print(f"Event ID:       {event_id}")
    print(f"Event date:     {event_date}")
    print(f"Latitude:       {latitude}")
    print(f"Longitude:      {longitude}")
    print(f"Spatial status: {spatial_status}")

    event_point = Point(float(longitude), float(latitude))

    # --------------------------------------------------------
    # Validate published coordinate against our event
    # --------------------------------------------------------

    published_coordinate_distance = distance_m(
        event_point,
        Point(
            PUBLISHED_LANDSLIDE_LON,
            PUBLISHED_LANDSLIDE_LAT,
        ),
    )

    print("\nPublished landslide-coordinate verification")
    print("-" * 70)
    print(
        "Distance from event JSON to published coordinate: "
        f"{published_coordinate_distance:.3f} m"
    )

    coordinate_match = published_coordinate_distance <= 10.0

    print(
        "Coordinate match: "
        + ("PASS" if coordinate_match else "FAIL")
    )

    if not coordinate_match:
        fail(
            "Event coordinate does not match the published landslide "
            "coordinate within the allowed tolerance."
        )

    # --------------------------------------------------------
    # Load candidate roads
    # --------------------------------------------------------

    print("\nLoading road candidates...")
    print("-" * 70)

    if not CANDIDATE_PATH.exists():
        fail(f"Candidate parquet does not exist: {CANDIDATE_PATH}")

    roads = gpd.read_parquet(CANDIDATE_PATH)

    print(f"Candidate rows: {len(roads)}")
    print(f"Road CRS:      {roads.crs}")

    require_columns(
        roads,
        [
            "osm_id",
            "highway",
            "distance_m",
            "geometry",
        ],
    )

    if roads.empty:
        fail("Candidate road dataset is empty.")

    # --------------------------------------------------------
    # Normalize CRS
    # --------------------------------------------------------

    if roads.crs is None:
        fail("Candidate road dataset has no CRS.")

    roads = roads.to_crs("EPSG:4326")

    # --------------------------------------------------------
    # Geometry QA
    # --------------------------------------------------------

    invalid_count = int((~roads.geometry.is_valid).sum())
    empty_count = int(roads.geometry.is_empty.sum())
    null_count = int(roads.geometry.isna().sum())

    print("\nCandidate geometry QA")
    print("-" * 70)
    print(f"Invalid geometries: {invalid_count}")
    print(f"Empty geometries:   {empty_count}")
    print(f"Null geometries:    {null_count}")

    if invalid_count:
        fail("Candidate dataset contains invalid geometries.")

    if empty_count:
        fail("Candidate dataset contains empty geometries.")

    if null_count:
        fail("Candidate dataset contains null geometries.")

    # --------------------------------------------------------
    # Metric projection
    # --------------------------------------------------------

    metric_roads = roads.to_crs("EPSG:32645")

    event_metric = (
        gpd.GeoSeries([event_point], crs="EPSG:4326")
        .to_crs("EPSG:32645")
        .iloc[0]
    )

    bridge_point = Point(
        BRIDGE_LON,
        BRIDGE_LAT,
    )

    bridge_metric = (
        gpd.GeoSeries([bridge_point], crs="EPSG:4326")
        .to_crs("EPSG:32645")
        .iloc[0]
    )

    # --------------------------------------------------------
    # Calculate independent spatial evidence
    # --------------------------------------------------------

    print("\nCalculating spatial evidence...")
    print("-" * 70)

    metric_roads["distance_to_landslide_m"] = (
        metric_roads.geometry.distance(event_metric)
    )

    metric_roads["distance_to_historical_bridge_m"] = (
        metric_roads.geometry.distance(bridge_metric)
    )

    metric_roads["within_landslide_review_radius"] = (
        metric_roads["distance_to_landslide_m"]
        <= LANDSLIDE_REVIEW_RADIUS_M
    )

    metric_roads["near_historical_bridge"] = (
        metric_roads["distance_to_historical_bridge_m"]
        <= BRIDGE_REVIEW_RADIUS_M
    )

    # --------------------------------------------------------
    # Evidence scoring
    # --------------------------------------------------------
    #
    # This score is NOT a machine-learning prediction.
    #
    # It is an auditable ranking mechanism for human/source
    # review. A score NEVER automatically creates a label.
    # --------------------------------------------------------

    def evidence_score(row) -> int:

        score = 0

        highway = clean_string(row.get("highway")).lower()

        # Real road classes only. Paths/footways/tracks should not
        # receive the road-network evidence bonus.
        if highway in {
            "motorway",
            "trunk",
            "primary",
            "secondary",
            "tertiary",
            "unclassified",
            "residential",
            "service",
        }:
            score += 1

        # Spatial proximity to the published landslide.
        if row["distance_to_landslide_m"] <= 1000:
            score += 1

        # Spatial proximity to historical bridge.
        if row["distance_to_historical_bridge_m"] <= BRIDGE_REVIEW_RADIUS_M:
            score += 2

        # Very close to historical bridge.
        if row["distance_to_historical_bridge_m"] <= 100:
            score += 1

        return score

    metric_roads["evidence_score"] = metric_roads.apply(
        evidence_score,
        axis=1,
    )

    # --------------------------------------------------------
    # Back to WGS84
    # --------------------------------------------------------

    roads = metric_roads.to_crs("EPSG:4326")

    # --------------------------------------------------------
    # Rank candidates
    # --------------------------------------------------------

    roads = roads.sort_values(
        by=[
            "evidence_score",
            "distance_to_historical_bridge_m",
            "distance_to_landslide_m",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    roads["candidate_rank"] = range(1, len(roads) + 1)

    # --------------------------------------------------------
    # Explicit automatic-confirmation policy
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Even if a road is extremely close to the historical bridge,
    # we do not automatically create a training label.
    #
    # A source-supported road identity is still required.
    # --------------------------------------------------------

    roads["automatic_confirmation_allowed"] = False

    roads["confirmed_affected"] = False

    roads["label_ready_for_training"] = False

    roads["confirmation_status"] = "MANUAL_SOURCE_REVIEW_REQUIRED"

    roads["evidence_note"] = (
        "Spatial evidence ranked from real event geometry and "
        "historical bridge coordinates. Road identity is not "
        "automatically inferred from distance."
    )

    # --------------------------------------------------------
    # Candidate statistics
    # --------------------------------------------------------

    near_bridge = roads[
        roads["distance_to_historical_bridge_m"]
        <= BRIDGE_REVIEW_RADIUS_M
    ].copy()

    near_landslide = roads[
        roads["distance_to_landslide_m"]
        <= LANDSLIDE_REVIEW_RADIUS_M
    ].copy()

    road_classes = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
        "unclassified",
        "residential",
        "service",
    }

    plausible_roads = roads[
        roads["highway"]
        .astype(str)
        .str.lower()
        .isin(road_classes)
    ].copy()

    # --------------------------------------------------------
    # Top candidate
    # --------------------------------------------------------

    top = roads.iloc[0]

    top_osm_id = clean_string(top["osm_id"])
    top_name = clean_string(top.get("name"))
    top_highway = clean_string(top.get("highway"))
    top_district = clean_string(top.get("district"))

    top_landslide_distance = float(
        top["distance_to_landslide_m"]
    )

    top_bridge_distance = float(
        top["distance_to_historical_bridge_m"]
    )

    top_score = int(top["evidence_score"])

    # --------------------------------------------------------
    # Build reproducible decision
    # --------------------------------------------------------

    decision = {
        "status": "REVIEW_REQUIRED",
        "confirmed_affected": False,
        "training_label_created": False,
        "automatic_confirmation_allowed": False,
        "reason": (
            "The source confirms road damage on the "
            "Passingdang-Mantam Road, but the available OSM "
            "candidate dataset does not independently establish "
            "the damaged road segment identity. Spatial proximity "
            "and historical bridge proximity are used only for "
            "candidate ranking."
        ),
    }

    # --------------------------------------------------------
    # Provenance
    # --------------------------------------------------------

    provenance = {
        "event_source": event.get("source"),
        "event_geometry_source": event.get("geometry_source"),
        "landslide_coordinate_source": LANDSLIDE_SOURCE,
        "road_damage_source": ROAD_SOURCE,
        "historical_bridge_source": BRIDGE_SOURCE,
        "historical_bridge_coordinates": {
            "latitude": BRIDGE_LAT,
            "longitude": BRIDGE_LON,
        },
        "synthetic_values_added": False,
        "manual_database_values_changed": False,
        "training_label_created": False,
    }

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    result = {
        "event_id": event_id,
        "event_date": event_date,
        "event_coordinates": {
            "latitude": float(latitude),
            "longitude": float(longitude),
            "crs": "EPSG:4326",
        },
        "source_evidence": {
            "landslide_source": LANDSLIDE_SOURCE,
            "reported_road": "Passingdang-Mantam Road",
            "reported_road_damage_length_m": 300,
            "automatic_confirmation_allowed": False,
        },
        "historical_bridge_anchor": {
            "name": BRIDGE_NAME,
            "latitude": BRIDGE_LAT,
            "longitude": BRIDGE_LON,
            "source": BRIDGE_SOURCE,
        },
        "mapping": {
            "input_candidate_rows": int(len(roads)),
            "near_landslide_rows": int(len(near_landslide)),
            "near_historical_bridge_rows": int(len(near_bridge)),
            "plausible_road_class_rows": int(len(plausible_roads)),
            "landslide_review_radius_m": LANDSLIDE_REVIEW_RADIUS_M,
            "bridge_review_radius_m": BRIDGE_REVIEW_RADIUS_M,
        },
        "top_candidate": {
            "candidate_rank": int(top["candidate_rank"]),
            "osm_id": top_osm_id,
            "name": top_name,
            "highway": top_highway,
            "district": top_district,
            "distance_to_landslide_m": top_landslide_distance,
            "distance_to_historical_bridge_m": top_bridge_distance,
            "evidence_score": top_score,
        },
        "decision": decision,
        "provenance": provenance,
    }

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Save candidate evidence table
    # --------------------------------------------------------

    export_columns = [
        "candidate_rank",
        "osm_id",
        "name",
        "highway",
        "district",
        "state",
        "distance_m",
        "distance_to_landslide_m",
        "distance_to_historical_bridge_m",
        "within_landslide_review_radius",
        "near_historical_bridge",
        "evidence_score",
        "automatic_confirmation_allowed",
        "confirmed_affected",
        "label_ready_for_training",
        "confirmation_status",
        "evidence_note",
        "geometry",
    ]

    export_columns = [
        c for c in export_columns
        if c in roads.columns
    ]

    evidence_gdf = roads[export_columns].copy()

    evidence_gdf.to_parquet(
        CONFIRMED_OUTPUT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EVIDENCE-BASED ROAD CONFIRMATION COMPLETE")
    print("=" * 70)

    print(f"\nCandidate roads:                 {len(roads)}")
    print(f"Near landslide (<1 km):         {len(near_landslide)}")
    print(
        "Near historical bridge (<250 m): "
        f"{len(near_bridge)}"
    )
    print(f"Plausible road classes:         {len(plausible_roads)}")

    print("\nTop spatial candidate:")
    print(f"  Rank:             {top['candidate_rank']}")
    print(f"  OSM ID:           {top_osm_id}")
    print(
        f"  Name:             "
        f"{top_name if top_name else '<unnamed>'}"
    )
    print(f"  Highway:          {top_highway}")
    print(f"  District:         {top_district}")
    print(
        f"  Landslide dist:   "
        f"{top_landslide_distance:.2f} m"
    )
    print(
        f"  Bridge dist:      "
        f"{top_bridge_distance:.2f} m"
    )
    print(f"  Evidence score:   {top_score}")

    print("\nDecision:")
    print("  Confirmed affected:      NO")
    print("  Training label created:  NO")
    print("  Automatic confirmation:  NO")
    print("  Status:                  REVIEW_REQUIRED")

    print("\nProvenance safeguards:")
    print("  Synthetic values:        NO")
    print("  Manual DB edits:         NO")
    print("  Source-backed event:     YES")
    print("  Source-backed road damage: YES")

    print("\nOutputs:")
    print(f"  Confirmation: {OUTPUT_PATH}")
    print(f"  Evidence:     {CONFIRMED_OUTPUT_PATH}")

    print("\nIMPORTANT:")
    print(
        "  This script does NOT create a training label."
    )
    print(
        "  Spatial proximity is used only to rank candidates."
    )
    print(
        "  A source-supported OSM road identity is still required."
    )

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()