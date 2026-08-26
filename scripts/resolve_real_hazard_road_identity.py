from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


# ============================================================
# NER-Nav — REAL Hazard → OSM Road Identity Resolver
#
# PURPOSE
# -------
# Resolve the real Passingdang-Mantam road corridor against the
# untouched OSM road database using independent geographic
# evidence.
#
# IMPORTANT
# ---------
# This script:
#   - NEVER modifies the road database
#   - NEVER creates synthetic road attributes
#   - NEVER labels a road from distance alone
#   - NEVER changes an OSM name
#   - NEVER creates a training label unless identity evidence
#     passes the explicit gate
#
# Outputs are evidence artifacts only.
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml"
    / "real_hazard"
    / "events.json"
)

# Fallback to the currently generated event file.
if not EVENT_FILE.exists():
    EVENT_FILE = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "ml"
        / "real_hazard_events.json"
    )

ROAD_DB = (
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
    / "ml"
    / "real_hazard"
)

CANDIDATES = OUTPUT_DIR / "road_identity_candidates.parquet"
REPORT = OUTPUT_DIR / "road_identity_resolution.json"
CORRIDOR_GEOJSON = OUTPUT_DIR / "road_identity_corridor_candidates.geojson"
AUDIT = OUTPUT_DIR / "road_identity_provenance_audit.json"


# ------------------------------------------------------------
# REAL SOURCE-BACKED ANCHORS
# ------------------------------------------------------------
#
# These are NOT values added to the road database.
#
# They are external source evidence used only for spatial
# comparison.
#
# Mantam historical vehicular suspension bridge:
# 27.536725 N, 88.492531 E
#
# Passingdang suspension bridge:
# 27.531941 N, 88.515095 E
#
# Landslide centre:
# 27.5397 N, 88.50068611111111 E
#
# Sources:
#   - NRSC / ISRO
#   - Martha, Roy & Kumar (2017)
#   - Bridgemeister historical bridge inventory
#
# ------------------------------------------------------------

LANDSLIDE = {
    "latitude": 27.5397,
    "longitude": 88.50068611111111,
    "source": "Martha, Roy & Kumar (2017), Current Science 113(7)",
}

MANTAM_BRIDGE = {
    "latitude": 27.536725,
    "longitude": 88.492531,
    "source": "Bridgemeister historical Mantam vehicular suspension bridge",
    "status": "destroyed_2016_08_13",
}

PASSINGDANG_BRIDGE = {
    "latitude": 27.531941,
    "longitude": 88.515095,
    "source": "Bridgemeister Passingdang Talung River suspension bridge",
}

REPORTED_ROAD = "Passingdang-Mantam Road"
REPORTED_DAMAGE_M = 300.0


# ------------------------------------------------------------
# SEARCH PARAMETERS
# ------------------------------------------------------------

# Candidate must be reasonably close to at least one real
# geographic anchor.
ANCHOR_SEARCH_RADIUS_M = 1000.0

# Candidates close to both bridge anchors are particularly
# interesting because the two bridges constrain the corridor.
BOTH_BRIDGE_RADIUS_M = 350.0

# A road that runs close to the straight corridor between the
# two independent bridge anchors receives corridor evidence.
CORRIDOR_DISTANCE_M = 300.0

# Landslide vicinity used for ranking only.
LANDSLIDE_RADIUS_M = 1500.0

# OSM road classes considered plausible motor-road candidates.
PLAUSIBLE_HIGHWAYS = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
}


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value).strip().lower()

    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("_", " ")

    text = re.sub(r"\s+", " ", text)

    return text


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def load_event() -> dict[str, Any]:
    if not EVENT_FILE.exists():
        raise FileNotFoundError(
            f"Real hazard event file not found:\n{EVENT_FILE}"
        )

    data = json.loads(EVENT_FILE.read_text(encoding="utf-8"))

    if "events" in data:
        events = data["events"]

        if not events:
            raise RuntimeError("Event file contains no events.")

        event = events[0]
    else:
        event = data

    return event


def choose_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    return None


def projected_distance_m(
    roads: gpd.GeoDataFrame,
    point: Point,
) -> pd.Series:
    """
    Return distances in metres by projecting to a local metric CRS.
    """

    if roads.crs is None:
        raise RuntimeError("Road database has no CRS.")

    point_gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[point],
        crs="EPSG:4326",
    )

    metric_crs = roads.estimate_utm_crs()

    if metric_crs is None:
        metric_crs = "EPSG:32645"

    roads_metric = roads.to_crs(metric_crs)
    point_metric = point_gdf.to_crs(metric_crs).geometry.iloc[0]

    return roads_metric.geometry.distance(point_metric)


def metric_geometry(
    roads: gpd.GeoDataFrame,
    geometry,
):
    metric_crs = roads.estimate_utm_crs()

    if metric_crs is None:
        metric_crs = "EPSG:32645"

    return roads.to_crs(metric_crs), gpd.GeoSeries(
        [geometry],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]


def build_corridor_line(
    roads: gpd.GeoDataFrame,
) -> LineString:
    """
    Construct a straight geographic reference corridor between
    the independent Passingdang and Mantam bridge anchors.

    This is a reference geometry only.
    It is NOT inserted into the road database.
    """

    p1 = Point(
        PASSINGDANG_BRIDGE["longitude"],
        PASSINGDANG_BRIDGE["latitude"],
    )

    p2 = Point(
        MANTAM_BRIDGE["longitude"],
        MANTAM_BRIDGE["latitude"],
    )

    metric_crs = roads.estimate_utm_crs()

    if metric_crs is None:
        metric_crs = "EPSG:32645"

    line = gpd.GeoSeries(
        [LineString([p1, p2])],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]

    return line


def line_distance_to_point(
    roads_metric: gpd.GeoDataFrame,
    point: Point,
) -> pd.Series:
    metric_crs = roads_metric.crs

    point_metric = gpd.GeoSeries(
        [point],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]

    return roads_metric.geometry.distance(point_metric)


def line_distance_to_corridor(
    roads_metric: gpd.GeoDataFrame,
    corridor_metric: LineString,
) -> pd.Series:
    return roads_metric.geometry.distance(corridor_metric)


def extract_event_coordinates(event: dict[str, Any]):
    coords = event.get("coordinates", {})

    lat = coords.get("latitude")
    lon = coords.get("longitude")

    if lat is None or lon is None:
        lat = LANDSLIDE["latitude"]
        lon = LANDSLIDE["longitude"]

    return float(lat), float(lon)


def source_road_evidence(event: dict[str, Any]) -> dict[str, Any]:
    road_damage = event.get("road_damage", {})

    reported_road = (
        road_damage.get("reported_road")
        or REPORTED_ROAD
    )

    damage_length = (
        road_damage.get("reported_damage_length_m")
        or REPORTED_DAMAGE_M
    )

    return {
        "reported_road": reported_road,
        "reported_damage_length_m": float(damage_length),
        "source_supports_road_damage": safe_bool(
            road_damage.get("reported", True)
        ),
    }


def calculate_identity_score(row: pd.Series) -> int:
    """
    IMPORTANT:
    This score ranks evidence.

    It does NOT itself authorize a training label.
    """

    score = 0

    # Very strong independent geographic evidence:
    if row["near_both_bridges"]:
        score += 5

    # Strong corridor evidence:
    if row["corridor_consistent"]:
        score += 3

    # Near the historical Mantam bridge:
    if row["near_mantam_bridge"]:
        score += 2

    # Near Passingdang bridge:
    if row["near_passingdang_bridge"]:
        score += 2

    # Near published landslide:
    if row["near_landslide"]:
        score += 1

    # Motor-road class:
    if row["plausible_highway"]:
        score += 1

    # Explicit OSM naming can help, but must NEVER be invented.
    name = normalize_text(row.get("name", ""))

    if "passingdang" in name or "passindang" in name:
        score += 4

    if "mantam" in name:
        score += 4

    if "passingdang" in name and "mantam" in name:
        score += 6

    return int(score)


def source_name_match(
    name: Any,
    ref: Any,
) -> bool:
    text = " ".join(
        [
            normalize_text(name),
            normalize_text(ref),
        ]
    )

    passingdang = (
        "passingdang" in text
        or "passindang" in text
        or "passing dang" in text
    )

    mantam = "mantam" in text

    return passingdang and mantam


def corridor_identity_test(row: pd.Series) -> bool:
    """
    Conservative identity rule.

    A candidate is considered a POSSIBLE corridor identity when:
      - it is close to the independent bridge corridor, AND
      - it is a plausible motor-road class.

    This is still not sufficient for automatic training.
    """

    return bool(
        row["corridor_consistent"]
        and row["plausible_highway"]
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("NER-Nav — REAL Hazard → OSM Road Identity Resolver")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load event
    # --------------------------------------------------------

    print("\nLoading real event...")

    event = load_event()

    event_id = event.get(
        "event_id",
        "nrsc_sikkim_mantam_2016_08_13",
    )

    event_name = event.get(
        "event_name",
        "So Bhir / Mantam Landslide",
    )

    event_date = event.get(
        "event_date",
        "2016-08-13",
    )

    latitude, longitude = extract_event_coordinates(event)

    source_evidence = source_road_evidence(event)

    print(f"Event: {event_id}")
    print(f"Event name: {event_name}")
    print(f"Event date: {event_date}")
    print(f"Reported road: {source_evidence['reported_road']}")
    print(
        f"Reported damage length: "
        f"{source_evidence['reported_damage_length_m']:.0f} m"
    )

    # --------------------------------------------------------
    # Verify source file
    # --------------------------------------------------------

    raw_pdf = (
        PROJECT_ROOT
        / "data"
        / "raw"
        / "hazards"
        / "Sikkim_Landslide_2016_NRSC.pdf"
    )

    if not raw_pdf.exists():
        # Handle possible escaped/renamed project path structure.
        alternatives = list(
            (
                PROJECT_ROOT
                / "data"
                / "raw"
                / "hazards"
            ).glob("*.pdf")
        )

        if alternatives:
            raw_pdf = alternatives[0]

    if not raw_pdf.exists():
        raise FileNotFoundError(
            "NRSC raw PDF could not be found."
        )

    pdf_hash = sha256_file(raw_pdf)

    print("\nChecking untouched raw source...")
    print(f"PDF: {raw_pdf}")
    print(f"SHA256: {pdf_hash}")

    # --------------------------------------------------------
    # Load untouched OSM road DB
    # --------------------------------------------------------

    print("\nLoading OSM road database...")

    if not ROAD_DB.exists():
        raise FileNotFoundError(
            f"Road database not found:\n{ROAD_DB}"
        )

    db_hash_before = sha256_file(ROAD_DB)

    roads = gpd.read_file(ROAD_DB)

    if roads.empty:
        raise RuntimeError("OSM road database contains zero rows.")

    if roads.crs is None:
        raise RuntimeError(
            "OSM road database has no CRS."
        )

    print(f"Rows: {len(roads)}")
    print(f"CRS: {roads.crs}")

    # --------------------------------------------------------
    # Geometry QA
    # --------------------------------------------------------

    invalid_count = int(
        (~roads.geometry.is_valid).sum()
    )

    empty_count = int(
        roads.geometry.is_empty.sum()
    )

    null_count = int(
        roads.geometry.isna().sum()
    )

    print("\nRoad geometry QA...")
    print(f"Invalid geometries: {invalid_count}")
    print(f"Empty geometries:   {empty_count}")
    print(f"Null geometries:    {null_count}")

    if invalid_count:
        raise RuntimeError(
            "Road database contains invalid geometries. "
            "Refusing to repair them automatically."
        )

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    osm_col = choose_column(
        roads,
        ["osm_id", "osmid", "osm_way_id"],
    )

    highway_col = choose_column(
        roads,
        ["highway", "road_type"],
    )

    name_col = choose_column(
        roads,
        ["name", "road_name"],
    )

    ref_col = choose_column(
        roads,
        ["ref", "road_ref"],
    )

    if osm_col is None:
        raise RuntimeError(
            "No OSM ID field found in road database."
        )

    if highway_col is None:
        raise RuntimeError(
            "No highway field found in road database."
        )

    # --------------------------------------------------------
    # Preserve original database hash.
    # --------------------------------------------------------

    original_columns = list(roads.columns)

    # --------------------------------------------------------
    # Convert to local metric CRS
    # --------------------------------------------------------

    metric_crs = roads.estimate_utm_crs()

    if metric_crs is None:
        metric_crs = "EPSG:32645"

    roads_metric = roads.to_crs(metric_crs)

    # --------------------------------------------------------
    # Independent source anchors
    # --------------------------------------------------------

    landslide_point = Point(
        longitude,
        latitude,
    )

    mantam_bridge_point = Point(
        MANTAM_BRIDGE["longitude"],
        MANTAM_BRIDGE["latitude"],
    )

    passingdang_bridge_point = Point(
        PASSINGDANG_BRIDGE["longitude"],
        PASSINGDANG_BRIDGE["latitude"],
    )

    landslide_metric = gpd.GeoSeries(
        [landslide_point],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]

    mantam_bridge_metric = gpd.GeoSeries(
        [mantam_bridge_point],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]

    passingdang_bridge_metric = gpd.GeoSeries(
        [passingdang_bridge_point],
        crs="EPSG:4326",
    ).to_crs(metric_crs).iloc[0]

    # Straight-line independent corridor reference.
    corridor_metric = LineString(
        [
            (
                passingdang_bridge_metric.x,
                passingdang_bridge_metric.y,
            ),
            (
                mantam_bridge_metric.x,
                mantam_bridge_metric.y,
            ),
        ]
    )

    print("\nIndependent geographic anchors:")
    print(
        "Landslide: "
        f"{latitude:.8f}, {longitude:.8f}"
    )
    print(
        "Mantam bridge: "
        f"{MANTAM_BRIDGE['latitude']:.8f}, "
        f"{MANTAM_BRIDGE['longitude']:.8f}"
    )
    print(
        "Passingdang bridge: "
        f"{PASSINGDANG_BRIDGE['latitude']:.8f}, "
        f"{PASSINGDANG_BRIDGE['longitude']:.8f}"
    )

    # --------------------------------------------------------
    # Spatial distances
    # --------------------------------------------------------

    print("\nCalculating spatial evidence...")

    result = roads.copy()

    result["distance_to_landslide_m"] = (
        roads_metric.geometry.distance(
            landslide_metric
        )
    )

    result["distance_to_historical_bridge_m"] = (
        roads_metric.geometry.distance(
            mantam_bridge_metric
        )
    )

    result["distance_to_passingdang_bridge_m"] = (
        roads_metric.geometry.distance(
            passingdang_bridge_metric
        )
    )

    result["distance_to_bridge_corridor_m"] = (
        roads_metric.geometry.distance(
            corridor_metric
        )
    )

    result["near_landslide"] = (
        result["distance_to_landslide_m"]
        <= LANDSLIDE_RADIUS_M
    )

    result["near_mantam_bridge"] = (
        result["distance_to_historical_bridge_m"]
        <= BOTH_BRIDGE_RADIUS_M
    )

    result["near_passingdang_bridge"] = (
        result["distance_to_passingdang_bridge_m"]
        <= BOTH_BRIDGE_RADIUS_M
    )

    result["near_both_bridges"] = (
        result["near_mantam_bridge"]
        & result["near_passingdang_bridge"]
    )

    result["corridor_consistent"] = (
        result["distance_to_bridge_corridor_m"]
        <= CORRIDOR_DISTANCE_M
    )

    result["within_anchor_search_radius"] = (
        result["distance_to_landslide_m"]
        <= ANCHOR_SEARCH_RADIUS_M
    ) | (
        result["distance_to_historical_bridge_m"]
        <= ANCHOR_SEARCH_RADIUS_M
    ) | (
        result["distance_to_passingdang_bridge_m"]
        <= ANCHOR_SEARCH_RADIUS_M
    )

    result["plausible_highway"] = (
        result[highway_col]
        .astype(str)
        .str.lower()
        .isin(PLAUSIBLE_HIGHWAYS)
    )

    # --------------------------------------------------------
    # Name/ref evidence
    # --------------------------------------------------------

    result["source_name_match"] = False

    for idx in result.index:

        name_value = (
            result.at[idx, name_col]
            if name_col
            else ""
        )

        ref_value = (
            result.at[idx, ref_col]
            if ref_col
            else ""
        )

        result.at[idx, "source_name_match"] = (
            source_name_match(
                name_value,
                ref_value,
            )
        )

    # --------------------------------------------------------
    # Identity ranking
    # --------------------------------------------------------

    result["identity_evidence_score"] = result.apply(
        calculate_identity_score,
        axis=1,
    )

    result["corridor_identity_candidate"] = result.apply(
        corridor_identity_test,
        axis=1,
    )

    # --------------------------------------------------------
    # Source-supported identity gate
    # --------------------------------------------------------
    #
    # CRITICAL:
    #
    # We deliberately DO NOT turn a corridor candidate into a
    # confirmed training label automatically.
    #
    # A source-supported identity requires either:
    #
    #   A) explicit OSM source name/reference match, OR
    #
    #   B) independent corridor evidence sufficiently strong
    #      to support manual/source review.
    #
    # The output therefore distinguishes:
    #
    #   CORRIDOR_CANDIDATE
    #   SOURCE_NAME_MATCH
    #   MANUAL_SOURCE_IDENTITY_REQUIRED
    #
    # No label is created here.
    # --------------------------------------------------------

    result["resolution_status"] = "OUTSIDE_REVIEW"

    result.loc[
        result["corridor_identity_candidate"],
        "resolution_status",
    ] = "CORRIDOR_CANDIDATE"

    result.loc[
        result["source_name_match"],
        "resolution_status",
    ] = "SOURCE_NAME_MATCH_REQUIRES_REVIEW"

    # A genuinely source-supported identity is only accepted
    # when the OSM metadata itself identifies the corridor.
    #
    # We do NOT infer identity solely from proximity.
    result["source_supported_identity"] = (
        result["source_name_match"]
    )

    # --------------------------------------------------------
    # Training gate
    # --------------------------------------------------------

    result["confirmed_affected"] = False
    result["label_ready_for_training"] = False
    result["automatic_confirmation_allowed"] = False

    result["evidence_note"] = (
        "Spatial evidence ranks the real OSM candidate only. "
        "No affected-road training label created."
    )

    result.loc[
        result["source_name_match"],
        "evidence_note",
    ] = (
        "OSM name/ref contains a source-road identity match. "
        "Independent source review is still required before "
        "training-label creation."
    )

    # --------------------------------------------------------
    # Filter useful candidates
    # --------------------------------------------------------

    candidate_mask = (
        result["within_anchor_search_radius"]
        | result["corridor_identity_candidate"]
        | result["source_name_match"]
    )

    candidates = result.loc[candidate_mask].copy()

    candidates = candidates.sort_values(
        [
            "source_name_match",
            "near_both_bridges",
            "corridor_consistent",
            "identity_evidence_score",
            "distance_to_historical_bridge_m",
            "distance_to_landslide_m",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
            True,
        ],
    )

    candidates["candidate_rank"] = range(
        1,
        len(candidates) + 1,
    )

    # --------------------------------------------------------
    # Select output columns
    # --------------------------------------------------------

    preferred_columns = [
        "candidate_rank",
        osm_col,
        name_col,
        ref_col,
        highway_col,
        "distance_to_landslide_m",
        "distance_to_historical_bridge_m",
        "distance_to_passingdang_bridge_m",
        "distance_to_bridge_corridor_m",
        "near_landslide",
        "near_mantam_bridge",
        "near_passingdang_bridge",
        "near_both_bridges",
        "corridor_consistent",
        "plausible_highway",
        "source_name_match",
        "corridor_identity_candidate",
        "identity_evidence_score",
        "source_supported_identity",
        "confirmed_affected",
        "label_ready_for_training",
        "automatic_confirmation_allowed",
        "resolution_status",
        "evidence_note",
        "geometry",
    ]

    preferred_columns = [
        c
        for c in preferred_columns
        if c is not None and c in candidates.columns
    ]

    candidates = candidates[preferred_columns].copy()

    # --------------------------------------------------------
    # Write candidate Parquet
    # --------------------------------------------------------

    candidates.to_parquet(
        CANDIDATES,
        index=False,
    )

    # --------------------------------------------------------
    # GeoJSON evidence artifact
    # --------------------------------------------------------

    geo_candidates = candidates.copy()

    # GeoJSON cannot safely represent all pandas extension types.
    for col in geo_candidates.columns:
        if col == "geometry":
            continue

        if str(geo_candidates[col].dtype).startswith(
            "datetime"
        ):
            geo_candidates[col] = (
                geo_candidates[col]
                .astype(str)
            )

    geo_candidates.to_file(
        CORRIDOR_GEOJSON,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Re-hash the database AFTER reading
    # --------------------------------------------------------

    db_hash_after = sha256_file(ROAD_DB)

    database_unchanged = (
        db_hash_before == db_hash_after
    )

    if not database_unchanged:
        raise RuntimeError(
            "CRITICAL: OSM road database hash changed "
            "during resolution."
        )

    # --------------------------------------------------------
    # Verify no source road database mutation
    # --------------------------------------------------------

    columns_unchanged = (
        original_columns == list(roads.columns)
    )

    if not columns_unchanged:
        raise RuntimeError(
            "CRITICAL: road database schema changed."
        )

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    spatial_candidates = len(candidates)

    source_name_matches = int(
        candidates["source_name_match"].sum()
    )

    corridor_candidates = int(
        candidates["corridor_identity_candidate"].sum()
    )

    source_supported = int(
        candidates["source_supported_identity"].sum()
    )

    real_labels = int(
        candidates["label_ready_for_training"].sum()
    )

    both_bridge_candidates = int(
        candidates["near_both_bridges"].sum()
    )

    # --------------------------------------------------------
    # Top candidate
    # --------------------------------------------------------

    top_candidate = None

    if not candidates.empty:

        top = candidates.iloc[0]

        top_candidate = {
            "candidate_rank": int(
                top["candidate_rank"]
            ),
            "osm_id": str(
                top[osm_col]
            ),
            "name": (
                None
                if name_col is None
                else str(top[name_col])
            ),
            "ref": (
                None
                if ref_col is None
                else str(top[ref_col])
            ),
            "highway": str(
                top[highway_col]
            ),
            "distance_to_landslide_m": float(
                top["distance_to_landslide_m"]
            ),
            "distance_to_historical_bridge_m": float(
                top["distance_to_historical_bridge_m"]
            ),
            "distance_to_passingdang_bridge_m": float(
                top["distance_to_passingdang_bridge_m"]
            ),
            "distance_to_bridge_corridor_m": float(
                top["distance_to_bridge_corridor_m"]
            ),
            "identity_evidence_score": int(
                top["identity_evidence_score"]
            ),
            "source_name_match": bool(
                top["source_name_match"]
            ),
            "corridor_identity_candidate": bool(
                top["corridor_identity_candidate"]
            ),
        }

    # --------------------------------------------------------
    # FINAL TRAINING GATE
    # --------------------------------------------------------

    #
    # Deliberately strict.
    #
    # The resolver does NOT create a label merely because a
    # candidate is near both bridges.
    #
    # The OSM name/ref must independently identify the reported
    # road before this stage can produce a source-supported
    # identity.
    #

    if source_supported > 0:

        decision = (
            "SOURCE_NAME_MATCH_REQUIRES_MANUAL_CONFIRMATION"
        )

        training_allowed = False

        reason = (
            "An OSM name/ref match exists, but a source-backed "
            "road identity still requires review before an "
            "affected-road training label can be created."
        )

    else:

        decision = (
            "BLOCKED_ROAD_IDENTITY_NOT_VERIFIED"
        )

        training_allowed = False

        reason = (
            "The real source establishes damage to the "
            "Passingdang-Mantam Road, while the current OSM "
            "database does not independently identify an OSM "
            "way by that road identity. Spatial corridor "
            "evidence is retained for review but is not "
            "converted into a training label."
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = {
        "schema_version": "2.0",

        "generated_at_utc": utc_now(),

        "event": {
            "event_id": event_id,
            "event_name": event_name,
            "event_date": event_date,
            "reported_road": source_evidence[
                "reported_road"
            ],
            "reported_damage_length_m": (
                source_evidence[
                    "reported_damage_length_m"
                ]
            ),
        },

        "published_coordinates": {
            "landslide": LANDSLIDE,
            "historical_mantam_bridge": MANTAM_BRIDGE,
            "passingdang_bridge": PASSINGDANG_BRIDGE,
        },

        "source_evidence": {
            "source_supports_road_damage": (
                source_evidence[
                    "source_supports_road_damage"
                ]
            ),
            "reported_road": source_evidence[
                "reported_road"
            ],
            "reported_damage_length_m": (
                source_evidence[
                    "reported_damage_length_m"
                ]
            ),
        },

        "road_database": {
            "path": str(
                ROAD_DB.relative_to(PROJECT_ROOT)
            ),
            "sha256_before": db_hash_before,
            "sha256_after": db_hash_after,
            "database_unchanged": database_unchanged,
            "schema_unchanged": columns_unchanged,
            "rows_loaded": int(len(roads)),
            "crs": str(roads.crs),
        },

        "candidate_counts": {
            "spatial_candidates": spatial_candidates,
            "both_bridge_candidates": (
                both_bridge_candidates
            ),
            "corridor_candidates": corridor_candidates,
            "source_name_matches": source_name_matches,
            "source_supported_identities": (
                source_supported
            ),
            "real_training_labels": real_labels,
        },

        "thresholds": {
            "anchor_search_radius_m": (
                ANCHOR_SEARCH_RADIUS_M
            ),
            "both_bridge_radius_m": (
                BOTH_BRIDGE_RADIUS_M
            ),
            "corridor_distance_m": (
                CORRIDOR_DISTANCE_M
            ),
            "landslide_radius_m": (
                LANDSLIDE_RADIUS_M
            ),
        },

        "top_candidate": top_candidate,

        "decision": {
            "status": decision,
            "training_allowed": training_allowed,
            "automatic_distance_confirmation": False,
            "automatic_bridge_confirmation": False,
            "automatic_training_label_creation": False,
            "reason": reason,
        },

        "provenance": {
            "synthetic_values_added": False,
            "manual_database_values_changed": False,
            "road_database_modified": False,
            "source_file_modified": False,
            "database_hash_verified_unchanged": (
                database_unchanged
            ),
            "automatic_label_created": False,
        },

        "outputs": {
            "candidate_parquet": str(
                CANDIDATES.relative_to(PROJECT_ROOT)
            ),
            "corridor_geojson": str(
                CORRIDOR_GEOJSON.relative_to(
                    PROJECT_ROOT
                )
            ),
            "resolution_report": str(
                REPORT.relative_to(PROJECT_ROOT)
            ),
            "provenance_audit": str(
                AUDIT.relative_to(PROJECT_ROOT)
            ),
        },
    }

    write_json(
        REPORT,
        report,
    )

    # --------------------------------------------------------
    # Provenance audit
    # --------------------------------------------------------

    audit = {
        "generated_at_utc": utc_now(),

        "database": {
            "path": str(
                ROAD_DB.relative_to(PROJECT_ROOT)
            ),
            "sha256_before": db_hash_before,
            "sha256_after": db_hash_after,
            "unchanged": database_unchanged,
        },

        "raw_source": {
            "path": str(
                raw_pdf.relative_to(PROJECT_ROOT)
            ),
            "sha256": pdf_hash,
            "unchanged": True,
        },

        "policy": {
            "synthetic_values_allowed": False,
            "manual_database_edits_allowed": False,
            "distance_only_confirmation_allowed": False,
            "bridge_only_confirmation_allowed": False,
            "source_supported_identity_required": True,
            "confirmed_affected_required": True,
        },

        "result": {
            "source_name_matches": source_name_matches,
            "corridor_candidates": corridor_candidates,
            "source_supported_identities": (
                source_supported
            ),
            "real_training_labels": real_labels,
            "training_allowed": training_allowed,
        },
    }

    write_json(
        AUDIT,
        audit,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("REAL ROAD IDENTITY RESOLUTION COMPLETE")
    print("=" * 70)

    print(
        f"Spatial candidates:       {spatial_candidates}"
    )
    print(
        f"Both-bridge candidates:   {both_bridge_candidates}"
    )
    print(
        f"Corridor candidates:      {corridor_candidates}"
    )
    print(
        f"Source-name matches:      {source_name_matches}"
    )
    print(
        f"Source-supported IDs:     {source_supported}"
    )
    print(
        f"Real training labels:     {real_labels}"
    )

    if top_candidate:
        print("\nTop corridor candidate:")
        print(
            f"  Rank:                    "
            f"{top_candidate['candidate_rank']}"
        )
        print(
            f"  OSM ID:                  "
            f"{top_candidate['osm_id']}"
        )
        print(
            f"  Name:                    "
            f"{top_candidate['name']}"
        )
        print(
            f"  Ref:                     "
            f"{top_candidate['ref']}"
        )
        print(
            f"  Highway:                 "
            f"{top_candidate['highway']}"
        )
        print(
            f"  Landslide distance:      "
            f"{top_candidate['distance_to_landslide_m']:.2f} m"
        )
        print(
            f"  Mantam bridge distance:  "
            f"{top_candidate['distance_to_historical_bridge_m']:.2f} m"
        )
        print(
            f"  Passingdang distance:    "
            f"{top_candidate['distance_to_passingdang_bridge_m']:.2f} m"
        )
        print(
            f"  Corridor distance:       "
            f"{top_candidate['distance_to_bridge_corridor_m']:.2f} m"
        )
        print(
            f"  Evidence score:          "
            f"{top_candidate['identity_evidence_score']}"
        )

    print("\nDecision:")
    print(
        f"  {decision}"
    )
    print(
        f"  Training allowed:        "
        f"{training_allowed}"
    )

    print("\nProvenance:")
    print("  Synthetic values:        False")
    print("  Manual DB edits:         False")
    print(
        f"  Road DB unchanged:       "
        f"{database_unchanged}"
    )
    print("  Automatic label:         False")

    print("\nOutputs:")
    print(f"  Candidates: {CANDIDATES}")
    print(f"  GeoJSON:    {CORRIDOR_GEOJSON}")
    print(f"  Report:     {REPORT}")
    print(f"  Audit:      {AUDIT}")

    print("\nSTATUS: PASS")

    if not training_allowed:
        print(
            "\nIMPORTANT:"
        )
        print(
            "  This is a REAL-DATA identity-resolution stage."
        )
        print(
            "  No synthetic road identity was created."
        )
        print(
            "  No OSM database value was changed."
        )
        print(
            "  No training label was created."
        )
        print(
            "  Manual/source-backed identity confirmation "
            "remains required."
        )


if __name__ == "__main__":
    main()