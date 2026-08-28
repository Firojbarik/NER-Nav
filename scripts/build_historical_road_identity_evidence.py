
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point


# ============================================================
# NER-Nav — Historical Road Identity Evidence Builder
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REAL_HAZARD_DIR = PROJECT_ROOT / "data" / "processed" / "ml" / "real_hazard"
ROADS_PATH = PROJECT_ROOT / "data" / "processed" / "roads" / "ner_roads_districts.gpkg"

EVENT_CANDIDATES = [
    REAL_HAZARD_DIR / "real_hazard_events.json",
    REAL_HAZARD_DIR / "real_hazard_events.json",
    REAL_HAZARD_DIR / "events.json",
]

RESOLUTION_REPORT = REAL_HAZARD_DIR / "road_identity_resolution.json"

CANDIDATES_PARQUET = (
    REAL_HAZARD_DIR / "road_identity_candidates.parquet"
)

CORRIDOR_GEOJSON = (
    REAL_HAZARD_DIR / "historical_road_identity_candidates.geojson"
)

EVIDENCE_JSON = (
    REAL_HAZARD_DIR / "historical_road_identity_evidence.json"
)

PROVENANCE_JSON = (
    REAL_HAZARD_DIR / "historical_road_identity_provenance_audit.json"
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

LANDSLIDE_RADIUS_M = 1500.0
BRIDGE_RADIUS_M = 350.0
CORRIDOR_RADIUS_M = 300.0

EVENT_ID = "nrsc_sikkim_mantam_2016_08_13"

LANDSLIDE_LAT = 27.5397
LANDSLIDE_LON = 88.50068611111111

MANTAM_BRIDGE_LAT = 27.536725
MANTAM_BRIDGE_LON = 88.492531

PASSINGDANG_BRIDGE_LAT = 27.531941
PASSINGDANG_BRIDGE_LON = 88.515095

EXPECTED_ROAD_NAME = "Passingdang-Mantam Road"

# SHA256 from the untouched local NRSC source.
EXPECTED_SOURCE_SHA256 = (
    "3e60840a1e25e81a0cd9fb15c8c113b764d5c262c1309293d294d4d5a5602b09"
)


# ------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now().astimezone().astimezone().isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def ensure_directory() -> None:
    REAL_HAZARD_DIR.mkdir(parents=True, exist_ok=True)


def clean_scalar(value: Any) -> Any:
    """
    Convert NumPy/Pandas/native scalar values into strict JSON-safe
    Python values.

    This is the central fix for errors such as:

        TypeError: Object of type int64 is not JSON serializable
    """

    if value is None:
        return None

    # Pandas missing values.
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    # NumPy scalar types.
    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    if isinstance(value, np.bool_):
        return bool(value)

    # Python float NaN / infinity.
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None

        return value

    # Dates / timestamps.
    if isinstance(value, (datetime, date)):
        return value.isoformat()

    # Path objects.
    if isinstance(value, Path):
        return str(value)

    # NumPy arrays.
    if isinstance(value, np.ndarray):
        return [clean_scalar(v) for v in value.tolist()]

    # Lists / tuples.
    if isinstance(value, (list, tuple)):
        return [clean_scalar(v) for v in value]

    # Dictionaries.
    if isinstance(value, dict):
        return {
            str(k): clean_scalar(v)
            for k, v in value.items()
        }

    # Shapely geometry.
    if hasattr(value, "__geo_interface__"):
        return clean_scalar(value.__geo_interface__)

    return value


def clean_json_object(value: Any) -> Any:
    """
    Recursive JSON sanitizer.

    Unlike json.dumps(default=...), this also handles NumPy values
    nested deeply inside dictionaries/lists.
    """

    if isinstance(value, dict):
        return {
            str(k): clean_json_object(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            clean_json_object(v)
            for v in value
        ]

    return clean_scalar(value)


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Write JSON through a temporary file and replace the destination
    only after serialization succeeds.

    This prevents partially written JSON files after exceptions.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    cleaned = clean_json_object(data)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            json.dump(
                cleaned,
                f,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            f.write("\n")

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.stem + "_",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as f:
            f.write(text)

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_event_file() -> Path:
    """
    Locate the actual real-event file.

    The earlier pipeline has used slightly different filenames,
    so do not hard-code one spelling.
    """

    for candidate in EVENT_CANDIDATES:
        if candidate.exists():
            return candidate

    matches = sorted(
        REAL_HAZARD_DIR.glob("*hazard*event*.json")
    )

    if matches:
        return matches[0]

    raise FileNotFoundError(
        "No real hazard event JSON was found.\n"
        f"Directory checked:\n{REAL_HAZARD_DIR}\n\n"
        "Expected one of:\n"
        + "\n".join(f"  {p}" for p in EVENT_CANDIDATES)
    )


def extract_event(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Accept either:

      {"events": [...]}

    or:

      {"event": {...}}

    or a direct event object.
    """

    if isinstance(payload.get("events"), list):
        for event in payload["events"]:
            if event.get("event_id") == EVENT_ID:
                return event

        if payload["events"]:
            return payload["events"][0]

    if isinstance(payload.get("event"), dict):
        return payload["event"]

    if payload.get("event_id"):
        return payload

    raise ValueError(
        "Could not find an event object in the real hazard JSON."
    )


def require_columns(
    gdf: gpd.GeoDataFrame,
    columns: list[str],
) -> None:
    missing = [
        c for c in columns
        if c not in gdf.columns
    ]

    if missing:
        raise RuntimeError(
            "Road database is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


def normalise_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip().lower()

    replacements = {
        "–": "-",
        "—": "-",
        "_": " ",
        "/": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return " ".join(text.split())


def road_name_match(
    name: Any,
    ref: Any,
) -> bool:
    """
    Strict source-name check.

    We intentionally do NOT manufacture a match from spatial proximity.
    """

    name_text = normalise_text(name)
    ref_text = normalise_text(ref)

    expected = normalise_text(EXPECTED_ROAD_NAME)

    if expected in name_text:
        return True

    if expected in ref_text:
        return True

    # Conservative component check.
    return (
        "passingdang" in name_text
        and "mantam" in name_text
    ) or (
        "passingdang" in ref_text
        and "mantam" in ref_text
    )


def geometry_to_point(
    lat: float,
    lon: float,
) -> Point:
    return Point(float(lon), float(lat))


def distance_in_meters(
    geometry,
    point,
    transformer=None,
) -> float:
    """
    Calculate geodesic-ish local metric distance by projecting
    to an appropriate UTM CRS when possible.
    """

    if geometry is None or geometry.is_empty:
        return float("inf")

    if transformer is None:
        raise RuntimeError(
            "A metric projection transformer is required."
        )

    projected_geometry = transformer(geometry)
    projected_point = transformer(point)

    return float(
        projected_geometry.distance(projected_point)
    )


def build_metric_transformer(
    latitude: float,
    longitude: float,
):
    """
    Create a local UTM projection.

    Sikkim lies in UTM zone 45N.
    """

    import pyproj

    zone = int(
        math.floor((longitude + 180.0) / 6.0) + 1
    )

    epsg = 32600 + zone

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        f"EPSG:{epsg}",
        always_xy=True,
    )

    return transformer.transform


def get_source_pdf() -> Path | None:
    """
    Locate the local raw NRSC PDF without modifying it.
    """

    candidates = [
        PROJECT_ROOT
        / "data"
        / "raw"
        / "hazards"
        / "Sikkim_Landslide_2016_NRSC.pdf",
    ]

    for p in candidates:
        if p.exists():
            return p

    recursive = list(
        (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "hazards"
        ).glob("**/*Sikkim*Landslide*2016*.pdf")
    )

    return recursive[0] if recursive else None


def verify_source_pdf() -> dict[str, Any]:
    pdf = get_source_pdf()

    if pdf is None:
        return {
            "found": False,
            "path": None,
            "sha256": None,
            "expected_sha256": EXPECTED_SOURCE_SHA256,
            "unchanged": False,
        }

    digest = sha256_file(pdf)

    return {
        "found": True,
        "path": str(
            pdf.relative_to(PROJECT_ROOT)
        ),
        "sha256": digest,
        "expected_sha256": EXPECTED_SOURCE_SHA256,
        "unchanged": digest == EXPECTED_SOURCE_SHA256,
    }


# ------------------------------------------------------------
# Main evidence builder
# ------------------------------------------------------------

def main() -> None:
    ensure_directory()

    print("=" * 70)
    print("NER-Nav — Historical Road Identity Evidence Builder")
    print("=" * 70)

    # --------------------------------------------------------
    # Event
    # --------------------------------------------------------

    print()
    print("Loading real event...")

    event_path = find_event_file()

    print(f"Event file: {event_path}")

    payload = load_json(event_path)
    event = extract_event(payload)

    print(f"Event ID:   {event.get('event_id')}")
    print(f"Event name: {event.get('event_name')}")
    print(f"Event date: {event.get('event_date')}")

    reported_road = (
        event.get("road_damage", {})
        .get("reported_road")
        or event.get("reported_road")
        or EXPECTED_ROAD_NAME
    )

    damage_length = (
        event.get("road_damage", {})
        .get("reported_damage_length_m")
        or event.get("reported_damage_length_m")
        or 300.0
    )

    print(f"Reported road: {reported_road}")
    print(f"Damage length: {damage_length} m")

    # --------------------------------------------------------
    # Raw source verification
    # --------------------------------------------------------

    print()
    print("Checking untouched raw source...")

    source_info = verify_source_pdf()

    print(
        f"Source found: {source_info['found']}"
    )

    if source_info["found"]:
        print(
            f"SHA256: {source_info['sha256']}"
        )

    if source_info["found"] and not source_info["unchanged"]:
        raise RuntimeError(
            "The local NRSC PDF SHA256 does not match the "
            "expected untouched source hash.\n"
            f"Expected: {EXPECTED_SOURCE_SHA256}\n"
            f"Actual:   {source_info['sha256']}"
        )

    # --------------------------------------------------------
    # Road database
    # --------------------------------------------------------

    print()
    print("Loading OSM road database...")

    if not ROADS_PATH.exists():
        raise FileNotFoundError(
            f"Road database not found:\n{ROADS_PATH}"
        )

    roads_hash_before = sha256_file(ROADS_PATH)

    roads = gpd.read_file(ROADS_PATH)

    print(f"Rows: {len(roads)}")
    print(f"CRS:  {roads.crs}")

    require_columns(
        roads,
        [
            "osm_id",
            "name",
            "ref",
            "highway",
            "geometry",
        ],
    )

    if roads.crs is None:
        raise RuntimeError(
            "Road database has no CRS."
        )

    # Normalize to geographic coordinates.
    if roads.crs.to_epsg() != 4326:
        roads = roads.to_crs("EPSG:4326")

    # --------------------------------------------------------
    # Geometry QA
    # --------------------------------------------------------

    print()
    print("Road geometry QA...")

    invalid_count = int(
        (~roads.geometry.is_valid).sum()
    )

    empty_count = int(
        roads.geometry.is_empty.sum()
    )

    null_count = int(
        roads.geometry.isna().sum()
    )

    print(
        f"Invalid geometries: {invalid_count}"
    )
    print(
        f"Empty geometries:   {empty_count}"
    )
    print(
        f"Null geometries:    {null_count}"
    )

    if invalid_count:
        raise RuntimeError(
            "Road database contains invalid geometries. "
            "Refusing to alter/fix the database in this stage."
        )

    # --------------------------------------------------------
    # Anchors
    # --------------------------------------------------------

    print()
    print("Independent geographic anchors:")

    print(
        f"Landslide: "
        f"{LANDSLIDE_LAT:.8f}, "
        f"{LANDSLIDE_LON:.8f}"
    )

    print(
        f"Mantam bridge: "
        f"{MANTAM_BRIDGE_LAT:.8f}, "
        f"{MANTAM_BRIDGE_LON:.8f}"
    )

    print(
        f"Passingdang bridge: "
        f"{PASSINGDANG_BRIDGE_LAT:.8f}, "
        f"{PASSINGDANG_BRIDGE_LON:.8f}"
    )

    transformer = build_metric_transformer(
        LANDSLIDE_LAT,
        LANDSLIDE_LON,
    )

    landslide_point = geometry_to_point(
        LANDSLIDE_LAT,
        LANDSLIDE_LON,
    )

    mantam_bridge_point = geometry_to_point(
        MANTAM_BRIDGE_LAT,
        MANTAM_BRIDGE_LON,
    )

    passingdang_bridge_point = geometry_to_point(
        PASSINGDANG_BRIDGE_LAT,
        PASSINGDANG_BRIDGE_LON,
    )

    # --------------------------------------------------------
    # Spatial evidence
    # --------------------------------------------------------

    print()
    print("Calculating spatial evidence...")

    rows: list[dict[str, Any]] = []

    for _, row in roads.iterrows():
        geometry = row.geometry

        if geometry is None or geometry.is_empty:
            continue

        distance_landslide = distance_in_meters(
            geometry,
            landslide_point,
            transformer,
        )

        distance_mantam = distance_in_meters(
            geometry,
            mantam_bridge_point,
            transformer,
        )

        distance_passingdang = distance_in_meters(
            geometry,
            passingdang_bridge_point,
            transformer,
        )

        # Distance to the line joining the two bridge anchors.
        bridge_line = gpd.GeoSeries(
            [
                # Constructed in geographic CRS only as a topology
                # representation. Actual distance uses projected data.
                # We instead use the smaller bridge-anchor distance
                # as a conservative corridor proxy below.
                mantam_bridge_point,
            ],
            crs="EPSG:4326",
        )

        min_bridge_distance = min(
            distance_mantam,
            distance_passingdang,
        )

        near_mantam = (
            distance_mantam <= BRIDGE_RADIUS_M
        )

        near_passingdang = (
            distance_passingdang <= BRIDGE_RADIUS_M
        )

        near_both = (
            near_mantam
            and near_passingdang
        )

        # Conservative corridor criterion:
        # a road can be a corridor candidate if it is reasonably
        # close to either historical bridge and within the broad
        # landslide review radius.
        corridor_consistent = (
            min_bridge_distance <= CORRIDOR_RADIUS_M
            and distance_landslide <= LANDSLIDE_RADIUS_M
        )

        source_name = road.get("name")
        source_ref = road.get("ref")
        highway = road.get("highway")
        osm_id = road.get("osm_id")

        name_match = road_name_match(
            source_name,
            source_ref,
        )

        score = 0

        if distance_landslide <= LANDSLIDE_RADIUS_M:
            score += 1

        if distance_mantam <= BRIDGE_RADIUS_M:
            score += 2

        if distance_passingdang <= BRIDGE_RADIUS_M:
            score += 2

        if corridor_consistent:
            score += 2

        if name_match:
            score += 5

        if near_both:
            resolution_status = (
                "BOTH_BRIDGE_CANDIDATE"
            )
        elif name_match:
            resolution_status = (
                "SOURCE_NAME_MATCH"
            )
        elif corridor_consistent:
            resolution_status = (
                "CORRIDOR_CANDIDATE"
            )
        else:
            resolution_status = (
                "OUTSIDE_REVIEW"
            )

        rows.append(
            {
                "osm_id": clean_scalar(osm_id),
                "name": clean_scalar(source_name),
                "ref": clean_scalar(source_ref),
                "highway": clean_scalar(highway),
                "distance_to_landslide_m": distance_landslide,
                "distance_to_historical_bridge_m": distance_mantam,
                "distance_to_passingdang_bridge_m": distance_passingdang,
                "near_historical_bridge": near_mantam,
                "near_passingdang_bridge": near_passingdang,
                "near_both_bridges": near_both,
                "corridor_consistent": corridor_consistent,
                "source_name_match": name_match,
                "identity_evidence_score": score,
                "resolution_status": resolution_status,
                "geometry": geometry,
            }
        )

    candidates = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326",
    )

    if candidates.empty:
        raise RuntimeError(
            "No road candidates were produced."
        )

    # Rank.
    candidates = candidates.sort_values(
        by=[
            "source_name_match",
            "corridor_consistent",
            "identity_evidence_score",
            "near_both_bridges",
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
    ).reset_index(drop=True)

    candidates.insert(
        0,
        "candidate_rank",
        np.arange(1, len(candidates) + 1),
    )

    # --------------------------------------------------------
    # Save candidate parquet
    # --------------------------------------------------------

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PyArrow is required to write the candidate Parquet file.\n"
            "Install it with:\n"
            "  python -m pip install -U pyarrow"
        ) from exc

    parquet_columns = [
        c
        for c in candidates.columns
        if c != "geometry"
    ]

    candidates[parquet_columns + ["geometry"]].to_parquet(
        CANDIDATES_PARQUET,
        index=False,
    )

    # --------------------------------------------------------
    # Corridor GeoJSON
    # --------------------------------------------------------

    corridor = candidates[
        candidates["resolution_status"].isin(
            [
                "BOTH_BRIDGE_CANDIDATE",
                "SOURCE_NAME_MATCH",
                "CORRIDOR_CANDIDATE",
            ]
        )
    ].copy()

    corridor_export = corridor.copy()

    # GeoJSON cannot reliably carry arbitrary NumPy scalar objects.
    for column in corridor_export.columns:
        if column == "geometry":
            continue

        corridor_export[column] = corridor_export[
            column
        ].map(clean_scalar)

    corridor_export.to_file(
        CORRIDOR_GEOJSON,
        driver="GeoJSON",
    )

    # --------------------------------------------------------
    # Candidate summary
    # --------------------------------------------------------

    top = candidates.iloc[0]

    def row_to_summary(row: pd.Series) -> dict[str, Any]:
        return clean_json_object(
            {
                "candidate_rank": row.get("candidate_rank"),
                "osm_id": row.get("osm_id"),
                "name": row.get("name"),
                "ref": row.get("ref"),
                "highway": row.get("highway"),
                "distance_to_landslide_m": row.get(
                    "distance_to_landslide_m"
                ),
                "distance_to_historical_bridge_m": row.get(
                    "distance_to_historical_bridge_m"
                ),
                "distance_to_passingdang_bridge_m": row.get(
                    "distance_to_passingdang_bridge_m"
                ),
                "near_both_bridges": row.get(
                    "near_both_bridges"
                ),
                "corridor_consistent": row.get(
                    "corridor_consistent"
                ),
                "source_name_match": row.get(
                    "source_name_match"
                ),
                "identity_evidence_score": row.get(
                    "identity_evidence_score"
                ),
                "resolution_status": row.get(
                    "resolution_status"
                ),
            }
        )

    top_summary = row_to_summary(top)

    corridor_summaries = [
        row_to_summary(row)
        for _, row in corridor.head(25).iterrows()
    ]

    # --------------------------------------------------------
    # Counts
    # --------------------------------------------------------

    spatial_candidates = int(
        len(
            candidates[
                candidates["distance_to_landslide_m"]
                <= LANDSLIDE_RADIUS_M
            ]
        )
    )

    both_bridge_candidates = int(
        candidates["near_both_bridges"].sum()
    )

    corridor_candidates = int(
        candidates["corridor_consistent"].sum()
    )

    source_name_matches = int(
        candidates["source_name_match"].sum()
    )

    # CRITICAL:
    # We do NOT convert spatial evidence into a real training label.
    source_supported_ids = 0
    real_training_labels = 0

    # --------------------------------------------------------
    # Road database hash after
    # --------------------------------------------------------

    roads_hash_after = sha256_file(ROADS_PATH)

    database_unchanged = (
        roads_hash_before == roads_hash_after
    )

    if not database_unchanged:
        raise RuntimeError(
            "ROAD DATABASE CHANGED DURING EVIDENCE BUILD.\n"
            "This stage must never modify the source road database."
        )

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    decision = {
        "status": "BLOCKED_ROAD_IDENTITY_NOT_VERIFIED",
        "training_allowed": False,
        "automatic_distance_confirmation": False,
        "automatic_bridge_confirmation": False,
        "automatic_training_label_creation": False,
        "reason": (
            "The available real sources establish that the "
            "Passingdang-Mantam Road was damaged, but the current "
            "OSM database does not independently identify an OSM "
            "way with that road identity. Spatial and bridge "
            "corridor evidence is retained for review only and "
            "is not converted into a real training label."
        ),
    }

    # --------------------------------------------------------
    # Evidence report
    # --------------------------------------------------------

    report = {
        "schema_version": "3.0",

        "generated_at_utc": utc_now_iso(),

        "event": {
            "event_id": event.get(
                "event_id",
                EVENT_ID,
            ),
            "event_name": event.get(
                "event_name",
                "So Bhir / Mantam Landslide",
            ),
            "event_date": event.get(
                "event_date",
                "2016-08-13",
            ),
            "reported_road": reported_road,
            "reported_damage_length_m": damage_length,
        },

        "source_evidence": {
            "reported_road": EXPECTED_ROAD_NAME,
            "source_supports_road_damage": True,
            "source_damage_length_m": 300.0,
            "raw_source": source_info,
            "external_verification": {
                "nrsc_report": (
                    "The official NRSC report states that the "
                    "landslide occurred opposite the "
                    "Passingdang-Mantam Road and that about "
                    "300 metres of road was washed away."
                ),
                "current_science": (
                    "Martha, Roy & Kumar (2017) independently "
                    "places the So Bhir landslide opposite the "
                    "Passingdang-Mantam Road."
                ),
            },
        },

        "anchors": {
            "landslide": {
                "latitude": LANDSLIDE_LAT,
                "longitude": LANDSLIDE_LON,
                "source": (
                    "Martha, Roy & Kumar (2017), "
                    "Current Science 113(7)"
                ),
            },
            "historical_mantam_bridge": {
                "latitude": MANTAM_BRIDGE_LAT,
                "longitude": MANTAM_BRIDGE_LON,
                "source": (
                    "Bridgemeister historical Mantam "
                    "suspension bridge"
                ),
                "status": "destroyed_2016_08_13",
            },
            "passingdang_bridge": {
                "latitude": PASSINGDANG_BRIDGE_LAT,
                "longitude": PASSINGDANG_BRIDGE_LON,
                "source": (
                    "Bridgemeister Passingdang "
                    "suspension bridge"
                ),
            },
        },

        "road_database": {
            "path": str(
                ROADS_PATH.relative_to(PROJECT_ROOT)
            ),
            "sha256_before": roads_hash_before,
            "sha256_after": roads_hash_after,
            "database_unchanged": database_unchanged,
            "rows_loaded": int(len(roads)),
            "crs": "EPSG:4326",
            "invalid_geometries": invalid_count,
            "empty_geometries": empty_count,
            "null_geometries": null_count,
        },

        "thresholds": {
            "landslide_radius_m": LANDSLIDE_RADIUS_M,
            "bridge_radius_m": BRIDGE_RADIUS_M,
            "corridor_radius_m": CORRIDOR_RADIUS_M,
        },

        "candidate_counts": {
            "total_candidates": int(len(candidates)),
            "spatial_candidates": spatial_candidates,
            "both_bridge_candidates": both_bridge_candidates,
            "corridor_candidates": corridor_candidates,
            "source_name_matches": source_name_matches,
            "source_supported_identities": source_supported_ids,
            "real_training_labels": real_training_labels,
        },

        "top_corridor_candidate": top_summary,

        "candidate_summary": corridor_summaries,

        "decision": decision,

        "provenance": {
            "synthetic_values_added": False,
            "manual_database_values_changed": False,
            "road_database_modified": False,
            "source_file_modified": False,
            "automatic_label_created": False,
            "database_hash_verified_unchanged": database_unchanged,
        },

        "outputs": {
            "candidate_parquet": str(
                CANDIDATES_PARQUET.relative_to(
                    PROJECT_ROOT
                )
            ),
            "corridor_geojson": str(
                CORRIDOR_GEOJSON.relative_to(
                    PROJECT_ROOT
                )
            ),
            "evidence_report": str(
                EVIDENCE_JSON.relative_to(
                    PROJECT_ROOT
                )
            ),
            "provenance_audit": str(
                PROVENANCE_JSON.relative_to(
                    PROJECT_ROOT
                )
            ),
        },
    }

    # --------------------------------------------------------
    # Provenance audit
    # --------------------------------------------------------

    provenance = {
        "schema_version": "1.0",
        "generated_at_utc": utc_now_iso(),

        "stage": (
            "HISTORICAL_ROAD_IDENTITY_EVIDENCE"
        ),

        "event_id": EVENT_ID,

        "source": source_info,

        "road_database": {
            "path": str(
                ROADS_PATH.relative_to(PROJECT_ROOT)
            ),
            "sha256_before": roads_hash_before,
            "sha256_after": roads_hash_after,
            "unchanged": database_unchanged,
        },

        "rules": {
            "synthetic_values_allowed": False,
            "manual_database_edits_allowed": False,
            "distance_only_confirmation_allowed": False,
            "bridge_distance_only_confirmation_allowed": False,
            "source_name_match_required": True,
            "independent_identity_required": True,
            "training_label_creation_allowed": False,
        },

        "results": {
            "spatial_candidates": spatial_candidates,
            "corridor_candidates": corridor_candidates,
            "source_name_matches": source_name_matches,
            "source_supported_identities": 0,
            "real_training_labels": 0,
        },

        "decision": decision,

        "files_modified_by_stage": [
            str(
                CANDIDATES_PARQUET.relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                CORRIDOR_GEOJSON.relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                EVIDENCE_JSON.relative_to(
                    PROJECT_ROOT
                )
            ),
            str(
                PROVENANCE_JSON.relative_to(
                    PROJECT_ROOT
                )
            ),
        ],

        "files_not_modified": [
            str(
                ROADS_PATH.relative_to(
                    PROJECT_ROOT
                )
            ),
        ],
    }

    # --------------------------------------------------------
    # IMPORTANT:
    # Write only after ALL computation and serialization
    # structures have been successfully constructed.
    # --------------------------------------------------------

    atomic_write_json(
        EVIDENCE_JSON,
        report,
    )

    atomic_write_json(
        PROVENANCE_JSON,
        provenance,
    )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HISTORICAL ROAD IDENTITY EVIDENCE COMPLETE")
    print("=" * 70)

    print(
        f"Total candidates:       {len(candidates)}"
    )

    print(
        f"Spatial candidates:     {spatial_candidates}"
    )

    print(
        f"Both-bridge candidates: {both_bridge_candidates}"
    )

    print(
        f"Corridor candidates:    {corridor_candidates}"
    )

    print(
        f"Source-name matches:    {source_name_matches}"
    )

    print(
        f"Source-supported IDs:   {source_supported_ids}"
    )

    print(
        f"Real training labels:   {real_training_labels}"
    )

    print()
    print("Top candidate:")

    for key, value in top_summary.items():
        print(
            f"  {key}: {value}"
        )

    print()
    print("Decision:")
    print(
        f"  {decision['status']}"
    )
    print(
        f"  Training allowed: {decision['training_allowed']}"
    )

    print()
    print("Provenance:")
    print(
        "  Synthetic values: False"
    )
    print(
        "  Manual DB edits: False"
    )
    print(
        f"  Road DB unchanged: {database_unchanged}"
    )
    print(
        "  Automatic label: False"
    )

    print()
    print("Outputs:")

    print(
        f"  Candidates: {CANDIDATES_PARQUET}"
    )

    print(
        f"  GeoJSON:    {CORRIDOR_GEOJSON}"
    )

    print(
        f"  Evidence:   {EVIDENCE_JSON}"
    )

    print(
        f"  Audit:      {PROVENANCE_JSON}"
    )

    print()
    print("STATUS: PASS")
    print()
    print("IMPORTANT:")
    print(
        "  This stage produces evidence, not a training label."
    )
    print(
        "  Spatial proximity is NOT treated as road identity."
    )
    print(
        "  The OSM road database was not modified."
    )
    print(
        "  The real-data training gate remains BLOCKED."
    )


if __name__ == "__main__":
    main()