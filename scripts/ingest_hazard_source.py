from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_HAZARD_DIR = PROJECT_ROOT / "data" / "raw" / "hazards"
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "ml" / "hazard_source_manifest.json"

SUPPORTED_EXTENSIONS = {
    ".csv",
    ".json",
    ".geojson",
    ".gpkg",
    ".parquet",
    ".feather",
}


DATE_FIELD_CANDIDATES = {
    "date",
    "event_date",
    "incident_date",
    "occurrence_date",
    "occurred_at",
    "timestamp",
    "datetime",
    "event_datetime",
    "start_date",
    "end_date",
    "year",
}

LAT_FIELD_CANDIDATES = {
    "lat",
    "latitude",
    "y",
    "y_coord",
    "lat_dd",
}

LON_FIELD_CANDIDATES = {
    "lon",
    "lng",
    "longitude",
    "x",
    "x_coord",
    "lon_dd",
}

EVENT_TYPE_CANDIDATES = {
    "event_type",
    "incident_type",
    "hazard_type",
    "hazard",
    "type",
    "landslide_type",
}

SEVERITY_CANDIDATES = {
    "severity",
    "severity_level",
    "damage_level",
    "impact",
    "magnitude",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def normalize_column_name(name: Any) -> str:
    text = str(name).strip().lower()

    replacements = {
        " ": "_",
        "-": "_",
        "/": "_",
        ".": "_",
        "(": "",
        ")": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalized_columns(columns: list[Any]) -> dict[str, str]:
    result: dict[str, str] = {}

    for column in columns:
        normalized = normalize_column_name(column)
        result[normalized] = str(column)

    return result


def find_candidate_columns(columns: list[Any]) -> dict[str, list[str]]:
    normalized = normalized_columns(columns)

    def matches(candidates: set[str]) -> list[str]:
        return [
            original
            for normalized_name, original in normalized.items()
            if normalized_name in candidates
        ]

    return {
        "date": matches(DATE_FIELD_CANDIDATES),
        "latitude": matches(LAT_FIELD_CANDIDATES),
        "longitude": matches(LON_FIELD_CANDIDATES),
        "event_type": matches(EVENT_TYPE_CANDIDATES),
        "severity": matches(SEVERITY_CANDIDATES),
    }


def inspect_tabular(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path, nrows=1000)

    elif suffix == ".json":
        df = pd.read_json(path)

    elif suffix == ".parquet":
        df = pd.read_parquet(path)

    elif suffix == ".feather":
        df = pd.read_feather(path)

    else:
        raise ValueError(f"Unsupported tabular format: {suffix}")

    candidates = find_candidate_columns(list(df.columns))

    return {
        "format": suffix.lstrip("."),
        "sample_rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "dtypes": {
            str(column): str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "candidate_fields": candidates,
        "sample_missing_values": {
            str(column): int(df[column].isna().sum())
            for column in df.columns
        },
    }


def inspect_geospatial(path: Path) -> dict[str, Any]:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "GeoPandas is required to inspect GeoJSON/GPKG sources."
        ) from exc

    suffix = path.suffix.lower()

    if suffix == ".geojson":
        gdf = gpd.read_file(path)

    elif suffix == ".gpkg":
        layers = gpd.list_layers(path)

        if len(layers) == 0:
            raise ValueError("GeoPackage contains no layers.")

        layer_name = str(layers.iloc[0]["name"])
        gdf = gpd.read_file(path, layer=layer_name)

    else:
        raise ValueError(f"Unsupported geospatial format: {suffix}")

    candidates = find_candidate_columns(
        [column for column in gdf.columns if column != gdf.geometry.name]
    )

    geometry_types = (
        gdf.geometry.geom_type.value_counts(dropna=False)
        .to_dict()
    )

    return {
        "format": suffix.lstrip("."),
        "layer": (
            str(layers.iloc[0]["name"])
            if suffix == ".gpkg"
            else None
        ),
        "rows": int(len(gdf)),
        "columns": [
            str(column)
            for column in gdf.columns
            if column != gdf.geometry.name
        ],
        "geometry_column": str(gdf.geometry.name),
        "geometry_types": {
            str(key): int(value)
            for key, value in geometry_types.items()
        },
        "crs": str(gdf.crs),
        "invalid_geometry_count": int(
            (~gdf.geometry.is_valid).sum()
        ),
        "empty_geometry_count": int(
            gdf.geometry.is_empty.sum()
        ),
        "candidate_fields": candidates,
    }


def inspect_source(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix in {".csv", ".json", ".parquet", ".feather"}:
        return inspect_tabular(path)

    if suffix in {".geojson", ".gpkg"}:
        return inspect_geospatial(path)

    raise ValueError(
        f"Unsupported source extension: {suffix}. "
        f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
    )


def build_manifest(
    source_path: Path,
    inspection: dict[str, Any],
    original_source_id: str,
    access_condition: str,
    license_condition: str,
) -> dict[str, Any]:
    return {
        "manifest_version": "1.0.0",
        "generated_at": utc_now(),
        "project": "NER-Nav",
        "source": {
            "source_id": original_source_id,
            "local_path": str(source_path.relative_to(PROJECT_ROOT)),
            "filename": source_path.name,
            "extension": source_path.suffix.lower(),
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "acquired_at": utc_now(),
            "access_condition": access_condition,
            "license_condition": license_condition,
        },
        "inspection": inspection,
        "label_readiness": {
            "geometry_available": False,
            "event_date_available": False,
            "event_type_available": False,
            "severity_available": False,
            "ready_for_road_mapping": False,
            "ready_for_training_labels": False,
        },
        "rules": [
            "Raw source must be preserved.",
            "Do not modify the original raw source.",
            "Do not scrape restricted government services.",
            "Do not fabricate missing event dates.",
            "Do not fabricate missing geometry.",
            "Do not train until spatial and temporal labeling is documented.",
        ],
    }


def update_label_readiness(manifest: dict[str, Any]) -> None:
    inspection = manifest["inspection"]

    candidates = inspection.get("candidate_fields", {})

    geometry_available = (
        "geometry_column" in inspection
        or (
            candidates.get("latitude")
            and candidates.get("longitude")
        )
    )

    event_date_available = bool(candidates.get("date"))
    event_type_available = bool(candidates.get("event_type"))
    severity_available = bool(candidates.get("severity"))

    manifest["label_readiness"] = {
        "geometry_available": bool(geometry_available),
        "event_date_available": event_date_available,
        "event_type_available": event_type_available,
        "severity_available": severity_available,
        "ready_for_road_mapping": bool(geometry_available),
        "ready_for_training_labels": bool(
            geometry_available and event_date_available
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NER-Nav real hazard source ingestion and schema audit."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Path to an officially acquired hazard source file. "
            "The file will be copied unchanged into data/raw/hazards."
        ),
    )

    parser.add_argument(
        "--source-id",
        default="nrsc_landslide_atlas",
        help="Stable source identifier.",
    )

    parser.add_argument(
        "--access-condition",
        default="officially_acquired_manual",
        help="Documented acquisition/access condition.",
    )

    parser.add_argument(
        "--license-condition",
        default="VERIFY_BEFORE_REUSE",
        help="Documented license/reuse condition.",
    )

    parser.add_argument(
        "--no-copy",
        action="store_true",
        help=(
            "Inspect an already-preserved raw file without copying it."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("NER-Nav — Real Hazard Source Ingestion")
    print("=" * 70)

    source = Path(args.source).expanduser().resolve()

    print()
    print("Source:")
    print(f"  {source}")

    if not source.exists():
        print()
        print("ERROR: Source file does not exist.")
        print()
        print(
            "Acquire an officially permitted source first and place it "
            "under data/raw/hazards/."
        )
        return 2

    if not source.is_file():
        print()
        print("ERROR: Source path is not a file.")
        return 2

    if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        print()
        print(
            f"ERROR: Unsupported extension: {source.suffix}"
        )
        print(
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
        return 2

    RAW_HAZARD_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.no_copy:
        raw_path = source
    else:
        raw_path = RAW_HAZARD_DIR / source.name

        if source != raw_path:
            print()
            print("Preserving raw source...")
            shutil.copy2(source, raw_path)

    print()
    print("Raw source:")
    print(f"  {raw_path}")

    print()
    print("SHA-256:")
    digest = sha256_file(raw_path)
    print(f"  {digest}")

    print()
    print("Inspecting source schema...")

    try:
        inspection = inspect_source(raw_path)

    except Exception as exc:
        print()
        print("ERROR: Source inspection failed.")
        print(f"Reason: {exc}")
        return 3

    manifest = build_manifest(
        source_path=raw_path,
        inspection=inspection,
        original_source_id=args.source_id,
        access_condition=args.access_condition,
        license_condition=args.license_condition,
    )

    update_label_readiness(manifest)

    with MANIFEST_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("SOURCE INSPECTION")
    print("-" * 70)

    print(
        f"Format:             "
        f"{inspection.get('format', 'unknown')}"
    )

    if "rows" in inspection:
        print(f"Rows:               {inspection['rows']}")

    if "sample_rows" in inspection:
        print(
            f"Sample rows:        "
            f"{inspection['sample_rows']}"
        )

    if "crs" in inspection:
        print(f"CRS:                {inspection['crs']}")

    if "geometry_column" in inspection:
        print(
            f"Geometry column:    "
            f"{inspection['geometry_column']}"
        )

    print()
    print("Candidate fields:")

    for key, values in inspection.get(
        "candidate_fields", {}
    ).items():
        print(
            f"  {key:<18}: "
            f"{values if values else 'NONE'}"
        )

    print()
    print("LABEL READINESS")
    print("-" * 70)

    readiness = manifest["label_readiness"]

    print(
        "Geometry available:       "
        f"{readiness['geometry_available']}"
    )

    print(
        "Event date available:     "
        f"{readiness['event_date_available']}"
    )

    print(
        "Event type available:     "
        f"{readiness['event_type_available']}"
    )

    print(
        "Severity available:       "
        f"{readiness['severity_available']}"
    )

    print(
        "Ready for road mapping:   "
        f"{readiness['ready_for_road_mapping']}"
    )

    print(
        "Ready for training labels:"
        f" {readiness['ready_for_training_labels']}"
    )

    print()
    print("Manifest:")
    print(f"  {MANIFEST_PATH}")

    print()
    print("=" * 70)
    print("HAZARD SOURCE INGESTION COMPLETE")
    print("=" * 70)

    if readiness["ready_for_training_labels"]:
        print("STATUS: SOURCE READY FOR SPATIAL/TEMPORAL LABELING")
    elif readiness["geometry_available"]:
        print(
            "STATUS: SOURCE AVAILABLE, "
            "BUT TEMPORAL LABELING INFORMATION IS INCOMPLETE"
        )
    else:
        print(
            "STATUS: SOURCE REQUIRES SCHEMA REVIEW "
            "BEFORE ROAD MAPPING"
        )

    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())