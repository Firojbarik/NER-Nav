#!/usr/bin/env python3
"""Build a deterministic, versioned road identity mapping from the OSM snapshot.

This creates an identity *mapping*, not a claim that OSM identities persist
across all future map edits.  The mapping records the source OSM ID and a
geometry fingerprint so later snapshots can preserve or review source-ID
history instead of treating a raw OSM way ID as a permanent application ID.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROAD_FILE = ROOT / "data/processed/roads/ner_roads_districts.gpkg"
OUTPUT = ROOT / "data/processed/roads/road_segment_identity.parquet"
META = ROOT / "data/processed/roads/road_segment_identity.json"
IDENTITY_VERSION = "ner-road-snapshot-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_hash(geometry) -> str:
    return hashlib.sha256(geometry.wkb).hexdigest()


def build_mapping(roads: gpd.GeoDataFrame) -> pd.DataFrame:
    if roads.crs is None or str(roads.crs).upper() != "EPSG:4326":
        raise ValueError(f"road snapshot must be EPSG:4326, got {roads.crs}")
    if "osm_id" not in roads.columns:
        raise ValueError("road snapshot lacks osm_id")
    if roads["osm_id"].isna().any():
        raise ValueError("road snapshot contains null osm_id")
    if roads.geometry.isna().any() or (~roads.geometry.is_valid).any():
        raise ValueError("road snapshot contains null or invalid geometry")

    records = []
    for row in roads.itertuples(index=False):
        osm_id = int(getattr(row, "osm_id"))
        geo_hash = geometry_hash(row.geometry)
        # Deterministic within this source snapshot and reviewable across
        # snapshots; source-ID history remains explicit rather than inferred.
        identity_key = f"{IDENTITY_VERSION}|{osm_id}|{geo_hash}"
        segment_id = "ner-rseg-" + hashlib.sha256(
            identity_key.encode("ascii")).hexdigest()[:24]
        records.append({
            "road_segment_id": segment_id,
            "source_osm_id": osm_id,
            "source_identity_type": "osm_way_snapshot",
            "geometry_sha256": geo_hash,
            "identity_status": "snapshot_deterministic_pending_history_review",
            "identity_version": IDENTITY_VERSION,
            "ref": getattr(row, "ref", None),
            "state": getattr(row, "state", None),
            "district": getattr(row, "district", None),
        })
    return pd.DataFrame(records)


def main() -> int:
    roads = gpd.read_file(ROAD_FILE)
    mapping = build_mapping(roads)
    mapping.to_parquet(OUTPUT, index=False)
    source_sha = sha256_file(ROAD_FILE)
    metadata = {
        "schema_version": "1.0.0",
        "identity_version": IDENTITY_VERSION,
        "source_road_snapshot": str(ROAD_FILE.relative_to(ROOT)),
        "source_road_snapshot_sha256": source_sha,
        "rows": int(len(mapping)),
        "unique_source_osm_ids": int(mapping["source_osm_id"].nunique()),
        "identity_status": "SNAPSHOT_ONLY_PENDING_SOURCE_ID_HISTORY",
        "global_identity_verified": False,
        "policy": {
            "raw_road_snapshot_immutable": True,
            "osm_ids_are_source_identifiers": True,
            "geometry_matching_across_snapshots_requires_review": True,
            "no_event_labels_created": True,
        },
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
