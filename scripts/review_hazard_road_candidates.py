from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_event.json"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_road_mapping.parquet"
)

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hazards"

OUT_PARQUET = (
    OUT_DIR
    / "nrsc_sikkim_mantam_2016_road_candidates.parquet"
)

OUT_CSV = (
    OUT_DIR
    / "nrsc_sikkim_mantam_2016_road_candidates.csv"
)

OUT_REVIEW = (
    OUT_DIR
    / "nrsc_sikkim_mantam_2016_road_review.json"
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    print("=" * 70)
    print("NER-Nav — Hazard Road Candidate Review")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Load event
    # ---------------------------------------------------------------
    print("\nLoading verified event...")
    print("-" * 70)

    if not EVENT_PATH.exists():
        fail(f"Event file not found: {EVENT_PATH}")

    with EVENT_PATH.open("r", encoding="utf-8") as f:
        event = json.load(f)

    event_id = event.get("event_id")
    event_date = event.get("event_date")
    latitude = event.get("latitude")
    longitude = event.get("longitude")

    if not event_id:
        fail("Missing event_id.")

    if latitude is None or longitude is None:
        fail("Verified event coordinates are missing.")

    print(f"Event ID:       {event_id}")
    print(f"Event date:     {event_date}")
    print(f"Latitude:       {latitude}")
    print(f"Longitude:      {longitude}")

    # ---------------------------------------------------------------
    # Load existing mapping
    # ---------------------------------------------------------------
    print("\nLoading road mapping...")
    print("-" * 70)

    if not MAPPING_PATH.exists():
        fail(f"Mapping file not found: {MAPPING_PATH}")

    mapping = gpd.read_parquet(MAPPING_PATH)

    if mapping.empty:
        fail("Road mapping file is empty.")

    print(f"Mapping rows:   {len(mapping)}")
    print(f"CRS:            {mapping.crs}")

    print("\nAvailable mapping columns:")
    for column in mapping.columns:
        print(f"  - {column}")

    # ---------------------------------------------------------------
    # Validate actual schema
    # ---------------------------------------------------------------
    required = [
        "osm_id",
        "highway",
        "name",
        "district",
        "state",
        "geometry",
    ]

    missing = [c for c in required if c not in mapping.columns]

    if missing:
        fail(
            "Mapping file is missing required columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------------
    # Detect the actual distance column
    # ---------------------------------------------------------------
    distance_candidates = [
        "distance_m",
        "distance",
        "distance_meters",
        "event_distance_m",
    ]

    distance_column = next(
        (c for c in distance_candidates if c in mapping.columns),
        None,
    )

    if distance_column is None:
        fail(
            "No distance column exists in the mapping output. "
            "Fix map_hazard_event_to_roads.py first so it persists "
            "the calculated event-to-road distance."
        )

    print(f"\nDistance column: {distance_column}")

    # ---------------------------------------------------------------
    # Normalize fields
    # ---------------------------------------------------------------
    candidates = mapping.copy()

    candidates["name"] = candidates["name"].fillna("").astype(str)
    candidates["highway"] = (
        candidates["highway"].fillna("").astype(str)
    )
    candidates["district"] = (
        candidates["district"].fillna("").astype(str)
    )
    candidates["state"] = candidates["state"].fillna("").astype(str)

    candidates["distance_m"] = pd.to_numeric(
        candidates[distance_column],
        errors="coerce",
    )

    if candidates["distance_m"].isna().all():
        fail("Distance column exists but contains no usable numeric values.")

    # ---------------------------------------------------------------
    # Search radius
    #
    # Existing mapper used 2,000 m.
    # ---------------------------------------------------------------
    SEARCH_RADIUS_M = 2000.0

    candidates["within_search_radius"] = (
        candidates["distance_m"] <= SEARCH_RADIUS_M
    )

    candidates = candidates[
        candidates["within_search_radius"]
    ].copy()

    if candidates.empty:
        fail(
            "No road candidates remain within the configured "
            f"{SEARCH_RADIUS_M:.0f} m search radius."
        )

    print(f"Search radius:   {SEARCH_RADIUS_M:.0f} m")
    print(f"Candidates:       {len(candidates)}")

    # ---------------------------------------------------------------
    # Evidence classification
    # ---------------------------------------------------------------
    #
    # IMPORTANT:
    # Distance alone NEVER establishes that a road was affected.
    #
    candidates["review_status"] = "CANDIDATE"
    candidates["source_supported"] = False
    candidates["confirmed_affected"] = False

    candidates["evidence_note"] = (
        "NRSC reports road damage on the Passingdang-Mantam Road. "
        "Current candidate has not been independently matched to "
        "that named road segment."
    )

    candidates["rejection_reason"] = ""

    # ---------------------------------------------------------------
    # Road-name evidence
    # ---------------------------------------------------------------
    #
    # Search conservatively. We do NOT treat partial arbitrary
    # matches as confirmation.
    #
    normalized_name = (
        candidates["name"]
        .str.lower()
        .str.replace("-", " ", regex=False)
        .str.replace("_", " ", regex=False)
        .str.strip()
    )

    name_match = normalized_name.str.contains(
        "passingdang",
        na=False,
    ) & normalized_name.str.contains(
        "mantam",
        na=False,
    )

    candidates.loc[name_match, "source_supported"] = True
    candidates.loc[name_match, "review_status"] = "SOURCE_SUPPORTED"

    candidates.loc[
        name_match,
        "evidence_note",
    ] = (
        "OSM name contains both Passingdang and Mantam. "
        "This is source-supporting evidence but still requires "
        "manual/geospatial confirmation before labeling."
    )

    # ---------------------------------------------------------------
    # Rank candidates
    # ---------------------------------------------------------------
    candidates = candidates.sort_values(
        by=["distance_m", "osm_id"],
        ascending=[True, True],
    ).reset_index(drop=True)

    candidates["candidate_rank"] = candidates.index + 1

    # ---------------------------------------------------------------
    # Keep review columns
    # ---------------------------------------------------------------
    review_columns = [
        "candidate_rank",
        "osm_id",
        "name",
        "highway",
        "district",
        "state",
        "distance_m",
        "within_search_radius",
        "review_status",
        "source_supported",
        "confirmed_affected",
        "rejection_reason",
        "evidence_note",
        "geometry",
    ]

    candidates = candidates[review_columns]

    # ---------------------------------------------------------------
    # Output directory
    # ---------------------------------------------------------------
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Save parquet
    # ---------------------------------------------------------------
    candidates.to_parquet(
        OUT_PARQUET,
        index=False,
    )

    # ---------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------
    candidates.drop(
        columns=["geometry"]
    ).to_csv(
        OUT_CSV,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    nearest = candidates.iloc[0]

    source_supported_count = int(
        candidates["source_supported"].sum()
    )

    confirmed_count = int(
        candidates["confirmed_affected"].sum()
    )

    # ---------------------------------------------------------------
    # Review manifest
    # ---------------------------------------------------------------
    review = {
        "event_id": event_id,
        "event_date": event_date,
        "event_coordinates": {
            "latitude": latitude,
            "longitude": longitude,
            "crs": "EPSG:4326",
        },
        "source_evidence": {
            "organization": "National Remote Sensing Centre, ISRO",
            "reported_road": "Passingdang-Mantam Road",
            "reported_road_damage": True,
            "reported_road_damage_length_m": 300,
            "automatic_confirmation_allowed": False,
        },
        "mapping": {
            "input_rows": int(len(mapping)),
            "candidate_rows": int(len(candidates)),
            "search_radius_m": SEARCH_RADIUS_M,
            "distance_column": distance_column,
        },
        "counts": {
            "source_supported": source_supported_count,
            "confirmed_affected": confirmed_count,
        },
        "nearest_candidate": {
            "osm_id": str(nearest["osm_id"]),
            "name": nearest["name"],
            "highway": nearest["highway"],
            "district": nearest["district"],
            "state": nearest["state"],
            "distance_m": float(nearest["distance_m"]),
        },
        "decision": {
            "status": "REVIEW_REQUIRED",
            "training_label_created": False,
            "reason": (
                "Candidates have been ranked using the existing "
                "event-to-road mapping. Distance and name evidence "
                "are not sufficient by themselves to declare an "
                "affected road segment."
            ),
        },
        "outputs": {
            "parquet": str(
                OUT_PARQUET.relative_to(PROJECT_ROOT)
            ),
            "csv": str(
                OUT_CSV.relative_to(PROJECT_ROOT)
            ),
            "review_json": str(
                OUT_REVIEW.relative_to(PROJECT_ROOT)
            ),
        },
    }

    with OUT_REVIEW.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            review,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ---------------------------------------------------------------
    # HARD OUTPUT VERIFICATION
    # ---------------------------------------------------------------
    expected_outputs = [
        OUT_PARQUET,
        OUT_CSV,
        OUT_REVIEW,
    ]

    missing_outputs = [
        str(path)
        for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        fail(
            "Output verification failed. Missing: "
            + ", ".join(missing_outputs)
        )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ROAD CANDIDATE REVIEW COMPLETE")
    print("=" * 70)

    print(f"Candidates:          {len(candidates)}")
    print(f"Source-supported:    {source_supported_count}")
    print(f"Confirmed affected:  {confirmed_count}")

    print("\nNearest candidate:")
    print(f"  OSM ID:       {nearest['osm_id']}")
    print(
        f"  Name:         "
        f"{nearest['name'] if nearest['name'] else '<unnamed>'}"
    )
    print(f"  Highway:      {nearest['highway']}")
    print(f"  District:     {nearest['district']}")
    print(f"  Distance:     {nearest['distance_m']:.2f} m")

    print("\nOutputs:")
    print(f"  Parquet: {OUT_PARQUET}")
    print(f"  CSV:     {OUT_CSV}")
    print(f"  Review:  {OUT_REVIEW}")

    print("\nIMPORTANT:")
    print("  No road was automatically marked affected.")
    print("  No training label was created.")
    print("  Manual/source-supported confirmation remains required.")

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()