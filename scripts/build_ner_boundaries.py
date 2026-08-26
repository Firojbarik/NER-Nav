from pathlib import Path
import json

import geopandas as gpd
from shapely.geometry import GeometryCollection


# ---------------------------------------------------------
# NER-Nav Phase 3.1
# Build validated NER administrative boundary datasets
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "admin_boundaries"
    / "source"
    / "91"
)

STATE_SOURCE = SOURCE_DIR / "STATE_BOUNDARY.shp"
DISTRICT_SOURCE = SOURCE_DIR / "DISTRICT_BOUNDARY.shp"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "admin_boundaries"
)

REPORT_DIR = PROJECT_ROOT / "reports"

STATE_OUTPUT = OUTPUT_DIR / "ner_states.gpkg"
DISTRICT_OUTPUT = OUTPUT_DIR / "ner_districts.gpkg"
QA_OUTPUT = REPORT_DIR / "admin_boundary_qa.json"


# MDoNER NER definition
NER_STATES = [
    "ARUNACHAL PRADESH",
    "ASSAM",
    "MANIPUR",
    "MEGHALAYA",
    "MIZORAM",
    "NAGALAND",
    "SIKKIM",
    "TRIPURA",
]


def geometry_type_counts(gdf):
    return {
        str(key): int(value)
        for key, value in gdf.geometry.geom_type.value_counts().items()
    }


def validate_geometries(gdf, name):
    empty_count = int(gdf.geometry.is_empty.sum())
    null_count = int(gdf.geometry.isna().sum())
    invalid_count = int((~gdf.geometry.is_valid).sum())

    if empty_count:
        raise ValueError(f"{name}: {empty_count} empty geometries found")

    if null_count:
        raise ValueError(f"{name}: {null_count} null geometries found")

    if invalid_count:
        raise ValueError(f"{name}: {invalid_count} invalid geometries found")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NER-Nav Phase 3.1 — Administrative Boundary Build")
    print("=" * 70)

    print("\nSource files:")
    print(f"  States:    {STATE_SOURCE}")
    print(f"  Districts: {DISTRICT_SOURCE}")

    if not STATE_SOURCE.exists():
        raise FileNotFoundError(f"Missing state boundary: {STATE_SOURCE}")

    if not DISTRICT_SOURCE.exists():
        raise FileNotFoundError(f"Missing district boundary: {DISTRICT_SOURCE}")

    # -----------------------------------------------------
    # Load source data
    # -----------------------------------------------------

    states = gpd.read_file(STATE_SOURCE)
    districts = gpd.read_file(DISTRICT_SOURCE)

    print("\nLoaded source datasets:")
    print(f"  States:    {states.shape}")
    print(f"  Districts: {districts.shape}")

    # Repair invalid source geometries in-memory.
    # Raw source files are never modified.
    invalid_districts = ~districts.geometry.is_valid

    if invalid_districts.any():
        print()
        print("Repairing invalid district geometries:")
        print(
            districts.loc[
                invalid_districts,
                ["OBJECTID", "STATE_UT", "DISTRICT", "DIST_LGD"]
            ].to_string(index=False)
        )

        districts.loc[invalid_districts, "geometry"] = (
            districts.loc[invalid_districts, "geometry"].make_valid()
        )

        remaining_invalid = (~districts.geometry.is_valid).sum()

        print(f"  Repaired: {invalid_districts.sum()}")
        print(f"  Remaining invalid: {remaining_invalid}")

        if remaining_invalid:
            raise ValueError(
                f"District geometry repair failed: "
                f"{remaining_invalid} invalid geometries remain"
            )

    print("\nSource CRS:")
    print(f"  States:    {states.crs}")
    print(f"  Districts: {districts.crs}")

    # -----------------------------------------------------
    # Basic source validation
    # -----------------------------------------------------

    validate_geometries(states, "Source states")
    validate_geometries(districts, "Source districts")

    if "STATE" not in states.columns:
        raise ValueError("STATE column missing from STATE_BOUNDARY")

    if "STATE_UT" not in districts.columns:
        raise ValueError("STATE_UT column missing from DISTRICT_BOUNDARY")

    # -----------------------------------------------------
    # Filter the eight NER states
    # -----------------------------------------------------

    ner_states = states[
        states["STATE"].isin(NER_STATES)
    ].copy()

    missing_states = sorted(
        set(NER_STATES) - set(ner_states["STATE"].tolist())
    )

    unexpected_states = sorted(
        set(ner_states["STATE"].tolist()) - set(NER_STATES)
    )

    if missing_states:
        raise ValueError(
            f"Missing NER states: {missing_states}"
        )

    if unexpected_states:
        raise ValueError(
            f"Unexpected states selected: {unexpected_states}"
        )

    if len(ner_states) != 8:
        raise ValueError(
            f"Expected 8 NER state records, found {len(ner_states)}"
        )

    # -----------------------------------------------------
    # Filter districts using the same state names
    # -----------------------------------------------------

    ner_districts = districts[
        districts["STATE_UT"].isin(NER_STATES)
    ].copy()

    district_states = set(
        ner_districts["STATE_UT"].dropna().unique()
    )

    missing_district_states = sorted(
        set(NER_STATES) - district_states
    )

    if missing_district_states:
        raise ValueError(
            "District dataset missing states: "
            f"{missing_district_states}"
        )

    # -----------------------------------------------------
    # Convert to WGS84 geographic CRS
    # -----------------------------------------------------

    ner_states = ner_states.to_crs("EPSG:4326")
    ner_districts = ner_districts.to_crs("EPSG:4326")

    # -----------------------------------------------------
    # Final geometry validation
    # -----------------------------------------------------

    validate_geometries(ner_states, "NER states")
    validate_geometries(ner_districts, "NER districts")

    # -----------------------------------------------------
    # Add NER metadata
    # -----------------------------------------------------

    ner_states["region"] = "NER"
    ner_districts["region"] = "NER"

    # -----------------------------------------------------
    # Calculate basic spatial statistics
    # -----------------------------------------------------

    state_bounds = ner_states.total_bounds.tolist()
    district_bounds = ner_districts.total_bounds.tolist()

    # Geographic bounds are:
    # minx, miny, maxx, maxy
    state_bounds_dict = {
        "min_lon": state_bounds[0],
        "min_lat": state_bounds[1],
        "max_lon": state_bounds[2],
        "max_lat": state_bounds[3],
    }

    district_bounds_dict = {
        "min_lon": district_bounds[0],
        "min_lat": district_bounds[1],
        "max_lon": district_bounds[2],
        "max_lat": district_bounds[3],
    }

    # -----------------------------------------------------
    # Save GeoPackages
    # -----------------------------------------------------

    if STATE_OUTPUT.exists():
        STATE_OUTPUT.unlink()

    if DISTRICT_OUTPUT.exists():
        DISTRICT_OUTPUT.unlink()

    ner_states.to_file(
        STATE_OUTPUT,
        layer="ner_states",
        driver="GPKG",
    )

    ner_districts.to_file(
        DISTRICT_OUTPUT,
        layer="ner_districts",
        driver="GPKG",
    )

    # -----------------------------------------------------
    # QA report
    # -----------------------------------------------------

    qa = {
        "dataset": "NER-Nav administrative boundaries",
        "source": {
            "state_boundary": str(STATE_SOURCE),
            "district_boundary": str(DISTRICT_SOURCE),
            "crs_original": "Lambert Conformal Conic / WGS84",
            "crs_output": "EPSG:4326",
        },
        "ner_definition": {
            "source": "Ministry of Development of North Eastern Region",
            "states_expected": NER_STATES,
            "state_count": len(ner_states),
        },
        "states": {
            "record_count": len(ner_states),
            "geometry_types": geometry_type_counts(ner_states),
            "bounds": state_bounds_dict,
            "invalid_geometry_count": int(
                (~ner_states.geometry.is_valid).sum()
            ),
            "empty_geometry_count": int(
                ner_states.geometry.is_empty.sum()
            ),
        },
        "districts": {
            "record_count": len(ner_districts),
            "geometry_types": geometry_type_counts(ner_districts),
            "bounds": district_bounds_dict,
            "invalid_geometry_count": int(
                (~ner_districts.geometry.is_valid).sum()
            ),
            "empty_geometry_count": int(
                ner_districts.geometry.is_empty.sum()
            ),
            "districts_by_state": {
                str(state): int(count)
                for state, count in (
                    ner_districts["STATE_UT"]
                    .value_counts()
                    .sort_index()
                    .items()
                )
            },
        },
        "outputs": {
            "states": str(STATE_OUTPUT),
            "districts": str(DISTRICT_OUTPUT),
            "qa_report": str(QA_OUTPUT),
        },
        "status": "PASS",
    }

    with QA_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            qa,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -----------------------------------------------------
    # Console summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILD SUCCESSFUL")
    print("=" * 70)

    print("\nNER states:")
    for state in NER_STATES:
        count = int(
            (ner_districts["STATE_UT"] == state).sum()
        )
        print(f"  {state:<22} {count:>3} districts")

    print("\nState records:")
    print(f"  {len(ner_states)}")

    print("District records:")
    print(f"  {len(ner_districts)}")

    print("\nOutput:")
    print(f"  {STATE_OUTPUT}")
    print(f"  {DISTRICT_OUTPUT}")
    print(f"  {QA_OUTPUT}")

    print("\nStatus: PASS")


if __name__ == "__main__":
    main()