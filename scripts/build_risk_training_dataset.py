from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROAD_FILE = Path(
    "data/processed/roads/ner_roads_districts.gpkg"
)

OUTPUT_DIR = Path(
    "data/processed/ml"
)

OUTPUT_FILE = OUTPUT_DIR / "risk_training_dataset.parquet"
QA_FILE = OUTPUT_DIR / "risk_training_dataset_qa.json"


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

RISK_CLASSES = {
    0: "low",
    1: "moderate",
    2: "high",
    3: "severe",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_number(value, default=0.0):
    if value is None:
        return default

    try:
        value = float(value)

        if not math.isfinite(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def normalize_text(value):
    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def highway_score(highway):
    """
    Baseline road-class risk prior.

    This is NOT a learned hazard label.
    It is only a deterministic prior used to construct
    the initial development dataset until real incident
    observations are available.
    """

    scores = {
        "motorway": 0.05,
        "trunk": 0.10,
        "primary": 0.15,
        "secondary": 0.20,
        "tertiary": 0.25,
        "tertiary_link": 0.25,
        "secondary_link": 0.20,
        "primary_link": 0.15,
        "residential": 0.30,
        "living_street": 0.35,
        "service": 0.40,
        "unclassified": 0.45,
        "road": 0.45,
        "track": 0.60,
    }

    return scores.get(str(highway).lower(), 0.40)


def classify_risk(score):
    """
    Convert baseline risk score into four development classes.

    IMPORTANT:
    These are synthetic/proxy labels.
    They must NOT be presented as historical disaster labels.
    """

    if score < 0.25:
        return 0

    if score < 0.40:
        return 1

    if score < 0.55:
        return 2

    return 3


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("NER-Nav — Risk Training Dataset Construction")
    print("=" * 70)

    if not ROAD_FILE.exists():
        raise FileNotFoundError(
            f"Road dataset not found: {ROAD_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Road dataset: {ROAD_FILE}")
    print(f"Output:       {OUTPUT_FILE}")
    print()

    # -----------------------------------------------------------------
    # Load roads
    # -----------------------------------------------------------------

    print("Loading road dataset...")

    roads = gpd.read_file(ROAD_FILE)

    print(f"Road records: {len(roads):,}")
    print(f"CRS:          {roads.crs}")

    required_columns = {
        "osm_id",
        "highway",
        "district",
        "state",
        "geometry",
    }

    missing = required_columns - set(roads.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if roads.crs is None:
        raise ValueError(
            "Road dataset has no CRS."
        )

    if str(roads.crs).upper() != "EPSG:4326":

        print("Reprojecting to EPSG:4326...")

        roads = roads.to_crs("EPSG:4326")

    # -----------------------------------------------------------------
    # Geometry QA
    # -----------------------------------------------------------------

    invalid = int(
        (~roads.geometry.is_valid).sum()
    )

    empty = int(
        roads.geometry.is_empty.sum()
    )

    print()
    print("Geometry QA:")
    print(f"  Invalid: {invalid:,}")
    print(f"  Empty:   {empty:,}")

    if invalid or empty:
        raise ValueError(
            "Geometry QA failed."
        )

    # -----------------------------------------------------------------
    # Feature construction
    # -----------------------------------------------------------------

    print()
    print("Building ML features...")

    records = []

    skipped = 0

    for idx, row in roads.iterrows():

        geom = row.geometry

        if geom is None or geom.is_empty:
            skipped += 1
            continue

        if geom.geom_type != "LineString":
            skipped += 1
            continue

        coords = list(geom.coords)

        if len(coords) < 2:
            skipped += 1
            continue

        start_lon, start_lat = coords[0]
        end_lon, end_lat = coords[-1]

        # Representative point
        point = geom.interpolate(
            0.5,
            normalized=True,
        )

        lon = safe_number(point.x)
        lat = safe_number(point.y)

        highway = normalize_text(
            row.get("highway")
        )

        district = normalize_text(
            row.get("district")
        )

        state = normalize_text(
            row.get("state")
        )

        surface = normalize_text(
            row.get("surface")
        )

        maxspeed = normalize_text(
            row.get("maxspeed")
        )

        lanes = normalize_text(
            row.get("lanes")
        )

        access = normalize_text(
            row.get("access")
        )

        bridge = normalize_text(
            row.get("bridge")
        )

        tunnel = normalize_text(
            row.get("tunnel")
        )

        lit = normalize_text(
            row.get("lit")
        )

        smoothness = normalize_text(
            row.get("smoothness")
        )

        tracktype = normalize_text(
            row.get("tracktype")
        )

        service = normalize_text(
            row.get("service")
        )

        oneway = normalize_text(
            row.get("oneway")
        )

        coverage_ratio = safe_number(
            row.get("coverage_ratio"),
            default=1.0,
        )

        intersection_length_m = safe_number(
            row.get("intersection_length_m"),
            default=0.0,
        )

        # -------------------------------------------------------------
        # Baseline proxy features
        # -------------------------------------------------------------

        road_risk_prior = highway_score(
            highway
        )

        bridge_flag = int(
            bridge in {
                "yes",
                "true",
                "1",
            }
        )

        tunnel_flag = int(
            tunnel in {
                "yes",
                "true",
                "1",
            }
        )

        lit_flag = int(
            lit in {
                "yes",
                "true",
                "1",
            }
        )

        service_flag = int(
            service is not None
        )

        low_coverage_flag = int(
            coverage_ratio < 0.75
        )

        # -------------------------------------------------------------
        # Initial synthetic development score
        # -------------------------------------------------------------

        risk_score = road_risk_prior

        if bridge_flag:
            risk_score += 0.10

        if tunnel_flag:
            risk_score += 0.05

        if low_coverage_flag:
            risk_score += 0.10

        if tracktype is not None:
            risk_score += 0.05

        risk_score = min(
            max(risk_score, 0.0),
            1.0,
        )

        risk_class = classify_risk(
            risk_score
        )

        records.append(
            {
                # Identity
                "osm_id": int(row["osm_id"]),
                "district": district,
                "state": state,

                # Spatial
                "longitude": lon,
                "latitude": lat,

                "start_longitude": safe_number(
                    start_lon
                ),
                "start_latitude": safe_number(
                    start_lat
                ),
                "end_longitude": safe_number(
                    end_lon
                ),
                "end_latitude": safe_number(
                    end_lat
                ),

                # Road attributes
                "highway": highway,
                "surface": surface,
                "maxspeed": maxspeed,
                "lanes": lanes,
                "access": access,
                "bridge": bridge,
                "tunnel": tunnel,
                "lit": lit,
                "smoothness": smoothness,
                "tracktype": tracktype,
                "service": service,
                "oneway": oneway,

                # Existing engineering features
                "intersection_length_m":
                    intersection_length_m,
                "coverage_ratio":
                    coverage_ratio,

                # Derived numerical features
                "bridge_flag":
                    bridge_flag,
                "tunnel_flag":
                    tunnel_flag,
                "lit_flag":
                    lit_flag,
                "service_flag":
                    service_flag,
                "low_coverage_flag":
                    low_coverage_flag,

                "road_risk_prior":
                    road_risk_prior,

                # Development target
                "risk_score":
                    risk_score,

                "risk_class":
                    risk_class,

                "risk_label":
                    RISK_CLASSES[risk_class],

                # Provenance
                "label_source":
                    "synthetic_road_prior",
            }
        )

        if len(records) % 50_000 == 0:
            print(
                f"  Processed records: "
                f"{len(records):,}"
            )

    # -----------------------------------------------------------------
    # DataFrame
    # -----------------------------------------------------------------

    print()
    print("Creating training dataframe...")

    df = pd.DataFrame(records)

    print(
        f"Training records: {len(df):,}"
    )

    if df.empty:
        raise RuntimeError(
            "Training dataset is empty."
        )

    # -----------------------------------------------------------------
    # Dataset QA
    # -----------------------------------------------------------------

    print()
    print("Dataset QA:")

    print(
        f"  Rows:             {len(df):,}"
    )

    print(
        f"  Columns:          {len(df.columns):,}"
    )

    print(
        f"  Skipped roads:    {skipped:,}"
    )

    print(
        f"  Unique OSM IDs:   "
        f"{df['osm_id'].nunique():,}"
    )

    print()
    print("Risk-class distribution:")

    class_counts = (
        df["risk_class"]
        .value_counts()
        .sort_index()
    )

    class_distribution = {}

    for class_id, label in RISK_CLASSES.items():

        count = int(
            class_counts.get(
                class_id,
                0,
            )
        )

        percentage = (
            100.0 * count / len(df)
        )

        class_distribution[
            str(class_id)
        ] = {
            "label": label,
            "count": count,
            "percentage": percentage,
        }

        print(
            f"  {class_id} "
            f"{label:<10} "
            f"{count:>10,} "
            f"({percentage:6.2f}%)"
        )

    # -----------------------------------------------------------------
    # Missing-value report
    # -----------------------------------------------------------------

    print()
    print("Missing-value summary:")

    missing_report = {}

    for column in df.columns:

        count = int(
            df[column].isna().sum()
        )

        if count:
            missing_report[column] = count

    if missing_report:

        for column, count in sorted(
            missing_report.items(),
            key=lambda x: -x[1],
        ):

            print(
                f"  {column:<25} "
                f"{count:,}"
            )

    else:

        print("  NONE")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------

    print()
    print("Writing training dataset...")

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    qa = {
        "dataset":
            "NER-Nav risk training dataset",

        "source":
            str(ROAD_FILE),

        "output":
            str(OUTPUT_FILE),

        "rows":
            int(len(df)),

        "columns":
            int(len(df.columns)),

        "skipped_records":
            int(skipped),

        "unique_osm_ids":
            int(df["osm_id"].nunique()),

        "risk_class_distribution":
            class_distribution,

        "missing_values":
            missing_report,

        "label_source":
            "synthetic_road_prior",

        "warning":
            (
                "risk_class is a synthetic development target "
                "derived from road attributes. It is not historical "
                "hazard/incident ground truth."
            ),

        "status":
            "PASS",
    }

    with QA_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            qa,
            f,
            indent=2,
        )

    # -----------------------------------------------------------------
    # Final output
    # -----------------------------------------------------------------

    print()
    print("=" * 70)
    print("RISK TRAINING DATASET COMPLETE")
    print("=" * 70)

    print(
        f"Rows:              {len(df):,}"
    )

    print(
        f"Features/columns:  {len(df.columns):,}"
    )

    print(
        f"Skipped:           {skipped:,}"
    )

    print(
        f"Dataset:           {OUTPUT_FILE}"
    )

    print(
        f"QA report:         {QA_FILE}"
    )

    print()
    print(
        "IMPORTANT: "
        "risk_class is currently synthetic/proxy."
    )

    print(
        "Real incident/hazard data must replace "
        "these labels before production claims."
    )

    print(
        "Status:             PASS"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()