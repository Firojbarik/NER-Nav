from pathlib import Path
import json

import geopandas as gpd
import pandas as pd


# ============================================================
# NER-Nav — Road to District Attribution
# ============================================================

BASE = Path(__file__).resolve().parents[1]

ROADS = BASE / "data" / "processed" / "roads" / "ner_roads.gpkg"
DISTRICTS = BASE / "data" / "processed" / "admin_boundaries" / "ner_districts.gpkg"

OUTPUT = BASE / "data" / "processed" / "roads" / "ner_roads_districts.gpkg"
QA_OUTPUT = BASE / "data" / "processed" / "roads" / "ner_roads_district_qa.json"


print("=" * 70)
print("NER-Nav — Road to District Attribution")
print("=" * 70)

print(f"\nRoads:      {ROADS}")
print(f"Districts:  {DISTRICTS}")
print(f"Output:     {OUTPUT}")

# ------------------------------------------------------------
# 1. Read inputs
# ------------------------------------------------------------

print("\nReading roads...")
roads = gpd.read_file(ROADS)

print("Reading districts...")
districts = gpd.read_file(DISTRICTS)

print(f"Road records:      {len(roads):,}")
print(f"District records:  {len(districts):,}")

# ------------------------------------------------------------
# 2. Validate input CRS
# ------------------------------------------------------------

if roads.crs is None:
    raise ValueError("Road layer has no CRS.")

if districts.crs is None:
    raise ValueError("District layer has no CRS.")

print(f"Road CRS:          {roads.crs}")
print(f"District CRS:      {districts.crs}")

if roads.crs != districts.crs:
    print("\nReprojecting districts to road CRS...")
    districts = districts.to_crs(roads.crs)

# ------------------------------------------------------------
# 3. Validate required fields
# ------------------------------------------------------------

required_road_fields = {"osm_id", "geometry"}
required_district_fields = {"DISTRICT", "STATE_UT", "DIST_LGD", "STATE_LGD", "geometry"}

missing_roads = required_road_fields - set(roads.columns)
missing_districts = required_district_fields - set(districts.columns)

if missing_roads:
    raise ValueError(f"Missing road fields: {sorted(missing_roads)}")

if missing_districts:
    raise ValueError(f"Missing district fields: {sorted(missing_districts)}")

# ------------------------------------------------------------
# 4. Basic geometry validation
# ------------------------------------------------------------

invalid_roads = (~roads.geometry.is_valid).sum()
empty_roads = roads.geometry.is_empty.sum()

invalid_districts = (~districts.geometry.is_valid).sum()
empty_districts = districts.geometry.is_empty.sum()

print("\nInput geometry QA:")
print(f"  Invalid roads:       {invalid_roads:,}")
print(f"  Empty roads:         {empty_roads:,}")
print(f"  Invalid districts:   {invalid_districts:,}")
print(f"  Empty districts:     {empty_districts:,}")

if invalid_roads or empty_roads:
    raise ValueError("Road input contains invalid or empty geometries.")

if invalid_districts or empty_districts:
    raise ValueError("District input contains invalid or empty geometries.")

# ------------------------------------------------------------
# 5. Preserve original road row identity
# ------------------------------------------------------------

roads = roads.reset_index(drop=True)
roads["_road_row"] = roads.index

districts = districts.reset_index(drop=True)
districts["_district_row"] = districts.index

# ------------------------------------------------------------
# 6. Candidate spatial join
# ------------------------------------------------------------

print("\nFinding candidate road/district intersections...")

candidates = gpd.sjoin(
    roads[["_road_row", "osm_id", "geometry"]],
    districts[
        [
            "_district_row",
            "DISTRICT",
            "STATE_UT",
            "STATE_LGD",
            "DIST_LGD",
            "geometry",
        ]
    ],
    how="left",
    predicate="intersects",
)

print(f"Candidate matches: {len(candidates):,}")

# ------------------------------------------------------------
# 7. Separate unmatched roads
# ------------------------------------------------------------

unmatched = candidates[candidates["_district_row"].isna()].copy()
matched = candidates[candidates["_district_row"].notna()].copy()

unmatched_roads = set(unmatched["_road_row"].astype(int))

print(f"Roads with no district candidate: {len(unmatched_roads):,}")

# ------------------------------------------------------------
# 8. Calculate intersection lengths
#
# IMPORTANT:
# EPSG:4326 is geographic coordinates, so length calculations
# should not be performed directly in degrees.
#
# We use a projected CRS temporarily.
# ------------------------------------------------------------

print("\nPreparing projected geometries for length calculation...")

# Use a suitable projected CRS automatically from the road layer.
# GeoPandas recommends estimating a local UTM CRS for this purpose.
projected_crs = roads.estimate_utm_crs()

if projected_crs is None:
    raise ValueError("Could not determine a projected CRS for length calculation.")

print(f"Projected CRS: {projected_crs}")

roads_projected = roads[["_road_row", "geometry"]].to_crs(projected_crs)

districts_projected = districts[
    [
        "_district_row",
        "geometry",
    ]
].to_crs(projected_crs)

# ------------------------------------------------------------
# 9. Compute actual intersection lengths
# ------------------------------------------------------------

print("Calculating road length inside candidate districts...")

# Merge projected geometries onto candidate pairs.
candidate_lengths = matched[
    [
        "_road_row",
        "_district_row",
        "DISTRICT",
        "STATE_UT",
        "STATE_LGD",
        "DIST_LGD",
    ]
].copy()

candidate_lengths = candidate_lengths.merge(
    roads_projected,
    on="_road_row",
    how="left",
)

candidate_lengths = candidate_lengths.merge(
    districts_projected,
    on="_district_row",
    how="left",
    suffixes=("_road", "_district"),
)

candidate_lengths["intersection_geometry"] = gpd.GeoSeries(
    candidate_lengths["geometry_road"].values,
    crs=projected_crs,
).intersection(
    gpd.GeoSeries(
        candidate_lengths["geometry_district"].values,
        crs=projected_crs,
    ),
    align=False,
)

intersection_geometries = gpd.GeoSeries(
    candidate_lengths["intersection_geometry"].values,
    crs=projected_crs,
)

candidate_lengths["intersection_length_m"] = (
    intersection_geometries.length
)

# ------------------------------------------------------------
# 10. Pick dominant district
# ------------------------------------------------------------

print("Selecting dominant district for each road...")

candidate_lengths = candidate_lengths.sort_values(
    by=["_road_row", "intersection_length_m", "_district_row"],
    ascending=[True, False, True],
)

best = (
    candidate_lengths
    .drop_duplicates("_road_row", keep="first")
    .copy()
)

# ------------------------------------------------------------
# 11. Calculate ambiguity / boundary statistics
# ------------------------------------------------------------

candidate_counts = (
    candidate_lengths
    .groupby("_road_row")
    .size()
)

multi_match_roads = int((candidate_counts > 1).sum())

road_total_lengths = roads_projected.set_index("_road_row").geometry.length

best = best.set_index("_road_row")

best["road_length_m"] = road_total_lengths.loc[best.index]

best["coverage_ratio"] = (
    best["intersection_length_m"] /
    best["road_length_m"]
).clip(lower=0, upper=1)

best["district_match_type"] = "single"

best.loc[
    best.index.isin(candidate_counts[candidate_counts > 1].index),
    "district_match_type",
] = "boundary_resolved"

# ------------------------------------------------------------
# 12. Build final road dataset
# ------------------------------------------------------------

print("\nBuilding final dataset...")

assignment_columns = [
    "DISTRICT",
    "STATE_UT",
    "STATE_LGD",
    "DIST_LGD",
    "intersection_length_m",
    "coverage_ratio",
    "district_match_type",
]

assignment = best[assignment_columns].copy()

assignment = assignment.rename(
    columns={
        "DISTRICT": "district",
        "STATE_UT": "state",
        "STATE_LGD": "state_lgd",
        "DIST_LGD": "dist_lgd",
    }
)

final = roads.join(assignment, on="_road_row")

# ------------------------------------------------------------
# 13. Mark unmatched roads
# ------------------------------------------------------------

final["district_match_type"] = final["district_match_type"].fillna(
    "unmatched"
)

final["coverage_ratio"] = final["coverage_ratio"].fillna(0.0)
final["intersection_length_m"] = final["intersection_length_m"].fillna(0.0)

# Remove internal processing field.
final = final.drop(columns=["_road_row"])

# ------------------------------------------------------------
# 14. Final geometry QA
# ------------------------------------------------------------

invalid_final = (~final.geometry.is_valid).sum()
empty_final = final.geometry.is_empty.sum()

print("\nFinal geometry QA:")
print(f"  Invalid geometries: {invalid_final:,}")
print(f"  Empty geometries:   {empty_final:,}")

if invalid_final or empty_final:
    raise ValueError("Final dataset contains invalid or empty geometries.")

# ------------------------------------------------------------
# 15. Attribution QA
# ------------------------------------------------------------

assigned_count = int(final["district"].notna().sum())
unmatched_count = int(final["district"].isna().sum())

single_count = int(
    (final["district_match_type"] == "single").sum()
)

boundary_count = int(
    (final["district_match_type"] == "boundary_resolved").sum()
)

print("\nAttribution QA:")
print(f"  Total roads:                  {len(final):,}")
print(f"  Assigned roads:               {assigned_count:,}")
print(f"  Unmatched roads:              {unmatched_count:,}")
print(f"  Single-district roads:        {single_count:,}")
print(f"  Boundary-resolved roads:     {boundary_count:,}")

# ------------------------------------------------------------
# 16. District distribution
# ------------------------------------------------------------

district_distribution = (
    final["district"]
    .value_counts(dropna=True)
    .sort_index()
    .to_dict()
)

state_distribution = (
    final["state"]
    .value_counts(dropna=True)
    .sort_index()
    .to_dict()
)

# ------------------------------------------------------------
# 17. Write GeoPackage
# ------------------------------------------------------------

print("\nWriting GeoPackage...")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

final.to_file(
    OUTPUT,
    layer="ner_roads_districts",
    driver="GPKG",
)

# ------------------------------------------------------------
# 18. Write QA report
# ------------------------------------------------------------

qa = {
    "dataset": "NER-Nav OSM road network with district attribution",
    "input_roads": str(ROADS),
    "input_districts": str(DISTRICTS),
    "output": str(OUTPUT),

    "crs_output": str(final.crs),
    "projected_crs_for_length": str(projected_crs),

    "roads": {
        "record_count": int(len(final)),
        "unique_osm_ids": int(final["osm_id"].nunique()),
        "invalid_geometry_count": int(invalid_final),
        "empty_geometry_count": int(empty_final),
    },

    "districts": {
        "record_count": int(len(districts)),
    },

    "attribution": {
        "assigned_count": assigned_count,
        "unmatched_count": unmatched_count,
        "single_district_count": single_count,
        "boundary_resolved_count": boundary_count,
        "candidate_match_count": int(len(candidate_lengths)),
        "multi_candidate_road_count": multi_match_roads,
        "assignment_coverage_percent": round(
            assigned_count / len(final) * 100,
            4,
        ),
    },

    "district_distribution": district_distribution,
    "state_distribution": state_distribution,

    "status": (
        "PASS"
        if (
            len(final) == len(roads)
            and final["osm_id"].nunique() == len(final)
            and invalid_final == 0
            and empty_final == 0
        )
        else "FAIL"
    ),
}

with open(QA_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(qa, f, indent=2, ensure_ascii=False)

# ------------------------------------------------------------
# 19. Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ROAD → DISTRICT ATTRIBUTION COMPLETE")
print("=" * 70)

print(f"Road records:       {len(final):,}")
print(f"Assigned:           {assigned_count:,}")
print(f"Unmatched:          {unmatched_count:,}")
print(f"Single district:    {single_count:,}")
print(f"Boundary resolved:  {boundary_count:,}")
print(f"GeoPackage:         {OUTPUT}")
print(f"QA report:          {QA_OUTPUT}")
print(f"Status:             {qa['status']}")
print("=" * 70)