from pathlib import Path
import json

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_event.json"
)

CANDIDATE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
    / "nrsc_sikkim_mantam_2016_road_candidates.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "hazards"
)

OUTPUT_MAP = OUTPUT_DIR / "nrsc_sikkim_mantam_2016_road_review_map.png"
OUTPUT_GEOJSON = OUTPUT_DIR / "nrsc_sikkim_mantam_2016_road_review.geojson"
OUTPUT_JSON = OUTPUT_DIR / "nrsc_sikkim_mantam_2016_road_review_map.json"


def fail(message):
    raise RuntimeError(message)


def main():
    print("=" * 70)
    print("NER-Nav — Hazard Road Candidate Review Map")
    print("=" * 70)

    if not EVENT_PATH.exists():
        fail(f"Event file not found: {EVENT_PATH}")

    if not CANDIDATE_PATH.exists():
        fail(f"Candidate Parquet not found: {CANDIDATE_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading verified event...")
    with EVENT_PATH.open("r", encoding="utf-8") as f:
        event = json.load(f)

    lat = event.get("latitude")
    lon = event.get("longitude")

    if lat is None or lon is None:
        fail("Event latitude/longitude are missing.")

    print(f"Event ID:  {event.get('event_id')}")
    print(f"Latitude:  {lat}")
    print(f"Longitude: {lon}")

    print("\nLoading candidate roads...")
    roads = gpd.read_parquet(CANDIDATE_PATH)

    print(f"Candidate rows: {len(roads)}")
    print(f"Road CRS:      {roads.crs}")

    required = [
        "osm_id",
        "highway",
        "distance_m",
        "within_search_radius",
        "geometry",
    ]

    missing = [c for c in required if c not in roads.columns]

    if missing:
        fail(
            "Candidate Parquet is missing required columns: "
            + ", ".join(missing)
        )

    if roads.empty:
        fail("Candidate road dataset is empty.")

    if roads.crs is None:
        fail("Candidate road CRS is missing.")

    roads = roads.copy()

    roads = roads[
        roads.geometry.notna()
        & ~roads.geometry.is_empty
    ].copy()

    if roads.empty:
        fail("No valid road geometries remain.")

    # Event point.
    event_gdf = gpd.GeoDataFrame(
        {
            "event_id": [event.get("event_id")],
            "event_name": [event.get("event_name")],
            "event_date": [event.get("event_date")],
        },
        geometry=[Point(lon, lat)],
        crs="EPSG:4326",
    )

    # Normalize CRS for output/review.
    roads_wgs84 = roads.to_crs("EPSG:4326")

    # Save GeoJSON so it can be opened in GIS/QGIS.
    roads_wgs84.to_file(
        OUTPUT_GEOJSON,
        driver="GeoJSON",
    )

    # Project to metric CRS for plotting a meaningful local buffer.
    metric_crs = "EPSG:32645"

    roads_metric = roads_wgs84.to_crs(metric_crs)
    event_metric = event_gdf.to_crs(metric_crs)

    # 2 km review area.
    event_buffer = event_metric.buffer(2000)

    print("\nCreating review map...")
    print("Review radius: 2,000 m")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Candidate roads.
    roads_metric.plot(
        ax=ax,
        linewidth=1.5,
    )

    # Event point.
    event_metric.plot(
        ax=ax,
        markersize=100,
        marker="*",
        zorder=5,
    )

    # Review boundary.
    gpd.GeoSeries(
        event_buffer,
        crs=metric_crs,
    ).boundary.plot(
        ax=ax,
        linewidth=1.0,
        linestyle="--",
    )

    # Label only the closest 10 candidates to avoid a cluttered map.
    closest = roads_metric.sort_values("distance_m").head(10)

    for _, row in closest.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        point = geom.interpolate(0.5, normalized=True)

        name = row.get("name")
        name = "" if name is None else str(name).strip()

        if not name or name.lower() == "nan":
            name = "<unnamed>"

        label = (
            f"OSM {row['osm_id']}\n"
            f"{row['highway']}\n"
            f"{row['distance_m']:.0f} m"
        )

        ax.annotate(
            label,
            xy=(point.x, point.y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )

    ax.set_title(
        "NER-Nav — So Bhir / Mantam Landslide\n"
        "Candidate Road Review — 2 km radius"
    )

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")

    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_MAP,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Review summary.
    nearest = roads.sort_values("distance_m").iloc[0]

    summary = {
        "event_id": event.get("event_id"),
        "event_name": event.get("event_name"),
        "event_date": event.get("event_date"),
        "event_coordinates": {
            "latitude": lat,
            "longitude": lon,
            "crs": "EPSG:4326",
        },
        "candidate_count": int(len(roads)),
        "review_radius_m": 2000,
        "nearest_candidate": {
            "osm_id": str(nearest["osm_id"]),
            "name": (
                ""
                if nearest.get("name") is None
                else str(nearest.get("name"))
            ),
            "highway": str(nearest.get("highway")),
            "distance_m": float(nearest["distance_m"]),
        },
        "decision": {
            "confirmed_affected": False,
            "training_label_created": False,
            "status": "MANUAL_REVIEW_REQUIRED",
            "reason": (
                "The source identifies the damaged Passingdang-Mantam "
                "Road, but the current OSM candidate dataset does not "
                "contain an independently verified name match. "
                "Distance alone is insufficient to create a training label."
            ),
        },
        "outputs": {
            "map": str(OUTPUT_MAP.relative_to(PROJECT_ROOT)),
            "geojson": str(OUTPUT_GEOJSON.relative_to(PROJECT_ROOT)),
            "summary": str(OUTPUT_JSON.relative_to(PROJECT_ROOT)),
        },
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("ROAD REVIEW MAP COMPLETE")
    print("=" * 70)

    print(f"\nCandidates:       {len(roads)}")
    print(f"Nearest OSM ID:   {nearest['osm_id']}")
    print(f"Nearest highway:  {nearest['highway']}")
    print(f"Nearest distance: {float(nearest['distance_m']):.2f} m")

    print("\nOutputs:")
    print(f"  Map:     {OUTPUT_MAP}")
    print(f"  GeoJSON: {OUTPUT_GEOJSON}")
    print(f"  Summary: {OUTPUT_JSON}")

    print("\nIMPORTANT:")
    print("  No road has been marked affected.")
    print("  No training label has been created.")
    print("  Manual/source-supported review is still required.")

    print("\nSTATUS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()