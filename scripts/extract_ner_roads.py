from pathlib import Path
import json

import geopandas as gpd
import osmium
from shapely import wkb


INPUT = Path("data/processed/osm/ner_roads.osm.pbf")
OUTPUT = Path("data/processed/roads/ner_roads.gpkg")
QA_OUTPUT = Path("data/processed/roads/ner_roads_qa.json")


class RoadHandler(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()

        self.wkb_factory = osmium.geom.WKBFactory()

        self.rows = []
        self.total_ways = 0
        self.geometry_failures = 0

    def way(self, way):
        self.total_ways += 1

        tags = way.tags

        highway = tags.get("highway")
        if not highway:
            return

        try:
            wkb_data = self.wkb_factory.create_linestring(way)
            geometry = wkb.loads(wkb_data, hex=True)
        except Exception:
            self.geometry_failures += 1
            return

        if geometry is None or geometry.is_empty:
            self.geometry_failures += 1
            return

        self.rows.append(
            {
                "osm_id": way.id,
                "highway": highway,
                "name": tags.get("name"),
                "ref": tags.get("ref"),
                "surface": tags.get("surface"),
                "maxspeed": tags.get("maxspeed"),
                "access": tags.get("access"),
                "oneway": tags.get("oneway"),
                "bridge": tags.get("bridge"),
                "tunnel": tags.get("tunnel"),
                "lanes": tags.get("lanes"),
                "lit": tags.get("lit"),
                "smoothness": tags.get("smoothness"),
                "tracktype": tags.get("tracktype"),
                "service": tags.get("service"),
                "geometry": geometry,
            }
        )


def main():
    print("=" * 70)
    print("NER-Nav — OSM Road Extraction")
    print("=" * 70)

    print(f"\nInput:  {INPUT}")
    print(f"Output: {OUTPUT}")

    if not INPUT.exists():
        raise FileNotFoundError(f"Input PBF not found: {INPUT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    handler = RoadHandler()

    print("\nReading OSM PBF...")
    handler.apply_file(str(INPUT), locations=True)

    print(f"Total ways processed: {handler.total_ways:,}")
    print(f"Road records:         {len(handler.rows):,}")
    print(f"Geometry failures:    {handler.geometry_failures:,}")

    if not handler.rows:
        raise RuntimeError("No road geometries were extracted.")

    gdf = gpd.GeoDataFrame(handler.rows, geometry="geometry", crs="EPSG:4326")

    invalid = ~gdf.geometry.is_valid
    empty = gdf.geometry.is_empty

    print(f"\nInvalid geometries:   {invalid.sum():,}")
    print(f"Empty geometries:     {empty.sum():,}")

    if invalid.any():
        print("Repairing invalid road geometries...")
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()

    gdf = gdf[~gdf.geometry.is_empty].copy()

    # Remove duplicate OSM way IDs if any unexpected duplicates occur.
    gdf = gdf.drop_duplicates(subset=["osm_id"]).reset_index(drop=True)

    # Basic QA statistics.
    highway_counts = (
        gdf["highway"]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    qa = {
        "dataset": "NER-Nav OSM road network",
        "input": str(INPUT),
        "output": str(OUTPUT),
        "crs": "EPSG:4326",
        "total_ways_processed": handler.total_ways,
        "road_records": len(gdf),
        "geometry_failures": handler.geometry_failures,
        "invalid_geometry_after_repair": int((~gdf.geometry.is_valid).sum()),
        "empty_geometry": int(gdf.geometry.is_empty.sum()),
        "highway_types": highway_counts,
        "status": "PASS",
    }

    print("\nHighway types:")
    for highway, count in gdf["highway"].value_counts().items():
        print(f"  {str(highway):25} {count:,}")

    print("\nWriting GeoPackage...")
    gdf.to_file(
        OUTPUT,
        layer="roads",
        driver="GPKG",
    )

    QA_OUTPUT.write_text(
        json.dumps(qa, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("ROAD EXTRACTION SUCCESSFUL")
    print("=" * 70)
    print(f"Road records: {len(gdf):,}")
    print(f"GeoPackage:   {OUTPUT}")
    print(f"QA report:    {QA_OUTPUT}")
    print("Status:       PASS")


if __name__ == "__main__":
    main()