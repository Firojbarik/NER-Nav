#!/usr/bin/env python3
"""Find the nearest OSM road ways for a given NH ref near a lat/lon in the
local NER road network (data/processed/roads/ner_roads.gpkg).

Usage:
    python scripts/probe_road_by_location.py --ref NH10 --lat 27.04 --lon 88.47
"""
from __future__ import annotations

import argparse

import geopandas as gpd

GPKG = "data/processed/roads/ner_roads.gpkg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--radius-km", type=float, default=30.0)
    args = ap.parse_args()

    roads = gpd.read_file(GPKG)
    sub = roads[roads["ref"].astype(str).str.contains(re_escape(args.ref), case=False, regex=False)]
    if sub.empty:
        print(f"No ways with ref matching '{args.ref}' in network")
        return 0

    sub = sub.copy()
    sub["cy"] = sub.geometry.centroid.y
    sub["cx"] = sub.geometry.centroid.x
    # approx distance in km (equirectangular)
    sub["dkm"] = (((sub["cy"] - args.lat) ** 2 + ((sub["cx"] - args.lon) * 0.72) ** 2) ** 0.5) * 111.0
    sub = sub[(sub["dkm"] <= args.radius_km)].nsmallest(args.k, "dkm")
    cols = [c for c in ["osm_id", "ref", "highway", "name", "dkm", "cy", "cx"] if c in sub.columns]
    print(sub[cols].to_string(index=False))
    return 0


def re_escape(s: str) -> str:
    import re
    return re.escape(s)


if __name__ == "__main__":
    raise SystemExit(main())
