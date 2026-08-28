#!/usr/bin/env python3
"""Estimate slope for roads with elevation but missing slope using open-elevation API.

Queries 4 cardinal neighbors (~500m) around each road centroid to estimate slope.
"""
import json
import time
import sys
import urllib.request
import numpy as np
import pandas as pd
from pathlib import Path

TERRAIN = Path('data/processed/terrain/road_terrain_features.parquet')
ROADS = Path('data/processed/roads/ner_roads_districts.gpkg')


def api_lookup(locations, retries=5):
    """Query open-elevation API for a batch of locations."""
    payload = json.dumps({"locations": locations}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                'https://api.open-elevation.com/api/v1/lookup',
                data=payload,
                headers={'Content-Type': 'application/json'},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())['results']
        except Exception as e:
            time.sleep(1.0 * (attempt + 1))
    return None


def main():
    terrain = pd.read_parquet(TERRAIN)
    terrain['osm_id'] = terrain['osm_id'].astype('int64')

    need_slope = terrain[terrain['elevation_m'].notna() & terrain['slope_degrees'].isna()]
    print(f'Need slope for {len(need_slope)} roads')

    if len(need_slope) == 0:
        print('Nothing to do')
        return

    import geopandas as gpd
    roads = gpd.read_file(ROADS)
    roads['osm_id'] = roads['osm_id'].astype('int64')
    road_centroids = roads.set_index('osm_id').geometry.centroid

    # Group osm_ids, batch query all neighbors
    BATCH = 4  # 4 neighbors per road
    updated = 0
    failed = 0
    osm_list = [int(o) for o in need_slope['osm_id']]

    for i in range(0, len(osm_list), 5):  # 5 roads = 20 lookups per batch
        osm_batch = osm_list[i:i + 5]
        locations = []
        for osm_id in osm_batch:
            c = road_centroids.get(osm_id)
            if c is None:
                continue
            lat, lon = c.y, c.x
            # ~555m at 25N = 0.005 deg
            d = 0.005
            locations.extend([
                {"latitude": lat + d, "longitude": lon},
                {"latitude": lat - d, "longitude": lon},
                {"latitude": lat, "longitude": lon + d},
                {"latitude": lat, "longitude": lon - d},
            ])

        results = api_lookup(locations)
        if results is None:
            failed += len(osm_batch)
            print(f'  batch {i // 25}: FAILED')
            continue

        # Distribute results back to roads (4 per road)
        idx = 0
        for osm_id in osm_batch:
            c = road_centroids.get(osm_id)
            if c is None:
                continue
            elevs = [r.get('elevation') for r in results[idx:idx + 4] if r.get('elevation') is not None]
            idx += 4
            if len(elevs) >= 2:
                max_diff = max(elevs) - min(elevs)
                # horizontal distance ~110m per 0.001 deg at equator, ~100m at 25N
                slope_deg = np.degrees(np.arctan(max_diff / 100.0))
                mask = terrain['osm_id'] == osm_id
                terrain.loc[mask, 'slope_degrees'] = slope_deg
                updated += 1
        print(f'  batch {i // 5}: updated {updated} total so far')
        time.sleep(0.3)

    print(f'Completed: updated={updated}, failed={failed}')
    terrain.to_parquet(TERRAIN, index=False)
    print('Saved')


if __name__ == '__main__':
    main()