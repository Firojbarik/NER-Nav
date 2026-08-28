#!/usr/bin/env python3
"""Query open-elevation API for missing road elevations and estimate slope."""
import json
import time
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
import numpy as np

DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
ROADS_FILE = Path("data/processed/roads/ner_roads_districts.gpkg")
TERRAIN_FILE = Path("data/processed/terrain/road_terrain_features.parquet")

# Find roads with NaN terrain in our dataset
ds = pd.read_parquet(DATASET)
terrain_nan = ds[ds['elevation_m'].isna()]
unique_osm_nan = terrain_nan['osm_id'].unique()
print(f'Unique osm_ids with NaN terrain in dataset: {len(unique_osm_nan)}')

# Get their centroids from road file
roads = gpd.read_file(ROADS_FILE)
roads['osm_id'] = roads['osm_id'].astype('int64')
missing_roads = roads[roads['osm_id'].isin(unique_osm_nan)]
print(f'Found {len(missing_roads)} missing roads in road file')

# Get centroids
centroids = missing_roads.geometry.centroid
locations = []
for idx, row in missing_roads.iterrows():
    c = centroids.loc[idx]
    locations.append({
        "osm_id": int(row["osm_id"]),
        "ref": row.get("ref", ""),
        "state": row.get("state", ""),
        "latitude": float(c.y),
        "longitude": float(c.x)
    })
    print(f'  osm_id={row["osm_id"]}, ref={row.get("ref","")}, state={row.get("state","")}, lat={c.y:.4f}, lon={c.x:.4f}')

# Query open-elevation API in batches
def query_elevation(locations_batch):
    data = json.dumps({"locations": locations_batch}).encode()
    req = urllib.request.Request(
        'https://api.open-elevation.com/api/v1/lookup',
        data=data,
        headers={'Content-Type': 'application/json', 'User-Agent': 'NER-Nav/1.0'}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f'API error: {e}')
        return None

# Process in batches of 50 (API limit)
batch_size = 50
all_results = []
for i in range(0, len(locations), batch_size):
    batch = locations[i:i+batch_size]
    print(f'Querying batch {i//batch_size + 1}/{(len(locations)-1)//batch_size + 1}...')
    result = query_elevation(batch)
    if result and 'results' in result:
        all_results.extend(result['results'])
    time.sleep(0.5)  # Be nice to the API

# Map results back to osm_ids
elevation_map = {}
for loc, res in zip(locations, all_results):
    elevation_map[loc['osm_id']] = res.get('elevation')

# Now estimate slope by querying nearby points (±0.001 deg ~ 100m)
print('\nEstimating slope from local gradients...')
slope_map = {}
for loc in locations:
    osm_id = loc['osm_id']
    lat, lon = loc['latitude'], loc['longitude']
    # Query 4 neighbors: N, S, E, W (~100m offset)
    neighbors = [
        {"latitude": lat + 0.001, "longitude": lon},    # North
        {"latitude": lat - 0.001, "longitude": lon},    # South
        {"latitude": lat, "longitude": lon + 0.001},    # East
        {"latitude": lat, "longitude": lon - 0.001},    # West
    ]
    result = query_elevation(neighbors)
    if result and 'results' in result:
        elevs = [r.get('elevation') for r in result['results'] if r.get('elevation') is not None]
        if len(elevs) >= 2:
            # Simple slope estimate: max elevation difference / distance
            max_diff = max(elevs) - min(elevs)
            # ~100m per 0.001 degree
            slope_deg = np.degrees(np.arctan(max_diff / 100.0))
            slope_map[loc['osm_id']] = slope_deg
        else:
            slope_map[osm_id] = np.nan
    else:
        slope_map[osm_id] = np.nan
    time.sleep(0.5)

# Load existing terrain data and update
terrain = pd.read_parquet(TERRAIN_FILE)
terrain['osm_id'] = terrain['osm_id'].astype('int64')

updated = 0
for osm_id, elev in elevation_map.items():
    if elev is not None:
        mask = terrain['osm_id'] == osm_id
        if mask.any():
            terrain.loc[mask, 'elevation_m'] = elev
            terrain.loc[mask, 'slope_degrees'] = slope_map.get(osm_id, np.nan)
            updated += 1

print(f'\nUpdated {updated} road segments with elevation/slope')

# Save updated terrain
terrain.to_parquet(TERRAIN_FILE, index=False)
print(f'Saved updated terrain to {TERRAIN_FILE}')

# Verify
print('\nVerification:')
terrain_check = pd.read_parquet(TERRAIN_FILE)
print(f'Total roads: {len(terrain_check)}')
print(f'NaN elevation: {terrain_check["elevation_m"].isna().sum()}')
print(f'NaN slope: {terrain_check["slope_degrees"].isna().sum()}')