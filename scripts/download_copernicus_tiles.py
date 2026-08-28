#!/usr/bin/env python3
"""Identify Copernicus DEM 1x1 tiles needed for road centroids above 25N,
then download them to data/raw/terrain/copernicus/."""
import json
import time
import urllib.request
import pandas as pd
from pathlib import Path

TERRAIN = Path('data/processed/terrain/road_terrain_features.parquet')


def get_centroids_above_25n():
    import geopandas as gpd
    terrain = pd.read_parquet(TERRAIN)
    terrain['osm_id'] = terrain['osm_id'].astype('int64')
    roads = gpd.read_file('data/processed/roads/ner_roads_districts.gpkg')
    roads['osm_id'] = roads['osm_id'].astype('int64')
    centroids = roads.set_index('osm_id').geometry.centroid
    pts = pd.DataFrame({
        'osm_id': [int(i) for i in centroids.index],
        'lat': [c.y for c in centroids],
        'lon': [c.x for c in centroids],
    })
    # Only roads present in the training terrain file
    pts = pts[pts['osm_id'].isin(terrain['osm_id'])]
    high = pts[pts['lat'] > 25.0]
    print(f'Roads above 25N (in training set): {len(high)}')
    return high


def main():
    out_dir = Path('data/raw/terrain/copernicus')
    out_dir.mkdir(parents=True, exist_ok=True)

    high = get_centroids_above_25n()
    if len(high) == 0:
        print('No roads above 25N found')
        return

    # Determine needed 1x1 tiles (lat floor N, lon floor E)
    tiles = set()
    for _, r in high.iterrows():
        lat = int(r['lat'])
        lon = int(r['lon'])
        tiles.add((lat, lon))

    print(f'Need {len(tiles)} 1x1 Copernicus tiles')
    for (lat, lon) in sorted(tiles):
        name = f'Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM'
        tif = f'{name}/{name}.tif'
        out = out_dir / f'{name}.tif'
        if out.exists():
            print(f'  {lat}N {lon}E: already have')
            continue
        url = f'https://copernicus-dem-30m.s3.amazonaws.com/{tif}'
        print(f'  Downloading {lat}N {lon}E ...', flush=True)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NER-Nav/1.0'})
            with urllib.request.urlopen(req, timeout=120) as resp, open(out, 'wb') as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            print(f'    saved {out.stat().st_size / (1024*1024):.1f} MB')
        except Exception as e:
            print(f'    FAILED: {type(e).__name__}: {e}')
        time.sleep(0.2)

    print('\nDone.')


if __name__ == '__main__':
    main()