#!/usr/bin/env python3
"""
NER-Nav: SRTM DEM Data Download Script

Downloads SRTM 30m elevation tiles for NER region
SRTM: Shuttle Radar Topography Mission
Resolution: 30m (1 arc-second)
License: Public Domain
"""

import os
import sys
from pathlib import Path
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TERRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "terrain" / "srtm"
RAW_TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

# NER Region bounding box
NER_BBOX = {
    "lat_min": 21.5,
    "lat_max": 29.5,
    "lon_min": 89.5,
    "lon_max": 97.5
}

# SRTM tiles are organized as 1°x1° tiles
# Tile naming: N##E### or S##W### format
# For NER we need tiles from N21E089 to N29E097

# SRTM via CGIAR-CSI (public access, no authentication required)
# Alternative: USGS EarthExplorer requires account
SRTM_BASE_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/"


def generate_tile_list():
    """Generate list of SRTM 5x5 degree tiles covering NER region."""

    # CGIAR-CSI provides SRTM in 5°x5° tiles
    # Tile naming: srtm_XX_YY where XX is column, YY is row
    # We need to map lat/lon to tile indices

    tiles = []

    # SRTM tiles start at -180°, 60°N and go 5° at a time
    # Column (X): floor((lon + 180) / 5) + 1
    # Row (Y): floor((60 - lat) / 5) + 1

    # For NER region: 21.5-29.5°N, 89.5-97.5°E
    # We need tiles covering this range

    lon_min = int(NER_BBOX["lon_min"] / 5) * 5
    lon_max = int(NER_BBOX["lon_max"] / 5) * 5 + 5
    lat_min = int(NER_BBOX["lat_min"] / 5) * 5
    lat_max = int(NER_BBOX["lat_max"] / 5) * 5 + 5

    for lon in range(lon_min, lon_max, 5):
        for lat in range(lat_min, lat_max, 5):
            col = int((lon + 180) / 5) + 1
            row = int((60 - lat) / 5) + 1
            tile_name = f"srtm_{col:02d}_{row:02d}"
            tiles.append(tile_name)

    return tiles


def download_srtm_tile(tile_name):
    """Download a single SRTM 5x5 degree tile."""

    filename = f"{tile_name}.zip"
    url = f"{SRTM_BASE_URL}{filename}"
    output_path = RAW_TERRAIN_DIR / filename

    # Skip if already downloaded
    if output_path.exists():
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  {tile_name}: SKIP (already exists, {file_size_mb:.1f} MB)")
        return "skipped"

    # Download
    try:
        print(f"  {tile_name}: Downloading...", end="", flush=True)
        urllib.request.urlretrieve(url, output_path)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f" OK ({file_size_mb:.1f} MB)")
        return "downloaded"

    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f" SKIP (tile not available)")
            return "not_available"
        else:
            print(f" FAILED (HTTP {e.code})")
            return "failed"

    except Exception as e:
        print(f" FAILED ({str(e)})")
        return "failed"


def main():
    print("="*70)
    print("SRTM DEM Data Download for NER-Nav")
    print("="*70)
    print()
    print("Target region:")
    print(f"  Latitude:  {NER_BBOX['lat_min']}°N to {NER_BBOX['lat_max']}°N")
    print(f"  Longitude: {NER_BBOX['lon_min']}°E to {NER_BBOX['lon_max']}°E")
    print()
    print("Data source: SRTM v4.1 (CGIAR-CSI)")
    print("Resolution: 90m (3 arc-second)")
    print("License: Public Domain")
    print(f"Output directory: {RAW_TERRAIN_DIR}")
    print()

    # Generate tile list
    tiles = generate_tile_list()
    print(f"Required tiles: {len(tiles)}")
    print(f"Tile coverage: {tiles[0]} to {tiles[-1]}")
    print()

    # Download tiles
    print("="*70)
    print("DOWNLOADING TILES")
    print("="*70)
    print()

    stats = {
        "downloaded": 0,
        "skipped": 0,
        "not_available": 0,
        "failed": 0
    }

    for tile_name in tiles:
        result = download_srtm_tile(tile_name)
        stats[result] += 1

    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Downloaded: {stats['downloaded']} tiles")
    print(f"Skipped (already exists): {stats['skipped']} tiles")
    print(f"Not available (water): {stats['not_available']} tiles")
    print(f"Failed: {stats['failed']} tiles")
    print()

    total_tiles = stats['downloaded'] + stats['skipped']
    print(f"Total tiles available: {total_tiles}/{len(tiles)}")
    print()

    if stats['failed'] > 0:
        print("WARNING: Some downloads failed.")
        print("You can re-run this script to retry.")
        print()

    print("NEXT STEPS:")
    print("1. Extract ZIP files: unzip data/raw/terrain/srtm/*.zip")
    print("2. Mosaic tiles: gdal_merge.py -o ner_dem.tif data/raw/terrain/srtm/*.tif")
    print("3. Extract terrain features: python scripts/process_terrain_features.py")
    print()
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        sys.exit(1)
