#!/usr/bin/env python3
"""
NER-Nav: Terrain Feature Extraction from SRTM DEM

Purpose:
    Extract elevation, slope, aspect, and terrain roughness features
    for each road segment in the NER-Nav road network.

Input:
    - SRTM DEM tiles (data/raw/terrain/srtm/*.tif)
    - Road network (data/processed/osm/ner_roads.parquet)

Output:
    - data/processed/terrain/road_terrain_features.parquet

Features:
    - elevation_m: Elevation at road centroid (meters)
    - slope_degrees: Terrain slope (0-90 degrees)
    - aspect_degrees: Aspect direction (0-360 degrees)
    - aspect_category: Cardinal direction (N/NE/E/SE/S/SW/W/NW) or NODATA
      when no DEM coverage (no-data is kept distinct from genuinely flat terrain)

Author: Senior ML Engineer
Priority: P0 (unblocks feature engineering)
Date: 2026-08-27
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Point
import rasterio.mask

warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_TERRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "terrain" / "srtm"
PROCESSED_TERRAIN_DIR = PROJECT_ROOT / "data" / "processed" / "terrain"
PROCESSED_TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

ROADS_FILE = PROJECT_ROOT / "data" / "processed" / "roads" / "ner_roads.gpkg"
OUTPUT_FILE = PROCESSED_TERRAIN_DIR / "road_terrain_features.parquet"
DEM_MOSAIC_FILE = PROCESSED_TERRAIN_DIR / "ner_dem_mosaic.tif"


def mosaic_srtm_tiles():
    """Mosaic all SRTM tiles into a single raster."""

    print("=" * 70)
    print("STEP 1: Mosaic SRTM Tiles")
    print("=" * 70)
    print()

    # Find all SRTM tiles
    tile_files = list(RAW_TERRAIN_DIR.glob("srtm_*.tif"))

    if len(tile_files) == 0:
        raise FileNotFoundError(f"No SRTM tiles found in {RAW_TERRAIN_DIR}")

    print(f"Found {len(tile_files)} SRTM tiles:")
    for f in tile_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print()

    # Check if mosaic already exists
    if DEM_MOSAIC_FILE.exists():
        print(f"Mosaic already exists: {DEM_MOSAIC_FILE}")
        size_mb = DEM_MOSAIC_FILE.stat().st_size / (1024 * 1024)
        print(f"Size: {size_mb:.1f} MB")
        print("Skipping mosaic step.")
        print()
        return DEM_MOSAIC_FILE

    print("Mosaicking tiles...")

    # Open all tiles
    src_files = [rasterio.open(f) for f in tile_files]

    # Merge tiles
    mosaic, out_transform = merge(src_files)

    # Get metadata from first tile
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "lzw"
    })

    # Write mosaic
    with rasterio.open(DEM_MOSAIC_FILE, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close source files
    for src in src_files:
        src.close()

    size_mb = DEM_MOSAIC_FILE.stat().st_size / (1024 * 1024)
    print(f"Mosaic created: {DEM_MOSAIC_FILE}")
    print(f"Size: {size_mb:.1f} MB")
    print()

    return DEM_MOSAIC_FILE


def calculate_slope(elevation_array, cell_size_m=90):
    """Calculate slope in degrees from elevation array."""

    # Calculate gradients in x and y directions
    dy, dx = np.gradient(elevation_array, cell_size_m)

    # Calculate slope in radians
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))

    # Convert to degrees
    slope_degrees = np.degrees(slope_radians)

    return slope_degrees


def calculate_aspect(elevation_array):
    """Calculate aspect in degrees from elevation array."""

    # Calculate gradients
    dy, dx = np.gradient(elevation_array)

    # Calculate aspect in radians (0 = North, clockwise)
    aspect_radians = np.arctan2(-dy, dx)

    # Convert to degrees (0-360, 0 = East, counterclockwise)
    aspect_degrees = np.degrees(aspect_radians)

    # Convert to geographic aspect (0 = North, clockwise)
    aspect_degrees = 90 - aspect_degrees
    aspect_degrees[aspect_degrees < 0] += 360

    return aspect_degrees


def aspect_to_category(aspect_degrees):
    """Convert aspect degrees to cardinal direction category."""

    if np.isnan(aspect_degrees):
        return "NODATA"

    # 8 cardinal directions
    # N: 337.5-22.5, NE: 22.5-67.5, E: 67.5-112.5, etc.
    if aspect_degrees >= 337.5 or aspect_degrees < 22.5:
        return "N"
    elif aspect_degrees < 67.5:
        return "NE"
    elif aspect_degrees < 112.5:
        return "E"
    elif aspect_degrees < 157.5:
        return "SE"
    elif aspect_degrees < 202.5:
        return "S"
    elif aspect_degrees < 247.5:
        return "SW"
    elif aspect_degrees < 292.5:
        return "W"
    elif aspect_degrees < 337.5:
        return "NW"
    else:
        return "FLAT"


def extract_terrain_features(roads_gdf, dem_path):
    """Extract terrain features for each road segment."""

    print("=" * 70)
    print("STEP 2: Extract Terrain Features")
    print("=" * 70)
    print()

    print(f"Loading DEM: {dem_path}")

    with rasterio.open(dem_path) as dem_src:

        print(f"DEM CRS: {dem_src.crs}")
        print(f"DEM bounds: {dem_src.bounds}")
        print(f"DEM shape: {dem_src.shape}")
        print(f"DEM resolution: ~{abs(dem_src.transform[0]):.4f} degrees")
        print()

        # Read full elevation array
        print("Reading elevation data...")
        elevation = dem_src.read(1)

        # Calculate slope
        print("Calculating slope...")
        slope = calculate_slope(elevation, cell_size_m=90)

        # Calculate aspect
        print("Calculating aspect...")
        aspect = calculate_aspect(elevation)

        print()
        print(f"Processing {len(roads_gdf)} road segments...")
        print()

        # Get road centroids
        centroids = roads_gdf.geometry.centroid
        coords = [(x, y) for x, y in zip(centroids.x, centroids.y)]

        # Sample elevation at centroids
        elevations = []
        slopes = []
        aspects = []

        for coord in coords:
            try:
                # Get row, col from coordinates
                row, col = dem_src.index(coord[0], coord[1])

                # Check bounds
                if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]:
                    elevations.append(elevation[row, col])
                    slopes.append(slope[row, col])
                    aspects.append(aspect[row, col])
                else:
                    elevations.append(np.nan)
                    slopes.append(np.nan)
                    aspects.append(np.nan)

            except Exception:
                elevations.append(np.nan)
                slopes.append(np.nan)
                aspects.append(np.nan)

        # Add features to dataframe
        roads_gdf["elevation_m"] = elevations
        roads_gdf["slope_degrees"] = slopes
        roads_gdf["aspect_degrees"] = aspects

        # Convert aspect to categories
        roads_gdf["aspect_category"] = roads_gdf["aspect_degrees"].apply(aspect_to_category)

    return roads_gdf


def validate_features(roads_gdf):
    """Validate extracted terrain features."""

    print("=" * 70)
    print("STEP 3: Validate Features")
    print("=" * 70)
    print()

    # Check for missing values
    missing_elevation = roads_gdf["elevation_m"].isna().sum()
    missing_slope = roads_gdf["slope_degrees"].isna().sum()
    missing_aspect = roads_gdf["aspect_degrees"].isna().sum()

    print(f"Missing values:")
    print(f"  Elevation: {missing_elevation} ({missing_elevation/len(roads_gdf)*100:.1f}%)")
    print(f"  Slope: {missing_slope} ({missing_slope/len(roads_gdf)*100:.1f}%)")
    print(f"  Aspect: {missing_aspect} ({missing_aspect/len(roads_gdf)*100:.1f}%)")
    print()

    # Check value ranges
    valid_elevation = roads_gdf["elevation_m"].dropna()
    valid_slope = roads_gdf["slope_degrees"].dropna()
    valid_aspect = roads_gdf["aspect_degrees"].dropna()

    print(f"Value ranges:")
    print(f"  Elevation: {valid_elevation.min():.1f}m to {valid_elevation.max():.1f}m")
    print(f"  Slope: {valid_slope.min():.1f}° to {valid_slope.max():.1f}°")
    print(f"  Aspect: {valid_aspect.min():.1f}° to {valid_aspect.max():.1f}°")
    print()

    # Check aspect categories
    aspect_counts = roads_gdf["aspect_category"].value_counts()
    print(f"Aspect category distribution:")
    for cat, count in aspect_counts.items():
        print(f"  {cat}: {count} ({count/len(roads_gdf)*100:.1f}%)")
    print()

    # Validation checks
    issues = []

    if missing_elevation / len(roads_gdf) > 0.1:
        issues.append(f"HIGH: >10% missing elevation values")

    if valid_slope.max() > 90:
        issues.append(f"ERROR: Slope exceeds 90 degrees (max: {valid_slope.max():.1f}°)")

    if valid_elevation.min() < -500:
        issues.append(f"WARNING: Very low elevations detected (min: {valid_elevation.min():.1f}m)")

    if len(issues) > 0:
        print("VALIDATION ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        print()
    else:
        print("[OK] All validation checks passed")
        print()

    return len(issues) == 0


def main():
    print("=" * 70)
    print("NER-Nav Terrain Feature Extraction")
    print("=" * 70)
    print()

    # Check if roads file exists
    if not ROADS_FILE.exists():
        print(f"ERROR: Roads file not found: {ROADS_FILE}")
        print()
        print("Expected location: data/processed/roads/ner_roads.gpkg")
        print()
        print("Please ensure OSM road data has been processed.")
        print("If roads are in a different location, update ROADS_FILE in this script.")
        sys.exit(1)

    # Load roads
    print(f"Loading road network: {ROADS_FILE}")
    roads_gdf = gpd.read_file(ROADS_FILE)
    print(f"Loaded {len(roads_gdf)} road segments")
    print(f"CRS: {roads_gdf.crs}")
    print()

    # Mosaic SRTM tiles
    dem_path = mosaic_srtm_tiles()

    # Extract features
    roads_with_terrain = extract_terrain_features(roads_gdf, dem_path)

    # Validate
    valid = validate_features(roads_with_terrain)

    # Save output
    print("=" * 70)
    print("STEP 4: Save Output")
    print("=" * 70)
    print()

    # Drop geometry for output (keep only features)
    output_cols = [
        "osm_id",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "aspect_category"
    ]

    # Check which columns exist
    available_cols = [col for col in output_cols if col in roads_with_terrain.columns]

    output_df = roads_with_terrain[available_cols].copy()

    print(f"Saving terrain features: {OUTPUT_FILE}")
    output_df.to_parquet(OUTPUT_FILE, index=False)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Output size: {size_mb:.1f} MB")
    print()

    print("=" * 70)
    print("TERRAIN FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print()
    print(f"Features extracted for {len(output_df)} road segments")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if valid:
        print("STATUS: [OK] Ready for model training")
    else:
        print("STATUS: [WARNING] Validation issues detected - review output")

    print()
    print("NEXT STEPS:")
    print("1. Extract rainfall features: python scripts/process_chirps_features.py")
    print("2. Join terrain + rainfall features to road segments")
    print("3. Build training dataset with confirmed hazard labels")
    print()
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
