#!/usr/bin/env python3
"""
Build a complete NER-consistent DEM mosaic from SRTM (<=25N) + Copernicus (>25N),
then recompute real elevation & slope for the training-set road centroids.
Replaces open-elevation point estimates with real raster values.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRTM_DIR = PROJECT_ROOT / "data/raw/terrain/srtm"
COP_DIR = PROJECT_ROOT / "data/raw/terrain/copernicus"
OUT_DIR = PROJECT_ROOT / "data/processed/terrain"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEM_FULL = OUT_DIR / "ner_dem_full.tif"
TERRAIN_TRAIN = OUT_DIR / "road_terrain_features_train.parquet"


def build_full_mosaic():
    if DEM_FULL.exists():
        print(f"Mosaic exists: {DEM_FULL}")
        return DEM_FULL

    srtm = sorted(SRTM_DIR.glob("srtm_*.tif"))
    cop = sorted(COP_DIR.glob("Copernicus_DSM_COG_10_*.tif"))
    print(f"SRTM tiles: {len(srtm)}, Copernicus tiles: {len(cop)}")

    srcs = [rasterio.open(f) for f in srtm + cop]
    mosaic, transform = merge(srcs)
    meta = srcs[0].meta.copy()
    meta.update(driver="GTiff", height=mosaic.shape[1], width=mosaic.shape[2],
                transform=transform, compress="lzw", bigtiff="IF_SAFER")
    with rasterio.open(DEM_FULL, "w", **meta) as dst:
        dst.write(mosaic)
    for s in srcs:
        s.close()
    with rasterio.open(DEM_FULL) as s:
        print(f"Mosaic bounds: {s.bounds}, shape: {s.shape}")
    print(f"Saved: {DEM_FULL} ({DEM_FULL.stat().st_size/(1024*1024):.0f} MB)")
    return DEM_FULL


def _slope_from_geographic_dem(elevation, transform, row_offset=0):
    """Calculate slope from a geographic-CRS DEM in degrees.

    ``np.gradient`` returns elevation change per degree when the DEM is in
    EPSG:4326.  Treating that value as elevation change per metre saturates
    the result at 90 degrees.  Convert each row's longitude degree spacing to
    metres before calculating the slope, and preserve DEM no-data as NaN.
    """
    values = np.ma.asarray(elevation, dtype="float32")
    filled = values.filled(np.nan)
    if filled.shape[0] < 2 or filled.shape[1] < 2:
        return np.full(filled.shape, np.nan, dtype="float32")

    pixel_x_deg = abs(float(transform.a))
    pixel_y_deg = abs(float(transform.e))
    rows = np.arange(row_offset, row_offset + filled.shape[0], dtype=float)
    lat = transform.f + (rows + 0.5) * transform.e
    x_m_per_deg = 111320.0 * np.cos(np.radians(lat))
    y_m_per_deg = 110574.0

    dz_dy_deg, dz_dx_deg = np.gradient(filled, pixel_y_deg, pixel_x_deg)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz_dx_m = dz_dx_deg / x_m_per_deg[:, None]
        dz_dy_m = dz_dy_deg / y_m_per_deg
        slope = np.degrees(np.arctan(np.hypot(dz_dx_m, dz_dy_m)))
    if np.ma.isMaskedArray(values):
        slope[np.ma.getmaskarray(values)] = np.nan
    return slope.astype("float32")


def sample_terrain(roads_gdf, dem_path, chunk_rows=512):
    with rasterio.open(dem_path) as dem:
        centroids = roads_gdf.geometry.centroid
        locations = []
        for geom in centroids:
            try:
                row, col = dem.index(geom.x, geom.y)
            except Exception:
                row, col = -1, -1
            locations.append((row, col))

        elevs = np.full(len(locations), np.nan, dtype="float32")
        slopes = np.full(len(locations), np.nan, dtype="float32")
        height, width = dem.height, dem.width
        valid = [(i, row, col) for i, (row, col) in enumerate(locations)
                 if 0 <= row < height and 0 <= col < width]

        # Process narrow horizontal strips so the 18,000 x 18,000 DEM does
        # not require several gigabytes for full-frame gradients.
        for core_start in range(0, height, chunk_rows):
            core_end = min(core_start + chunk_rows, height)
            wanted = [(i, row, col) for i, row, col in valid
                      if core_start <= row < core_end]
            if not wanted:
                continue
            read_start = max(0, core_start - 1)
            read_end = min(height, core_end + 1)
            window = Window(0, read_start, width, read_end - read_start)
            values = np.ma.asarray(
                dem.read(1, window=window, masked=True), dtype="float32")
            elevation_values = values.filled(np.nan)
            slope = _slope_from_geographic_dem(
                values, dem.window_transform(window))
            for i, row, col in wanted:
                local_row = row - read_start
                elevs[i] = elevation_values[local_row, col]
                slopes[i] = slope[local_row, col]

    roads_gdf = roads_gdf.copy()
    roads_gdf["elevation_m"] = elevs
    roads_gdf["slope_degrees"] = slopes
    return roads_gdf


def main():
    dem = build_full_mosaic()

    # Load training terrain file (osm_ids we care about)
    train_terrain = pd.read_parquet(OUT_DIR / "road_terrain_features.parquet")
    train_terrain["osm_id"] = train_terrain["osm_id"].astype("int64")

    # Load roads with geometry
    roads = gpd.read_file(PROJECT_ROOT / "data/processed/roads/ner_roads_districts.gpkg")
    roads["osm_id"] = roads["osm_id"].astype("int64")
    train_ids = set(train_terrain["osm_id"])
    relevant = roads[roads["osm_id"].isin(train_ids)].copy()

    sampled = sample_terrain(relevant, dem)

    # Merge back into the terrain file, only overwriting where we have new raster values
    merge_map = sampled.set_index("osm_id")[["elevation_m", "slope_degrees"]]
    merged = train_terrain.set_index("osm_id")
    merged["elevation_m"] = merge_map["elevation_m"].reindex(merged.index)
    merged["slope_degrees"] = merge_map["slope_degrees"].reindex(merged.index)
    merged = merged.reset_index()

    merged.to_parquet(OUT_DIR / "road_terrain_features.parquet", index=False)
    print(f"\nSaved updated terrain features: {OUT_DIR / 'road_terrain_features.parquet'}")
    print(f"NaN elevation: {merged['elevation_m'].isna().sum()}")
    print(f"NaN slope: {merged['slope_degrees'].isna().sum()}")
    print(f"Elevation min/max: {merged['elevation_m'].min():.0f}/{merged['elevation_m'].max():.0f}")
    print(f"Slope min/max: {merged['slope_degrees'].min():.1f}/{merged['slope_degrees'].max():.1f}")


if __name__ == "__main__":
    main()
