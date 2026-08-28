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
from rasterio.warp import Resampling

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


def sample_terrain(roads_gdf, dem_path):
    with rasterio.open(dem_path) as dem:
        elev_arr = dem.read(1)
        # nodata mask
        nodata = dem.nodata
        # np.gradient on full array with proper cell size
        # get approximate cell size in degrees
        cell_deg_x = abs(dem.transform[0])
        cell_deg_y = abs(dem.transform[4])
        # convert to metres (~111000 * cos(lat)); average ~111320
        avg_lat = (dem.bounds.top + dem.bounds.bottom) / 2
        m_per_deg = 111320 * np.cos(np.radians(avg_lat))
        dy_deg, dx_deg = np.gradient(elev_arr, cell_deg_y, cell_deg_x)
        slope = np.degrees(np.arctan(np.sqrt((dx_deg * m_per_deg)**2 + (dy_deg * m_per_deg)**2) / (m_per_deg * 1)))

        elevs, slopes = [], []
        for geom in roads_gdf.geometry.centroid:
            x, y = geom.x, geom.y
            try:
                row, col = dem.index(x, y)
                if 0 <= row < elev_arr.shape[0] and 0 <= col < elev_arr.shape[1]:
                    ev = elev_arr[row, col]
                    if nodata is not None and ev == nodata:
                        ev = np.nan
                    elevs.append(ev)
                    sl = slope[row, col]
                    if np.isnan(ev):
                        sl = np.nan
                    slopes.append(sl)
                else:
                    elevs.append(np.nan); slopes.append(np.nan)
            except Exception:
                elevs.append(np.nan); slopes.append(np.nan)
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