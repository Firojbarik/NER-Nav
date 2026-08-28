# DEM/Terrain Data Acquisition & Feature Engineering Guide

**Project:** NER-Nav  
**Purpose:** Guide acquisition and processing of elevation/slope data for landslide susceptibility modeling  
**Status:** Ingestion Plan - P0 Priority

---

## 1. Requirements

Elevation and slope are critical predictors for landslide and terrain-related road disruptions.

**Target Region:** North Eastern Region (8 states)  
**Coordinate Coverage:**  
- Latitude: 21.5°N to 29.5°N  
- Longitude: 89.5°E to 97.5°E

**Required Resolution:** 30m (1 arc-second) minimum

---

## 2. Data Source Options

### Option A: SRTM (Shuttle Radar Topography Mission) - RECOMMENDED
* **Provider:** NASA/USGS
* **URL:** https://earthexplorer.usgs.gov/
* **Resolution:** 30m (1 arc-second)
* **Coverage:** Global
* **License:** Public Domain (no restrictions)
* **Format:** GeoTIFF (.tif)
* **Decision Category:** B (Authoritative Scientific Source)

### Option B: Bhuvan CartoDEM
* **Provider:** NRSC/ISRO (Bhuvan)
* **URL:** https://bhuvan.nrsc.gov.in/
* **Resolution:** 30m
* **Coverage:** India
* **License:** Free for research/development (verify commercial use)
* **Format:** GeoTIFF
* **Decision Category:** A (Government Source)

**Recommendation:** Use SRTM for unrestricted licensing, or CartoDEM for potentially higher accuracy in Indian terrain.

---

## 3. Required Features

For each road segment, derive:

| Feature | Unit | Calculation | Purpose |
|---------|------|-------------|---------|
| `elevation_m` | meters | DEM value at road centroid | Base elevation |
| `slope_degrees` | degrees | Gradient calculation | Landslide susceptibility |
| `elevation_range_100m` | meters | Max-min elevation within 100m buffer | Local terrain roughness |
| `aspect_category` | categorical | N/NE/E/SE/S/SW/W/NW | Exposure direction |

---

## 4. Processing Pipeline

```
SRTM GeoTIFF tiles
    ↓
Mosaic tiles covering NER
    ↓
Reproject to EPSG:4326 (if needed)
    ↓
Extract elevation at road centroids
    ↓
Calculate slope using gradient
    ↓
Calculate aspect
    ↓
Join to road segments (osm_id)
    ↓
Save to data/processed/ml/terrain_features.parquet
```

---

## 5. Implementation

**File:** `scripts/process_terrain_features.py`

```python
import rasterio
import geopandas as gpd
import numpy as np
from pathlib import Path

def calculate_slope(elevation_array, cell_size_m):
    """Calculate slope in degrees from elevation raster."""
    dy, dx = np.gradient(elevation_array, cell_size_m)
    slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_degrees = np.degrees(slope_radians)
    return slope_degrees

def extract_features(dem_path, roads_gdf):
    """Extract elevation and slope for road segments."""
    with rasterio.open(dem_path) as src:
        # Get elevation at road centroids
        centroids = roads_gdf.geometry.centroid
        coords = [(x, y) for x, y in zip(centroids.x, centroids.y)]
        elevations = [x[0] for x in src.sample(coords)]
        
        # Calculate slope
        elevation_array = src.read(1)
        slope_array = calculate_slope(elevation_array, 30)  # 30m resolution
        slopes = [x[0] for x in src.sample(coords)]
    
    roads_gdf['elevation_m'] = elevations
    roads_gdf['slope_degrees'] = slopes
    
    return roads_gdf
```

---

## 6. Validation

Before use, validate:
- No negative elevations (except valid sea-level areas)
- Slope values: 0° ≤ slope ≤ 90°
- No missing values for road segments in NER region
- CRS matches road network (EPSG:4326)

---

## 7. Next Actions

1. Download SRTM tiles covering NER from USGS EarthExplorer
2. Store in `data/raw/terrain/srtm/`
3. Mosaic tiles into single GeoTIFF
4. Run extraction script
5. Validate feature ranges
6. Document provenance in DATA_SOURCES.md

**Estimated effort:** 4-6 hours (including download time)

---

**End of DEM Acquisition Guide**
