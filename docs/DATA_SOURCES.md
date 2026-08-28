# NER-Nav Real Data Sources Registry

**Generated:** 2026-08-26  
**Auditor:** Senior ML Engineer / Data Architect  
**Project:** AI-Based Smart Logistics and Accessibility Intelligence Platform for NER

---

## Decision Matrix and Rules

For every dataset, the project enforces the following source classification rule:
- **A = Authoritative Government Source** (e.g., IMD, ISRO, NRSC, Survey of India, NESAC, GSI, National Data Portal or state PWDs)
- **B = Authoritative Scientific/Institutional Source** (e.g., academic journals, NASA, ESA, Copernicus, USGS)
- **C = Reputable Open Geospatial Source** (e.g., OpenStreetMap)
- **D = Third-Party Dataset** (e.g., Kaggle, GitHub repositories, personal blogs)

**Preference Hierarchy:**  
`A > B > C > D`

*If any Class D sources are proposed, their inclusion must be formally justified, and their licensing, spatial accuracy, and timestamp resolution verified.*

---

## 1. Road Network Data

### OpenStreetMap (OSM) NER Road Extract
* **Provider:** OpenStreetMap Foundation
* **Official URL:** [https://www.openstreetmap.org/](https://www.openstreetmap.org/)
* **Access Method:** Geofabrik download endpoint
* **API/File/WMS/WFS:** PBF format / extract file
* **Geographic Coverage:** North Eastern Region (Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura)
* **Temporal Coverage:** Current road geometry (updated dynamically by community)
* **Resolution:** Node coordinates in WGS84, road-layer complexity down to street level
* **Update Frequency:** Dynamically updated. Local snapshot downloaded on 2026-08-23
* **License:** Open Database License (ODbL) v1.0 (requires attribution)
* **Usage Restrictions:** Permitted for commercial and non-commercial project use; derived datasets must remain open under ODbL
* **Data Format:** GPKG / Parquet (processed)
* **Important Fields:** `osm_id` (stable identifier), `highway`, `geometry`, `surface`, `maxspeed`, `lanes`, `bridge`, `tunnel`
* **Known Limitations:** Attribute sparsity is very high in NER (e.g., `surface` is 93.3% missing, `maxspeed` is 99.7% missing). Geometric connectivity is generally good but minor rural roads may be incomplete.
* **Downloaded:** Yes
* **Processing Status:** ✅ COMPLETED. Extracted to `data/processed/roads/ner_roads.gpkg` and spatially joined with districts to `ner_roads_districts.gpkg`.
* **Used for Training:** Yes (as the base spatial network layer)
* **Reason:** Comprehensive open-source road network with global support and excellent GIS tools integration.
* **Decision Category:** **C (Reputable Open Geospatial Source)**

---

## 2. Administrative Boundaries Data

### Survey of India / OGD India District & State Boundaries
* **Provider:** Survey of India / Government of India Open Government Data (OGD)
* **Official URL:** [https://data.gov.in](https://data.gov.in)
* **Access Method:** Open portal download
* **API/File/WMS/WFS:** Shapefile (.shp) format
* **Geographic Coverage:** India (National)
* **Temporal Coverage:** 2021-2023 administrative boundaries
* **Resolution:** District-level and State-level boundaries
* **Update Frequency:** Annual or post-administrative changes
* **License:** Government Open Data License (GODL) - India
* **Usage Restrictions:** Attribution required; free redistribution and use permitted.
* **Data Format:** (.shp) original, GPKG (processed)
* **Important Fields:** `OBJECTID`, `STATE_UT`, `STATE_LGD`, `DISTRICT`, `DIST_LGD`, `geometry`
* **Known Limitations:** Boundary conflicts can occur due to newly declared districts not updated in the national dataset.
* **Downloaded:** Yes
* **Processing Status:** ✅ COMPLETED. Processed to `data/processed/admin_boundaries/ner_states.gpkg` and `ner_districts.gpkg`.
* **Used for Training:** Yes (for spatial indexing, grouping, and administrative features)
* **Reason:** Official admin boundary shapefiles from Survey of India.
* **Decision Category:** **A (Authoritative Government Source)**

---

## 3. Disruption Events & Disruption Target Data (Ground Truth)

### NRSC Sikkim Landslide Event (2016)
* **Provider:** National Remote Sensing Centre (NRSC), ISRO
* **Official URL:** [https://www.nrsc.gov.in/](https://www.nrsc.gov.in/)
* **Access Method:** Downloaded official technical report pdf
* **API/File/WMS/WFS:** Technical report (PDF) file
* **Geographic Coverage:** Mantam village region, North Sikkim
* **Temporal Coverage:** Event date: 2016-08-13
* **Resolution:** Single event point coordinate + impact narrative (13.30 IST)
* **Update Frequency:** Event-specific / static report
* **License:** Government of India usage terms. Use for research and development.
* **Usage Restrictions:** Permitted with attribution; commercial exploitation of raw reports restricted.
* **Data Format:** PDF (original), JSON (processed)
* **Important Fields:** Event details in text; coordinate 27.5397°N, 88.5007°E
* **Known Limitations:** Only represents a single regional event. Does not contain explicit GIS mapping linking the landslide footprint to specific OSM road ID segments. The exact washed-away section coordinates must be verified dynamically.
* **Downloaded:** Yes (saved as `data/raw/hazards/Sikkim_Landslide_2016_NRSC.pdf`)
* **Processing Status:** ⚠️ PARTIALLY COMPLETED. Data parsed and structured to `data/processed/hazards/nrsc_sikkim_mantam_2016_event.json`. Spatial mapping to road candidates exists, but confirmation to stable OSM segment ID is **blocked**.
* **Used for Training:** No (target labels are 0 for the real model)
* **Reason:** Ground-truth evidence for real-world landslide incident.
* **Decision Category:** **A (Authoritative Government Source)**

### NRSC / ISRO Landslide Atlas of India (EXPECTED)
* **Provider:** National Remote Sensing Centre (NRSC), ISRO
* **Official URL:** [https://www.nrsc.gov.in/nrscnew/resources_atlas_landslide.php](https://www.nrsc.gov.in/nrscnew/resources_atlas_landslide.php)
* **Access Method:** Request-based download or WMS services from Bhuvan portal
* **API/File/WMS/WFS:** Geospatial database / shapefiles
* **Geographic Coverage:** Landslide-prone hilly areas of India, including the Himalayas and North Eastern Region
* **Temporal Coverage:** 1998 - 2022
* **Resolution:** Localized spatial polygons/points indicating landslide occurrences
* **Update Frequency:** Irregular/Atlas release
* **License:** Commercial evaluation terms apply; free download for academic/public work with citation.
* **Usage Restrictions:** License terms must be confirmed before model training and backend integration.
* **Data Format:** Geospatial Database (usually Shapefile or GPKG)
* **Important Fields:** Lat/Lon, Date, Landslide Type, Severity, Rainfall Association
* **Known Limitations:** Mainly contains historical events; may lack precise timestamps (has event-date but rarely event-hour). Spatial coordinates target the landslide crown or slide center, which might sit up to 500m away from the road centerlines.
* **Downloaded:** No
* **Processing Status:** ❌ PENDING. Awaiting acquisition permission or public release files.
* **Used for Training:** No (Awaiting download)
* **Reason:** Authoritative national baseline for real landslides.
* **Decision Category:** **A (Authoritative Government Source)**

---

## 4. Meteorological Prediction Features (Weather / Rainfall)

### India Meteorological Department (IMD) Historical Rainfall Data
* **Provider:** India Meteorological Department, Ministry of Earth Sciences, Govt. of India
* **Official URL:** [https://mausam.imd.gov.in/](https://mausam.imd.gov.in/)
* **Access Method:** Official IMD Data Portal, grid download API, or district warnings
* **API/File/WMS/WFS:** CSV, API, or Gridded NetCDF files
* **Geographic Coverage:** North Eastern Region (station and gridded coverage)
* **Temporal Coverage:** Historical observations (2015-2024 range desired)
* **Resolution:** Gridded (0.25° x 0.25° or 1° x 1°) or district-average values
* **Update Frequency:** Daily / Weekly
* **License:** Request / Purchase and public OGD terms
* **Usage Restrictions:** Public domain for generic statistics; high-resolution daily station/mesh grids require verification of data use agreements.
* **Data Format:** NetCDF (.nc) or CSV tables
* **Important Fields:** Date, District_Name, Daily_Rainfall_mm, Rainfall_Departure_Percent
* **Known Limitations:** Station data is sparse in remote parts of NER (Arunachal Pradesh). Grid interpolation might smooth out heavy local cloudbursts. Daily updates could have a reporting delay of 24-48 hours.
* **Downloaded:** No
* **Processing Status:** ❌ PENDING (still the preferred A-class source; acquisition dependent on access terms / API selection).
* **Used for Training:** No
* **Reason:** Primary triggering factor for landslides and floods.
* **Decision Category:** **A (Authoritative Government Source)**

### ⚠️ VERIFIED 2026-08-27: CHIRPS v2.0 Daily Rainfall (PARTIALLY INGESTED)
* **Provider:** Climate Hazards Center, UC Santa Barbara (CHIRPS v2.0)
* **Official URL:** [https://data.chc.ucsb.edu/products/CHIRPS-2.0/](https://data.chc.ucsb.edu/products/CHIRPS-2.0/)
* **Access Method:** Direct download (automated via `scripts/download_chirps_rainfall.py`)
* **API/File/WMS/WFS:** Daily GeoTIFF (.tif.gz)
* **Geographic Coverage:** Global (includes NER)
* **Temporal Coverage:** **106 days of 2015 (2015-01-01 → 2015-04-16) + 356 days (2017-2025) acquired for the real temporal training set** (30-day lookback of every 12-event prediction sample). These are real CHIRPS v2.0 rasters downloaded from `data.chc.ucsb.edu` via `scripts/download_chirps_dates.py`. Full 2015-2024 daily archive (~3,650 days) not fully acquired — only the dates needed by the confirmed temporal samples.
* **Resolution:** 0.05° (~5 km)
* **Update Frequency:** Daily (archive)
* **License:** Public Domain
* **Usage Restrictions:** None
* **Data Format:** GeoTIFF-compressed (raw), Parquet (processed)
* **Important Fields:** rainfall_1day, rainfall_3day, rainfall_7day, rainfall_14day, rainfall_30day (mm), feature_date, rainfall_days_available
* **Known Limitations (VERIFIED):** (1) `road_rainfall_features.parquet` currently contains a **single feature_date (2015-04-16)** — the extraction script now supports arbitrary anchor dates and true multi-year time-series via `--dates/--start/--end`; full 2015-2024 still pending download. (2) Fixed 2026-08-27: no-data is now NaN (never 0) and windows are calendar-date-anchored (see `ml/features/rainfall.py` + tests). (3) CHIRPS has a known high-elevation precipitation bias.
* **Downloaded:** Yes (partial)
* **Processing Status:** ✅ PROCESSED. `road_rainfall_features.parquet` (2015 snapshot, synthetic harness) + `road_rainfall_features_temporal.parquet` (real 2017-2025, 44 anchors × 289,841 roads).
* **Used for Training:** Yes — real CHIRPS rainfall features (1/3/7/14/30-day) are used in the real temporal risk model (`real_temporal_risk_dataset.parquet`).
* **Reason:** Precipitative trigger for landslide/flood disruption. Used as interim scientific source until IMD (A-class) or a complete CHIRPS archive is acquired.
* **Decision Category:** **B (Authoritative Scientific Source)**
* **Next action:** resume `download_chirps_rainfall.py` to complete 2015-2024, then run the multi-date extraction for the modelling window.

---

## 5. Geological & Terrain Prediction Features (DEM / Slope)

### ⚠️ VERIFIED 2026-08-27: SRTM v4 Terrain (INGESTED)
* **Provider:** CGIAR-CSI / SRTM v4.1
* **Official URL:** [https://srtm.csi.cgiar.org/](https://srtm.csi.cgiar.org/)
* **Access Method:** Direct download (automated via `scripts/download_srtm_dem.py`)
* **API/File/WMS/WFS:** GeoTIFF tiles
* **Geographic Coverage:** NER bounding box approx 21.5°N-29.5°N, 89.5°E-97.5°E (6 tiles: srtm_54_08, 54_09, 55_08, 55_09, 56_08, 56_09)
* **Temporal Coverage:** Static (SRTM v4.1)
* **Resolution:** 90 m (3 arc-second)
* **Update Frequency:** Static
* **License:** Public Domain (unrestricted)
* **Usage Restrictions:** None
* **Data Format:** GeoTIFF (raw), GeoTIFF mosaic 187 MB (processed)
* **Important Fields:** elevation_m, slope_degrees, aspect_degrees, aspect_category
* **Known Limitations:** **Only 32.5% (94,327/289,841) of road segments have elevation** — the OSM NER extract extends beyond the downloaded tiles. `aspect_category` uses `NODATA` for non-covered roads (no-data is kept distinct from flat terrain). The dead `elevation_range_100m` placeholder (previously constant `0.0`) was dropped from the catalog.
* **Downloaded:** Yes
* **Processing Status:** ✅ COMPLETED → `data/processed/terrain/road_terrain_features.parquet`
* **Used for Training:** Yes — terrain features (elevation_m, slope_degrees) are used in both the real static risk model and the real temporal risk model.
* **Reason:** Baseline terrain/slope features for landslide susceptibility.
* **Decision Category:** **B (Authoritative Scientific Source)**

### Bhuvan / ISRO / USGS Digital Elevation Model (DEM) — NOT USED (SRTM adopted)
* **Provider:** NRSC / ISRO (Bhuvan portal) or NASA/USGS (SRTM)
* **Official URL:** [https://bhuvan.nrsc.gov.in/](https://bhuvan.nrsc.gov.in/) or [https://earthexplorer.usgs.gov/](https://earthexplorer.usgs.gov/)
* **Status:** ❌ NOT ADOPTED. SRTM v4 (public domain, 90 m) was ingested instead of CartoDEM. If 30 m or sub-arc-second terrain is later required, CartoDEM / ALOS AW3D30 / Copernicus GLO-30 should be evaluated with license verification.

---

## 6. Traffic & Congestion Prediction Features

### OpenStreetMap & OpenTraffic Datasets (PROPOSED)
* **Provider:** OpenTraffic project or local DOTs
* **Official URL:** Not available / under evaluation
* **Access Method:** To be determined
* **API/File/WMS/WFS:** REST API or CSV logs
* **Geographic Coverage:** NER Urban centers (Guwahati, Gangtok, Shillong, etc.)
* **Temporal Coverage:** Real-time / hourly logs
* **Resolution:** Road segment-level velocities
* **Update Frequency:** Live
* **License:** ODbL or proprietary
* **Usage Restrictions:** To be evaluated
* **Data Format:** JSON/CSV
* **Important Fields:** segment_id, average_speed, travel_delay_minutes
* **Downloaded:** No
* **Processing Status:** ❌ DATA GAP. Not available.
* **Used for Training:** No
* **Reason:** Real-world traffic congestion features for route planning.
* **Decision Category:** **C (Reputable Open Geospatial Source)**
* **Alternative:** If data is missing, we must declare a DATA GAP and exclude traffic and delay estimation from the version 1 baseline.

---

## Summary of Dataset Decision Provenance

| ID | Dataset Name | Decision Category | Primary Source | Current Status (verified 2026-08-27) |
|----|--------------|-------------------|----------------|----------------|
| 1 | OSM Road Network | **C (Open)** | OpenStreetMap Foundation | ✅ Ingested |
| 2 | Boundary Shapefiles | **A (Govt)** | Survey of India | ✅ Ingested |
| 3 | Sikkim Landslide PDF | **A (Govt)** | NRSC / ISRO | ⚠️ Ingested, Confirmation Blocked |
| 4 | Landslide Atlas | **A (Govt)** | NRSC / ISRO | ❌ Pending Acquisition (DATA GAP) |
| 5 | IMD Grid Rainfall | **A (Govt)** | India Meteorological Dept | ❌ Pending Selection (DATA GAP) |
| 6 | CHIRPS v2.0 Rainfall | **B (Sci)** | UCSB Climate Hazards Center | ✅ Ingested (106 days 2015 + 356 days 2017-2025 for temporal training set; real CHIRPS) |
| 7 | SRTM v4 Terrain | **B (Sci)** | CGIAR-CSI | ✅ Ingested & processed (32.5% road coverage) |
| 8 | Landslide Atlas / DEM (Bhuvan) | **A (Govt)** | NRSC / ISRO | ❌ Not adopted (SRTM used instead) |
| 9 | Traffic Speed Logs | **C (Open)** | OpenTraffic (Unconfirmed) | ❌ Missing (DATA GAP) |
| 10 | Trip Delay Logs | **C/D** | Not Identified | ❌ Missing (DATA GAP) |

---

## Provenance Enforcement Log

Every dataset processed into the local database or pipeline must be updated in this registry. The creation of synthetic data under labels `real_incident_data` or `traffic_data` is strictly forbidden. 

For pipeline baseline demonstrations to run, they must explicitly source their label columns from `"synthetic_road_prior"` to separate baseline testing from actual real-world inference checks.

**Provenance Safeguards Checklist:**
- [x] No database credentials in Git repository.
- [x] Every raw file has a recorded SHA256 checksum in `data/raw/hazards/SOURCE_MANIFEST.json`.
- [x] Manual coordinate entries are flagged and cross-referenced with scientific publications.
- [x] Pre-processing pipelines remain identical for model baseline validation and inference runs.
- [ ] Real-data training blocked until source-supported real road labels are established (currently 0 confirmed). Terrain (SRTM) and partial rainfall (CHIRPS) are ingested; IMD/Bhuvan not required as dependencies once raw labels complete.
- [ ] Documentation synchronized with on-disk artifacts (terrain/rainfall status corrected 2026-08-27).

**End of Data Sources Document**
