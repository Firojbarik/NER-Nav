# NER-Nav ML Data Catalog

Generated from repository evidence on 2026-08-28. **DATA GAP** means the
repository does not contain enough evidence to make the stronger claim.

## Active Data Products

### OpenStreetMap North-Eastern Zone road network

| Field | Value |
|---|---|
| Source / provider | OpenStreetMap contributors; Geofabrik distribution |
| Dataset | north-eastern-zone-latest.osm.pbf |
| Official source | https://download.geofabrik.de/asia/india/north-eastern-zone.html |
| License / usage | ODbL 1.0; attribution and share-alike obligations apply |
| Local acquisition | 2026-08-23 snapshot |
| Date range | Snapshot; source edit history is not retained in the model table |
| Geographic coverage | Geofabrik North-Eastern Zone extract |
| Spatial resolution | Vector ways/nodes; variable contributor precision |
| Temporal resolution | Snapshot |
| Update frequency | Upstream updates regularly; this repository has no automated refresh SLA |
| Fields used | osm_id, ref, highway, bridge, geometry, state, district |
| Units / CRS | Geometry processed to EPSG:4326; metric operations use projected CRS |
| Rows | 289,841 road ways; 289,841 unique OSM IDs |
| Missingness | Active model table is restricted to selected corridors; OSM tag missingness varies |
| Duplicates / geometry | Road QA reports no duplicate OSM IDs, invalid, or empty geometries |
| Limitations | OSM way IDs can change after edits and are not a durable internal segment identity; road status, surface, traffic, and closure reporting are incomplete |
| Training use | Road identity, class, bridge flag, district/state attribution |

Local source checksum: 98bd62a8a52dcbb1616e934e6e40569df5a794e79458df6fd0aa637e3ac654ac.

### Survey of India administrative boundaries

| Field | Value |
|---|---|
| Source / provider | Survey of India |
| Dataset | State and district administrative boundary shapefiles |
| Official source | https://surveyofindia.gov.in/pages/administrative-boundary-data-base-abdb- |
| License / usage | Product-specific Survey of India terms apply; repository lacks the accepted license/declaration record: **DATA GAP** |
| Date range / epoch | **DATA GAP** in local manifest |
| Geographic coverage | Eight NER states; 131 processed districts |
| Spatial resolution | Vector administrative boundaries; source scale/accuracy not recorded locally |
| Temporal resolution | Administrative snapshot |
| Update frequency | No repository refresh schedule |
| Fields used | State, district, geometry |
| Units / CRS | Processed EPSG:4326 |
| Missingness / duplicates | Road attribution QA reports 100% assignment; source attribute quality is inconsistent |
| Known limitations | District names contain encoding/substitution artifacts and embedded line breaks; boundary edition is not versioned in a complete manifest |
| Training use | State and district context only; currently not direct model features |

Five raw shapefiles occupy about 210 MB. The exact local download transaction
and usage terms must be retained before production redistribution.

### CHIRPS v2 daily rainfall

| Field | Value |
|---|---|
| Source / provider | Climate Hazards Center, UC Santa Barbara, with USGS/FEWS NET collaboration |
| Dataset | CHIRPS v2.0 daily 0.05 degree rainfall |
| Official source | https://www.chc.ucsb.edu/data/chirps |
| License / usage | Public domain dedication stated by provider |
| Date range | Upstream 1981 to near-present; active model anchors 2017-07-01 to 2025-09-13 |
| Geographic coverage | Quasi-global 50S-50N, including NER |
| Spatial resolution | 0.05 degree, approximately 5 km |
| Temporal resolution | Daily |
| Update frequency | Daily/near-real-time upstream; no production fetch SLA here |
| Fields used | 1, 3, 7, 14, and 30-day rainfall totals; available-day count |
| Units | Millimetres |
| Local extent | 573 raw GeoTIFF archives, about 2.12 GB; 63 processed anchor dates; 18,259,983 road/date rows |
| Active dataset missingness | 23.37% for every rainfall window |
| Duplicates | Active ML sample keys have no duplicates |
| Known limitations | Gridded areal estimate, sparse gauges, complex-terrain/extreme-rain bias; processed feature_date is stored as a string; freshness/issue timestamp is absent |
| Training use | Primary time-varying trigger features |

The provider has announced CHIRPS v3 and says v2 production ends after
December 2026. Migration must be evaluated as a new dataset version; v2 and v3
must not be mixed silently.

### SRTM v4.1 terrain

| Field | Value |
|---|---|
| Source / provider | CGIAR-CSI distribution of NASA SRTM |
| Dataset | Six SRTM v4.1 GeoTIFF tiles and processed road terrain table |
| Official reference | https://www.earthdata.nasa.gov/data/instruments/srtm |
| License / usage | Repository says public domain; exact CGIAR download terms/acquisition manifest are missing: **DATA GAP** |
| Date range | Static mission-derived elevation |
| Geographic coverage | Partial NER coverage from six local tiles |
| Spatial resolution | 90 m / 3 arc-second product |
| Temporal resolution / updates | Static |
| Fields used | Elevation, slope; aspect retained outside active model |
| Units | Metres and degrees |
| Local extent | Six GeoTIFFs, about 433 MB; 289,841 processed road rows |
| Active dataset missingness | 70.11% elevation and slope missing |
| Duplicates | One terrain row per OSM ID in processed table |
| Known limitations | Severe geographic non-coverage; void/interpolation and terrain age; road-level aggregation method must remain versioned |
| Training use | Elevation, slope, missingness flag, rainfall interactions |

### Historical road-disruption observations

| Field | Value |
|---|---|
| Source / provider | Mixed NRSC/government/media evidence and manually reviewed mappings |
| Dataset | 23 event tuples in ml/features/temporal_design.py; 13 local processed event JSON records |
| License / usage | Several local event records say VERIFY_BEFORE_REUSE; complete raw-source licenses are **DATA GAP** |
| Date range | 2017-07-15 to 2025-09-14 |
| Geographic coverage | Eight states, nine NH reference corridors; highly uneven event counts |
| Spatial resolution | Event mapped to an OSM way |
| Temporal resolution | Date-level; event-hour and report-delay generally unavailable |
| Update frequency | Manual |
| Fields used | Event ID/date, affected OSM way, NH reference |
| Missingness | Raw file, checksum, access time, exact source URL, or district is absent for multiple records |
| Duplicates | Active model sample keys have no duplicates; clustered same-day events remain dependent observations |
| Known limitations | Code-maintained event list is not a complete source manifest; media evidence is not equivalent to official incident ground truth; observation/reporting bias is unmeasured |
| Training use | Positive event anchors only |

### Real temporal risk modelling table

| Field | Value |
|---|---|
| Dataset | data/processed/ml/real_temporal_risk_dataset.parquet |
| Dataset version | SHA256 2cccd364ad2b5ca1abfeefbe1c71fffb313ef546e1653eb2279e913d19f01ace |
| Construction | Real event dates/road mappings joined to CHIRPS and SRTM/OSM features |
| Unit of prediction | Road source identifier at a prediction date |
| Target | A recorded disruption on the mapped road in (prediction_time, prediction_time + 7 days] |
| Rows | 184 samples, 23 event groups, 115 unique road source IDs |
| Labels | 69 positives, 115 negatives |
| Positive construction | Three anchors per event at 1, 3, and 7 days before the same event |
| Negative construction | 23 same-road controls at event-14 days; 92 sampled corridor roads labelled assumed-unaffected |
| Positive rate | 37.5% |
| Geographic coverage | Eight states, nine corridors; not representative of the full 289,841-road network |
| Missingness | Rainfall 23.37%; terrain 70.11% |
| Duplicate sample keys | 0 |
| Known limitations | Multiple rows per event are not independent events; zero source-confirmed unaffected negatives; OSM way identity is mutable; no incident observation-completeness denominator |

## Label Confidence

| Label type | Count | Confidence |
|---|---:|---|
| Recorded affected road within seven days | 69 samples from 23 events | Limited to event evidence and road mapping quality |
| Same-road event outside seven-day horizon | 23 | The named event is outside the horizon, but absence of any other disruption is unverified |
| Assumed-unaffected corridor road | 92 | Low; not reported is not did not occur |
| Source-confirmed unaffected road/window | 0 | **DATA GAP** |

The dataset therefore does not provide reliable negative ground truth for a
production probability model.

## Candidate Source Evaluation

| Candidate | Realness / authority | NER and target fit | License / access | Duplicate risk | Production decision |
|---|---|---|---|---|---|
| NRSC/NDEM Landslide Inventory and Atlas | Government satellite inventory | High for landslides; road impact and exact event time may require mapping | Portal terms and machine reuse must be cleared | High against existing event list | P1 acquire through authorized channel |
| IMD historical gridded/station rainfall | Government meteorological observation | High; potentially improves NER extremes | Product selection and use terms unresolved | Overlaps CHIRPS in time, not semantics | P1 evaluate; do not blend before bias/units study |
| CHIRPS v3 | Authoritative scientific product | Full NER; continuity after v2 | Public provider access | Full overlap with v2 | P1 migration experiment as a new version |
| Survey of India current ABDB | Government boundary standard | Full NER | Exact accepted usage terms must be retained | Replaces, not appends to, old edition | P1 versioned refresh |
| GSI/NRSC landslide warnings/inventories | Government geohazard sources | Potentially high | Access and redistribution review needed | Cross-source duplicates likely | P1 source reconciliation |
| State PWD / DDMA road closure logs | Operational ground truth | Highest target compatibility if structured | **DATA GAP**; agreements required | Duplicate reports across agencies | P0/P1 priority for positives and verified negatives |
| Traffic/travel-time logs | Real if obtained from operators | Needed only for delay model | No source identified | Unknown | **DATA GAP**; do not build delay model |
| Field reports | Real observations, not automatic truth | High local relevance | Consent, retention, verification policy required | Duplicate/copy reports likely | Build verified-observation workflow before labels |

## Catalog Controls

- No source enters training without provider, acquisition time, local path,
  SHA256, license state, spatial/temporal coverage, and schema.
- Unknown values remain **DATA GAP**.
- Dataset versions are content hashes.
- Source identifiers and stable internal segment identifiers are separate.
- Synthetic fixtures are restricted to software tests and never enter this
  catalog as performance evidence.
