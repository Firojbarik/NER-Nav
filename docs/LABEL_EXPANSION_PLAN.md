# Real Label Expansion Plan — Path to Honest Production Model

## Current State (2026-08-28)
- **24 confirmed events** across 10 NH corridors (72 positives, 120 negatives, 192 total samples).
- **Features**: 5 rainfall windows + 6 rainfall ratios + 3 slope-rainfall interactions + elevation + slope.
- **Rainfall coverage now 100%** (was 73%): fixed a pipeline bug where `rebuild_pipeline.py` wrote extracted rainfall features to `road_rainfall_features.parquet` while the dataset builder read the separate stale `road_rainfall_features_temporal.parquet`. Feature extraction now writes the `_temporal` file the builder consumes, so every sample has complete real CHIRPS rainfall for all 5 windows.
- **Hackathon/demo gate PASSED (post-fix, honest complete data)**: test_events=4 >= 3, ROC-AUC=0.746 >= 0.65, avg_precision=0.643 >= 0.50.
- **Grouped-CV (stable generalization estimate) is WEAKER than the single-split pass**: chronological expanding-window grouped CV (15 folds, each event group used as a future test block) pools to **ROC-AUC 0.563 (range 0.167-0.867, std 0.186), avg-precision 0.462 (range 0.306-0.810)**. The pooled ROC-AUC 0.563 is **below the 0.65 demo threshold** — i.e. the single-split demo pass (0.746) is driven by which few events land in the test split, NOT by stable generalization. See "Grouped-CV finding" below.
- **Production gate FAILED**: only 4 future test events (requires >= 30); recall below the 0.70 production floor.
- **STATUS: HACKATHON_DEMO_READY_LIMITED_CONFIDENCE** (formally passes the single-split demo gate, but grouped-CV shows genuine generalization is below the demo bar; treat the pass as a milestone, not a robust skill claim). This is not production-safe.
- **Note**: XGBoost still does not beat every comparable baseline (LR remains competitive); demo-gate pass is a credible-but-limited milestone, not production readiness.
- **Caveat on the pre-fix pass**: the earlier reported 0.673/0.507 was computed on data with 27% missing rainfall (NaN features); the corrected complete-data model is 0.746/0.643. Adding a single hard new test event drags the tiny 3-4 event test set down sharply (NH-37 Lakhipur 2025-10-12 -> ROC-AUC 0.556; NH-315A Kathalguri 2026-06-24 -> ROC-AUC 0.467), so each added test event must be vetted for coverage AND generalization risk.

## Structural finding: demo gate is not robust to test-set growth (2026-08-28)
Three independent experiments (2017-2025 events, then +NH-37 Lakhipur, then +NH-315A Kathalguri) confirm that **the demo gate's pass/fail is dominated by which 3-4 chronologically-latest events land in the held-out test set**, not by genuine model skill. A model trained on ~18-20 events cannot rank a genuinely new low-rainfall corridor (e.g. flat NH-37 Cachar or low-rainfall NH-315A Dibrugarh) against the baseline, so one atypical test example swings ROC-AUC from 0.746 to ~0.47. Consequently, blindly adding coverage-complete events is counterproductive: it keeps collapsing the gate without improving real generalization. The path to a robust production gate is a substantially larger training set (many more real events in train, not just test) plus corridor-diverse negatives, not cherry-picking single test events. CHIRPS data IS available for 2026 (verified on server through 2026-06+), so 2026 events are technically viable — the blocker is generalization, not data availability.

## Grouped-CV finding: the "demo pass" does not reflect stable generalization (2026-08-28)
Added a chronological expanding-window **grouped cross-validation** (`grouped_cv_evaluate` in `scripts/train_production_risk_model.py`, reported under `report.grouped_cv` and `report.robustness`). Each fold trains on all events before a block, validates on the next block, and predicts the following block as a future test; scores are pooled out-of-fold across 15 folds so **every event group serves as a future test set at least once**.
- **Pooled ROC-AUC = 0.563** (below the 0.65 demo bar), pooled avg-precision = 0.462.
- High variance: ROC-AUC 0.17-0.87, AP 0.31-0.81.
- **Conclusion**: the single-split demo pass (0.746/0.643) overstates the model; averaged across many future-test blocks the model clears neither the ROC-AUC nor (marginally) the AP bar. The model is genuinely weak at forecasting a *new* future block. To raise the stable estimate into demo territory requires a larger, corridor-diverse **training** set — not adding events to the test split, which only widens the gap between the single-split number and the CV estimate.

## Known Data Gaps

### Terrain — training dataset COMPLETE via Copernicus DEM (0 NaN, real raster)
- **Root cause (solved)**: SRTM DEM tiles only cover to 25N, but NER roads extend to ~29N (Arunachal Pradesh).
- **Mitigation (applied 2026-08-28)**: Downloaded Copernicus GLO-30 DEM 1x1° tiles (25-30N, 88-97E) from AWS Open Data `copernicus-dem-30m` bucket, merged with existing SRTM (<=25N) into `ner_dem_full.tif` (85-100E, 15-30N, 30m). Re-sampled real elevation/slope for all training-set roads. **0 NaN** elevation and slope; real raster values (not open-elevation point estimates).
- **Caveat**: Terrain completeness alone does not lift AUC; metrics are driven mostly by the small test set (4 events, 12 positives) and corridor diversity. After the rainfall-coverage pipeline fix, corrected complete-data demo-gate metrics are AUC=0.746 / AP=0.643, but still on a tiny test set and sensitive to which events land in test.
- **Remaining gap**: the full 195K-road OSM network still lacks raster terrain in the shared network terrain file (only the ~30 training corridors reprocessed); water areas are masked in Copernicus.

### Flat-terrain corridors underperform
- NH27 (Assam) and NH208A (Tripura) have near-zero or inverted separation.
- **Root cause**: The model learns "steep slope + rain = landslide" but flat-terrain disruption mechanisms (drainage, soil saturation, upstream hydrology) lack features.
- **Fix**: Add upstream catchment features (drainage density, contributing area) from DEM + flow direction analysis.

## Target NH Corridors (underrepresented or missing)

| NH | State | Events | Priority | Notes |
|----|-------|--------|----------|-------|
| NH102B | Manipur/Mizoram | 1 | HIGH | Guite Road (Churachandpur-Aizawl); added 2025-06-02 Sinzawl. |
| NH37 | Manipur | 4 | DONE | Good coverage. |
| NH27 | Assam | 4 | MEDIUM | Flat terrain — needs drainage features. |
| NH29 | Nagaland | 4 | DONE | Good coverage. |
| NH6 | Meghalaya/Mizoram | 3 | DONE | Good coverage. |
| NH2 | Assam/Nagaland | 1 | MEDIUM | Add more Upper Assam events. |
| NH13 | Arunachal | 4 | DONE | Tezpur-Tawang segmented (Bana/Seppa/Tawang). |
| NH10 | Sikkim | 1 | HIGH | Chungthang high-altitude corridor. |
| NH36 | Assam | 0 | HIGH | Tinsukia-Dibrugarh; tea-country erosion. |
| NH12 | Arunachal | 0 | HIGH | Tezpur-Tawang; high-altitude landslide corridor. |
| NH208A | Tripura | 1 | LOW | Flat terrain — needs different features. |

## How to Find and Confirm a New Event

### Step 1: Search for hazard events
Sources (search by NH + state + year):
- **IMD heavy rainfall alerts**: https://mausam.imd.gov.in/
- **NDMA event reports**: https://ndma.gov.in/
- **ASSAM Flood Control Board**: https://floodample.gov.in/
- **State PWD/PTA announcements** (social media, press releases).
- **News**: Search `"NH{X}" "landslide" "blocked" "{state}" 2024` or `2025`.

### Step 2: Confirm the affected road's OSM ID
Use **Overpass Turbo** (https://overpass-turbo.eu/):
```
way["ref"="NH{X}"](around:5000,{lat},{lon});
out ids;
```

### Step 3: Validate and add
```bash
python scripts/add_confirmed_event.py \
    --event_id {state}_{location}_{YYYY}_{MM}_{DD} \
    --osm_id {OSM_WAY_ID} \
    --ref NH{X} \
    --event_date {YYYY-MM-DD} \
    --apply
```

### Step 4: Rebuild
```bash
python scripts/rebuild_pipeline.py
```

## Seasonal / monsoon feature experiment (2026-08-28)
Added leakage-safe seasonal/monsoon features derived ONLY from the real
prediction timestamp (known exactly at decision time; identical at training
build and inference): `month`, `seasonal_sin`, `seasonal_cos`,
`monsoon_active`, `days_into_monsoon` (`ml/features/seasonal.py`). Threaded
through `build_real_temporal_dataset.py`, `train_production_risk_model.py`,
and `predict_real_temporal.py`, with unit tests.

**Result (honest): the stable generalization estimate did NOT improve.**
With 21 features (16 prior + 5 seasonal), grouped expanding-window CV pooled
ROC-AUC = **0.536** (min 0.067 / max 0.933), vs 0.563 before — a within-noise
change, and single-split test ROC-AUC moved 0.569 -> 0.585. Diagnosis: within
the current data, positives and their corridor negatives are sampled at the
SAME prediction date, so season/monsoon features are identical inside each
sample group and add essentially no local separation; and real NER events are
already so monsoon-concentrated that the features carry limited cross-group
rank signal. The finding strengthens the existing conclusion in
docs/ML_PRODUCTION_READINESS_REPORT.md: the bottleneck is event/sample count
and corridor diversity, not missing features. The features are retained (they
are physically motivated, leakage-safe, and part of the immutable 21-feature
schema) but they are not expected to move the gate alone.

