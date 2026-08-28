# Event Collection Step-by-Step Workflow (Manual)

Goal: collect and confirm enough real hazard events (and corridor-diverse
negatives) to lift the production-risk model from `GATED_LOW_CONFIDENCE` to a
defensible demo/production evidence level.

Current reality (2026-08-28, model `2026-08-28_160006_ce797e78`):
- 26 confirmed events, grouped-CV pooled ROC-AUC 0.536.
- Bottleneck is **training-set size + corridor diversity**, not features
  (adding seasonal features empirically did not move the gate).
- Production gate needs **>= 30 future TEST events**; demo needs >= 3 test
  events with ROC-AUC >= 0.65 / AP >= 0.50 (currently failing).
- Real NER event coverage is concentrated in monsoon months (Jun-Sep). Diversify
  by corridor AND by season where possible.

======================================================================
## THE LOOP (repeat ~10-15 times)
======================================================================
For each candidate event you find:

  1. FIND an event (source)
  2. CONFIRM the affected road's OSM id (Overpass)
  3. VALIDATE + ADD it (`add_confirmed_event.py`)
  4. REBUILD + grind metrics (`rebuild_pipeline.sh`)
  5. REVIEW the gate & grouped-CV report
  6. RECORD provenance in docs + event JSON

======================================================================
## STEP 1 - FIND an event (source priority)
======================================================================
Search by NH + state + year. Best real sources:

- IMD heavy-rain / landslide alerts:  https://mausam.imd.gov.in/
- NDMA event reports:                https://ndma.gov.in/
- Assam Flood Control (floodample):  https://floodample.gov.in/
- State PWD / Public Works press releases + X (Twitter) feeds.
- News: `"NH{ref}" "landslide|flood|blocked|washout" "{state}" {year}`

Target list by corridor gap (add MOST to these first — they are
underrepresented and add the corridor diversity the CV needs):

  HIGH (1-2 events, underrepresented AND present in the local network):
    NH510 Sikkim (1; 29 ways in network)
    NH102B Manipur/Mizoram (1; 56 ways)
    NH208A Tripura (1; 24 ways)
    NH10  Sikkim (2; 25 ways)
    NH2   (2; 211 ways) - add more Upper Assam
  MEDIUM (deepen good corridors, but DIVERSIFY the mechanism/season):
    NH27  Assam (4) - prefer FLAT-TERRAIN FLOOD events, not just landslides
    NH13  Arunachal (4)
    NH29  Nagaland (4)
    NH37  Manipur (4)
    NH6   (3)

  NOT YET ACTIONABLE: NH36 (Assam), NH12 (Arunachal) have **0 ways in the
  local OSM network** (`ner_roads.gpkg`), so events on them cannot join to
  features yet. If you find events there, record them but do NOT add until the
  road network is ingested for those refs (separate task).

Seasonal diversification: prefer to log events in **non-monsoon** months
(Nov-Apr) if any source reports one; this adds the seasonal contrast the
monsoon-concentrated set lacks.

Record for each candidate BEFORE doing anything else:
  event_id, date, state, location, suspected NH, source URL, a short note.

======================================================================
## STEP 2 - CONFIRM the affected road's OSM id
======================================================================
Use Overpass Turbo:  https://overpass-turbo.eu/

For a road that was blocked/damaged at or near a location (lat, lon):

  way["ref"="NH27"](around:6000, {lat},{lon});
  out ids;

If you know the exact way segment, refine by name/highway:

  way["ref"="NH27"]["highway"="trunk"](around:6000, {lat},{lon});
  out ids;

Notes:
- Write down the OSM `id` (e.g. a 8-9 digit number). Use the **way id**.
- If two segments are candidates, pick the one whose geometry/name best
  matches the news description (bridge, village, mountain pass).
- The id MUST exist in the project's road network (check next step) so the
  event can join to features.

======================================================================
## STEP 3 - VERIFY the OSM id is in the local network (REQUIRED)
======================================================================
Before adding, confirm the road is actually in `ner_roads.gpkg`, otherwise the
event will silently produce no features. Run:

```bash
python -c "
import geopandas as gpd
r = gpd.read_file('data/processed/roads/ner_roads.gpkg')
print(r[r['osm_id'] == {YOUR_OSM_ID}][['osm_id','ref','highway','name']])
"
```

Expected: one row, ref matches the NH you expect. If empty, re-pick the OSM id.

Also confirm terrain + rainfall coverage for that road exist:

```bash
python -c "
import pandas as pd
t = pd.read_parquet('data/processed/terrain/road_terrain_features.parquet')
print('terrain:', t[t['osm_id']=={YOUR_OSM_ID}][['elevation_m','slope_degrees']])
"
```

======================================================================
## STEP 4 - VALIDATE + ADD the event
======================================================================
```bash
python scripts/add_confirmed_event.py \
    --event_id {state}_{location}_{YYYY}_{MM}_{DD} \
    --osm_id {YOUR_OSM_ID} \
    --ref NH{XX} \
    --event_date {YYYY-MM-DD}
```

This is a dry run: it validates event_id uniqueness, osm_id, ISO date
(>= 2017-01-01), and the `NH\d+[A-Z]?` ref pattern. Fix any errors, then apply:

```bash
python scripts/add_confirmed_event.py \
    --event_id {state}_{location}_{YYYY}_{MM}_{DD} \
    --osm_id {YOUR_OSM_ID} \
    --ref NH{XX} \
    --event_date {YYYY-MM-DD} \
    --apply
```

Naming convention (match existing ids):
  state_location_year_month_day   e.g.
  assam_dhansiri_2024_06_12
  arunachal_seppa_bypass_2024_07_20
  tripura_agartala_bypass_2025_06_01

======================================================================
## STEP 5 - REBUILD + grind metrics
======================================================================
Rebuild the dataset, retrain the honest-gated model, and run the tests:

```bash
python scripts/rebuild_pipeline.py
```

That chains: download needed CHIRPS dates -> extract rainfall features ->
rebuild dataset -> retrain with gates -> run unit tests.

(If the download is slow/unneeded, skip it:)
```bash
python scripts/rebuild_pipeline.py --skip-download
```

======================================================================
## STEP 6 - REVIEW the gate & grouped-CV report
======================================================================
Read the newest report:

```bash
python -c "
import json
tag = json.load(open('data/models/prod_latest.json'))['model_version']
r = json.load(open(f'data/models/prod_report_{tag}.json'))
print('status:', r['status'])
print('test metrics:', r['test_metrics']['roc_auc'], r['test_metrics']['avg_precision'])
print('demo gate:', r['gates']['demo']['status'], r['gates']['demo']['reasons'])
print('prod gate:', r['gates']['production']['reasons'])
cv = r['grouped_cv']
print('grouped-CV pooled ROC-AUC:', cv.get('pooled_roc_auc'), 'AP:', cv.get('pooled_avg_precision'))
print('n test events:', r['split']['n_test_events'])
"
```

Interpretation:
- Watch the **grouped-CV pooled ROC-AUC** as the stable signal. You want it to
  rise toward >= 0.65 (demo bar).
- Watch **n test events** grow toward >= 30 (production bar) over many sessions
  and BEFORE trusting any single-split number.
- In the report's `gates.demo.reasons` you will see which metric is blocking.
- IMPORTANT expectation (from previous experiments): one or two new events can
  swing the 3-4-event single-split test down sharply. Do NOT judge the change
  by one run; judge by 5-10 new events and by grouped-CV trend.

======================================================================
## STEP 7 - RECORD provenance (DATA GAP #2)
======================================================================
The report flags that only ~13 of 26 events have a processed `*_event.json`.
For every event you add, create its provenance file so the label registry is
complete. Example structure (match `data/raw/hazards/` conventions):

  data/raw/hazards/{event_id}_source.json
  {
    "event_id": "...",
    "event_date": "...",
    "location": "...",
    "state": "...",
    "affected_road_ref": "NH27",
    "osm_id": 123456789,
    "source": {
      "type": "NEWS|GOV_REPORT|SCIENTIFIC|IMD|NDMA|STATE_PWD",
      "url": "https://...",
      "organization": "...",
      "acquired_at_utc": "...",
      "checked": true
    },
    "road_damage": {
      "reported": true,
      "source_supported": true,
      "confirmed_affected": true,
      "label_ready_for_training": true
    }
  }

Also append a row to `data/raw/hazards/ner_hazard_events_database.csv`
(event_id, event_date, event_type, state, location_name, latitude, longitude,
affected_road, source, severity, notes).

======================================================================
## STEP 8 - COMMIT the milestone
======================================================================
When you have a meaningful batch (e.g. +5 events), commit the pipeline code +
provenance + new model bundle so each milestone is reproducible:

```bash
git add ml/ scripts/ tests/ docs/ data/raw/hazards/ data/models/
git add -u
git commit -m "..."
```
(Exclude the large re-fetchable raw rasters — already gitignored.)

======================================================================
## Corridor-diverse NEGATIVES (the second gap)
======================================================================
The model currently has ZERO source-confirmed negative windows — negatives are
"assumed unaffected" corridor roads. The honest upgrade is to add negatives
from operations data proving a road was OPEN/no-event during a monsoon window:

1. Sources that prove "road X open / no disruption on date D":
   - State PWD circulars / closure lists
   - District Disaster Management Authority (DDMA) daily status
   - Traffic/police notifications, national highway operators
   - Verified media: "traffic normal on NH-Y on {date}"
2. For each: pick the OSM id of that road + a date where no event is recorded,
   and add it as a negative. Design: reuse the same `osm_id`, set
   `--event_date` to a known-clear reference, and mark it clearly.
3. This is lower priority than more positives but materially hardens the label
   quality the report keeps flagging.

======================================================================
## Success criteria
======================================================================
- Grouped-CV pooled ROC-AUC trending >= 0.65 after ~10-15 new events.
- n test events growing toward >= 30.
- Every event has a source-provenance file (closes data gap #2).
- Prefer corridor-diversity (NH36, NH12, NH510, NH102B, NH208A, NH10) over
  re-deepening already-good corridors.

Recommended pace: 1 loop (~40-60 min) per event; 5-10 events/session.
