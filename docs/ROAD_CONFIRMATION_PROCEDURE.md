# Road Confirmation Procedure for Hazard Events

**Project:** NER-Nav  
**Purpose:** Establish defensible road-segment labels from hazard event reports  
**Status:** Draft procedure for implementation

---

## Problem Statement

The NRSC Sikkim Mantam 2016 landslide report provides:
- ✅ Event location: 27.5397°N, 88.5007°E
- ✅ Descriptive road name: "Passingdang-Mantam Road"
- ✅ Impact description: "300m road washed away"
- ❌ Specific OSM way IDs
- ❌ GPS tracks of affected sections
- ❌ Before/after satellite imagery links

**Current status:** 35 road candidates identified within proximity, but **0 have confirmed road identity**.

**Blocker:** Cannot train real-data model until road segments have `source_supported=true AND confirmed_affected=true`.

---

## Confirmation Criteria

For a road segment to become a **positive training label**, it must satisfy:

```
source_supported = true
    AND
confirmed_affected = true
```

### `source_supported` Requirements

At least ONE of:
1. **Named match:** OSM road `name` or `ref` field matches the hazard report road name
2. **Spatial evidence:** Road segment intersects published landslide footprint polygon
3. **Published coordinates:** Scientific paper provides GPS track of affected road section
4. **Government records:** Official road closure records list the specific route/segment
5. **Satellite imagery:** Before/after imagery shows visible road damage at segment location

### `confirmed_affected` Requirements

At least ONE of:
1. **Direct impact statement:** Report explicitly states "road at [coordinates] was damaged/blocked"
2. **Temporal correlation:** Road closure records match event date ±3 days
3. **Visual confirmation:** Satellite imagery shows damage/debris on road surface
4. **Field verification:** Post-event field survey confirms damage (if available)
5. **Expert review:** Domain expert (civil engineer, geologist, or government official) confirms match

---

## Procedure for Sikkim Event

### Step 1: Named Road Match (PRIORITY)

**Action:**
```sql
SELECT osm_id, name, ref, highway, ST_AsText(geometry)
FROM ner_roads_districts
WHERE (
    LOWER(name) LIKE '%passingdang%'
    OR LOWER(name) LIKE '%mantam%'
    OR LOWER(ref) LIKE '%passingdang%'
    OR LOWER(ref) LIKE '%mantam%'
)
AND state = 'Sikkim'
AND ST_Distance(
    geometry::geography,
    ST_SetSRID(ST_MakePoint(88.50068611, 27.5397), 4326)::geography
) < 5000;  -- 5km radius
```

**If match found:**
- Mark `source_supported = true` (named match evidence)
- Record evidence: `"OSM name field matches report road name"`

### Step 2: Proximity + Road Type Filter

**Action:**
```sql
SELECT osm_id, name, highway, 
       ST_Distance(
           geometry::geography,
           ST_SetSRID(ST_MakePoint(88.50068611, 27.5397), 4326)::geography
       ) as distance_m
FROM ner_roads_districts
WHERE state = 'Sikkim'
  AND highway IN ('trunk', 'primary', 'secondary', 'tertiary', 'unclassified')
  AND ST_Distance(
      geometry::geography,
      ST_SetSRID(ST_MakePoint(88.50068611, 27.5397), 4326)::geography
  ) < 2000  -- 2km radius
ORDER BY distance_m ASC
LIMIT 10;
```

**Manual review required:**
- Visual inspection on OpenStreetMap at 27.5397°N, 88.5007°E
- Identify roads connecting Passingdang and Mantam villages
- Check road continuity (is there a logical path?)

### Step 3: Satellite Imagery Review (RECOMMENDED)

**Data sources:**
- Google Earth historical imagery (2016-08-12 before, 2016-08-15 after)
- Sentinel-2 archive (free, 10m resolution, ESA Copernicus)
- Planet Labs imagery (if accessible)

**Procedure:**
1. Load OSM road candidate geometries into QGIS
2. Overlay with 2016-08-13 ± 7 days satellite imagery
3. Identify visible road damage, debris, or missing road sections
4. Match damage locations to OSM segment IDs

**If visual damage confirmed:**
- Mark `confirmed_affected = true`
- Record evidence: `"Satellite imagery shows road damage at [coordinates] on [date]"`

### Step 4: Cross-Reference Published Paper

**Source:** Martha, Roy & Kumar (2017), Current Science 113(7)  
**URL:** https://www.researchgate.net/publication/320280207

**Action:**
- Extract any additional road/bridge coordinates from paper
- Check for maps showing affected infrastructure
- Verify reported impact area matches OSM candidates

### Step 5: Expert Review (FALLBACK)

**If Steps 1-4 are inconclusive:**
- Document uncertainty in confirmation metadata
- Mark candidates as `requires_expert_review = true`
- Contact:
  - North Sikkim District PWD office
  - NRSC researchers (paper authors)
  - Local disaster management authority

**Do NOT:**
- Mark roads as `confirmed_affected` based solely on proximity
- Use distance-only heuristics as positive labels
- Assume the closest road is automatically the correct road

---

## Implementation Script

**File:** `scripts/manual_confirm_sikkim_road.py`

```python
# Pseudocode structure

import geopandas as gpd
import pandas as pd

# Load candidates
candidates = pd.read_parquet(
    'data/processed/hazards/nrsc_sikkim_mantam_2016_road_candidates.parquet'
)

# Step 1: Check for name matches
name_matches = candidates[
    candidates['name'].str.contains('Passingdang|Mantam', case=False, na=False)
    | candidates['ref'].str.contains('Passingdang|Mantam', case=False, na=False)
]

if len(name_matches) > 0:
    print(f"FOUND {len(name_matches)} named matches:")
    print(name_matches[['osm_id', 'name', 'ref', 'distance_m']])
    
    # Prompt for manual confirmation
    for idx, row in name_matches.iterrows():
        confirm = input(
            f"Confirm OSM ID {row['osm_id']} ({row['name']}) as affected road? (y/n): "
        )
        if confirm.lower() == 'y':
            candidates.loc[idx, 'source_supported'] = True
            candidates.loc[idx, 'source_evidence'] = 'OSM name matches report'
            
            confirm_affected = input(
                "Confirm this road was affected by the landslide? (y/n): "
            )
            if confirm_affected.lower() == 'y':
                candidates.loc[idx, 'confirmed_affected'] = True
                candidates.loc[idx, 'confirmation_evidence'] = 'Manual expert review'

# Step 2: If no name matches, review top 5 closest roads
if len(name_matches) == 0:
    top_candidates = candidates.nsmallest(5, 'distance_m')
    print("\nNo named matches. Top 5 closest roads:")
    print(top_candidates[['osm_id', 'name', 'highway', 'distance_m']])
    print("\nREQUIRES MANUAL REVIEW:")
    print("1. Open OpenStreetMap at 27.5397°N, 88.5007°E")
    print("2. Identify roads connecting Passingdang and Mantam")
    print("3. Check satellite imagery for visible damage")
    print("4. Update confirmation fields manually")

# Save updated candidates
candidates.to_parquet(
    'data/processed/hazards/nrsc_sikkim_mantam_2016_confirmed_road.parquet'
)

# Report training gate status
real_labels = (
    (candidates['source_supported'] == True)
    & (candidates['confirmed_affected'] == True)
).sum()

print(f"\n{'='*60}")
print(f"Real training labels available: {real_labels}")
if real_labels > 0:
    print("STATUS: TRAINING GATE UNBLOCKED")
else:
    print("STATUS: TRAINING GATE STILL BLOCKED")
print(f"{'='*60}")
```

---

## Quality Assurance

Before marking any road as a real label:

**Checklist:**
- [ ] Evidence is traceable to authoritative source
- [ ] Coordinates verified against multiple sources
- [ ] Temporal alignment confirmed (event date matches report)
- [ ] No synthetic values added
- [ ] Confirmation method documented in metadata
- [ ] Uncertainty quantified (if applicable)

**Red flags (DO NOT CONFIRM):**
- Distance-only match with no supporting evidence
- Assumed impact without verification
- Circular reasoning (using model output as ground truth)
- Manual coordinate guessing

---

## Metadata Schema

For each confirmed road, record:

```json
{
  "osm_id": 123456789,
  "event_id": "nrsc_sikkim_mantam_2016_08_13",
  "source_supported": true,
  "source_evidence": "OSM name 'Mantam Road' matches NRSC report",
  "confirmed_affected": true,
  "confirmation_evidence": "Satellite imagery shows road damage at 27.5398°N, 88.5005°E on 2016-08-14",
  "confirmation_method": "visual_satellite_imagery",
  "confidence_level": "high",
  "confirmed_by": "manual_review",
  "confirmed_at": "2026-08-27T00:00:00Z",
  "notes": "Visible debris and missing road section in Sentinel-2 imagery"
}
```

---

## Next Steps

1. **Immediate:** Run named-match query against OSM database
2. **If no match:** Load OpenStreetMap at event coordinates, identify road network topology
3. **Acquire satellite imagery:** Download Sentinel-2 or Google Earth historical imagery for visual confirmation
4. **Document results:** Update confirmation file with evidence and confidence levels
5. **Re-run training gate:** Verify `real_label_rows > 0` before proceeding to model training

---

## Expected Outcome

**Minimum viable labels:** 1-5 confirmed road segments from Sikkim event

**Ideal outcome:** 10-20 confirmed road segments from 3-5 additional events

**Training readiness:** Once ≥10 positive labels exist with ≥100 negative labels (roads near events but unaffected)

---

**Status:** Procedure documented, awaiting implementation

**Assigned:** ML Engineer / Data Engineer

**Priority:** P0 (blocks real-data training)

**Estimated effort:** 2-4 hours for Sikkim event confirmation
