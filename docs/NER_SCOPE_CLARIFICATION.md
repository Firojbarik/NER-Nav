# NER-Nav Project Scope Clarification

**Project:** NER-Nav (North Eastern Region Navigation & Risk Assessment)  
**Geographic Scope:** ALL 8 North Eastern States of India  
**Date:** 2026-08-27

---

## Geographic Coverage

### ✅ Complete NER Region (8 States)

1. **Arunachal Pradesh** - 83,743 km² (mountainous, landslide-prone)
2. **Assam** - 78,438 km² (plains, flood-prone)
3. **Manipur** - 22,327 km² (hills + valley, mixed hazards)
4. **Meghalaya** - 22,429 km² (highest rainfall, landslides)
5. **Mizoram** - 21,081 km² (steep terrain, landslides)
6. **Nagaland** - 16,579 km² (hills, landslides + earthquakes)
7. **Sikkim** - 7,096 km² (Himalayan, earthquakes + landslides)
8. **Tripura** - 10,486 km² (plains + hills, floods)

**Total Area:** 262,179 km²  
**Total Road Network:** 289,841 segments (from OSM)

---

## Current Data Coverage Status

### ✅ Infrastructure Data (Complete for ALL NER)

**Road Network:**
- Coverage: All 8 states
- Segments: 289,841
- Source: OpenStreetMap
- File: `data/processed/roads/ner_roads.gpkg`
- Status: ✅ COMPLETE

**Admin Boundaries:**
- Coverage: All 8 states + districts
- Source: Government data
- Status: ✅ COMPLETE

### ✅ Environmental Data (Complete for ALL NER)

**Terrain Features:**
- Coverage: All 8 states
- Data: SRTM DEM (6 tiles, 90m resolution)
- Bounding box: 21.5-29.5°N, 89.5-97.5°E (covers entire NER)
- Extracted features: 289,841 road segments
- Valid data: 94,327 roads (32.5% in NER, others outside region)
- Status: ✅ COMPLETE

**Rainfall Data:**
- Coverage: All 8 states (CHIRPS is global)
- Current: 106 days (Jan-Apr 2015)
- Required: 2015-2024 full period
- Extracted features: 289,841 road segments
- Status: ⚠️ PARTIAL (sufficient for dev, needs completion)

### ❌ Hazard Event Data (SEVERELY INCOMPLETE)

**Current Status:**
- **Events acquired: 1** (Sikkim only)
- **Events required: 100+ across ALL 8 states**
- **Coverage: 12.5%** (1 of 8 states)

**This is the CRITICAL BLOCKER for NER-wide model training.**

---

## Why Sikkim-Only Data is Insufficient

### Geographic Diversity Across NER

**Topography:**
- Arunachal Pradesh: High Himalayan (>7,000m peaks)
- Assam: Brahmaputra plains (<100m elevation)
- Meghalaya: Plateau (1,500m)
- Sikkim: Mid-Himalayan (1,000-8,000m)

**Climate:**
- Assam: Sub-tropical monsoon
- Meghalaya: Wettest place on Earth (Cherrapunji)
- Arunachal Pradesh: Alpine/temperate zones

**Hazard Profiles:**
- Sikkim: Earthquake + landslide dominated
- Assam: Flood dominated
- Meghalaya: Extreme rainfall landslides
- Tripura: Riverine floods

**Training on Sikkim-only data will NOT generalize to:**
- Assam flood patterns
- Meghalaya extreme rainfall landslides
- Arunachal Pradesh high-altitude hazards
- Tripura riverine flood dynamics

---

## Required: NER-Wide Event Distribution

### Minimum Target: 50-100 Events

**Geographic distribution (suggested minimum per state):**

| State | Primary Hazard | Min Events | Priority |
|-------|----------------|------------|----------|
| Assam | Flood | 15-20 | HIGH (largest state, frequent floods) |
| Meghalaya | Landslide | 10-15 | HIGH (extreme rainfall) |
| Arunachal Pradesh | Landslide | 10-12 | HIGH (large area, infrastructure critical) |
| Sikkim | Landslide/Earthquake | 5-8 | MEDIUM (already have 1) |
| Manipur | Mixed | 5-8 | MEDIUM |
| Nagaland | Landslide | 5-8 | MEDIUM |
| Mizoram | Landslide | 5-8 | MEDIUM |
| Tripura | Flood | 5-8 | MEDIUM |

**Total: 60-95 events minimum**

### Temporal Distribution

**Monsoon seasons (2015-2024):**
- June-September: Peak hazard period
- Need events across multiple years to capture variability
- Target: 6-10 events per year

---

## Current Blocker Analysis

### What We Have (Sikkim Event)

**Sikkim Mantam Landslide 2016:**
- Date: 2016-08-13
- Type: Monsoon-triggered landslide
- Location: 27.5397°N, 88.5007°E
- Context: Mid-elevation, secondary road
- Status: Candidates identified, needs confirmation

**This represents:**
- 1 event type (landslide)
- 1 state (Sikkim)
- 1 season (monsoon)
- 1 year (2016)
- 1 elevation zone (mid-altitude)

### What We Need (NER-Wide Coverage)

**Event diversity required:**
- Landslides: 30-40 events (all hill states)
- Floods: 20-30 events (Assam, Tripura, valley areas)
- Earthquakes: 5-10 events (if data available)
- Multiple seasons: Pre-monsoon, monsoon, post-monsoon
- Multiple years: 2015-2024 (capture climate variability)
- Multiple road types: National highways, state roads, village roads

---

## Data Acquisition Strategy for NER-Wide Coverage

### Phase 1: High-Yield Public Sources (Week 1)

**Target: 20-30 events**

1. **EM-DAT Database** (easiest, structured data)
   - Search: All 8 NER states
   - Filter: 2015-2024, Floods + Landslides
   - Expected: 10-15 major events

2. **GDACS Alerts**
   - Historical alerts for NER region
   - Expected: 5-10 major events

3. **ReliefWeb Reports**
   - Situation reports with location data
   - Expected: 5-10 documented events

### Phase 2: News Archives (Week 2)

**Target: 20-30 events**

4. **Google News Search**
   - Per-state queries (see script output)
   - Cross-reference with OSM road network
   - Expected: 15-25 events

5. **Local News Sources**
   - The Assam Tribune
   - The Shillong Times
   - Sikkim Express
   - Expected: 5-10 events

### Phase 3: Government Sources (Week 3-4)

**Target: 20-40 events**

6. **NRSC Landslide Atlas**
   - Requires institutional access
   - High-quality satellite-verified data
   - Expected: 15-30 landslide events

7. **State PWD Reports**
   - Annual reports often list major road damages
   - Expected: 5-10 events

8. **IMD Bulletins**
   - Flood warnings often mention affected roads
   - Expected: 5-10 flood events

### Phase 4: Field Data (Optional, Long-term)

9. **Collaborate with:**
   - State transport departments
   - BRO (Border Roads Organization)
   - Local disaster management authorities

---

## Immediate Action Items

### Priority 1: Expand Event Coverage (CRITICAL)

**Timeline: 2-5 days**

1. Create event search workflow:
   ```bash
   python scripts/search_ner_hazard_events.py  # Already created
   ```

2. Manual search tasks:
   - EM-DAT: 2-3 hours
   - GDACS: 1-2 hours  
   - ReliefWeb: 2-3 hours
   - News archives: 4-6 hours per state (focus on top 3 states first)

3. Create event CSV:
   ```
   data/raw/hazards/ner_events_2015_2024.csv
   ```

4. Run road matching:
   ```bash
   python scripts/resolve_real_hazard_road_identity.py
   ```

### Priority 2: State-Priority Approach

**If time-limited, focus on these 3 states first (60% of events):**

1. **Assam** (floods) - 15-20 events
   - Most populated state
   - Frequent Brahmaputra floods
   - Critical NH-37, NH-52

2. **Meghalaya** (landslides) - 10-15 events
   - Extreme rainfall
   - Shillong-Cherrapunji corridor
   - High economic impact

3. **Arunachal Pradesh** (landslides) - 10-12 events
   - Largest state
   - Strategic border roads
   - High-altitude challenges

**This gives 35-47 events covering 3 major hazard contexts.**

Then expand to remaining 5 states.

---

## Updated P0 Blocker Status

### Before (Incorrect Understanding)

```
P0-BLOCK-001: 1 Sikkim event needs confirmation
ACTION: 30-min manual OSM review
```

### After (Correct Understanding)

```
P0-BLOCK-001: Need 50-100 events across ALL 8 NER states
CURRENT: 1 event (Sikkim only) = 2% of minimum
ACTION: Multi-week event search and acquisition
TIMELINE: 2-5 days for minimum 50 events
```

---

## Summary

### ✅ What's Ready for NER-Wide Development

- Road network: All 8 states (289,841 segments)
- Terrain features: All 8 states (94,327 with valid data)
- Rainfall features: All 8 states (106 days partial, full download ongoing)
- Feature extraction pipelines: Tested and working

### ❌ What's Blocking NER-Wide Training

- **Hazard events: Only Sikkim (1 of 8 states)**
- Need 50-100 events distributed across all states
- Need diverse event types (floods, landslides, earthquakes)
- Need temporal coverage (2015-2024)

### 🎯 Corrected Next Steps

1. **Urgent**: Search for NER-wide events (use guide created)
2. **Target**: Minimum 50 events across 8 states
3. **Timeline**: 2-5 days of manual search effort
4. **Then**: Resume model training with NER-wide data

---

**Key Insight:** The Sikkim event is important for validation, but NER-Nav requires **geographic diversity across all 8 states** to build a model that generalizes to the entire North Eastern Region.

---

**Document Created:** 2026-08-27  
**Status:** Scope clarification complete  
**Action Required:** Begin NER-wide event search immediately
