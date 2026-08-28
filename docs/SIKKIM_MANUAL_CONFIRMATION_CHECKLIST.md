# Manual Road Confirmation Checklist - Sikkim Mantam 2016

**Event:** NRSC Sikkim Mantam Landslide (2016-08-13)  
**Purpose:** Manually confirm which OSM road segments were affected  
**Estimated Time:** 30 minutes  
**Priority:** P0 (unblocks training)

---

## Event Context

**Location:** 27.5397°N, 88.5007°E (landslide depletion zone center)  
**Reported Impact:**
- 300m of "Passingdang-Mantam Road" washed away
- Kanka bridge submerged
- 8 villages cut off: Tingvong, Lingdem, Laven, Kayeem, Lingzya, Bay, Sakyong Pentong, Ruklu Kayeem

**Source:** NRSC report + Martha et al. (2017), Current Science 113(7)

---

## Top 3 Priority Candidates for Manual Review

### Candidate 1: OSM ID 1215136364
- **Type:** secondary road
- **Distance:** 443m from landslide center
- **OpenStreetMap Link:** https://www.openstreetmap.org/way/1215136364
- **Inspect:** https://www.openstreetmap.org/?mlat=27.5397&mlon=88.5007#map=16/27.5397/88.5007

**Review checklist:**
- [ ] Does this road segment lie between Passingdang and Mantam villages?
- [ ] Does this road pass through or near the landslide coordinates?
- [ ] Is this a logical main connecting route for the affected villages?
- [ ] Does the road alignment suggest it could have experienced damage?

**Confirmation decision:**
- If YES to ≥3 questions above → Mark as `confirmed_affected=true`
- If NO or UNCERTAIN → Move to Candidate 2

---

### Candidate 2: OSM ID 599418823
- **Type:** secondary road
- **Distance:** 530m from landslide center
- **OpenStreetMap Link:** https://www.openstreetmap.org/way/599418823
- **Inspect:** https://www.openstreetmap.org/?mlat=27.5397&mlon=88.5007#map=16/27.5397/88.5007

**Review checklist:**
- [ ] Does this road segment connect to the affected villages?
- [ ] Is this road spatially aligned with the reported impact zone?
- [ ] Could this be an alternate route or part of the same Passingdang-Mantam corridor?

**Confirmation decision:**
- If YES to ≥2 questions → Mark as `confirmed_affected=true`
- If UNCERTAIN → Mark as `requires_further_investigation=true`

---

### Candidate 3: OSM ID 266906129
- **Type:** tertiary road
- **Distance:** 857m from landslide center
- **OpenStreetMap Link:** https://www.openstreetmap.org/way/266906129
- **Inspect:** https://www.openstreetmap.org/?mlat=27.5397&mlon=88.5007#map=16/27.5397/88.5007

**Review checklist:**
- [ ] Does this tertiary road provide access to cut-off villages?
- [ ] Is this road downstream/downslope from the landslide?

**Confirmation decision:**
- If YES → Mark as `possibly_affected=true` (secondary impact)
- If NO → Mark as `unaffected=true`

---

## Manual Confirmation Procedure

### Step 1: Open OpenStreetMap Inspector

Visit: https://www.openstreetmap.org/?mlat=27.5397&mlon=88.5007#map=15/27.5397/88.5007

**What to look for:**
1. Road network topology connecting villages
2. Main road corridor heading north/south from event center
3. Bridge locations (look for bridge=yes tags)
4. Road connectivity patterns

### Step 2: Cross-Reference with Villages

**Search for affected villages on OSM:**
- Tingvong
- Lingdem
- Laven
- Mantam village

**Question:** Which road(s) provide the main access to these villages from the south?

### Step 3: Identify Logical Main Route

**Key insight:** The report states the road was washed away, cutting off 8 villages. This suggests:
- The affected road is the PRIMARY access route (likely secondary or tertiary highway)
- The road lies between the landslide site and the cut-off villages
- Alternative routes do not exist (otherwise villages wouldn't be "cut off")

**Expected pattern:** Look for a single main road corridor passing through or very near 27.5397°N, 88.5007°E that connects to multiple villages to the north.

### Step 4: Document Findings

Fill out the confirmation table below:

| OSM ID | Confirmed? | Evidence | Confidence |
|--------|-----------|----------|------------|
| 1215136364 | ☐ YES ☐ NO | Visual topology match / Proximity / Bridge presence | ☐ High ☐ Medium ☐ Low |
| 599418823 | ☐ YES ☐ NO | Visual topology match / Proximity | ☐ High ☐ Medium ☐ Low |
| 266906129 | ☐ YES ☐ NO | Visual topology match / Proximity | ☐ High ☐ Medium ☐ Low |

---

## Confirmation Metadata Template

For each confirmed road, record:

```json
{
  "osm_id": 1215136364,
  "source_supported": true,
  "source_evidence": "NRSC report describes 'Passingdang-Mantam Road'; visual inspection confirms this OSM segment is the main road corridor in the area",
  "confirmed_affected": true,
  "confirmation_evidence": "Road passes within 443m of landslide center at 27.5397°N, 88.5007°E; topology indicates this is the primary access route to affected villages; no alternative routes visible",
  "confirmation_method": "manual_osm_inspection",
  "confidence_level": "medium",
  "confirmed_by": "manual_review",
  "confirmed_at": "2026-08-26T20:45:00Z",
  "uncertainty_notes": "OSM road lacks name attribute; confirmation based on topology and proximity; satellite imagery would increase confidence",
  "recommended_followup": "Acquire Sentinel-2 imagery from 2016-08-14 to visually confirm damage"
}
```

---

## Implementation: Update Confirmation File

After manual review, run:

```python
import pandas as pd

# Load candidates
df = pd.read_parquet(
    'data/processed/hazards/nrsc_sikkim_mantam_2016_confirmed_road.parquet'
)

# Example: Confirm OSM ID 1215136364
df.loc[df['osm_id'] == 1215136364, 'source_supported'] = True
df.loc[df['osm_id'] == 1215136364, 'confirmed_affected'] = True
df.loc[df['osm_id'] == 1215136364, 'confirmation_evidence'] = (
    'Manual OSM inspection: road passes within 443m of landslide; '
    'topology indicates primary access route to affected villages'
)
df.loc[df['osm_id'] == 1215136364, 'confidence_level'] = 'medium'

# Save updated file
df.to_parquet(
    'data/processed/hazards/nrsc_sikkim_mantam_2016_confirmed_road.parquet'
)

print(f"Confirmed roads: {df['confirmed_affected'].sum()}")
```

---

## Expected Outcome

**Minimum:** 1-2 confirmed road segments  
**Ideal:** 3-5 confirmed road segments

**Training readiness:**
- With 1-2 confirmed positives + 30-33 confirmed negatives (unaffected candidates) = ~35 labeled examples
- This is sufficient to **unblock Phase 2** (combine with rainfall + terrain features)
- More events needed for robust model (target: 100+ positive labels)

---

## After Confirmation

1. Re-run training gate audit:
   ```bash
   python scripts/audit_real_hazard_training_gate.py
   ```

2. Expected output:
   ```
   Real training labels: 1-3
   STATUS: TRAINING GATE UNBLOCKED (if ≥1 label)
   ```

3. Proceed to Phase 2: Build real training dataset with rainfall + terrain features

---

## Conservative Approach (If Uncertain)

**If visual inspection is inconclusive:**

1. Mark top 1-2 candidates as `confidence_level='low'`
2. Flag for follow-up with satellite imagery
3. DO NOT use low-confidence labels for final model evaluation
4. Use only as development/testing labels
5. Prioritize acquiring additional well-documented events (Option C)

---

**Status:** Manual confirmation checklist prepared  
**Next Action:** Human reviewer performs OSM inspection (30 minutes)  
**Alternative:** Proceed with rainfall/terrain acquisition in parallel

---

**End of Manual Confirmation Checklist**
