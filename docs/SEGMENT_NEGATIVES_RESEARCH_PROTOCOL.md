# Segment-Scoped Negative Observations — Web Research Protocol

Target: replace the 24 corridor-level road-status controls in
`data/raw/hazards/negative_observations.csv` with **segment-scoped
unaffected-road observations**, and grow the count so the production model can
rank a genuinely healthy high-rainfall stretch against a disrupted one.

This is the audit's current P0:
> "Segment-scope the observation-backed negative evidence: replace
> corridor-level road-status rows with segment-level unaffected observations
> in negative_observations.csv"

Dependency note (Track B, NOT this doc): the MODEL gate
(`ROC-AUC >= 0.75 AND recall >= 0.7` over `>= 30` independent future test
events) cannot be satisfied by negatives alone. It additionally needs a much
larger confirmed-event corpus. Event expansion is covered by
`docs/EVENT_COLLECTION_WORKFLOW.md`, `docs/LABEL_EXPANSION_PLAN.md`, and
`scripts/add_confirmed_event.py`. This protocol only produces the negative half
of the separation.

---

## 1. Admission contract (exact, from code)

`validate_negative_observation_file()` + `validate_negative_observation_quality()`
in `ml/data/training_gate.py` and the builder in
`scripts/build_production_dataset.py`.

### 1.1 CSV schema (10 columns, all required)

| column | rule |
|---|---|
| `observation_id` | unique; pattern `nh<X>_seg_<YYYYMMDD>_<NNN>` |
| `osm_id` | positive integer; must exist in `data/processed/roads/ner_roads_districts.gpkg` |
| `ref` | `ref` value of that OSM way (e.g. `NH37`) |
| `prediction_time` | ISO date; anchor for features |
| `observation_window_end` | `>= prediction_time + 7 days` (full-horizon coverage) |
| `source_url` | real HTTP URL |
| `source_publish_date` | ISO date; **must be `<= prediction_time`** |
| `observation_status` | UPPERCASE; must NOT contain `CLEAR`, `REOPEN`, `RESUMED`, `RESTRICT` (substring check) |
| `observation_scope` | must start with `segment_` (e.g. `segment_observed_open`) |
| `evidence_note` | exact, verbatim-justified citation of the affirmative language |

### 1.2 Ingestion

Each row becomes exactly ONE negative sample (`label=0`,
`label_source=real_observed_unaffected`, `sample_kind=observed_open_negative`)
whose `osm_id` is merged against the road/terrain network and whose
`prediction_time` is merged against `road_rainfall_features_temporal.parquet`.

### 1.3 Time/coverage constraints

- Rainfall lookback is 30 days before `prediction_time`, so valid anchors are
  **2015-02-01 .. 2026-07-24** (CHIRPS archive covers 2015-01-01 .. 2026-07-31;
  window end must also be within it).
- Recommended anchoring: `prediction_time = source_publish_date`,
  `observation_window_end = source_publish_date + 7`. This keeps
  `publish <= prediction` and the window rule simple and honest: the source
  affirms normal movement on the publish date and the 7-day band that follows
  is a documented curation judgement, not a field survey.

### 1.4 Hard fails (do not submit such rows)

- Absence-based evidence ("no disruption reported", "not on the blocked
  list") — explicitly rejected ("Absence from an incident feed ... not
  evidence of non-occurrence").
- Same-road time offsets of confirmed events.
- Status that implies prior/ongoing disruption:
  - `CLEAR...` (cleared, clearance)
  - `REOPEN...` (reopened, reopening)
  - `RESUMED` (resumed)
  - `RESTRICT...` (restricted, restrictions) — e.g. night-time curbs
- Corridor-level scope (`corridor_traffic_resumed`, `corridor_single_lane`,
  ...) — these are precisely what the P0 asks us to replace.
- `source_publish_date > prediction_time`.
- Window shorter than 7 days.
- Modelled/synthetic/generated rows of any kind.

---

## 2. Evidence archetypes that qualify

A row qualifies only if the source **affirmatively** reports FULL normal
movement on a *named stretch / bounded segment* with no clearance, no
restriction, and no reopened framing. Four archetypes:

1. **Rest-of-corridor affirmation.** A disruption is confined to spot X; the
   article affirms the rest of the same NH is open/normal:
   - "vehicular movement on the rest of NH-6 remained normal"
   - "only [spot] is affected; movement continues uninterrupted on the
     remaining stretch"
   Maps to a DIFFERENT OSM way on the same `ref`, away from X.

2. **Proactive status advisory.** PTA/NHIDCL/DM/police traffic bulletin during
   a weather event:
   - "NH-37 remains open. Vehicular movement normal."
   - "No restrictions on movement along NH-10."
   (No clearance/reopen/resumed/restrict wording.)

3. **Queued-independent passage report.** An eyewitness/official quote that a
   specific stretch was passable throughout a monsoon spell while nearby
   stretches were not.

4. **Multi-route statement.** A report that alternative commuting on a given
   corridor "continues normally" where the corridor serves as a lifeline and
   the report explicitly rules out restrictions to through-traffic.

Worked example of a **near-miss that must be EXCLUDED**:
`data/raw/assamtribune_2026_nh315a.html` ("vehicular movement remains
unaffected for now") FAILS because the same article imposes 6pm-6am movement
restrictions, reports narrowed/collapsed roadway ("RESTRICT"/restriction), and
therefore cannot support a clean 7-day unaffected window. Keep such sources as
documentation only.

---

## 3. Search procedure

### 3.1 Domains (already in the repo's source pool plus confirmed outlets)

- assamtribune.com, eastmojo.com, nenow.in (north east now), nagalandpost.com,
  theshillongtimes.com, easternmirrornagaland.com, morungexpress.com,
  thenortheasttoday.com, sikkimexpress.com, imphaltimes.com
- State government press releases (`*.gov.in`): Meghalaya PWD, Manipur PWD,
  Nagaland PWD/DMR, Sikkim NHIDCL/PWD, Arunachal PWD, Assam NHIDCL.

### 3.2 Query templates (news engine of choice)

```
"NH-6" landslide Meghalaya "traffic normal" 2024
"NH-37" Manipur landslide "movement normal" 2025
"NH-102B" Mizoram "rest of the" highway traffic
"NH-10" Sikkim "vehicular movement" normal 2025
"NH-13" Arunachal "open to traffic" 2024
"NH-29" Nagaland "vehicular movement" continues
"NH-2" Kohima "traffic moving normally"
"NH-27" Assam "no disruption" movement highway
"NH-208A" Tripura traffic movement normal
site:*.gov.in {district} "traffic" "open" "normal" highway 2025
```

Search windows: monsoon months (May-Oct) of 2015-2026, prioritising the
current dataset's weakest corridors (section 4).

### 3.3 Capture rule

For every qualifying article, save the HTML into `data/raw/` using the
existing naming convention (`<outlet>_<YYYY>_<road>_<place>.html`), then add
the CSV row. The saved file is the auditable source; the URL in the CSV must
resolve to it.

---

## 4. Target corridors & anchors

| NH | State(s) | Priority | Anchor window | Note |
|----|----------|----------|---------------|------|
| NH102B | Manipur/Mizoram | HIGH | 2025 monsoon | Only 2 positives; Guite Rd |
| NH10 / NH510 | Sikkim | HIGH | 2024-2025 monsoon | Chungthang/Bardang corridor |
| NH37 | Manipur | HIGH | 2024-2025 monsoon | Strong positives; needs healthy-segment contrast |
| NH13 | Arunachal | HIGH | 2024-2025 monsoon | Tezpur-Tawang segments |
| NH6 | Meghalaya/Mizoram | MEDIUM | 2024-2025 monsoon | Jowai-Malidor, Kuliang |
| NH29 | Nagaland | MEDIUM | 2024-2025 monsoon | Dimapur-Kohima |
| NH2 | Nagaland/Assam | MEDIUM | 2025 monsoon | Kohima-Mao |
| NH27 | Assam | LOW | 2024-2025 monsoon | Flat; needs affected-vs-open same-day pairs |
| NH208A | Tripura | LOW | 2025 monsoon | Flat |

Target diversity: for each corridor, prefer (a) a high-rainfall hill stretch
with affirmed normal movement, (b) a flat stretch affirmed open during a flood
window, and (c) a stretch geographically adjacent to a confirmed positive.

---

## 5. Producing a row

1. Resolve the named healthy stretch to an OSM way id
   (Overpass Turbo):
   ```
   way["ref"="NH{X}"](around:10000,{lat},{lon});
   out ids;
   ```
2. Verify the id exists in `data/processed/roads/ner_roads_districts.gpkg`
   and record its `ref`.
3. Add the row with `observation_scope=segment_observed_open`,
   `observation_status=NORMAL_MOVEMENT_CONFIRMED` (or
   `SEGMENT_OPEN_TRAFFIC_NORMAL`), and an `evidence_note` that quotes the
   affirmative phrase verbatim and names the stretch + publish date.
4. Re-run the validators (section 6) immediately; do not batch-add rows
   and validate late.

---

## 6. Validation, dataset, and audit loop

```powershell
# 1. file schema + quality gates
.venv\Scripts\python.exe -m unittest tests.test_training_input_gates

# 2. rebuild the production dataset (new negatives become samples)
.venv\Scripts\python.exe scripts\build_production_dataset.py

# 3. retrain the production model with the honest gate
.venv\Scripts\python.exe scripts\train_production_risk_model.py

# 4. re-audit (MODEL is the gate that must move; DOCUMENTATION/etc must not regress)
.venv\Scripts\python.exe scripts\audit_ml_production_readiness.py

# 5. full test suite
.venv\Scripts\python.exe -m unittest discover -s tests
```

Working parquet/QA/manifests are gitignored; only the MD, the new HTML source
files, and `data/raw/hazards/negative_observations.csv` will require commits.

---

## 7. Batch milestones (revised after execution)

- **Batch 0 (EXECUTED, platonic target 18 rows)**: dual rows for the 3 HIGH
  corridors in their 2025-monsoon windows were the planned volume, but the
  admissible yield was exactly **1 row** (`nh2_seg_20241126_001`, NH-2 via
  Sangai 2024-11-26 — see section 8). Full rebuild/retrain/audit/tests green
  (MODEL the only remaining gate).
- **Batch 1+ (revised)**: no fixed volume targets. Continue hunting archetype
  1/4 affirmations at the sustainable rate (~1 row per corridor-year) across
  NH6/NH29/NH2/NH13/NH10/NH102B for 2024-2025; stop when two consecutive
  search rounds of each corridor's monsoon yield no new admissible source.
- Negatives are a supporting stream, not the MODEL-gate engine:
  close MODEL via Track B (`docs/EVENT_COLLECTION_WORKFLOW.md`).

## 8. General findings (executed Batches, 2026-09)

Research was executed across the news pool + official bulletins for the HIGH
and MEDIUM corridors (section 4), covering 2023-2026 monsoon windows. Final
yield: exactly ONE admissible row (see 8.2). The protocol's original volume
targets are not reachable from this evidence ecosystem.

### 8.1 What dominates the 2023-2026 press for these corridors

- **Manipur NH-2/NH-37**: the ethnic-conflict regime, not weather. Recurring
  bullets of the form "N vehicles facilitated on NH-2 / M on NH-37" are
  security-force convoy-escort operations (indiatodayne 2024-08-03: 273/64;
  krctimes 2024-05-09: 198/183; sentinel 2024-10-05: 169/172). Armed escort =
  restricted/controlled movement, not clean normal movement → EXCLUDE.
- **Sikkim NH-10/NH-510**: restriction-then-restoration chains (May-Oct 2025
  fully documented). Proactive advisory affirmations without restrict/reopen
  wording are absent.
- **NH-102B**: standstill reports; the complement corridor (NH-2 Tedim Rd) is
  conflict-blocked. No clean window.
- **NH-29 (Sep 2024)**: official travel advisories impose time-slots /
  one-way / light-only passage → restricted → EXCLUDE.
- **NH-6/NH-13**: chronic disruption + restoration/clearance framing
  ("traffic restored", "reopens after", "movement restricted"). Clean-window
  affirmations of the *rest of the corridor* are rare and always paired with
  active-work language.

### 8.2 The admissible row (type 4, multi-route/cross-corridor)

- Sangai Express, 2024-11-26: NH-37 (Thongju/Churachandpur) disrupted ↔ NH-2
  Imphal-Dimapur affirmed flowing 200-250 trucks/day. Row
  `nh2_seg_20241126_001`, osm 44886096, source saved to
  `data/raw/thesangaiexpress_2024_nh2_imphal_dimapur.html`. The affirmation
  is the neutral complement inside a disruption article — the only archetype
  that reliably appears.

### 8.3 Excluded near-misses worth keeping on file

- NH-2 "stranded vehicles moving after MDA bypass at Phesama" (e-pao
  2025-06-05) — post-repair reopen framing; also cuts the window feature
  history (Phesama cutoff 2025-05-31).
- "NH-2 never blocked" claim by KZC (TOI 2025-09-05) — stakeholder advocacy
  in an active dispute, not a neutral road-status report.
- All NH-37/NH-2 convoy-facilitation bulletins (see 8.1) — escort regime.
- Sangai 2026-08-30 "resumption" framing — excluded by keyword rule.

### 8.4 Ceiling and implication

- Admissible segment-scoped affirmations occur at ~1 per corridor-year in this
  ecosystem (roughly one clean archetype-1/4 report per corridor per monsoon).
- The MODEL gate (`>= 30` independent future test events) cannot be met by
  negatives at any practically reachable volume. Closing P0 still leaves the
  gate red. Gating progress therefore requires **Track B** (event expansion,
  `docs/EVENT_COLLECTION_WORKFLOW.md`); negatives continue to be added as
  found, at the reduced, evidence-honest rate enforced here.

---

## 10. Training contamination diagnosis (2026-09)

### 10.1 The 24 inadmissible rows poison the model signal

All 24 corridor-level negatives carry clearance/reopening status
(`OPEN_AFTER_CLEARANCE`, `FULLY_REOPENED`, `REOPENED_REGULATED`) and describe
roads that WERE disrupted. They were nonetheless ingested as training
negatives (`label=0`). The consequence:

- Negative rows have **higher** mean rainfall than positives (rainfall_30day:
  465mm vs 359mm), teaching the model "high rain = safe" — an inverted
  signal.
- Grouped-CV pooled ROC-AUC = 0.31 (worse than chance 0.5).
- `enforce_training_input_gate()` checks file schema and label sources but
  does NOT call `validate_negative_observation_quality()` — so the poisoned
  rows pass into training.

### 10.2 Why training cannot be fixed without additional clean negatives

Excluding the 24 poisoned rows from the dataset (keeping only the 1 clean
segment-scoped negative) leaves exactly **1 negative sample**. Under the
hard audit constraints:

- `VALIDATION` gate: `n_test_events >= 30`
- `CALIBRATION` gate: `n_val_events >= 10`

With 63 event groups (38 positive + 25 negative observations), the single
clean negative sits at chronological group ~33. The maximum training fold
size is 63 − 30 − 10 = **23 groups**, which cannot contain the clean
negative. The training fold therefore has **zero** negatives, and XGBoost
cannot fit (`Empty dataset at worker` / `predict_proba[:,1]` IndexError).

### 10.3 Implication

The `GATED_LOW_CONFIDENCE` state with a 0.31 grouped-CV is the **honest,
trainable state given current data**. Any attempt to improve the MODEL gate
requires at minimum:

1. More clean, segment-scoped negatives (≥2 for a viable split, ideally ≥15
   for meaningful two-class training) from **operational sources** (PWD/DDMA
   daily "no disruption" lists, NHIDCL advisories, satellite change detection)
   — the press/bulletin ecosystem cannot supply them.
2. More confirmed positive events (Track B, `EVENT_COLLECTION_WORKFLOW.md`).

The negative-contamination fix (section 10.1) should be applied once the
clean-negative pool reaches ≥20 rows.

---

## 11. Honesty rules (non-negotiable)

1. No fabricated, inferred, or absence-derived rows.
2. The `evidence_note` quotes the article; the HTML is saved in `data/raw/`.
3. A row is submitted only if it passes the quality gate as-is; never loosen
   the gate to admit a row.
4. If a source shows ANY restriction/clearance/narrowing language, it is an
   exclusion, not evidence — keep it as a documented near-miss.