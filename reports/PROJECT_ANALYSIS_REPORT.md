# NER-Nav — Complete Project Analysis Report

**Problem Statement:** AI-Based Smart Logistics and Accessibility Intelligence Platform for the North Eastern Region (ID 26002, MDoNER)
**Analysis date:** 2026-09-19
**Repo state:** branch `ml/production-ready-model`, HEAD `0937986` (worktree has 3 modified files + untracked prod bundles)

---

## 1. Executive Summary

| Dimension | Assessment |
|---|---|
| **Overall readiness** | ML research pipeline complete & reproducible; application platform (dashboards, routing, GPS tracking, mobile field reporting, alerts) is **scaffold-only**. |
| **Production model** | `GATED_LOW_CONFIDENCE`, not deployable. Latest run (2026-09-12) regressed to test ROC-AUC **0.50**, recall **0.0** — the model now emits no alerts at all. |
| **Demo model** | `HACKATHON_DEMO_READY_LIMITED_CONFIDENCE` (2026-09-11_n62_p114, grouped-CV ROC-AUC ~0.555) — served by the live FastAPI demo feed. Research use only. |
| **Honesty / engineering discipline** | Excellent. Failed-closed gates, no synthetic labels in training, hash-verified artifacts, event-disjoint chronological splits, full leakage analysis, and unusually candid documentation. |
| **Problem-statement coverage (a–h)** | **2 of 8 features implemented** (hazard-risk prediction; basic alert flag in API). Remaining 6 (GPS tracking, alternate routing + ETA, field reporting, dashboards, multilingual + offline sync, live weather/gov integration) are empty scaffolds. |

**Verdict: a genuinely strong, honest AI/ML research foundation with a live demo API — but still a *research project*, not yet a *platform*. It cannot be entered as a complete solution to PS 26002 in its current form.** The decisive blocker is ground-truth data scarcity (verified negatives + independent future test events), *not* code.

---

## 2. Alignment with the Problem Statement

### 2.1 Required platform capabilities (Description a–h)

| # | Required capability | Status | Evidence |
|---|---|---|---|
| a | Real-time road/bridge/transport accessibility monitoring | **Not implemented** | No live road-status store; only a static `demo_risk_feed.json` snapshot from `data/predictions/`. DB has a single `nodes` table (no segments/alerts/tracking). |
| b | Predict disruptions (landslide, flood, heavy rain, road damage, congestion) | **Implemented (research)** | 7-day disruption-risk XGBoost on 21 features (rainfall windows, terrain, seasonal, interactions); `scripts/predict_real_temporal.py`, `/api/v1/predict`. **Gated — not production-safe.** |
| c | Alternate-route suggestions + estimated delays | **Not implemented** | `routing/` (`algorithms/`, `graph/`, `services/`, `weights/`) contains only `.gitkeep`. Route API schemas exist but no route endpoint/router logic. Delay/ETA prediction explicitly blocked (no travel-time labels). |
| d | GPS-based vehicle tracking of essential goods | **Not implemented** | No GPS ingestion, no vehicle DB, no movement feeds anywhere in code. |
| e | Automated alerts (blocked roads, inaccessible regions, delayed deliveries, high-risk corridors) | **Partial** | `operating_decision: ALERT/NO_ALERT` + risk bands (LOW/MEDIUM/HIGH) in the API contract and `risk_feed()`; **no notification engine, no push/email/SMS, no delivery-delay alerts.** |
| f | Field officials upload geo-tagged updates, photos, incident reports | **Not implemented** | `mobile/src/{components,offline,screens,services,sync}/` all empty scaffolds. |
| g | Centralized dashboards (district connectivity, bottlenecks, emergency routes, real-time supply movement) | **Not implemented** | `frontend/src/{components,features,hooks,maps,pages,services}/` empty scaffolds. Only raw JSON risk feed exists. |
| h | Multilingual notifications + offline sync for low-network areas | **Not implemented** | `mobile/src/offline/` and `sync/` empty. No i18n. |

### 2.2 Expected-solution components

| Expected deliverable | Status |
|---|---|
| AI-powered route prediction & optimization engine | Prediction engine: done (gated). Route optimization: **missing**. |
| GIS-enabled accessibility monitoring dashboard | **Missing** (maps/ empty). |
| GPS-based vehicle tracking system | **Missing**. |
| Real-time alert & notification mechanism | Flag-level only; no mechanism. |
| Mobile/web app for field-level reporting & monitoring | **Missing**. |
| Weather API / transport DB / gov monitoring integration | Weather **offline ingestion** done (CHIRPS, IMD, Open-Meteo); **no live API integration**, no transport/government feeds. |
| Cloud infrastructure, secure data, offline support | Docker-compose PostGIS scaffold + SQLAlchemy migration framework. No cloud deployment, no auth, no role-based access, no offline client. |

---

## 3. Repository / Architecture Analysis

### 3.1 What exists and works

```
backend/            FastAPI service — LIVE
  main.py           app factory + /health
  api/routes.py     GET /api/v1/models, GET /api/v1/feed, POST /api/v1/predict
  services/risk_model.py   hash-verified bundle loading, XGBoost scoring,
                           isotonic calibrator, TreeSHAP top-5 factors,
                           risk bands, strict tag authorisation
  services/features.py    input validation, derived-feature recomputation
  schemas/predict.py      pydantic request/response contracts
  models/node.py, db/session.py, core/config.py   PostGIS stub (nodes)
ml/                 Feature engineering, labels, splits, training
  features/temporal_design.py   54 verified event/road tuples + design rules
  features/{rainfall,seasonal,missingness,compose}.py
  labels/generate.py, splits/temporal.py, data/training_gate.py
scripts/            80+ scripts (acquisition, dataset build, train, predict,
                    audit, runbook)
data/               Raw (OSM, SRTM, CHIRPS, IMD, events, NHAI, MoRTH, CWC,
                    news), processed, models (bundles), predictions
database/           Alembic migrations + PostGIS
tests/              18 test modules
docs/               26 documents describing sources, gates, monitoring,
                    remediation runbooks
```

### 3.2 The only fully wired vertical slice

The pipeline that is real and reproducible end-to-end:

```
RAW (OSM + SRTM/Copernicus + CHIRPS/IMD + event registry + negatives)
  → build_real_temporal_dataset → labeled feature matrix
  → temporal split (39 train / 10 val / 30 test events)
  → XGBoost → calibration → threshold → versioned, hash-locked bundle
  → FastAPI demo feed (GET /api/v1/feed) + single-road POST /api/v1/predict
```

This is the honest strength of the project. Everything *is* clockwork except the model signal.

### 3.3 Scaffolds (empty — `.gitkeep` only)

`frontend/src/**`, `mobile/src/**`, `routing/**` (all), `simulators/`, `backend/tests/`, `database/schemas/`, `database/seeds/`, `ml/{evaluation,inference,models,training}/`.

The folder structure anticipates the full platform, but none of these microservices/views are implemented.

---

## 4. Data & ML Deep Dive

### 4.1 Data holdings (substantial)

- **Road network:** OSM/Geofabrik NER extract — **289,841 ways**, extracted to `ner_roads.gpkg`.
- **Terrain:** SRTM v4 + SRTMGL1 1-arc-sec + Copernicus GLO-30 (`ner_dem_full.tif`).
- **Precipitation:** CHIRPS v2 daily rasters (2015–2026, ~44,000+ committed files), IMD gridded 0.25° (2001–2025), Open-Meteo station series.
- **Ground truth:** 54 confirmed disruption events (event→OSM-way resolved, source-dated), 25 observation-backed negative records (`negative_observations.csv`, each with source URL + 7-day coverage window).
- **Infrastructure context:** NHAI toll plazas, MoRTH NH black-spots, CWC river telemetry at NH crossings, IFI/EM-DAT inventories.
- Licensing tracked per source (DATA_SOURCES.md decision matrix A/B/C/D).

### 4.2 Label design (sound)

- Target: disruption on an OSM way within a 7-day horizon.
- Positives at offsets −1/−3/−7 days before each verified event.
- Same-road control at −14 days; corridor negatives from a real unaffected-road pool.
- **Training gate (`training_gate.py`) fails closed**: any `assumed_unaffected` negative, unverified event URL, missing registry entry, or negative without full 7-day window coverage blocks training. Synthetic data is explicitly confined to tests.

### 4.3 Latest production run (2026-09-12_021329_a5ae56cb)

| Metric | Value | Gate |
|---|---|---|
| Status | `GATED_LOW_CONFIDENCE` | — |
| Samples | 187 (162 pos / 25 neg) across 79 event groups | — |
| Test events | 30 (of 30 required) | ✅ count gate |
| Test ROC-AUC | **0.50** | ❌ (need ≥0.65 demo / ≥0.75 prod) |
| Test AP | 0.892 | ✅ |
| Test recall | **0.0** (66 FNs, threshold 0.857) | ❌ (need ≥0.70) |
| Grouped-CV pooled AUC (36 folds) | **0.352** | ❌ |
| Advanced-model advantage | Unproven (LogReg AUC 0.623 > XGB 0.50) | ❌ |

**Diagnostic reading:** the model has *degenerated to predicting everyone safe*. Train ROC-AUC 0.50 and validation 0.50 with FNR 1.0 mean the current hyperparameters (depth-1, heavily regularized, threshold set by `top_k` on a validation set that produced an alert rate of 0.0) are producing a near-constant / unusable classifier. The regression is *new* relative to earlier runs (e.g., the 2026-09-11 demo model reached grouped-CV AUC ~0.555 and the Aug-28 run 0.537) and should be treated as a model-selection/threshold regression worth investigating, in addition to the standing data gaps.

### 4.4 Root causes of low confidence (confirmed by docs)

1. **Verified negatives were the historic blocker; now partially fixed** — 25 observation-backed negatives exist, but 4 of 7 corridor groups are still "OPEN_AFTER_CLEARANCE / REOPENED" statuses that the quality gate itself flags as not clean segment-level negatives (`validate_negative_observation_quality`). Density remains far below what calibration needs.
2. **Independent future test events** — now 30 chronologically future events (numeric gate met), but with 187 samples the signal is still thin and split variance is high.
3. **Weak, non-robust signal** — grouped-CV pooled AUC 0.35–0.55 across runs; simple rainfall-threshold and logistic baselines match or beat XGBoost.
4. **Terrain missingness** — patterned; some states 0% coverage historically (Copernicus fixed training corridors, not the full network).
5. **Mutable OSM way IDs** — used as the segment identity; stable versioned `road_segment_id` is a documented data gap.
6. **Non-monsoon months** — Jan/Apr/Nov/Dec have essentially zero confirmed events.

---

## 5. Engineering Quality Assessment

**Strengths (genuinely above typical project standard):**
- **Honesty-by-design:** gates fail closed; `production_ready: false` is machine-enforced, not claimed; caveats travel with every served model.
- **Reproducibility:** lockfile (67 pinned deps), immutable content-hashed bundles, dataset/feature/model/policy SHA256 chains, `working_tree_dirty` provenance, clean-tree hash-identical rebuild demonstrated.
- **Anti-leakage:** strictly-past rainfall windows, event-disjoint chronological splits, validation-only threshold/calibration fit, runtime leakage guard.
- **Inference rigour:** OOD flags from frozen training profiles, weather-freshness states, missing-terrain flag, non-causal SHAP disclaimers, controlled errors for adversarial inputs.
- **Documentation:** 26 docs form a genuinely useful audit trail (`ML_PRODUCTION_READINESS_REPORT`, `ML_AUDIT`, `MONITORING_STRATEGY`, `RETRAINING_STRATEGY`, `DATA_GAP_REMEDIATION`).

**Weaknesses / risk areas:**
- **Model regression going the wrong direction** — the newest prod bundle is the *worst* classifier yet (recall 0). Reverting the prod pointer to the demo/higher-AUC bundle, or re-running with sensible thresholds, would at least restore a non-degenerate demo.
- **Scaffold platform** — PS 26002 asks for a platform; ~75% of the surface area is unimplemented.
- **`backend/app/db/session.py`** constructs an engine at import with `postgres_password` required — the app **will not fully start without `.env`**, and no graceful degradation exists for the DB (though current routes do not hit the DB, so the API itself boots).
- **Tests:** 98 tests reported historically; I re-ran the non-GIS subset (73 tests) — all pass. On this machine the rasterio/SRTM-dependent modules fail to import because a Windows App-Control policy blocks `rasterio._base.dll` (environment issue, not code). Note: `python -m unittest discover -s tests` takes >5 min (heavy GIS/XLST suites) → CI timeout risk.
- **Demo feed model mismatch:** `demo_latest.json` → `2026-09-11_n62_p114`, while `risk_model.py` default tag is `2026-08-28_151030_b7486dea`; both bundles exist and are served correctly (env override wins), but the dual-tagline adds drift risk.
- **Trace hygiene:** earlier audits found production traces mixed with `test_segment_*` fixtures and feature-version drift (documented in ML_COMPLETION_REPORT phase 16).
- **Uncommitted work:** publish the 3 pending events + regen bundles, or intentionally keep them as a WIP checkpoint.

---

## 6. Feature-by-Feature PS-26002 Scorecard (weighted)

| PS-26002 requirement | Weight | Score |
|---|---|---|
| AI hazard/disruption prediction (b) | High | 80% (research-level, honest, gated) |
| Data & weather ingestion / GIS foundation | High | 85% |
| Alert decision output (e, partial) | Medium | 30% |
| Alternate routing + ETA (c) | High | 5% |
| GPS tracking of essential goods (d) | High | 0% |
| Field reporting mobile app + offline sync (f, h) | High | 0% |
| Dashboards — district connectivity, supply movement (g) | High | 0% |
| Multilingual notifications (h) | Medium | 0% |
| Cloud/secure/ops (auth, observability, SLA) | Medium | 15% |

**Effective platform completeness: ≈ 22–25%** of the statement's full scope — driven almost entirely by the ML/back-end slice.

---

## 7. Recommended Roadmap (priority order)

**Phase 1 — Stop the bleed (this week)**
1. Re-run the prod trainer with the *previous* demo/higher-performing configuration (or restore `demo_latest` as the demo pointer) so the served demo model is not the degenerate recall-0 bundle; commit the 3 pending events + regenerated bundles with the model card explaining the regression.
2. Make `db/session.py` lazy so the API boots without a DB; surface `/health` + `/api/v1/models` cleanly.

**Phase 2 — Prove the core model (highest value for the competition)**
3. Stabilize the XGBoost hyperparameters/threshold policy (current depth-1/`top_k` threshold gives recall 0) and compare against LogReg + rainfall baselines transparently.
4. Grow *clean segment-level* verified negatives via PWD/DDMA/NHIDCL closure logs (the quality gate already rejects corridor clearance statuses) and record them in `negative_observations.csv`.
5. Add a live/forward-validation loop that collects genuinely future events and joins delayed ground truth (monitor_model.py exists but is not scheduled).

**Phase 3 — Claim the platform scope (high effort, high PS score)**
6. **Routes:** implement `routing/` on the OSM graph (OSMnx + NetworkX) with risk-weighted edge costs from the model → alternate route + ETA deliverable (c).
7. **Dashboards (g):** build a minimal web frontend (MapLibre/Leaflet + a district-connectivity table + risk filter over `/api/v1/feed`) — even 1 page is enough to demonstrate GIS accessibility monitoring (a).
8. **Mobile field reporting (f/h):** a single form screen (geo-tagged incident upload + photo) with offline queue and JSON sync to the backend proves both (f) and offline (h).
9. **Alerts (e):** email/Telegram/SMS hook from `operating_decision` for HIGH/MEDIUM segments.
10. **Ops/security:** add API auth + rate limiting, monitoring/alerting on the documented strategy, and an initialize-db + deploy path (Docker Compose already scaffolds PostGIS).

**Phase 4 — Polish**
11. Multilingual notification strings (Assamese, Bengali, Bodo, etc.) for the mobile/alert layer.
12. Live weather API integration (Open-Meteo forecast) to move from offline rasters to near-real-time feature freshness.
13. Pfinal CI that excludes rasterio-dependent modules or caches heavy data so `unittest discover` completes under a reasonable timeout.

---

## 8. The One-Paragraph Verdict

NER-Nav contains the best-executed part of this problem that is hardest to do honestly — a real-data, reproducible, leakage-controlled disruption-risk model with verified-event ground truth, served through a live, well-contracted demo API and supported by unusually candid documentation. What it does *not* yet have is everything that makes it a "platform" to a judge: alternate routing, GPS tracking, field-reporting apps, dashboards, alerts, multilingual/offline support, and secure ops. The fastest path to a strong submission is (1) fix the current degenerate prod bundle, (2) keep adding clean verified negatives and future test events to lift the gates, and (3) build the thinnest working versions of routing and dashboard/reporting that consume the existing API — that triples statement coverage without weakening the rigorous core.