# NER-Nav

**AI-Based Smart Logistics and Accessibility Intelligence Platform for the North Eastern Region (India)**

Predicts **7-day road-disruption risk** (landslide, flood, weather) on National Highway (NH) corridors across the 8 North Eastern Region states, at OSM road-segment granularity. Served via a FastAPI demo API.

---

## Status

The end-to-end ML pipeline is **complete and reproducible**, but the model is **gated below production confidence** — the decisive blocker is verified ground-truth labels, not code or features.

| Component | State |
|---|---|
| Feature pipeline (terrain, rainfall, seasonal) | Complete |
| Event label registry (32 verified positives) | Complete |
| Source-backed negatives (24) | Complete |
| Chronological split / training gates | Complete |
| FastAPI service | Live |
| Model quality | `GATED_LOW_CONFIDENCE` (test ROC-AUC 0.50) |

See `docs/ML_PRODUCTION_READINESS_REPORT.md`, `docs/ML_CONFIDENCE_TIERS_BLOCKERS.md`, and `docs/ML_DATA_GAP_REMEDIATION.md` for the honest, detailed status and the label-remediation runbook.

---

## Repository layout

```
backend/          FastAPI service (API, risk model serving, feature derivation)
ml/               Feature engineering, labels, splits, gates, training
scripts/          Ingestion, training, prediction, verification pipelines
data/             Raw inputs, processed datasets, model bundles, predictions
database/         SQLAlchemy migrations (PostGIS)
stories/ docs/    Acquisition reports, provenance, mission handoffs
simulators/       (scaffold)
frontend/ mobile/ (scaffold)
routing/          (scaffold)
tests/            Backend + ML tests
```

## Quickstart

Python `>=3.12,<3.13`. Install pinned deps and run the API:

```bash
pip install -r requirements.lock
uvicorn backend.app.main:app --reload
```

Or run the full stack (PostGIS + API) with Docker:

```bash
docker compose up --build
```

Key API endpoints:

- `GET  /api/v1/models` — current model versions and gating status
- `GET  /api/v1/feed`  — segment-level risk feed from the demo bundle
- `POST /api/v1/predict` — predict disruption risk for a segment

## Training the production model

```bash
python scripts/train_production_risk_model.py
```

The pipeline enforces fail-closed integrity rules: no synthetic labels, no inferred provenance, chronological event-group splits, and gated release. Model bundles ship with SHA256 verification.

## Data sources

Raw inputs live in `data/raw/` with source attribution and licensing tracked in `docs/DATA_SOURCES.md`. Notable holdings:

- **Road network:** OSM NER extract (289k segments)
- **Terrain:** SRTM v4 + SRTMGL1 1-arc-sec + Copernicus GLO-30 DEM (`ner_dem_full.tif`)
- **Precipitation:** CHIRPS v2.0 daily (2015–2026), IMD gridded (2001–2025), Open-Meteo station series
- **Events:** `ner_news_events.csv` + source HTML articles, EM-DAT (India), IFI flood inventory
- **Infrastructure:** NHAI toll plazas, MoRTH NH black spots, CWC river-water-level telemetry at NH crossings

Large re-fetchable rasters (`.grd`, `.hgt`, `.tif`, `.gz`, `.shp`) are gitignored and documented rather than committed.

## Documentation index

| Doc | Purpose |
|---|---|
| `docs/ML_PRODUCTION_READINESS_REPORT.md` | Model metrics, gates, honest status |
| `docs/ML_CONFIDENCE_TIERS_BLOCKERS.md` | Confidence tiers and remaining blockers |
| `docs/ML_DATA_GAP_REMEDIATION.md` | Runbook to reach HIGH confidence (labels + identity) |
| `docs/LABEL_EXPANSION_PLAN.md` | Corridor-priority table for label expansion |
| `docs/DATA_SOURCES.md` | Source registry, licensing, decision rules |
| `docs/NEGATIVE_OBSERVATION_PROVENANCE.md` | Source-backed negative labels |