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
# 🗺️ NER-Nav

[![Python 3.12–3.14](https://img.shields.io/badge/python-3.12--3.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI 0.141](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostGIS 17-3.5](https://img.shields.io/badge/PostGIS-17--3.5-336791?logo=postgresql&logoColor=white)](https://postgis.net/)
[![Docker](https://img.shields.io/badge/Docker-required-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen?logo=github)](https://github.com/your-org/NER-Nav/pulls)

**AI-Based Smart Logistics & Accessibility Intelligence Platform for India's North Eastern Region**

NER-Nav transforms the eight-state North Eastern Region's logistics from reactive to predictive. By fusing **ML hazard prediction**, **GIS-based spatial routing**, **real-time GPS fleet tracking**, and an **offline-first mobile field-reporting app**, the platform recalculates the safest delivery routes before physical blockages are ever reported — keeping essential goods like medicines, food rations, and disaster-relief supplies moving even during monsoon season.

*Four-layer microservices architecture: Client → API Gateway → AI & Logic → Data & Persistence*

---

## 📋 Table of Contents

- [🗺️ NER-Nav](#️-ner-nav)
  - [📋 Table of Contents](#-table-of-contents)
  - [🌏 About](#-about)
  - [📊 Impact at a Glance](#-impact-at-a-glance)
  - [👥 Who Is This For?](#-who-is-this-for)
  - [✨ Key Features](#-key-features)
  - [🛠️ Tech Stack](#️-tech-stack)
  - [🏗️ Architecture Overview](#️-architecture-overview)
  - [⚡ Quick Start (TL;DR)](#-quick-start-tldr)
  - [🚀 Installation](#-installation)
    - [Prerequisites](#prerequisites)
    - [1. Clone the Repository](#1-clone-the-repository)
    - [2. Set Up a Virtual Environment](#2-set-up-a-virtual-environment)
    - [3. Install Dependencies](#3-install-dependencies)
    - [4. Configure Environment Variables](#4-configure-environment-variables)
    - [5. Start the Database](#5-start-the-database)
    - [6. Run Database Migrations](#6-run-database-migrations)
  - [▶️ Running the App](#️-running-the-app)
  - [📡 API Reference](#-api-reference)
  - [📁 Project Structure](#-project-structure)
  - [🧪 Running Tests](#-running-tests)
  - [🛣️ Roadmap](#️-roadmap)
  - [🔧 Troubleshooting](#-troubleshooting)
  - [🔒 Security](#-security)
  - [🤝 Contributing](#-contributing)
  - [🙏 Acknowledgments](#-acknowledgments)
  - [📬 Contact](#-contact)

---

## 🌏 About

The North Eastern Region of India — spanning eight states and over 262,000 sq. km — faces chronic logistics disruptions driven by monsoon-season landslides, flash floods, and the near-total absence of real-time supply-chain visibility. Essential commodities routinely reach remote districts days late, with life-threatening consequences.

NER-Nav addresses this by building the region's first **predictive** logistics intelligence platform. Its core innovation — **Dynamic Graph-Weighted Routing** — injects XGBoost-generated hazard scores directly as edge weights into a live road graph. When risk on a segment exceeds threshold, A\* automatically reroutes supply convoys *before* a physical blockage is even reported.

---

## 📊 Impact at a Glance

| Metric | Baseline | Target |
|---|---|---|
| Route reroute decision time | 4–6 hours (manual) | **< 5 minutes** (automated AI) |
| Incident reporting lag | 12–24 hours | **< 15 minutes** (offline sync) |
| ML hazard prediction accuracy | 0% (absent) | **> 85% F1-score** |
| Districts with live connectivity data | 0 of 86 | **86 of 86** NER districts |
| Fleet visibility coverage | 0% | **100%** of registered vehicles |
| Supply delivery delay events | No baseline | **≥ 30% reduction** |

---

## 👥 Who Is This For?

| Persona | Role | Core Need |
|---|---|---|
| 🖥️ **Command Center Operator** | MDoNER / State Disaster Authority | Real-time district map, fleet visibility, predictive hazard alerts |
| 🚛 **Logistics / Fleet Driver** | Essential-goods convoy driver | AI-safe routing, in-app hazard reroutes, cargo status |
| 📡 **Field Official** | Local disaster authority officer | Offline incident reporting with geo-tagged photos, zero-network capable |
| 🏛️ **District Collector** | District administration | Connectivity status board, escalation to PWD, supply-chain gap view |

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🤖 **AI Route Engine** | XGBoost risk scores injected as graph edge weights; A\* auto-reroutes supply convoys when hazard score > 0.7 |
| 🗺️ **GIS Dashboard** | React + Mapbox GL JS — live fleet tracklines, hazard overlays, district connectivity heatmap |
| 📱 **Offline Field App** | React Native/PWA with SQLite queue; geo-tagged photo + GPS; auto-syncs on reconnect |
| 🔔 **Real-Time Alerts** | WebSocket push + SMS for road blocks, delivery delays, and high-risk corridors |
| 🚛 **Fleet Tracker** | Live vehicle positions with automatic reroute broadcast to driver and command center |
| 🌦️ **Weather Integration** | IMD + OpenWeather APIs polled every 15 min; triggers ML risk recalculation on change |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python FastAPI 0.141 + Uvicorn (async, WebSocket-native) |
| **Database** | PostgreSQL 17 + PostGIS 3.5 · Spatial queries: `ST_DWithin`, `ST_Intersects` |
| **Migrations** | Alembic |
| **AI / ML** | Scikit-Learn · XGBoost · NetworkX (A\* routing) · Celery (async tasks) |
| **Cache / Pub-Sub** | Redis (route cache + WebSocket broker) |
| **Auth** | JWT + OAuth 2.0 (roles: field official / logistics manager / admin) |
| **Frontend** | React.js + Tailwind CSS + Mapbox GL JS *(planned)* |
| **Mobile** | React Native / PWA + SQLite *(planned)* |
| **Infrastructure** | Docker · AWS ECS + RDS + S3 + CloudFront |
| **Weather** | IMD API (primary) · OpenWeather API (failover) |

---

## 🏗️ Architecture Overview

NER-Nav follows a four-layer decoupled architecture:

```
[ Layer 1 ]  Client & Presentation
              React.js + Mapbox GL JS (Web) · React Native / PWA (Mobile)

[ Layer 2 ]  API Gateway & App Server
              Python FastAPI · REST + WebSocket · JWT Auth · Rate Limiting

[ Layer 3 ]  AI & Business Logic
              XGBoost (Hazard Prediction) · NetworkX (A* Routing) · Celery (Async Jobs)

[ Layer 4 ]  Data & Persistence
              PostgreSQL + PostGIS · Redis Cache · AWS S3 · IMD / OpenWeather API
```

**How Dynamic Graph-Weighted Routing works:**

```
Weather alert fires (IMD API)
        │
        ▼
XGBoost computes Risk_Multiplier for affected road segments
        │
        ▼
New edge weight = Base_Distance_km × Risk_Multiplier
        │
        ▼
A* re-runs on updated graph → safest alternate route computed
        │
        ▼
Route saved to PostGIS → WebSocket broadcast to dashboard + driver
        │
        ▼
End-to-end latency: < 5 seconds
```

![GIS Dashboard Screenshot](docs/images/gis-dashboard.png)
*GIS command-center dashboard — district connectivity, hazard overlays, and live fleet positions*

![Field Reporter App Screenshot](docs/images/field-reporter-app.png)
*Offline-first Field Reporter App — incident queue flushes automatically on network restore*

---

## ⚡ Quick Start (TL;DR)

Already familiar with FastAPI and Docker? Here's the shortest path to a running server:

```bash
git clone https://github.com/your-org/NER-Nav.git && cd NER-Nav
python3.14 -m venv .venv-314 && source .venv-314/bin/activate   # Windows: .\.venv-314\Scripts\Activate.ps1
pip install -e .
cp .env.example .env          # edit POSTGRES_PASSWORD
docker compose up -d
alembic upgrade head
uvicorn backend.app.main:app --reload
# → http://127.0.0.1:8000
```

For the full walkthrough, read on.

---

## 🚀 Installation

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.12 – 3.14 | Use `py -3.14` on Windows or `python3.14` on macOS/Linux |
| Docker Desktop | Latest | Required to run the PostGIS database container |
| Git | Any | — |

> **Windows users:** If running Activate.ps1 fails with an execution-policy error, run this once in an admin PowerShell session:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

> **Linux users:** `psycopg[binary]` bundles libpq. If you see a linker error, install the system dependency with `sudo apt install libpq-dev` then retry `pip install -e .`.

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/NER-Nav.git
cd NER-Nav
```

---

### 2. Set Up a Virtual Environment

The project targets Python 3.14. The virtual-environment is named `.venv-314` (the `314` reflects the Python minor version, making it easy to manage multiple interpreters in the same workspace).

```powershell
# Windows (PowerShell)
py -3.14 -m venv .venv-314
.\.venv-314\Scripts\Activate.ps1
```

```bash
# macOS / Linux
python3.14 -m venv .venv-314
source .venv-314/bin/activate
```

---

### 3. Install Dependencies

```bash
python -m pip install -e .
```

This installs NER-Nav in editable mode, so local source changes are reflected immediately without reinstalling.

---

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values. The table below documents every variable:

| Variable | Required | Default | Description |
|---|---|---|---|
| `APP_NAME` | No | `NER-Nav API` | Display name returned by the health endpoint |
| `APP_ENV` | No | `development` | Runtime environment (`development` / `production`) |
| `APP_VERSION` | No | `0.1.0` | Semantic version string |
| `POSTGRES_HOST` | Yes | `127.0.0.1` | PostgreSQL host (use `127.0.0.1`, **not** `localhost`, to force IPv4) |
| `POSTGRES_PORT` | Yes | `5432` | PostgreSQL port |
| `POSTGRES_DB` | Yes | `ner_nav` | Database name |
| `POSTGRES_USER` | Yes | `ner_nav` | Database user |
| `POSTGRES_PASSWORD` | **Yes** | `change_me` | ⚠️ **Must be changed** — use a strong password |
| `DATABASE_URL` | Yes | — | Full DSN — must stay in sync with the four `POSTGRES_*` vars above |

> ⚠️ **Never commit `.env` to version control.** It is listed in `.gitignore` by default. Only `.env.example` (which contains no secrets) should be committed.

---

### 5. Start the Database

```bash
# Start the PostGIS container in the background
docker compose up -d

# Confirm it is healthy before continuing (~10–15 seconds)
docker compose ps

# Stream logs if you need to debug
docker compose logs -f postgres
```

This starts a PostGIS 17-3.5 container (`ner-nav-postgres`) at `localhost:5432`. The container runs a health check every 5 seconds; wait for `Status: healthy` before proceeding.

**Other useful Docker commands:**

```bash
docker compose stop          # Pause the container (data preserved)
docker compose down          # Stop and remove the container
docker compose down -v       # Stop, remove container AND wipe the volume (full reset)
```

---

### 6. Run Database Migrations

**Apply all pending migrations to head:**

```bash
alembic upgrade head
```

**Create a new migration after modifying a SQLAlchemy model:**

```bash
alembic revision --autogenerate -m "describe your change"
alembic upgrade head
```

**Roll back the last migration:**

```bash
alembic downgrade -1
```

---

## ▶️ Running the App

```powershell
# Windows
.\.venv-314\Scripts\Activate.ps1
uvicorn backend.app.main:app --reload
```

```bash
# macOS / Linux
source .venv-314/bin/activate
uvicorn backend.app.main:app --reload
```

**Verify it's running:**

```bash
curl http://127.0.0.1:8000/api/v1/health
# Expected: {"status": "ok", ...}
```

| URL | Description |
|---|---|
| http://127.0.0.1:8000 | Browser UI (health check widget) |
| http://127.0.0.1:8000/docs | Interactive Swagger UI |
| http://127.0.0.1:8000/redoc | ReDoc API documentation |

---

## 📡 API Reference

> Full OpenAPI spec is auto-generated at `/docs` when the server is running.

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| `GET` | `/api/v1/health` | None | Backend + database connectivity check |
| `POST` | `/api/v1/routes/optimize` | JWT | Request AI-optimized route between two nodes |
| `GET` | `/api/v1/incidents` | JWT | List all field-reported and ML-predicted incidents |
| `POST` | `/api/v1/incidents` | JWT | Submit a new geo-tagged incident report |
| `GET` | `/api/v1/fleet` | JWT | Live fleet positions (WebSocket alternative) |
| `WS` | `/ws/fleet` | JWT | WebSocket stream — real-time fleet & route updates |
| `WS` | `/ws/alerts` | JWT | WebSocket stream — road-block and hazard alerts |

---

## 📁 Project Structure

```
NER-Nav/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI application entrypoint
│       ├── routers/             # Route handlers (incidents, fleet, alerts, …)
│       ├── models/              # SQLAlchemy ORM models
│       ├── schemas/             # Pydantic request/response schemas
│       └── services/            # Business logic (routing engine, ML inference, …)
├── database/
│   └── migrations/
│       ├── versions/            # Alembic revision scripts
│       └── env.py               # Alembic runtime environment
├── ml/
│   ├── models/                  # Trained model artifacts — git-ignored (*.pkl, *.joblib)
│   └── data/                    # Training data — git-ignored
├── frontend/                    # React.js + Mapbox GL JS dashboard (planned)
├── mobile/                      # React Native / PWA Field Reporter App (planned)
├── tests/
│   ├── unit/                    # Unit tests for services and utilities
│   └── integration/             # Integration tests against live DB
├── docs/
│   └── images/                  # README screenshots and diagrams
├── .env.example                 # Environment variable template (committed)
├── .gitignore
├── alembic.ini                  # Alembic configuration
├── docker-compose.yml           # PostGIS container definition
└── pyproject.toml               # Project metadata & dependencies
```

---

## 🧪 Running Tests

```bash
# Install test dependencies (if not already present)
pip install -e ".[dev]"

# Run the full test suite
pytest

# Run with coverage report
pytest --cov=backend --cov-report=term-missing

# Run only unit tests (no DB required)
pytest tests/unit/

# Run only integration tests (PostGIS container must be running)
pytest tests/integration/
```

---

## 🛣️ Roadmap

| Phase | Feature | Timeline |
|---|---|---|
| **MVP (current)** | AI Route Engine · GIS Dashboard · Offline Field App · Alert System · Mock Fleet Tracker | — |
| **Phase 2** | MDoNER transport DB integration · Real IoT GPS hardware for government fleets | Month 3–5 |
| **Phase 3** | Multilingual UI (Assamese, Manipuri, Mizo, Nepali, Hindi) · Drone last-mile optimization | Month 5–6 |
| **Phase 4** | Satellite imagery change detection · State-level admin portal with budget analytics | Month 8–12 |

Track progress and vote on features in [GitHub Issues](https://github.com/your-org/NER-Nav/issues).

---

## 🔧 Troubleshooting

**`docker compose ps` shows container as `starting` or `unhealthy`**
The PostGIS initialization takes 10–20 seconds on first run. Wait and re-run `docker compose ps`. If it stays unhealthy, check logs: `docker compose logs postgres`.

**`alembic upgrade head` fails with "connection refused"**
The database container isn't ready yet, or `DATABASE_URL` in `.env` doesn't match your Docker settings. Confirm `POSTGRES_HOST=127.0.0.1` (not `localhost`) and that the container is healthy.

**`Activate.ps1` blocked on Windows**
Run this once in an admin PowerShell: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`, then retry activation.

**`ModuleNotFoundError: No module named 'backend'`**
The virtual environment is not activated, or the package wasn't installed. Run `pip install -e .` inside the activated environment.

**Port 5432 already in use**
Another PostgreSQL instance is running locally. Stop it, or change `POSTGRES_PORT` in `.env` and `docker-compose.yml` to a free port (e.g., `5433`).

**`psycopg` linker error on Linux**
Install the system library: `sudo apt install libpq-dev`, then `pip install -e .` again.

---

## 🔒 Security

- All API endpoints (except `/api/v1/health`) require a valid **JWT** bearer token.
- Tokens are signed with RS256; role-based access enforces separation between field official, logistics manager, and admin roles.
- Traffic is encrypted with **TLS 1.3** in transit; data at rest uses **AES-256** (AWS RDS encryption enabled by default).
- Never commit `.env`, `*.pem`, `*.key`, or any credential files — all are covered by `.gitignore`.
- Rotate `POSTGRES_PASSWORD` and JWT signing keys before any production deployment.

To report a security vulnerability, please email **[security@ner-nav.example.com](mailto:security@ner-nav.example.com)** rather than opening a public issue.

---

## 🤝 Contributing

Contributions are welcome! Here's how to get involved:

1. **Fork** the repository and create a feature branch: `git checkout -b feat/your-feature`
2. **Code** your changes, following the existing style — run `ruff check .` before pushing
3. **Test** your work: `pytest` must pass with no regressions
4. **Commit** with a conventional commit message: `git commit -m "feat: describe what you added"`
5. **Open a Pull Request** against `main` — include a short description of what changed and why

For bugs or feature requests, please [open an issue](https://github.com/your-org/NER-Nav/issues) first so we can discuss the approach.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md) before submitting.

---

## 🙏 Acknowledgments

- **[MDoNER](https://mdoner.gov.in/)** — Ministry of Development of North Eastern Region, for defining Problem Statement ID 26002
- **[NDMA](https://ndma.gov.in/)** — National Disaster Management Authority, for historical landslide and disruption datasets
- **[ISRO Bhuvan](https://bhuvan.nrsc.gov.in/)** — For GIS road-network and geological vulnerability data for the NER
- **[IMD](https://mausam.imd.gov.in/)** — Indian Meteorological Department, for the real-time rainfall and flood warning API
- **[OpenWeather](https://openweathermap.org/)** — Commercial weather API used as IMD failover

---

## 📬 Contact

Have questions or want to collaborate? Reach out at **[team@ner-nav.example.com](mailto:team@ner-nav.example.com)** or open a discussion in the [GitHub repository](https://github.com/your-org/NER-Nav/discussions).

---

*Built with ❤️ for the people of India's North Eastern Region.*
