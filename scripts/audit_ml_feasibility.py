from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
ML_ROOT = DATA_ROOT / "processed" / "ml"

OUTPUT_JSON = ML_ROOT / "ml_feasibility_audit.json"
OUTPUT_MD = ML_ROOT / "ml_feasibility_audit.md"


SOURCE_REGISTRY = [
    {
        "id": "nrsc_landslide_atlas",
        "name": "NRSC / ISRO Landslide Atlas",
        "category": "real_incident_data",
        "status": "FEASIBLE_PENDING_ACQUISITION",
        "access_mode": "official_portal_or_download",
        "automation_allowed": "unknown",
        "coverage": "India landslide inventory, 1998-2022; includes seasonal, event-based and route-wise inventories",
        "ner_relevance": "HIGH",
        "license_status": "MUST_VERIFY_BEFORE_REUSE",
        "notes": (
            "Official NRSC source. Approximately 80,000 landslides are "
            "reported in the atlas. Acquisition path must be documented."
        ),
        "url": "https://www.nrsc.gov.in/nrscnew/resources_atlas_landslide.php",
    },
    {
        "id": "ndem_landslide_inventory",
        "name": "NDEM / NRSC Seasonal Landslide Inventory",
        "category": "real_incident_data",
        "status": "FEASIBLE_WITH_ACCESS_CONTROLS",
        "access_mode": "official_portal",
        "automation_allowed": "DO_NOT_SCRAPE",
        "coverage": "Seasonal and event-based landslide information",
        "ner_relevance": "HIGH",
        "license_status": "ACCESS_TERMS_MUST_BE_REVIEWED",
        "notes": (
            "NDEM exposes landslide inventory and related hazard services. "
            "NDEM terms restrict unauthorized automated access."
        ),
        "url": "https://ndem.nrsc.gov.in/geological_lsinventory.php",
    },
    {
        "id": "ndem_hazard",
        "name": "NDEM / NRSC Historical Hazard Services",
        "category": "hazard",
        "status": "FEASIBLE_WITH_ACCESS_CONTROLS",
        "access_mode": "official_portal",
        "automation_allowed": "DO_NOT_SCRAPE",
        "coverage": "Historical flood, landslide, hazard and geo-hazard products",
        "ner_relevance": "HIGH",
        "license_status": "ACCESS_TERMS_MUST_BE_REVIEWED",
        "notes": (
            "Useful for historical hazard enrichment. Access permissions "
            "and permitted reuse must be verified."
        ),
        "url": "https://ndem.nrsc.gov.in/",
    },
    {
        "id": "imd_rainfall",
        "name": "India Meteorological Department rainfall data",
        "category": "rainfall",
        "status": "FEASIBLE_PENDING_DATASET_SELECTION",
        "access_mode": "official_source",
        "automation_allowed": "MUST_VERIFY",
        "coverage": "Historical rainfall observations/information",
        "ner_relevance": "HIGH",
        "license_status": "MUST_VERIFY_BEFORE_REUSE",
        "notes": (
            "Need to select the exact historical machine-readable product "
            "and document spatial/temporal resolution."
        ),
        "url": "https://mausam.imd.gov.in/imd_latest/contents/rainfallinformation.php",
    },
    {
        "id": "ndem_dem",
        "name": "NDEM / NRSC DEM products",
        "category": "terrain",
        "status": "FEASIBLE_PENDING_ACQUISITION",
        "access_mode": "official_portal_or_service",
        "automation_allowed": "MUST_VERIFY",
        "coverage": "DEM products including 30 m and other resolutions",
        "ner_relevance": "HIGH",
        "license_status": "MUST_VERIFY_BEFORE_REUSE",
        "notes": (
            "Elevation can be converted into slope/terrain features. "
            "Acquisition and reuse conditions must be recorded."
        ),
        "url": "https://ndem.nrsc.gov.in/",
    },
    {
        "id": "nrsc_geology",
        "name": "NRSC / ISRO geological geospatial resources",
        "category": "geological_vulnerability",
        "status": "PARTIALLY_FEASIBLE",
        "access_mode": "official_portal_or_dataset",
        "automation_allowed": "MUST_VERIFY",
        "coverage": "Geological and geomorphological information",
        "ner_relevance": "HIGH",
        "license_status": "MUST_VERIFY_BEFORE_REUSE",
        "notes": (
            "Need to identify a concrete spatial vulnerability layer "
            "appropriate for road-segment enrichment."
        ),
        "url": "https://www.nrsc.gov.in/nrscnew/resources_atlas_geology.php",
    },
    {
        "id": "traffic",
        "name": "Historical traffic / congestion observations",
        "category": "traffic_data",
        "status": "MISSING",
        "access_mode": "NOT_IDENTIFIED",
        "automation_allowed": "UNKNOWN",
        "coverage": "No verified project-local historical traffic dataset",
        "ner_relevance": "HIGH",
        "license_status": "NOT_APPLICABLE",
        "notes": (
            "Must identify a legitimate source or explicitly mark traffic "
            "as unavailable for the first real-data model."
        ),
        "url": None,
    },
    {
        "id": "delay",
        "name": "Historical travel-time / delay observations",
        "category": "delay_target",
        "status": "MISSING",
        "access_mode": "NOT_IDENTIFIED",
        "automation_allowed": "UNKNOWN",
        "coverage": "No verified project-local delay target",
        "ner_relevance": "HIGH",
        "license_status": "NOT_APPLICABLE",
        "notes": (
            "Required for the secondary delay regression model. "
            "Do not synthesize a production target."
        ),
        "url": None,
    },
]


REQUIRED_CAPABILITIES = {
    "real_incident_data": {
        "required": True,
        "purpose": "Historical disruption/incident labels",
    },
    "rainfall": {
        "required": True,
        "purpose": "Weather and cumulative rainfall features",
    },
    "terrain": {
        "required": True,
        "purpose": "Elevation and slope features",
    },
    "geological_vulnerability": {
        "required": True,
        "purpose": "Geological/terrain susceptibility enrichment",
    },
    "traffic_data": {
        "required": True,
        "purpose": "Congestion and traffic-related risk/delay features",
    },
    "delay_target": {
        "required": True,
        "purpose": "Secondary travel-delay regression target",
    },
}


LOCAL_EXPECTED_ARTIFACTS = [
    DATA_ROOT / "processed" / "graph" / "ner_road_graph.pkl",
    DATA_ROOT / "processed" / "roads" / "ner_roads_districts.gpkg",
    ML_ROOT / "risk_training_dataset.parquet",
    ML_ROOT / "risk_model.json",
    ML_ROOT / "risk_model_preprocessor.joblib",
]


def classify_local_foundation() -> dict:
    result = {}

    for path in LOCAL_EXPECTED_ARTIFACTS:
        key = path.name
        result[key] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
        }

    return result


def build_capability_status() -> dict:
    status = {}

    for capability, config in REQUIRED_CAPABILITIES.items():
        matching = [
            source
            for source in SOURCE_REGISTRY
            if source["category"] == capability
        ]

        status[capability] = {
            "required": config["required"],
            "purpose": config["purpose"],
            "sources": [source["id"] for source in matching],
            "source_statuses": [source["status"] for source in matching],
        }

    return status


def determine_gates(capability_status: dict) -> dict:
    gates = {}

    real_incident = capability_status["real_incident_data"]["source_statuses"]
    rainfall = capability_status["rainfall"]["source_statuses"]
    terrain = capability_status["terrain"]["source_statuses"]
    geology = capability_status["geological_vulnerability"]["source_statuses"]
    traffic = capability_status["traffic_data"]["source_statuses"]
    delay = capability_status["delay_target"]["source_statuses"]

    gates["real_disruption_training"] = {
        "status": (
            "BLOCKED"
            if (
                not real_incident
                or all(
                    s not in {
                        "FOUND",
                        "FEASIBLE_ACQUIRED",
                    }
                    for s in real_incident
                )
            )
            else "PENDING_VALIDATION"
        ),
        "reason": (
            "Real incident/hazard records must be acquired, normalized, "
            "spatially mapped to road segments, temporally aligned and "
            "converted into defensible labels before training."
        ),
    }

    gates["terrain_enrichment"] = {
        "status": "PENDING_ACQUISITION",
        "reason": (
            "A DEM source has been identified, but the project does not "
            "yet contain a verified terrain dataset."
        ),
    }

    gates["geological_enrichment"] = {
        "status": "PENDING_ACQUISITION",
        "reason": (
            "A geological source family exists, but a concrete reusable "
            "vulnerability layer has not yet been acquired and validated."
        ),
    }

    gates["rainfall_enrichment"] = {
        "status": "PENDING_ACQUISITION",
        "reason": (
            "Rainfall is required, but the exact historical machine-readable "
            "dataset and spatial/temporal alignment have not yet been locked."
        ),
    }

    gates["delay_model"] = {
        "status": "BLOCKED",
        "reason": (
            "No verified historical delay/travel-time target currently exists."
        ),
    }

    gates["traffic_features"] = {
        "status": "BLOCKED",
        "reason": (
            "No verified historical traffic/congestion dataset currently exists."
        ),
    }

    gates["synthetic_baseline"] = {
        "status": "AVAILABLE",
        "reason": (
            "The synthetic road-prior dataset/model exists and may be used "
            "only for pipeline development, testing and demonstration."
        ),
    }

    return gates


def build_markdown(report: dict) -> str:
    lines = []

    lines.append("# NER-Nav ML Feasibility Audit")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")

    lines.append("## Decision")
    lines.append("")
    lines.append(
        "**REAL-DATA TRAINING GATE: BLOCKED until real hazard labels are "
        "acquired and mapped to road segments.**"
    )
    lines.append("")
    lines.append(
        "The current synthetic-label model remains a development baseline "
        "and must not be presented as real-world predictive performance."
    )
    lines.append("")

    lines.append("## Required capabilities")
    lines.append("")
    lines.append("| Capability | Required | Current status |")
    lines.append("|---|---:|---|")

    for capability, item in report["capabilities"].items():
        statuses = ", ".join(item["source_statuses"]) or "NONE"
        lines.append(
            f"| {capability} | "
            f"{'YES' if item['required'] else 'NO'} | {statuses} |"
        )

    lines.append("")

    lines.append("## Feasibility gates")
    lines.append("")
    lines.append("| Gate | Status |")
    lines.append("|---|---|")

    for name, gate in report["gates"].items():
        lines.append(f"| {name} | **{gate['status']}** |")

    lines.append("")

    lines.append("## Verified source registry")
    lines.append("")

    for source in report["sources"]:
        lines.append(f"### {source['name']}")
        lines.append("")
        lines.append(f"- ID: `{source['id']}`")
        lines.append(f"- Category: `{source['category']}`")
        lines.append(f"- Status: **{source['status']}**")
        lines.append(f"- Access: `{source['access_mode']}`")
        lines.append(f"- Automation: `{source['automation_allowed']}`")
        lines.append(f"- NER relevance: `{source['ner_relevance']}`")
        lines.append(f"- Licensing: `{source['license_status']}`")
        lines.append(f"- Coverage: {source['coverage']}")
        lines.append(f"- Notes: {source['notes']}")

        if source["url"]:
            lines.append(f"- Official source: {source['url']}")

        lines.append("")

    lines.append("## Local foundation")
    lines.append("")

    for name, item in report["local_foundation"].items():
        status = "FOUND" if item["exists"] else "MISSING"
        lines.append(f"- `{name}`: **{status}**")

    lines.append("")

    lines.append("## Rules")
    lines.append("")
    lines.append(
        "1. Do not scrape NDEM without explicit authorization."
    )
    lines.append(
        "2. Do not fabricate incident labels."
    )
    lines.append(
        "3. Do not fabricate traffic observations."
    )
    lines.append(
        "4. Do not fabricate delay targets."
    )
    lines.append(
        "5. Preserve raw source data before transformation."
    )
    lines.append(
        "6. Record source, acquisition date, license/access condition, "
        "spatial resolution and temporal resolution."
    )
    lines.append(
        "7. Do not train the final disruption model until the target "
        "definition and temporal labeling procedure are documented."
    )
    lines.append("")

    lines.append("## Next micro-step")
    lines.append("")
    lines.append(
        "**STEP 2: Real hazard-event source acquisition and local raw-data "
        "ingestion design.**"
    )
    lines.append("")
    lines.append(
        "First priority: obtain an officially permitted landslide/hazard "
        "inventory covering NER, preserve it under data/raw, inspect its "
        "schema and determine whether event geometry/time is sufficient to "
        "label the existing road segments."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    print("=" * 70)
    print("NER-Nav — ML Requirements & Feasibility Audit")
    print("=" * 70)
    print()

    ML_ROOT.mkdir(parents=True, exist_ok=True)

    capability_status = build_capability_status()
    gates = determine_gates(capability_status)
    local_foundation = classify_local_foundation()

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "decision": "REAL_DATA_TRAINING_BLOCKED",
        "sources": SOURCE_REGISTRY,
        "capabilities": capability_status,
        "gates": gates,
        "local_foundation": local_foundation,
        "rules": [
            "Do not scrape NDEM without explicit authorization.",
            "Do not fabricate incident labels.",
            "Do not fabricate traffic observations.",
            "Do not fabricate delay targets.",
            "Preserve raw source data before transformation.",
            "Document source and licensing/access conditions.",
            "Perform temporal and spatial leakage checks before training.",
        ],
    }

    OUTPUT_JSON.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    OUTPUT_MD.write_text(
        build_markdown(report),
        encoding="utf-8",
    )

    print("Required ML capabilities")
    print("-" * 70)

    for capability, item in capability_status.items():
        statuses = ", ".join(item["source_statuses"]) or "NONE"
        print(f"{capability:30s}: {statuses}")

    print()
    print("Feasibility gates")
    print("-" * 70)

    for name, gate in gates.items():
        print(f"{name:30s}: {gate['status']}")

    print()
    print("Local foundation")
    print("-" * 70)

    for name, item in local_foundation.items():
        status = "FOUND" if item["exists"] else "MISSING"
        print(f"{name:45s}: {status}")

    print()
    print("=" * 70)
    print("ML FEASIBILITY AUDIT COMPLETE")
    print("=" * 70)
    print(f"JSON:   {OUTPUT_JSON}")
    print(f"Report: {OUTPUT_MD}")
    print()
    print("REAL-DATA TRAINING GATE: BLOCKED")
    print()
    print(
        "Next: acquire officially permitted real hazard-event data, "
        "preserve raw data, inspect schema, then map events to road segments."
    )
    print("=" * 70)


if __name__ == "__main__":
    main()