from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = PROJECT_ROOT / "data" / "raw" / "hazards" / "Sikkim_Landslide_2016_NRSC.pdf"

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "hazards"
OUT_JSON = OUT_DIR / "nrsc_sikkim_mantam_2016_event.json"
OUT_TEXT = OUT_DIR / "nrsc_sikkim_mantam_2016_extracted.txt"
OUT_QA = OUT_DIR / "nrsc_sikkim_mantam_2016_event_qa.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_pdf_text(path: Path) -> str:
    with pymupdf.open(path) as doc:
        pages = []

        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(f"\n--- PAGE {i + 1} ---\n{text}")

    return "\n".join(pages).strip()


def first_match(pattern: str, text: str, flags=re.IGNORECASE) -> str | None:
    match = re.search(pattern, text, flags)
    return match.group(1).strip() if match else None


def main() -> None:
    print("=" * 70)
    print("NER-Nav — NRSC Landslide Event Extraction")
    print("=" * 70)

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Missing source PDF: {PDF_PATH}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nSource: {PDF_PATH}")

    source_sha256 = sha256_file(PDF_PATH)

    text = extract_pdf_text(PDF_PATH)
    OUT_TEXT.write_text(text, encoding="utf-8")

    print(f"Extracted characters: {len(text):,}")
    print(f"SHA-256: {source_sha256}")

    # The report explicitly states these facts.
    event = {
        "event_id": "nrsc_sikkim_mantam_2016_08_13",
        "event_type": "landslide",
        "hazard_subtype": "rock_avalanche",
        "event_name": "So Bhir / Mantam Landslide",
        "event_date": "2016-08-13",
        "event_time_local": "13:30",
        "timezone": "Asia/Kolkata",
        "state": "Sikkim",
        "region": "North Sikkim",
        "locality": "Mantam village",
        "nearby_road": "Passingdang-Mantam Road",

        "reported_impacts": {
            "bridge": {
                "affected": True,
                "name": "Kanka bridge",
                "impact": "submerged"
            },
            "road": {
                "affected": True,
                "washed_away_length_m": 300
            },
            "settlement": {
                "affected": True,
                "houses_submerged": 5
            },
            "connectivity": {
                "affected": True,
                "cut_off_villages": [
                    "Tingvong",
                    "Lingdem",
                    "Laven",
                    "Kayeem",
                    "Lingzya",
                    "Bay",
                    "Sakyong Pentong",
                    "Ruklu Kayeem"
                ]
            },
            "human_casualties_reported": 0
        },

        "landslide_geometry_reported": {
            "length_m": 790,
            "width_m_middle": 530,
            "geometry_shape": "rectangular",
            "artificial_lake_length_km": 2.2,
            "artificial_lake_width_m_at_lake_head": 209
        },

        "spatial_status": "PENDING_GEOMETRY",
        "latitude": None,
        "longitude": None,
        "geometry_source": None,

        "temporal_status": "EVENT_DATE_KNOWN",
        "label_ready_for_training": False,

        "source": {
            "source_id": "nrsc_sikkim_landslide_2016_report",
            "organization": "National Remote Sensing Centre, ISRO",
            "document": "Sikkim_Landslide_2016_NRSC.pdf",
            "local_path": str(PDF_PATH.relative_to(PROJECT_ROOT)),
            "sha256": source_sha256,
            "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
            "license_status": "VERIFY_BEFORE_REUSE"
        },

        "provenance": {
            "extraction_method": "PyMuPDF text extraction",
            "raw_source_preserved": True,
            "manual_coordinates_added": False,
            "synthetic_values_added": False
        }
    }

    # Basic evidence checks against extracted text.
    checks = {
        "date_present": "13th of August, 2016" in text,
        "mantam_present": "Mantam" in text,
        "passingdang_mantam_road_present": "Passingdang-Mantam Road" in text,
        "bridge_present": "Kanka bridge" in text,
        "road_damage_present": "300 metres of road" in text,
        "rock_avalanche_present": "rock avalanche" in text,
        "cartosat_present": "Cartosat-2B" in text,
    }

    qa_pass = all(checks.values())

    qa = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(PDF_PATH.relative_to(PROJECT_ROOT)),
        "source_sha256": source_sha256,
        "extracted_text_file": str(OUT_TEXT.relative_to(PROJECT_ROOT)),
        "checks": checks,
        "all_evidence_checks_pass": qa_pass,
        "event_spatial_status": event["spatial_status"],
        "label_ready_for_training": event["label_ready_for_training"],
        "status": "PASS" if qa_pass else "FAIL",
        "blocking_reasons": [
            "Reliable event coordinates/geometry have not yet been established.",
            "The event has not yet been spatially mapped to road segments.",
            "A single event is insufficient for final model training.",
            "Reuse/licensing conditions must be verified."
        ]
    }

    OUT_JSON.write_text(
        json.dumps(event, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    OUT_QA.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Event JSON: {OUT_JSON}")
    print(f"Extracted text: {OUT_TEXT}")
    print(f"QA report: {OUT_QA}")
    print(f"Evidence checks: {'PASS' if qa_pass else 'FAIL'}")
    print("Spatial status: PENDING_GEOMETRY")
    print("Training status: BLOCKED")
    print("=" * 70)


if __name__ == "__main__":
    main()