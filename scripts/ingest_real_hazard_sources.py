
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pypdf import PdfReader


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PDF = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hazards"
    / "Sikkim_Landslide_2016_NRSC.pdf"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

EVENTS_OUTPUT = OUTPUT_DIR / "real_hazard_events.json"
MANIFEST_OUTPUT = OUTPUT_DIR / "real_hazard_source_ingestion_manifest.json"


# This is provenance metadata only.
# It does NOT claim that the URL was accessed by this script.
OFFICIAL_SOURCE_URL = (
    "https://www.nrsc.gov.in/"
)


EVENT_ID = "nrsc_sikkim_mantam_2016_08_13"


# These values are already part of the project's verified source record.
#
# IMPORTANT:
# This ingestion script does not invent these values and does not write
# anything into the road database.
EVENT_RECORD = {
    "event_id": EVENT_ID,
    "event_name": "So Bhir / Mantam Landslide",
    "event_date": "2016-08-13",

    "coordinates": {
        "latitude": 27.5397,
        "longitude": 88.50068611111111,
        "crs": "EPSG:4326",
        "coordinate_source": (
            "Martha, Roy & Kumar (2017), Current Science 113(7), "
            "Assessment of the valley-blocking 'So Bhir' landslide "
            "near Mantam village, North Sikkim, India, using satellite images."
        ),
        "coordinate_verification": "PUBLISHED_SOURCE_CROSS_CHECK",
    },

    "hazard": {
        "type": "LANDSLIDE",
        "name": "So Bhir / Mantam Landslide",
    },

    "road_damage": {
        "reported": True,

        # This is a source-backed project fact.
        # It is NOT treated as proof that an OSM segment has been identified.
        "reported_road": "Passingdang-Mantam Road",

        "reported_damage_length_m": 300,

        # Critical training safeguards.
        "road_identity_verified": False,
        "source_supported": False,
        "confirmed_affected": False,
        "label_ready_for_training": False,

        "automatic_confirmation_allowed": False,
    },
}


# ============================================================================
# Utility functions
# ============================================================================

def utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    """Calculate SHA256 without modifying the source file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def extract_pdf_text(path: Path) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract text from every PDF page.

    The extracted text is evidence inspection only.
    It is never written back into the PDF.
    """
    reader = PdfReader(str(path))

    pages: list[dict[str, Any]] = []
    text_parts: list[str] = []

    for index, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception as exc:
            page_text = ""
            pages.append(
                {
                    "page": index,
                    "characters": 0,
                    "extraction_error": str(exc),
                }
            )
            continue

        pages.append(
            {
                "page": index,
                "characters": len(page_text),
                "extraction_error": None,
            }
        )

        text_parts.append(page_text)

    return "\n".join(text_parts), pages


def normalize_text(value: str) -> str:
    """
    Normalize PDF text for searching.

    Handles common PDF extraction differences such as:
      Passingdang - Mantam
      Passingdang-Mantam
      PASSINGDANG
      line breaks inside words
    """
    value = value.replace("\u00ad", "")
    value = value.replace("\r", "\n")

    # Collapse whitespace.
    value = re.sub(r"\s+", " ", value)

    return value.strip().lower()


def search_term(text: str, pattern: str) -> bool:
    """Case-insensitive regex search."""
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def inspect_pdf_evidence(source_text: str) -> dict[str, Any]:
    """
    Inspect what is actually present in the extracted PDF text.

    IMPORTANT:
    Absence of an exact phrase does NOT cause ingestion to fail.

    The purpose of this function is to preserve the distinction between:
      1. evidence found in extracted PDF text;
      2. verified project metadata from the source record.
    """
    normalized = normalize_text(source_text)

    checks = {
        "landslide": search_term(
            normalized,
            r"\blandslide\b",
        ),

        "mantam": search_term(
            normalized,
            r"\bmantam\b",
        ),

        "passingdang": search_term(
            normalized,
            r"\bpassingdang\b",
        ),

        "road": search_term(
            normalized,
            r"\broad\b",
        ),

        "year_2016": search_term(
            normalized,
            r"\b2016\b",
        ),

        "august": search_term(
            normalized,
            r"\baugust\b",
        ),

        "damage": search_term(
            normalized,
            r"\bdamag(?:e|ed|ing)\b",
        ),

        "three_hundred": search_term(
            normalized,
            r"\b300\b|\bthree hundred\b",
        ),
    }

    # Try several real-text forms of the road name.
    road_name_patterns = [
        r"passingdang\s*[-–—]\s*mantam",
        r"passingdang\s+mantam",
        r"mantam\s+road",
        r"passingdang",
    ]

    road_name_match = None

    for pattern in road_name_patterns:
        match = re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        )

        if match:
            road_name_match = {
                "pattern": pattern,
                "matched_text": match.group(0),
            }
            break

    # Pull useful snippets around terms actually found in the PDF.
    snippets: list[dict[str, str]] = []

    snippet_patterns = [
        r"landslide",
        r"mantam",
        r"passingdang",
        r"road",
        r"damage",
    ]

    seen_snippets: set[str] = set()

    for pattern in snippet_patterns:
        for match in re.finditer(
            pattern,
            normalized,
            flags=re.IGNORECASE,
        ):
            start = max(0, match.start() - 180)
            end = min(len(normalized), match.end() + 300)

            snippet = normalized[start:end].strip()

            if snippet and snippet not in seen_snippets:
                seen_snippets.add(snippet)

                snippets.append(
                    {
                        "term": pattern,
                        "snippet": snippet,
                    }
                )

            if len(snippets) >= 12:
                break

        if len(snippets) >= 12:
            break

    return {
        "normalized_character_count": len(normalized),
        "term_presence": checks,
        "reported_road_exact_phrase_found": road_name_match is not None,
        "reported_road_match": road_name_match,
        "snippets": snippets,
    }


def build_event(
    *,
    source_sha256: str,
    source_text: str,
    pdf_pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build the real-source event record.

    No road segment is selected here.
    No OSM value is generated here.
    No training label is generated here.
    """
    pdf_evidence = inspect_pdf_evidence(source_text)

    event = {
        **EVENT_RECORD,

        "source": {
            "organization": "National Remote Sensing Centre, ISRO",
            "document": SOURCE_PDF.name,
            "local_path": str(
                SOURCE_PDF.relative_to(PROJECT_ROOT)
            ),
            "sha256": source_sha256,

            # Provenance reference only.
            "official_source_url": OFFICIAL_SOURCE_URL,

            "source_type": "RAW_LOCAL_SOURCE",
            "source_file_unchanged": True,
        },

        "pdf_extraction": {
            "extraction_status": "PASS",
            "page_count": len(pdf_pages),
            "extracted_character_count": len(source_text),
            "pages": pdf_pages,
            "evidence_inspection": pdf_evidence,
        },

        "provenance": {
            "synthetic_values_added": False,
            "manual_database_values_changed": False,
            "source_file_modified": False,
            "road_database_modified": False,
            "training_label_created": False,
        },

        "training_gate": {
            "source_supported": False,
            "confirmed_affected": False,
            "label_ready_for_training": False,
            "training_allowed": False,

            "status": (
                "BLOCKED_ROAD_IDENTITY_NOT_VERIFIED"
            ),

            "reason": (
                "The real hazard source has been ingested, but "
                "the source evidence has not independently established "
                "the identity of an OSM road segment. The reported road "
                "name is retained as source metadata only. Spatial "
                "proximity alone must not create a training label."
            ),
        },

        "ingestion": {
            "ingested_at_utc": utc_now(),
            "method": "PDF_TEXT_EXTRACTION_AND_SOURCE_RECORD_VALIDATION",
        },
    }

    return event


def build_manifest(
    *,
    source_sha256: str,
    source_text: str,
    pdf_pages: list[dict[str, Any]],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Create an auditable ingestion manifest."""
    return {
        "generated_at_utc": utc_now(),

        "project_root": str(PROJECT_ROOT),

        "source": {
            "path": str(
                SOURCE_PDF.relative_to(PROJECT_ROOT)
            ),
            "filename": SOURCE_PDF.name,
            "size_bytes": SOURCE_PDF.stat().st_size,
            "sha256": source_sha256,
            "source_type": "RAW_LOCAL_SOURCE",
        },

        "pdf_extraction": {
            "status": "PASS",
            "page_count": len(pdf_pages),
            "characters": len(source_text),
        },

        "event": {
            "event_id": event["event_id"],
            "event_date": event["event_date"],
            "event_name": event["event_name"],
        },

        "road_evidence": {
            "reported_road": event["road_damage"]["reported_road"],
            "reported_damage_length_m": event[
                "road_damage"
            ]["reported_damage_length_m"],

            "road_identity_verified": False,
            "source_supported": False,
            "confirmed_affected": False,
            "label_ready_for_training": False,
        },

        "provenance_policy": {
            "synthetic_values_added": False,
            "manual_database_values_changed": False,
            "source_file_modified": False,
            "road_database_modified": False,
            "automatic_training_label_created": False,
        },

        "training_gate": {
            "decision": "BLOCKED_ROAD_IDENTITY_NOT_VERIFIED",
            "training_allowed": False,
            "reason": (
                "Real source ingestion succeeded, but a real "
                "source-supported road identity has not yet been "
                "established."
            ),
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write deterministic human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("NER-Nav — Real Hazard Source Ingestion")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Source validation
    # ------------------------------------------------------------------

    print("Checking official raw source...")
    print("-" * 70)

    if not SOURCE_PDF.exists():
        raise FileNotFoundError(
            f"Required raw source does not exist:\n{SOURCE_PDF}"
        )

    if not SOURCE_PDF.is_file():
        raise RuntimeError(
            f"Expected a file but found something else:\n{SOURCE_PDF}"
        )

    source_sha256 = sha256_file(SOURCE_PDF)

    print(f"Source: {SOURCE_PDF}")
    print(f"SHA256: {source_sha256}")
    print()

    # ------------------------------------------------------------------
    # PDF extraction
    # ------------------------------------------------------------------

    print("Extracting source text...")
    print("-" * 70)

    source_text, pdf_pages = extract_pdf_text(SOURCE_PDF)

    if not source_text.strip():
        raise RuntimeError(
            "No extractable text was found in the PDF. "
            "Do not create an event from an empty extraction."
        )

    print(f"Extracted characters: {len(source_text):,}")
    print(f"PDF pages inspected:  {len(pdf_pages)}")
    print()

    # ------------------------------------------------------------------
    # Evidence inspection
    # ------------------------------------------------------------------

    print("Inspecting source evidence...")
    print("-" * 70)

    evidence = inspect_pdf_evidence(source_text)

    for key, value in evidence["term_presence"].items():
        print(f"{key:24}: {value}")

    print(
        f"{'Exact road phrase found':24}: "
        f"{evidence['reported_road_exact_phrase_found']}"
    )

    if evidence["reported_road_match"]:
        print(
            "Road-text match:        "
            f"{evidence['reported_road_match']['matched_text']}"
        )
    else:
        print(
            "Road-text match:        "
            "NOT FOUND IN EXTRACTED PDF TEXT"
        )

    print()

    # ------------------------------------------------------------------
    # Important provenance decision
    # ------------------------------------------------------------------

    print("Building real-source event record...")
    print("-" * 70)

    event = build_event(
        source_sha256=source_sha256,
        source_text=source_text,
        pdf_pages=pdf_pages,
    )

    manifest = build_manifest(
        source_sha256=source_sha256,
        source_text=source_text,
        pdf_pages=pdf_pages,
        event=event,
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        EVENTS_OUTPUT,
        {
            "schema_version": "1.0",
            "generated_at_utc": utc_now(),
            "events": [event],
        },
    )

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("REAL HAZARD SOURCE INGESTION COMPLETE")
    print("=" * 70)
    print()

    print("Event:")
    print(f"  ID:              {event['event_id']}")
    print(f"  Name:            {event['event_name']}")
    print(f"  Date:            {event['event_date']}")

    print()
    print("Coordinates:")
    print(
        f"  Latitude:        "
        f"{event['coordinates']['latitude']}"
    )
    print(
        f"  Longitude:       "
        f"{event['coordinates']['longitude']}"
    )
    print(
        f"  CRS:             "
        f"{event['coordinates']['crs']}"
    )

    print()
    print("Reported road:")
    print(
        f"  Name:            "
        f"{event['road_damage']['reported_road']}"
    )
    print(
        f"  Damage length:   "
        f"{event['road_damage']['reported_damage_length_m']} m"
    )

    print()
    print("PDF extraction:")
    print(
        f"  Characters:      "
        f"{len(source_text):,}"
    )
    print(
        f"  Exact road text: "
        f"{evidence['reported_road_exact_phrase_found']}"
    )

    print()
    print("Training gate:")
    print(
        f"  Source supported: "
        f"{event['training_gate']['source_supported']}"
    )
    print(
        f"  Confirmed affected:"
        f" {event['training_gate']['confirmed_affected']}"
    )
    print(
        f"  Label ready:      "
        f"{event['training_gate']['label_ready_for_training']}"
    )
    print(
        f"  Training allowed: "
        f"{event['training_gate']['training_allowed']}"
    )
    print(
        f"  Status:            "
        f"{event['training_gate']['status']}"
    )

    print()
    print("Provenance safeguards:")
    print(
        f"  Synthetic values:  "
        f"{event['provenance']['synthetic_values_added']}"
    )
    print(
        f"  Manual DB edits:   "
        f"{event['provenance']['manual_database_values_changed']}"
    )
    print(
        f"  Source modified:   "
        f"{event['provenance']['source_file_modified']}"
    )
    print(
        f"  Road DB modified:  "
        f"{event['provenance']['road_database_modified']}"
    )
    print(
        f"  Training label:    "
        f"{event['provenance']['training_label_created']}"
    )

    print()
    print("Outputs:")
    print(f"  Events:   {EVENTS_OUTPUT}")
    print(f"  Manifest: {MANIFEST_OUTPUT}")

    print()
    print("IMPORTANT:")
    print(
        "  The exact road-name phrase was NOT required to be present "
        "in PDF text extraction."
    )
    print(
        "  The reported road remains source metadata, not an OSM "
        "road identity."
    )
    print(
        "  No road segment was confirmed."
    )
    print(
        "  No training label was created."
    )
    print(
        "  No synthetic value was added."
    )
    print(
        "  No database value was changed."
    )

    print()
    print("STATUS: PASS")
    print(
        "INGESTION PASS ≠ TRAINING READY"
    )


if __name__ == "__main__":
    main()
