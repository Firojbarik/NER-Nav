from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_HAZARDS_DIR = PROJECT_ROOT / "data" / "raw" / "hazards"
PROCESSED_HAZARDS_DIR = PROJECT_ROOT / "data" / "processed" / "hazards"
PROCESSED_ML_DIR = PROJECT_ROOT / "data" / "processed" / "ml"

SOURCE_MANIFEST_PATH = PROCESSED_ML_DIR / "real_hazard_source_manifest.json"
TRAINING_GATE_PATH = PROCESSED_ML_DIR / "real_hazard_training_gate.json"


SUPPORTED_RAW_EXTENSIONS = {
    ".pdf",
    ".csv",
    ".json",
    ".geojson",
    ".parquet",
    ".gpkg",
    ".xlsx",
    ".xls",
    ".zip",
    ".tif",
    ".tiff",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return -1


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def inspect_tabular_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "inspection_status": "NOT_INSPECTED",
        "rows": None,
        "columns": [],
        "error": None,
    }

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
            result["inspection_status"] = "PASS"
            result["columns"] = [str(c) for c in df.columns]

            # Obtain actual row count without loading everything.
            with path.open("rb") as handle:
                row_count = sum(1 for _ in handle)

            result["rows"] = max(row_count - 1, 0)

        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)

            result["inspection_status"] = "PASS"

            if isinstance(obj, list):
                result["rows"] = len(obj)

                if obj and isinstance(obj[0], dict):
                    result["columns"] = sorted(
                        {str(k) for item in obj if isinstance(item, dict) for k in item}
                    )

            elif isinstance(obj, dict):
                result["rows"] = 1
                result["columns"] = sorted(str(k) for k in obj.keys())

        elif suffix == ".geojson":
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)

            result["inspection_status"] = "PASS"

            if isinstance(obj, dict):
                features = obj.get("features", [])

                if isinstance(features, list):
                    result["rows"] = len(features)

                    properties: set[str] = set()

                    for feature in features[:100]:
                        if isinstance(feature, dict):
                            props = feature.get("properties", {})

                            if isinstance(props, dict):
                                properties.update(str(k) for k in props)

                    result["columns"] = sorted(properties)

        elif suffix == ".parquet":
            df = pd.read_parquet(path, engine="pyarrow")

            result["inspection_status"] = "PASS"
            result["rows"] = int(len(df))
            result["columns"] = [str(c) for c in df.columns]

        elif suffix in {".xlsx", ".xls"}:
            excel = pd.ExcelFile(path)

            result["inspection_status"] = "PASS"
            result["columns"] = [str(x) for x in excel.sheet_names]
            result["rows"] = None

        else:
            result["inspection_status"] = "SKIPPED"

    except Exception as exc:
        result["inspection_status"] = "ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"

    return result


def inventory_raw_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if not RAW_HAZARDS_DIR.exists():
        return records

    for path in sorted(RAW_HAZARDS_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_RAW_EXTENSIONS:
            continue

        record: dict[str, Any] = {
            "path": relative_path(path),
            "filename": path.name,
            "extension": path.suffix.lower(),
            "size_bytes": safe_size(path),
            "sha256": sha256_file(path),
            "modified_at_utc": datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            ).isoformat(),
            "source_type": "RAW_LOCAL_SOURCE",
        }

        if path.suffix.lower() in {
            ".csv",
            ".json",
            ".geojson",
            ".parquet",
            ".xlsx",
            ".xls",
        }:
            record["schema_inspection"] = inspect_tabular_file(path)

        else:
            record["schema_inspection"] = {
                "inspection_status": "NOT_APPLICABLE",
                "rows": None,
                "columns": [],
                "error": None,
            }

        records.append(record)

    return records


def find_confirmation_files() -> list[Path]:
    if not PROCESSED_HAZARDS_DIR.exists():
        return []

    files: list[Path] = []

    for path in PROCESSED_HAZARDS_DIR.rglob("*confirmed_road.parquet"):
        if path.is_file():
            files.append(path)

    return sorted(files)


def inspect_confirmation_file(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": relative_path(path),
        "status": "ERROR",
        "rows": 0,
        "confirmed_rows": 0,
        "source_supported_rows": 0,
        "real_label_rows": 0,
        "columns": [],
        "error": None,
    }

    try:
        df = pd.read_parquet(path, engine="pyarrow")

        record["status"] = "PASS"
        record["rows"] = int(len(df))
        record["columns"] = [str(c) for c in df.columns]

        if "source_supported" in df.columns:
            record["source_supported_rows"] = int(
                df["source_supported"].fillna(False).astype(bool).sum()
            )

        if "confirmed_affected" in df.columns:
            record["confirmed_rows"] = int(
                df["confirmed_affected"].fillna(False).astype(bool).sum()
            )

        # A real training label requires BOTH:
        #
        #   source_supported == True
        #   confirmed_affected == True
        #
        # This prevents a proximity-ranked candidate from becoming
        # a positive label automatically.
        if {
            "source_supported",
            "confirmed_affected",
        }.issubset(df.columns):
            real_label_mask = (
                df["source_supported"].fillna(False).astype(bool)
                & df["confirmed_affected"].fillna(False).astype(bool)
            )

            record["real_label_rows"] = int(real_label_mask.sum())

        elif "confirmation_status" in df.columns:
            status = (
                df["confirmation_status"]
                .fillna("")
                .astype(str)
                .str.upper()
            )

            record["real_label_rows"] = int(
                status.isin(
                    {
                        "SOURCE_CONFIRMED",
                        "CONFIRMED_AFFECTED",
                        "REAL_LABEL",
                    }
                ).sum()
            )

    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"

    return record


def build_training_gate(
    raw_sources: list[dict[str, Any]],
    confirmations: list[dict[str, Any]],
) -> dict[str, Any]:

    total_real_labels = sum(
        int(item.get("real_label_rows", 0))
        for item in confirmations
    )

    total_confirmed = sum(
        int(item.get("confirmed_rows", 0))
        for item in confirmations
    )

    total_source_supported = sum(
        int(item.get("source_supported_rows", 0))
        for item in confirmations
    )

    raw_source_count = len(raw_sources)

    if raw_source_count == 0:
        decision = "BLOCKED_NO_RAW_HAZARD_SOURCE"

        reason = (
            "No supported raw hazard source files were found under "
            "data/raw/hazards. Acquire an officially permitted real "
            "hazard inventory and preserve the original source before "
            "training."
        )

    elif total_real_labels == 0:
        decision = "BLOCKED_NO_REAL_ROAD_LABELS"

        reason = (
            "Raw real hazard sources exist, but no road segment currently "
            "has both source_supported=true and confirmed_affected=true. "
            "The model must not be trained until source-supported road "
            "identities are established."
        )

    else:
        decision = "READY_FOR_REAL_DATASET_CONSTRUCTION"

        reason = (
            "At least one source-supported confirmed road label exists. "
            "The next stage may construct a temporal/spatially leakage-safe "
            "training dataset."
        )

    return {
        "generated_at_utc": utc_now(),
        "decision": decision,
        "reason": reason,
        "rules": {
            "synthetic_values_allowed": False,
            "manual_database_edits_allowed": False,
            "distance_only_confirmation_allowed": False,
            "source_supported_confirmation_required": True,
            "confirmed_affected_required": True,
        },
        "counts": {
            "raw_source_files": raw_source_count,
            "confirmation_files": len(confirmations),
            "source_supported_rows": total_source_supported,
            "confirmed_rows": total_confirmed,
            "real_label_rows": total_real_labels,
        },
        "training_allowed": total_real_labels > 0,
        "current_model_status": (
            "REAL_DATA_MODEL_NOT_READY"
            if total_real_labels == 0
            else "REAL_LABELS_AVAILABLE_FOR_DATASET_CONSTRUCTION"
        ),
        "next_required_stage": (
            "Acquire/map additional real events and establish "
            "source-supported road identities."
            if total_real_labels == 0
            else
            "Build leakage-safe temporal training dataset, then evaluate "
            "baseline and XGBoost models."
        ),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )


def main() -> int:
    print("=" * 70)
    print("NER-Nav — Real Hazard Training Gate Audit")
    print("=" * 70)
    print()

    print("Project root:")
    print(f"  {PROJECT_ROOT}")
    print()

    print("Raw hazard directory:")
    print(f"  {RAW_HAZARDS_DIR}")
    print()

    print("Scanning raw source files...")
    print("-" * 70)

    raw_sources = inventory_raw_sources()

    print(f"Raw source files found: {len(raw_sources)}")

    for item in raw_sources:
        print(
            f"  {item['filename']} "
            f"({item['size_bytes']} bytes) "
            f"SHA256={item['sha256'][:16]}..."
        )

    print()

    print("Scanning existing road-confirmation artifacts...")
    print("-" * 70)

    confirmation_paths = find_confirmation_files()

    confirmations = [
        inspect_confirmation_file(path)
        for path in confirmation_paths
    ]

    print(f"Confirmation files found: {len(confirmations)}")

    for item in confirmations:
        print(
            f"  {item['path']}: "
            f"rows={item['rows']}, "
            f"source_supported={item['source_supported_rows']}, "
            f"confirmed={item['confirmed_rows']}, "
            f"REAL_LABEL={item['real_label_rows']}"
        )

    print()

    manifest = {
        "generated_at_utc": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "raw_hazard_directory": relative_path(RAW_HAZARDS_DIR),
        "raw_sources": raw_sources,
        "confirmation_artifacts": confirmations,
        "provenance_policy": {
            "raw_files_hashed": True,
            "synthetic_values_added": False,
            "manual_database_values_changed": False,
            "labels_created_by_this_script": False,
        },
    }

    write_json(SOURCE_MANIFEST_PATH, manifest)

    gate = build_training_gate(
        raw_sources=raw_sources,
        confirmations=confirmations,
    )

    write_json(TRAINING_GATE_PATH, gate)

    print("=" * 70)
    print("REAL HAZARD TRAINING GATE")
    print("=" * 70)

    print(f"Decision:             {gate['decision']}")
    print(
        f"Raw source files:     "
        f"{gate['counts']['raw_source_files']}"
    )
    print(
        f"Confirmation files:   "
        f"{gate['counts']['confirmation_files']}"
    )
    print(
        f"Source-supported:     "
        f"{gate['counts']['source_supported_rows']}"
    )
    print(
        f"Confirmed affected:   "
        f"{gate['counts']['confirmed_rows']}"
    )
    print(
        f"Real training labels: "
        f"{gate['counts']['real_label_rows']}"
    )
    print()

    print("Provenance safeguards:")
    print("  Synthetic values:        NO")
    print("  Manual DB edits:         NO")
    print("  Automatic labels:        NO")
    print("  Distance-only labels:    NO")
    print("  Source-supported labels: REQUIRED")
    print()

    print("Outputs:")
    print(f"  Manifest: {SOURCE_MANIFEST_PATH}")
    print(f"  Gate:     {TRAINING_GATE_PATH}")
    print()

    print("Decision reason:")
    print(f"  {gate['reason']}")
    print()

    if gate["training_allowed"]:
        print("STATUS: PASS")
        return 0

    print("STATUS: BLOCKED")
    print()
    print(
        "Do NOT train the final real-data hazard model yet."
    )

    return 2


if __name__ == "__main__":
    sys.exit(main())