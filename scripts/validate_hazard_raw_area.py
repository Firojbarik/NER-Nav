from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hazards"
REGISTRY = RAW_DIR / "source_registry.json"


ALLOWED_RAW_EXTENSIONS = {
    ".csv",
    ".json",
    ".geojson",
    ".gpkg",
    ".shp",
    ".zip",
    ".parquet",
    ".tif",
    ".tiff",
    ".xlsx",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def load_registry() -> dict:
    if not REGISTRY.exists():
        raise FileNotFoundError(
            f"Source registry not found: {REGISTRY}"
        )

    with REGISTRY.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError("Registry root must be a JSON object.")

    if "sources" not in data:
        raise ValueError("Registry must contain 'sources'.")

    if not isinstance(data["sources"], list):
        raise ValueError("'sources' must be a list.")

    return data


def inspect_raw_files() -> list[dict]:
    records: list[dict] = []

    for path in sorted(RAW_DIR.rglob("*")):
        if not path.is_file():
            continue

        if path.name.lower() in {
            "readme.md",
            "readme.mdx",
            ".gitkeep",
            "source_registry.json",
        }:
            continue

        relative = path.relative_to(PROJECT_ROOT)

        records.append(
            {
                "path": str(relative),
                "name": path.name,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "supported_extension": (
                    path.suffix.lower() in ALLOWED_RAW_EXTENSIONS
                ),
            }
        )

    return records


def validate_registry(data: dict) -> list[str]:
    errors: list[str] = []

    seen_ids: set[str] = set()

    for index, source in enumerate(data["sources"]):
        if not isinstance(source, dict):
            errors.append(
                f"sources[{index}] must be an object."
            )
            continue

        source_id = source.get("source_id")

        if not source_id:
            errors.append(
                f"sources[{index}] missing source_id."
            )
            continue

        if source_id in seen_ids:
            errors.append(
                f"Duplicate source_id: {source_id}"
            )

        seen_ids.add(source_id)

        for required in (
            "name",
            "organization",
            "category",
            "status",
            "official_url",
            "license_status",
        ):
            if required not in source:
                errors.append(
                    f"{source_id}: missing required field '{required}'."
                )

    return errors


def main() -> None:
    print("=" * 70)
    print("NER-Nav — Raw Hazard Area Validation")
    print("=" * 70)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw hazard directory: {RAW_DIR}")
    print()

    registry = load_registry()

    errors = validate_registry(registry)

    files = inspect_raw_files()

    print("Registry:")
    print("-" * 70)
    print(f"Sources registered: {len(registry['sources'])}")

    for source in registry["sources"]:
        print(
            f"  {source['source_id']:<32} "
            f"{source['status']}"
        )

    print()
    print("Raw datasets:")
    print("-" * 70)

    if not files:
        print("  No raw hazard dataset acquired yet.")

    unsupported = 0

    for record in files:
        status = (
            "SUPPORTED"
            if record["supported_extension"]
            else "REVIEW"
        )

        if not record["supported_extension"]:
            unsupported += 1

        print(
            f"  {record['path']}"
            f" | {record['size_bytes']:,} bytes"
            f" | {status}"
        )

    if unsupported:
        errors.append(
            f"{unsupported} raw file(s) have unsupported extensions."
        )

    print()
    print("Integrity:")
    print("-" * 70)

    for record in files:
        print(
            f"  {record['name']}: "
            f"SHA256={record['sha256']}"
        )

    print()
    print("Validation:")
    print("-" * 70)

    if errors:
        for error in errors:
            print(f"  ERROR: {error}")

        raise SystemExit(1)

    print("  Registry schema: PASS")
    print("  Raw directory:   PASS")
    print("  Integrity scan:  PASS")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "raw_directory": str(RAW_DIR),
        "registered_sources": len(registry["sources"]),
        "raw_dataset_count": len(files),
        "datasets": files,
        "status": "PASS",
    }

    output = PROJECT_ROOT / "data" / "processed" / "ml"
    output.mkdir(parents=True, exist_ok=True)

    report_path = output / "hazard_raw_area_validation.json"

    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print()
    print("=" * 70)
    print("RAW HAZARD AREA VALIDATION COMPLETE")
    print("=" * 70)
    print(f"Report: {report_path}")
    print()
    print("IMPORTANT:")
    print("No real hazard dataset has been accepted for training.")
    print("Acquire an officially permitted source before proceeding.")
    print("=" * 70)


if __name__ == "__main__":
    main()