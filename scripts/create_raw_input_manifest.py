#!/usr/bin/env python3
"""Generate a complete raw-to-model input checksum manifest.

Covers every raw acquisition (weather rasters, incident CSVs, HTML sources,
negative observations, event registry) plus the processed geo/terrain inputs
that feed dataset construction. Writes a single manifest JSON with per-file
SHA-256 digests and a self-consistency manifest hash.

Usage:
    python scripts/create_raw_input_manifest.py          # dry run, prints summary
    python scripts/create_raw_input_manifest.py --apply  # write the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Root trees captured in the manifest (raw acquisitions + ML-relevant processed).
INPUT_ROOTS = [
    "data/raw",
    "data/processed/roads",
    "data/processed/terrain",
    "data/processed/weather",
    "data/processed/graph",
    "data/processed/admin",
    "data/processed/ml",
]
# Data files that are outputs, not inputs, of the ML pipeline (excluded).
EXCLUDE_PREFIXES = [
    "data/processed/ml/real_temporal_risk_dataset.parquet",
    "data/processed/ml/real_temporal_risk_dataset_qa.json",
    "data/processed/ml/raw_input_manifest.json",
    "data/processed/ml/real_temporal_risk_model_results_latest.json",
    "data/processed/ml/readiness_audits/",
    "data/processed/ml/evaluation_runs/",
    "data/processed/ml/monitoring/",
    "data/predictions/",
    "data/models/",
]
OUTPUT = ROOT / "data/processed/ml/raw_input_manifest.json"
MIN_EXPECTED_FILES = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files() -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for root_str in INPUT_ROOTS:
        root = ROOT / root_str
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if any(rel.startswith(pre) or f"/{rel}" == pre
                   for pre in EXCLUDE_PREFIXES):
                continue
            if path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the manifest to data/processed/ml/raw_input_manifest.json")
    args = ap.parse_args()

    files = collect_files()
    print(f"Collecting digests for {len(files)} files ...")
    entries = {}
    for i, path in enumerate(files, 1):
        rel = path.relative_to(ROOT).as_posix()
        entries[rel] = sha256_file(path)
        if i % 200 == 0:
            print(f"  hashed {i}/{len(files)}")

    content = dict(entries)
    payload = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(content),
        "input_roots": INPUT_ROOTS,
        "files": content,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    if len(content) < MIN_EXPECTED_FILES:
        print(f"ERROR: manifest too small ({len(content)} < {MIN_EXPECTED_FILES})", file=sys.stderr)
        return 2

    if not args.apply:
        print(f"Dry run: {len(content)} files, manifest_sha256="
              f"{payload['manifest_sha256'][:12]}..."
        )
        return 0

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(content)} digests to {OUTPUT}")
    print(f"manifest_sha256={payload['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())