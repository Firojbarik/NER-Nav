#!/usr/bin/env python3
"""Create a deterministic checksum manifest for the ML training inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/ml/ml_input_manifest.json"

EXPECTED = [
    "data/processed/roads/ner_roads_districts.gpkg",
    "data/processed/roads/road_segment_identity.parquet",
    "data/processed/terrain/road_terrain_features.parquet",
    "data/processed/weather/road_rainfall_features_temporal.parquet",
    "data/processed/ml/event_label_registry.json",
    "data/raw/hazards/negative_observations.csv",
    "requirements.lock",
    "ml/features/temporal_design.py",
    "ml/features/seasonal.py",
    "ml/data/training_gate.py",
    "ml/features/rainfall.py",
    "scripts/build_real_temporal_dataset.py",
    "scripts/process_chirps_features.py",
    "scripts/rebuild_terrain_full.py",
    "scripts/train_production_risk_model.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict:
    files = []
    for relative in EXPECTED:
        path = ROOT / relative
        item = {"path": relative, "exists": path.exists()}
        if path.exists():
            item.update({"size_bytes": path.stat().st_size,
                         "sha256": sha256_file(path)})
        files.append(item)
    payload = {
        "schema_version": "1.0.0",
        "purpose": "Complete raw-to-model input checksum manifest",
        "deterministic": True,
        "python_requires": ">=3.12,<3.13",
        "files": files,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def main() -> int:
    manifest = build_manifest()
    OUTPUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    missing = [item["path"] for item in manifest["files"] if not item["exists"]]
    print(json.dumps({"manifest": str(OUTPUT.relative_to(ROOT)),
                      "manifest_sha256": manifest["manifest_sha256"],
                      "missing": missing}, indent=2))
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
