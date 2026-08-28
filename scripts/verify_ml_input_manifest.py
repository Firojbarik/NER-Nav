#!/usr/bin/env python3
"""Verify every file in the deterministic ML input manifest is unchanged."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/ml/ml_input_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path = MANIFEST) -> dict:
    if not path.exists():
        return {"status": "FAIL", "errors": [f"missing manifest: {path}"]}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    declared_hash = manifest.get("manifest_sha256")
    content = dict(manifest)
    content.pop("manifest_sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"))
    if declared_hash != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
        errors.append("manifest SHA256 does not match its contents")
    for item in manifest.get("files", []):
        source = ROOT / item["path"]
        if not source.exists():
            errors.append(f"missing input: {item['path']}")
        elif sha256_file(source) != item.get("sha256"):
            errors.append(f"SHA256 mismatch: {item['path']}")
    return {"status": "PASS" if not errors else "FAIL",
            "manifest_sha256": manifest.get("manifest_sha256"),
            "errors": errors}


def main() -> int:
    result = verify()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
