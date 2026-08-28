#!/usr/bin/env python3
"""Build a demo risk feed for the NER-Nav hackathon dashboard.

Computes actual CHIRPS rainfall features + real terrain for a curated set of
National Highway segments and scores them with the demo-ready model.  Output is
a single JSON file that the web dashboard reads directly (no live API needed on
stage, no network dependencies).

Usage:
    python scripts/build_demo_risk_feed.py [--tag 2026-08-28_151030_b7486dea]
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.rainfall import aggregate_windows  # noqa: E402

from scripts.predict_real_temporal import (  # noqa: E402
    _calibrate,
    risk_level,
    validate_and_prepare,
)

RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
ROADS_FILE = PROJECT_ROOT / "data" / "processed" / "roads" / "ner_roads.gpkg"
TERRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "terrain" / "road_terrain_features.parquet"
DATASET_FILE = PROJECT_ROOT / "data" / "processed" / "ml" / "real_temporal_risk_dataset.parquet"
MODEL_DIR = PROJECT_ROOT / "data" / "models"
OUTPUT = PROJECT_ROOT / "data" / "predictions" / "demo_risk_feed.json"

WINDOWS = [1, 3, 7, 14, 30]
LOOKBACK_DAYS = 30
DEFAULT_TAG = "2026-08-28_151030_b7486dea"


def discover_chirps_files() -> list[tuple[date, Path]]:
    found: list[tuple[date, Path]] = []
    for year_dir in RAW_WEATHER_DIR.glob("chirps_*"):
        if not year_dir.is_dir():
            continue
        for gz in year_dir.glob("chirps-v2.0.*.tif.gz"):
            stem = gz.stem
            rel = stem.removeprefix("chirps-v2.0.").removesuffix(".tif")
            try:
                d = datetime.strptime(rel, "%Y.%m.%d").date()
            except ValueError:
                continue
            found.append((d, gz))
    found.sort(key=lambda t: t[0])
    return found


def decompress(gz_path: Path, out_dir: Path) -> Path:
    out_tif = out_dir / gz_path.name.replace(".gz", "")
    if not out_tif.exists():
        with gzip.open(gz_path, "rb") as f_in, open(out_tif, "wb") as f_out:
            import shutil
            shutil.copyfileobj(f_in, f_out, length=1 << 20)
    return out_tif


def sample_points(tif_path: Path, coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        inv = ~src.transform
        cols, rows = inv * (coords[:, 0], coords[:, 1])
        cols = np.rint(np.asarray(cols)).astype(np.int64)
        rows = np.rint(np.asarray(rows)).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.any(inside):
            return out
        band = src.read(1)
        raw = np.full(n, np.nan, dtype=np.float64)
        raw[inside] = band[rows[inside], cols[inside]]
    valid = ~np.isnan(raw) & (raw >= 0)
    out[valid] = raw[valid]
    return out


def compute_rainfall(coords: np.ndarray, anchor: date,
                     available: dict[date, Path], decompressed: dict[date, Path]) -> pd.DataFrame:
    needed = [anchor - timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)]
    present = [(d, anchor - d) for d in needed if d in available]
    if not present:
        raise RuntimeError(f"no CHIRPS data within 30-day lookback of {anchor}")

    offsets = np.array([-(anchor - d).days for d, _ in present], dtype=np.int64)
    daily = np.full((coords.shape[0], len(present)), np.nan, dtype=np.float64)
    for j, (d, _) in enumerate(present):
        tif = decompressed.get(d) or (available[d] if d in available else None)
        if tif is None:
            continue
        try:
            daily[:, j] = sample_points(tif, coords)
        except Exception:
            continue

    agg = aggregate_windows(daily, offsets, WINDOWS)
    df = pd.DataFrame({f"rainfall_{w}day": agg[w] for w in WINDOWS})
    return df


def pick_segments(limit: int) -> pd.DataFrame:
    ds = pd.read_parquet(DATASET_FILE)
    terrain = pd.read_parquet(TERRAIN_FILE)[["osm_id", "elevation_m", "slope_degrees"]]
    roads = gpd.read_file(ROADS_FILE)
    roads["osm_id"] = roads["osm_id"].astype("int64")

    # One representative segment per (ref) covering the full NER corridor set.
    segs = (ds.groupby("ref")["osm_id"].first().reset_index())
    merged = segs.merge(terrain, on="osm_id", how="left")
    geom = roads.set_index("osm_id")["geometry"].to_dict()
    merged["geometry"] = merged["osm_id"].map(geom)
    merged = merged.dropna(subset=["geometry"])
    meta = (ds[["osm_id", "state", "district"]].drop_duplicates("osm_id"))
    merged = merged.merge(meta, on="osm_id", how="left")
    merged = gpd.GeoDataFrame(merged, geometry=merged["geometry"],
                              crs=roads.crs)
    return merged.head(limit)


def scores_for_tag(tag: str) -> dict:
    report_path = MODEL_DIR / f"prod_report_{tag}.json"
    feature_path = MODEL_DIR / f"prod_real_temporal_{tag}_features.json"
    model_path = MODEL_DIR / f"prod_real_temporal_{tag}.ubj"
    calib_path = MODEL_DIR / f"prod_real_temporal_{tag}_calib.json"

    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    spec = json.loads(feature_path.read_text(encoding="utf-8"))
    calibration = (json.loads(calib_path.read_text(encoding="utf-8"))
                   if calib_path.exists() else None)
    threshold = float(report.get("test_threshold", 0.5))
    policy_path = MODEL_DIR / f"risk_policy_{tag}.json"
    risk_policy = json.loads(policy_path.read_text(encoding="utf-8"))

    def score(features: dict) -> float:
        import xgboost as xgb  # noqa: F811
        X = pd.DataFrame([{name: features.get(name) for name in spec["features"]}])
        raw = float(model.predict_proba(X)[:, 1][0])
        return _calibrate(raw, calibration)

    return {
        "model": model, "spec": spec, "calibration": calibration,
        "threshold": threshold, "risk_policy": risk_policy,
        "report": report, "score": score,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--anchor-date", default=None,
                        help="YYYY-MM-DD prediction anchor; defaults to the latest CHIRPS day")
    parser.add_argument("--limit", type=int, default=14)
    args = parser.parse_args()

    available_files = discover_chirps_files()
    if not available_files:
        print("ERROR: no CHIRPS files found", file=sys.stderr)
        return 2
    avail_map = {d: p for d, p in available_files}
    latest = sorted(avail_map)[-1]
    anchor = (datetime.strptime(args.anchor_date, "%Y-%m-%d").date()
              if args.anchor_date else latest)
    print(f"Anchor date: {anchor} (latest CHIRPS: {latest})")

    segments = pick_segments(args.limit)
    coords = np.column_stack(
        [segments.geometry.centroid.x.values, segments.geometry.centroid.y.values])

    bundle = scores_for_tag(args.tag)
    with tempfile.TemporaryDirectory(prefix="chirps_demo_") as td:
        decompressed = {}
        needed = [anchor - timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)]
        for d in needed:
            if d in avail_map and d not in decompressed:
                decompressed[d] = decompress(avail_map[d], Path(td))
        rain = compute_rainfall(coords, anchor, avail_map, decompressed)

    rows = []
    for i, seg in segments.iterrows():
        base = {
            "rainfall_1day": rain.loc[i, "rainfall_1day"],
            "rainfall_3day": rain.loc[i, "rainfall_3day"],
            "rainfall_7day": rain.loc[i, "rainfall_7day"],
            "rainfall_14day": rain.loc[i, "rainfall_14day"],
            "rainfall_30day": rain.loc[i, "rainfall_30day"],
            "elevation_m": seg["elevation_m"],
            "slope_degrees": seg["slope_degrees"],
        }
        try:
            prepared, quality = validate_and_prepare(
                base, bundle["spec"]["features"],
                f"{anchor.isoformat()}T12:00:00+05:30",
                weather_observed_at=None,
                feature_profile=bundle["spec"].get("training_feature_profile", {}))
            probability = bundle["score"](prepared)
            level = risk_level(probability, bundle["risk_policy"])
            decision = "ALERT" if probability >= bundle["threshold"] else "NO_ALERT"
            rows.append({
                "osm_id": int(seg["osm_id"]),
                "ref": str(seg["ref"]),
                "state": str(seg.get("state") or ""),
                "district": str(seg.get("district") or ""),
                "latitude": round(float(seg.geometry.centroid.y), 6),
                "longitude": round(float(seg.geometry.centroid.x), 6),
                "elevation_m": None if pd.isna(seg["elevation_m"]) else float(seg["elevation_m"]),
                "slope_degrees": None if pd.isna(seg["slope_degrees"]) else float(seg["slope_degrees"]),
                "rainfall": {
                    "1day": round(float(rain.loc[i, "rainfall_1day"]), 2),
                    "3day": round(float(rain.loc[i, "rainfall_3day"]), 2),
                    "7day": round(float(rain.loc[i, "rainfall_7day"]), 2),
                    "14day": round(float(rain.loc[i, "rainfall_14day"]), 2),
                    "30day": round(float(rain.loc[i, "rainfall_30day"]), 2),
                },
                "disruption_probability": round(probability, 4),
                "risk_level": level,
                "operating_decision": decision,
                "confidence_flags": quality["confidence_flags"],
            })
        except ValueError as exc:
            print(f"WARN: segment {seg['osm_id']} ({seg['ref']}) skipped: {exc}",
                  file=sys.stderr)

    feed = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "anchor_date": anchor.isoformat(),
        "model_version": args.tag,
        "model_status": bundle["report"].get("status"),
        "operating_threshold": bundle["threshold"],
        "risk_policy": bundle["risk_policy"],
        "caveat": bundle["report"].get("caveat", ""),
        "segments": sorted(rows, key=lambda r: r["disruption_probability"], reverse=True),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(feed, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} scored segments to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())