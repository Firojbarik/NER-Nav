#!/usr/bin/env python3
"""
NER-Nav: CHIRPS Rainfall Feature Extraction (corrected)

Correctness fixes over the prior version (F-3, F-7 from docs/ML_AUDIT.md):
  * F-3  No-data is NaN, never 0: a missing/outside-coverage CHIRPS pixel is
         "unknown", not "no rain". Valid 0 mm cells stay 0.
  * F-7  Aggregates are calendar-date-anchored relative to each prediction
         timestamp. A 3-day window always means the 3 calendar days strictly
         before the anchor, so gaps cannot shift values into the wrong slot.
  * Temporal leakage guard: the anchor day itself (offset 0) and all future
         days are excluded from every window.

Output (long format, one row per (osm_id, feature_date)):
    osm_id, feature_date,
    rainfall_1day, rainfall_3day, rainfall_7day, rainfall_14day, rainfall_30day,
    rainfall_days_available   # count of present+valid observed days in the 30-day lookback

Contract (defensible, explicit):
  * Only days with a CHIRPS file on disk are considered observed. Days with no
    file (outside the downloaded range / failed download) are simply not in the
    lookback sum; `rainfall_days_available` records how many days contributed.
  * If a day has a file but is unreadable OR its pixel is no-data, that day is
    NaN. The strict aggregation treats any window containing such a day as NaN
    (never fabricate "no rain" from unknown data); pass --allow-partial to sum
    the valid days instead. A row is emitted for an anchor only if at least one
    lookback day is present.

Author: Senior ML Engineer
Date: 2026-08-27
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from ml.features.rainfall import (  # noqa: E402
    aggregate_windows,
    available_day_counts,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
PROCESSED_WEATHER_DIR = PROJECT_ROOT / "data" / "processed" / "weather"
PROCESSED_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

ROADS_FILE = PROJECT_ROOT / "data" / "processed" / "roads" / "ner_roads.gpkg"
DEFAULT_OUTPUT = PROCESSED_WEATHER_DIR / "road_rainfall_features.parquet"

LOOKBACK_DAYS = 30
WINDOWS = [1, 3, 7, 14, 30]


def discover_chirps_files() -> List[Tuple[date, Path]]:
    """Return sorted list of (calendar_date, gz_path) for all available files."""
    found: List[Tuple[date, Path]] = []
    for year_dir in RAW_WEATHER_DIR.glob("chirps_*"):
        if not year_dir.is_dir():
            continue
        for gz in year_dir.glob("chirps-v2.0.*.tif.gz"):
            # e.g. chirps-v2.0.2015.01.01.tif.gz
            stem = gz.stem  # strips .gz -> chirps-v2.0.2015.01.01.tif
            rel = stem.removeprefix("chirps-v2.0.").removesuffix(".tif")
            try:
                d = datetime.strptime(rel, "%Y.%m.%d").date()
            except ValueError:
                print(f"WARNING: unparsable CHIRPS filename (skipping): {gz.name}")
                continue
            found.append((d, gz))
    found.sort(key=lambda t: t[0])
    return found


def decompress(gz_path: Path, out_dir: Path) -> Path:
    """Decompress a .tif.gz into out_dir, returning the .tif path.

    Reuses an already-decompressed copy if present so the 30-day rolling windows
    across anchors are not repeatedly uncompressed.
    """
    out_tif = out_dir / gz_path.name.replace(".gz", "")
    if not out_tif.exists():
        with gzip.open(gz_path, "rb") as f_in, open(out_tif, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1 << 20)
    return out_tif


def sample_roads(tif_path: Path, road_coords: np.ndarray) -> np.ndarray:
    """Sample a CHIRPS precip band at road centroids -> (n_roads,) float.

    Negative/nodata and out-of-bounds/error samples become NaN (unknown),
    never 0. Valid 0 mm pixels stay 0 (no rain that day).

    road_coords : (n, 2) array of (lon, lat) in EPSG:4326.
    """
    n = road_coords.shape[0]
    out = np.full(n, np.nan, dtype=np.float64)
    with rasterio.open(tif_path) as src:
        height, width = src.height, src.width
        # Vectorized pixel lookup via the inverse affine transform.
        inv = ~src.transform
        cols, rows = inv * (road_coords[:, 0], road_coords[:, 1])
        cols = np.rint(np.asarray(cols)).astype(np.int64)
        rows = np.rint(np.asarray(rows)).astype(np.int64)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        if not np.any(inside):
            return out
        band = src.read(1)
        raw = np.full(n, np.nan, dtype=np.float64)
        raw[inside] = band[rows[inside], cols[inside]]
    # CHIRPS nodata sentinel: any strictly negative value means "no data".
    valid = ~np.isnan(raw) & (raw >= 0)
    out[valid] = raw[valid]
    return out


def build_roads():
    if not ROADS_FILE.exists():
        raise FileNotFoundError(f"Roads file not found: {ROADS_FILE}")
    gdf = gpd.read_file(ROADS_FILE)
    if "osm_id" not in gdf.columns:
        raise ValueError("Roads file must contain an osm_id column")
    coords = np.column_stack(
        [gdf.geometry.centroid.x.values, gdf.geometry.centroid.y.values]
    )
    return gdf["osm_id"].astype(np.int64).values, coords, gdf.crs


def resolve_dates(available: List[Tuple[date, Path]], args) -> List[date]:
    """Determine the anchor dates to process."""
    avail_dates = [d for d, _ in available]
    if not avail_dates:
        raise SystemExit("No CHIRPS files found. Run scripts/download_chirps_rainfall.py first.")

    if args.dates:
        anchors = sorted({datetime.strptime(d, "%Y-%m-%d").date() for d in args.dates})
        # Only keep anchors that have at least one available day in lookback.
        valid = [a for a in anchors if any(a - d <= timedelta(days=LOOKBACK_DAYS) and a > d for d in avail_dates)]
        if not valid:
            raise SystemExit(
                "None of --dates have any CHIRPS data within the 30-day lookback."
            )
        return valid

    if args.start and args.end:
        s = datetime.strptime(args.start, "%Y-%m-%d").date()
        e = datetime.strptime(args.end, "%Y-%m-%d").date()
        anchors = [s + timedelta(days=i) for i in range((e - s).days + 1)]
    else:
        # Default: every available file date is an anchor (predict at each
        # observed day boundary).
        anchors = avail_dates

    # Restrict to anchors that actually have at least one lookback day present.
    return [a for a in anchors if any(a - d <= timedelta(days=LOOKBACK_DAYS) and a > d for d in avail_dates)]


def sample_available_day(
    d: date,
    available: Dict[date, Path],
    decompressed: Dict[date, Path],
) -> Optional[np.ndarray]:
    """Sample one raw day for all roads; None if that file can't be read."""
    tif = decompressed.get(d) or (available[d] if d in available else None)
    if tif is None:
        return None
    try:
        return sample_roads(tif, road_coords_global)
    except Exception as exc:  # keep the pipeline alive on a bad tile
        print(f"WARNING: failed to read CHIRPS day {d}: {exc}")
        return None


# module-global roads cache (populated in main) so per-day sampling keeps them
road_coords_global: Optional[np.ndarray] = None
osm_ids_global: Optional[np.ndarray] = None


def process_anchor(
    anchor: date,
    available: Dict[date, Path],
    decompressed: Dict[date, Path],
    allow_partial: bool,
    road_chunk: int,
) -> Optional[pd.DataFrame]:
    """Compute rainfall features for one anchor date over all roads.

    Returns a long-format DataFrame [(osm_id, feature_date, *windows...)] or
    None if no lookback days were available.
    """
    # Days in the 30-day lookback that have raw files, anchored by offset.
    needed_dates = [anchor - timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)]
    present = [(d, anchor - d) for d in needed_dates if d in available]  # (date, offset)
    if not present:
        return None

    n_roads = road_coords_global.shape[0]
    # Offsets must be NEGATIVE: number of days this observation precedes the
    # anchor. `anchor - d` is positive, so negate it (core selects offsets < 0).
    offsets = np.array([-(anchor - d).days for d, _ in present], dtype=np.int64)
    daily = np.full((n_roads, len(present)), np.nan, dtype=np.float64)

    for j, (d, _) in enumerate(present):
        col = sample_available_day(d, available, decompressed)
        if col is not None:
            daily[:, j] = col

    rows_list: List[pd.DataFrame] = []
    for r0 in range(0, n_roads, road_chunk):
        r1 = min(r0 + road_chunk, n_roads)
        sub = daily[r0:r1]
        agg = aggregate_windows(sub, offsets, WINDOWS, allow_partial=allow_partial)
        days_avail = available_day_counts(sub)
        df = pd.DataFrame(
            {
                "osm_id": osm_ids_global[r0:r1],
                "feature_date": anchor.isoformat(),
            }
        )
        for w in WINDOWS:
            df[f"rainfall_{w}day"] = agg[w]
        df["rainfall_days_available"] = days_avail
        rows_list.append(df)
    return pd.concat(rows_list, ignore_index=True)


def main():
    global road_coords_global, osm_ids_global

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dates", nargs="*", metavar="YYYY-MM-DD",
                        help="Explicit anchor (prediction) dates to process.")
    parser.add_argument("--start", metavar="YYYY-MM-DD",
                        help="Inclusive start of anchor-date range.")
    parser.add_argument("--end", metavar="YYYY-MM-DD",
                        help="Inclusive end of anchor-date range.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Output parquet path.")
    parser.add_argument("--road-chunk", type=int, default=20000,
                        help="Roads processed per memory chunk.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Use partial-window sums (sum available days).")
    args = parser.parse_args()

    osm_ids, coords, crs = build_roads()
    road_coords_global = coords
    osm_ids_global = osm_ids
    print(f"Loaded {len(osm_ids)} road segments (CRS {crs})")

    available = discover_chirps_files()
    print(f"Found {len(available)} CHIRPS files "
          f"({available[0][0]} -> {available[-1][0]})")
    avail_map = {d: p for d, p in available}

    anchors = resolve_dates(available, args)
    if not anchors:
        raise SystemExit("No anchor dates have data within the 30-day lookback.")
    print(f"Processing {len(anchors)} anchor dates: "
          f"{anchors[0]} -> {anchors[-1]}")
    if len(anchors) > 500:
        print("NOTE: large anchor set; this is expected to take a while. "
              "Use --dates/--start/--end to bound it.")

    with tempfile.TemporaryDirectory(prefix="chirps_") as td:
        decompressed: Dict[date, Path] = {}
        chunks: List[pd.DataFrame] = []
        for i, anchor in enumerate(anchors):
            # ensure needed raw files are decompressed once
            needed = [anchor - timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)]
            for d in needed:
                if d in avail_map and d not in decompressed:
                    decompressed[d] = decompress(avail_map[d], Path(td))
            sub = process_anchor(anchor, avail_map, decompressed, args.allow_partial,
                                 args.road_chunk)
            if sub is not None:
                chunks.append(sub)
            if (i + 1) % 20 == 0 or (i + 1) == len(anchors):
                print(f"  {i + 1}/{len(anchors)} anchors done", flush=True)

    if not chunks:
        raise SystemExit("No features produced.")
    out = pd.concat(chunks, ignore_index=True)
    print(f"Writing {len(out)} rows to {args.output}")
    out.to_parquet(args.output, index=False)
    print("Rows per feature_date complete.")
    print("OK")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
