#!/usr/bin/env python3
"""
NER-Nav: Targeted CHIRPS download for the real temporal prediction design.

Downloads ONLY the calendar days needed for the 30-day rainfall lookback of the
real temporal samples (see ml/features/temporal_design.py). This is the LEAN,
real-data path: ~356 global daily rasters (2017-2025) instead of the full
2016-2024 archive (~3,300). Files are placed in data/raw/weather/chirps_YYYY/
so scripts/process_chirps_features.py can discover them unchanged.

Skips any file already present (resumable). Existing 2015 files are untouched.
"""

from __future__ import annotations

import sys
import urllib.request
import urllib.error
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.temporal_design import needed_chirps_dates  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"


def download_date(d: date, overwrite: bool = False) -> str:
    """Return 'downloaded' | 'skipped' | 'failed' for one calendar day."""
    year_dir = RAW_WEATHER_DIR / f"chirps_{d.year}"
    year_dir.mkdir(exist_ok=True)
    filename = f"chirps-v2.0.{d.strftime('%Y.%m.%d')}.tif.gz"
    output_path = year_dir / filename
    if output_path.exists() and not overwrite:
        return "skipped"
    url = f"{CHIRPS_BASE_URL}{d.year}/{filename}"
    try:
        urllib.request.urlretrieve(url, output_path)
        return "downloaded"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing_404"
        return f"http {e.code}"
    except Exception as e:  # network etc.
        output_path.unlink(missing_ok=True)
        return f"error {e}"


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-download files that already exist.")
    args = parser.parse_args()

    dates = needed_chirps_dates()
    print(f"Target dates: {len(dates)} ({dates[0]} -> {dates[-1]})")

    stats = {"downloaded": 0, "skipped": 0}
    problems = []
    for i, d in enumerate(dates, 1):
        status = download_date(d, overwrite=args.overwrite)
        stats.setdefault(status, 0)
        stats[status] += 1
        if status == "downloaded":
            print(f"[{i}/{len(dates)}] {d} OK")
        elif status != "skipped":
            problems.append(f"{d}: {status}")
            print(f"[{i}/{len(dates)}] {d} {status}")
        if i % 50 == 0:
            print(f"  ... {i}/{len(dates)} done (dl={stats['downloaded']})", flush=True)

    print()
    print("SUMMARY:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  {p}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
