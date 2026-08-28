#!/usr/bin/env python3
"""
NER-Nav: CHIRPS Rainfall Data Download Script

Downloads CHIRPS daily precipitation data for NER region (2015-2024)
CHIRPS: Climate Hazards Group InfraRed Precipitation with Station data
Resolution: 0.05 degrees (~5km)
License: Public Domain
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)

# NER Region bounding box
NER_BBOX = {
    "lat_min": 21.5,
    "lat_max": 29.5,
    "lon_min": 89.5,
    "lon_max": 97.5
}

# CHIRPS data URL pattern
# CHIRPS provides global daily data in GeoTIFF format
CHIRPS_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/"


def download_chirps_yearly(year):
    """Download CHIRPS data for a specific year."""

    print(f"\n{'='*70}")
    print(f"Downloading CHIRPS data for {year}")
    print(f"{'='*70}")

    year_dir = RAW_WEATHER_DIR / f"chirps_{year}"
    year_dir.mkdir(exist_ok=True)

    # CHIRPS organizes data by year and provides daily files
    # URL format: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/tifs/p05/YYYY/chirps-v2.0.YYYY.MM.DD.tif.gz

    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)

    downloaded = 0
    skipped = 0
    failed = 0

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y.%m.%d")
        filename = f"chirps-v2.0.{date_str}.tif.gz"

        url = f"{CHIRPS_BASE_URL}{year}/{filename}"
        output_path = year_dir / filename

        # Skip if already downloaded
        if output_path.exists():
            skipped += 1
            current_date += timedelta(days=1)
            continue

        # Download with progress indicator
        try:
            print(f"  Downloading {date_str}...", end="", flush=True)
            urllib.request.urlretrieve(url, output_path)
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f" OK ({file_size_mb:.1f} MB)")
            downloaded += 1

        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f" SKIP (not available)")
            else:
                print(f" FAILED (HTTP {e.code})")
            failed += 1

        except Exception as e:
            print(f" FAILED ({str(e)})")
            failed += 1

        current_date += timedelta(days=1)

    print(f"\nYear {year} summary:")
    print(f"  Downloaded: {downloaded} files")
    print(f"  Skipped (already exists): {skipped} files")
    print(f"  Failed: {failed} files")

    return downloaded, skipped, failed


def main():
    print("="*70)
    print("CHIRPS Rainfall Data Download for NER-Nav")
    print("="*70)
    print()
    print("Target region:")
    print(f"  Latitude:  {NER_BBOX['lat_min']}°N to {NER_BBOX['lat_max']}°N")
    print(f"  Longitude: {NER_BBOX['lon_min']}°E to {NER_BBOX['lon_max']}°E")
    print()
    print("Data source: CHIRPS v2.0 (Climate Hazards Group)")
    print("Resolution: 0.05° (~5km)")
    print("License: Public Domain")
    print(f"Output directory: {RAW_WEATHER_DIR}")
    print()

    # Download data for 2015-2024
    years = range(2015, 2025)

    total_downloaded = 0
    total_skipped = 0
    total_failed = 0

    for year in years:
        d, s, f = download_chirps_yearly(year)
        total_downloaded += d
        total_skipped += s
        total_failed += f

    print()
    print("="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Total downloaded: {total_downloaded} files")
    print(f"Total skipped: {total_skipped} files")
    print(f"Total failed: {total_failed} files")
    print()

    if total_failed > 0:
        print("WARNING: Some downloads failed. This may be due to:")
        print("  - Network issues")
        print("  - Data not yet available for recent dates")
        print("  - CHIRPS server maintenance")
        print()
        print("You can re-run this script to retry failed downloads.")
        print()

    print("NEXT STEPS:")
    print("1. Extract .gz files: gunzip data/raw/weather/chirps_*/*.gz")
    print("2. Crop to NER region and create daily rainfall features")
    print("3. Run: python scripts/process_chirps_features.py")
    print()
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}")
        sys.exit(1)
