#!/usr/bin/env python3
"""
NER-Nav: IMD Rainfall Data Download Helper

Purpose:
    Guide and automate IMD gridded rainfall data acquisition for NER region.

Status: P0 Priority (blocks real-data training)
Author: Senior ML Engineer
Date: 2026-08-26
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_WEATHER_DIR = PROJECT_ROOT / "data" / "raw" / "weather"
RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)


def print_acquisition_guide():
    """Print step-by-step IMD data acquisition instructions."""

    print("=" * 70)
    print("IMD Gridded Rainfall Data Acquisition Guide")
    print("=" * 70)
    print()

    print("TARGET: IMD High Resolution Daily Gridded Rainfall (0.25° x 0.25°)")
    print("PERIOD: 2015-2024 (9 years for train/val/test)")
    print("REGION: North Eastern Region")
    print("FORMAT: NetCDF (.nc) or Binary (.grd + .ctl)")
    print()

    print("=" * 70)
    print("ACQUISITION STEPS")
    print("=" * 70)
    print()

    print("STEP 1: Register at IMD Data Portal")
    print("-" * 70)
    print("Official IMD Data Sources:")
    print()
    print("  Option A: IMD Pune Climate Data Portal")
    print("  URL: https://imdpune.gov.in/")
    print("  - Navigate to 'Data' -> 'Gridded Data'")
    print("  - Register for academic/research access")
    print("  - Request daily gridded rainfall product")
    print()
    print("  Option B: India Meteorological Department Open Data Portal")
    print("  URL: https://data.gov.in/")
    print("  - Search: 'IMD rainfall gridded'")
    print("  - Download available datasets")
    print()
    print("  Option C: Contact IMD directly")
    print("  Email: director@imdpune.gov.in")
    print("  Subject: Request for gridded rainfall data (academic research)")
    print()

    print("STEP 2: Specify Data Requirements")
    print("-" * 70)
    print("When requesting data, specify:")
    print()
    print("  Spatial Coverage:")
    print("    Latitude:  21.5°N to 29.5°N")
    print("    Longitude: 89.5°E to 97.5°E")
    print("    (North Eastern Region bounding box)")
    print()
    print("  Temporal Coverage:")
    print("    Start: 2015-01-01")
    print("    End:   2024-12-31")
    print()
    print("  Desired Format: NetCDF (.nc) preferred")
    print("  Alternative:    Binary GrADS format (.grd + .ctl)")
    print()

    print("STEP 3: Download and Store Raw Files")
    print("-" * 70)
    print(f"Store downloaded files in: {RAW_WEATHER_DIR}")
    print()
    print("Expected file structure:")
    print("  data/raw/weather/")
    print("    - imd_gridded_rainfall_2015.nc")
    print("    - imd_gridded_rainfall_2016.nc")
    print("    - ...")
    print("    - imd_gridded_rainfall_2024.nc")
    print()
    print("OR (if single merged file):")
    print("  data/raw/weather/")
    print("    - imd_gridded_rainfall_2015_2024.nc")
    print()

    print("STEP 4: Verify Downloaded Data")
    print("-" * 70)
    print("After download, run:")
    print()
    print("  python scripts/verify_imd_rainfall.py")
    print()
    print("This will check:")
    print("  - File format validity")
    print("  - Coordinate coverage (lat/lon ranges)")
    print("  - Temporal coverage (date ranges)")
    print("  - Variable names and units")
    print("  - Missing data percentage")
    print()

    print("=" * 70)
    print("ALTERNATIVE: Public Rainfall Datasets (If IMD Unavailable)")
    print("=" * 70)
    print()
    print("If IMD access is restricted, consider these alternatives:")
    print()
    print("  1. CHIRPS (Climate Hazards Group InfraRed Precipitation)")
    print("     URL: https://data.chc.ucsb.edu/products/CHIRPS-2.0/")
    print("     Resolution: 0.05° (~5km)")
    print("     Coverage: Global, 1981-present")
    print("     License: Public domain")
    print("     Decision Category: B (Scientific/Institutional)")
    print()
    print("  2. GPM IMERG (Global Precipitation Measurement)")
    print("     URL: https://gpm.nasa.gov/data/directory")
    print("     Resolution: 0.1° (~10km)")
    print("     Coverage: Global, 2000-present")
    print("     License: Open (NASA)")
    print("     Decision Category: B (Scientific/Institutional)")
    print()
    print("  3. ERA5 Precipitation (Copernicus/ECMWF)")
    print("     URL: https://cds.climate.copernicus.eu/")
    print("     Resolution: 0.25° (~25km)")
    print("     Coverage: Global, 1979-present")
    print("     License: Copernicus License")
    print("     Decision Category: B (Scientific/Institutional)")
    print()
    print("NOTE: IMD is preferred (Category A - Government Source).")
    print("      Use alternatives only if IMD is genuinely inaccessible.")
    print()

    print("=" * 70)
    print("NEXT ACTIONS")
    print("=" * 70)
    print()
    print("1. Register at IMD portal OR choose alternative source")
    print("2. Download rainfall data for NER region (2015-2024)")
    print(f"3. Store in: {RAW_WEATHER_DIR}")
    print("4. Run: python scripts/verify_imd_rainfall.py")
    print("5. Proceed to feature engineering once verified")
    print()
    print("ESTIMATED TIME: 1-3 days (depending on data access approval)")
    print()
    print("=" * 70)
    print("DATA GAP STATUS")
    print("=" * 70)
    print()

    # Check if any rainfall data exists
    rainfall_files = list(RAW_WEATHER_DIR.glob("*.nc")) + list(RAW_WEATHER_DIR.glob("*.grd"))

    if len(rainfall_files) > 0:
        print(f"[OK] FOUND {len(rainfall_files)} rainfall file(s) in {RAW_WEATHER_DIR}")
        print()
        for f in rainfall_files:
            print(f"   - {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
        print()
        print("STATUS: READY FOR VERIFICATION")
        print("NEXT: Run python scripts/verify_imd_rainfall.py")
    else:
        print(f"[X] NO rainfall data found in {RAW_WEATHER_DIR}")
        print()
        print("STATUS: DATA ACQUISITION REQUIRED")
        print("BLOCKER: P0 - Cannot train real-data model without rainfall features")
        print()
        print("NEXT: Follow steps above to acquire IMD data")

    print()
    print("=" * 70)


def main():
    """Main entry point."""
    print_acquisition_guide()
    return 0


if __name__ == "__main__":
    sys.exit(main())
