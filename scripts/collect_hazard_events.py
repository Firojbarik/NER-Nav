#!/usr/bin/env python3
"""
NER-Nav: Hazard Event Data Collection Workflow

This script provides a structured workflow for manually collecting
hazard event data from reliable sources across all 8 NER states.

Since automated scraping of disaster databases often fails or requires
API keys, this script guides manual data collection with validation.

Author: Senior ML Engineer
Date: 2026-08-27
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAZARDS_DIR = PROJECT_ROOT / "data" / "raw" / "hazards"
HAZARDS_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_FILE = HAZARDS_DIR / "ner_hazard_events_database.csv"

NER_STATES = [
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura"
]

def print_collection_workflow():
    """Print step-by-step data collection workflow."""

    print("=" * 70)
    print("NER-NAV: HAZARD EVENT DATA COLLECTION WORKFLOW")
    print("=" * 70)
    print()

    print("OBJECTIVE: Collect 50-100 verified hazard events across 8 NER states")
    print("CURRENT: 1 event (Sikkim)")
    print("TARGET: 50+ events minimum")
    print()

    print("=" * 70)
    print("PHASE 1: HIGH-PRIORITY SOURCES (Target: 30-40 events)")
    print("=" * 70)
    print()

    print("SOURCE 1: ReliefWeb (Easiest, No Registration)")
    print("-" * 70)
    print("URL: https://reliefweb.int/updates")
    print()
    print("Search Process:")
    print("  1. Go to ReliefWeb")
    print("  2. Use Advanced Search:")
    print("     - Country: India")
    print("     - Date: 2015-01-01 to 2024-12-31")
    print("     - Keywords: Try each state name + 'flood' or 'landslide'")
    print()
    print("Example searches:")
    print("  - 'Assam flood' (expect: 20-30 reports)")
    print("  - 'Meghalaya landslide' (expect: 10-15 reports)")
    print("  - 'Arunachal Pradesh landslide' (expect: 5-10 reports)")
    print()
    print("What to extract from each report:")
    print("  - Event date (look for specific dates in text)")
    print("  - Location (district/village names)")
    print("  - Event type (flood/landslide/earthquake)")
    print("  - Road impacts (if mentioned)")
    print()
    print("Expected yield: 15-25 events with good location data")
    print()

    print("SOURCE 2: Google News Archives")
    print("-" * 70)
    print("URL: https://news.google.com/")
    print()
    print("Search queries per state (use date range: 2015-2024):")
    print()
    for state in NER_STATES:
        print(f"  '{state} landslide highway blocked'")
        print(f"  '{state} flood road closed'")
    print()
    print("Tips:")
    print("  - Look for local news sources (The Assam Tribune, Shillong Times)")
    print("  - Government press releases often have exact locations")
    print("  - Look for phrases like 'NH-37 closed', 'Shillong road blocked'")
    print()
    print("Expected yield: 10-20 events")
    print()

    print("SOURCE 3: Wikipedia Disaster Lists")
    print("-" * 70)
    print("URLs to check:")
    print("  - https://en.wikipedia.org/wiki/Floods_in_India")
    print("  - https://en.wikipedia.org/wiki/Landslides_in_India")
    print("  - Search: 'List of floods in India' or 'Assam floods history'")
    print()
    print("Expected yield: 5-10 major events")
    print()

    print("=" * 70)
    print("PHASE 2: INSTITUTIONAL SOURCES (Requires More Effort)")
    print("=" * 70)
    print()

    print("SOURCE 4: India Meteorological Department (IMD)")
    print("-" * 70)
    print("URL: https://mausam.imd.gov.in/")
    print()
    print("Navigation:")
    print("  - Go to 'Services' -> 'Heavy Rainfall Warnings Archive'")
    print("  - Select NER states and download PDF bulletins")
    print("  - Extract: Date, location, rainfall amounts")
    print()
    print("Expected yield: 10-15 flood events")
    print()

    print("SOURCE 5: National Disaster Management Authority (NDMA)")
    print("-" * 70)
    print("URL: https://ndma.gov.in/")
    print()
    print("Check:")
    print("  - Situation Reports")
    print("  - State-wise disaster data")
    print()
    print("Expected yield: 5-10 major events")
    print()

    print("=" * 70)
    print("DATA ENTRY TEMPLATE")
    print("=" * 70)
    print()

    print("For EACH event you find, record in CSV:")
    print()
    print("Required fields:")
    print("  event_id: ner_[state_abbrev]_[type]_YYYY_MM_DD")
    print("  event_date: YYYY-MM-DD")
    print("  event_type: landslide | flood | earthquake")
    print("  state: [Full state name]")
    print("  location_name: [District/Village]")
    print("  latitude: [Decimal degrees - use Google Maps if needed]")
    print("  longitude: [Decimal degrees]")
    print("  affected_road: [Road name/number if mentioned]")
    print("  source: [URL or citation]")
    print("  severity: low | moderate | high | severe")
    print("  notes: [Brief description]")
    print()

    print("Example entry:")
    print("  event_id: ner_assam_flood_2020_07_15")
    print("  event_date: 2020-07-15")
    print("  event_type: flood")
    print("  state: Assam")
    print("  location_name: Kaziranga, Golaghat District")
    print("  latitude: 26.5775")
    print("  longitude: 93.1711")
    print("  affected_road: NH-37")
    print("  source: https://reliefweb.int/report/india/assam-floods-july-2020")
    print("  severity: high")
    print("  notes: NH-37 submerged, Kaziranga national park flooded")
    print()

    print("=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)
    print()
    print("Before adding event to database, verify:")
    print("  [OK] Date is specific (not just year)")
    print("  [OK] Location has coordinates or can be geocoded")
    print("  [OK] Source is credible (news, govt, NGO)")
    print("  [OK] Event actually affected roads/infrastructure")
    print("  [OK] Falls within 2015-2024 time period")
    print()

    print("=" * 70)
    print("ESTIMATED TIMELINE")
    print("=" * 70)
    print()
    print("Phase 1 (ReliefWeb + News): 6-8 hours")
    print("  - ReliefWeb search: 2-3 hours")
    print("  - News archive search: 3-4 hours")
    print("  - Data entry: 1-2 hours")
    print("  Expected: 25-35 events")
    print()
    print("Phase 2 (Govt sources): 4-6 hours")
    print("  - IMD bulletins: 2-3 hours")
    print("  - NDMA reports: 2-3 hours")
    print("  Expected: 15-25 events")
    print()
    print("TOTAL: 10-14 hours for 40-60 events")
    print()

    print("=" * 70)
    print("QUICK START RECOMMENDATION")
    print("=" * 70)
    print()
    print("If time-limited, focus on these 3 states first:")
    print()
    print("1. ASSAM (Floods)")
    print("   Search: ReliefWeb 'Assam flood 2015-2024'")
    print("   Expected: 15-20 events")
    print("   Time: 2-3 hours")
    print()
    print("2. MEGHALAYA (Landslides)")
    print("   Search: News archives 'Meghalaya landslide road'")
    print("   Expected: 10-12 events")
    print("   Time: 2-3 hours")
    print()
    print("3. ARUNACHAL PRADESH (Landslides)")
    print("   Search: ReliefWeb + News 'Arunachal landslide highway'")
    print("   Expected: 8-10 events")
    print("   Time: 2-3 hours")
    print()
    print("TOTAL: 33-42 events in 6-9 hours")
    print()
    print("This gives sufficient coverage of 3 major hazard contexts.")
    print("Other 5 states can be added later for improved generalization.")
    print()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print()
    print("1. Start with ReliefWeb search for Assam floods")
    print("2. Record events in: data/raw/hazards/ner_hazard_events_database.csv")
    print("3. Once you have 10-15 events, test the pipeline:")
    print("   python scripts/resolve_real_hazard_road_identity.py")
    print("4. Continue collecting until 50+ events")
    print("5. Build training dataset")
    print()
    print("=" * 70)
    print()
    print(f"Database file: {DATABASE_FILE}")
    print()
    if DATABASE_FILE.exists():
        df = pd.read_csv(DATABASE_FILE)
        print(f"Current events in database: {len(df)}")
        print()
        if len(df) > 0:
            print("State distribution:")
            print(df['state'].value_counts())
    else:
        print("Database not yet created. Will be created on first entry.")
    print()


def validate_event_entry(event_dict):
    """Validate a single event entry."""

    required_fields = [
        'event_id', 'event_date', 'event_type', 'state',
        'location_name', 'latitude', 'longitude', 'source'
    ]

    issues = []

    # Check required fields
    for field in required_fields:
        if field not in event_dict or not event_dict[field]:
            issues.append(f"Missing required field: {field}")

    # Validate date format
    if 'event_date' in event_dict:
        try:
            datetime.strptime(event_dict['event_date'], '%Y-%m-%d')
        except ValueError:
            issues.append("event_date must be in YYYY-MM-DD format")

    # Validate coordinates
    if 'latitude' in event_dict:
        try:
            lat = float(event_dict['latitude'])
            if not (21.0 <= lat <= 30.0):
                issues.append(f"Latitude {lat} outside NER range (21-30°N)")
        except ValueError:
            issues.append("latitude must be numeric")

    if 'longitude' in event_dict:
        try:
            lon = float(event_dict['longitude'])
            if not (88.0 <= lon <= 98.0):
                issues.append(f"Longitude {lon} outside NER range (88-98°E)")
        except ValueError:
            issues.append("longitude must be numeric")

    # Validate state
    if 'state' in event_dict and event_dict['state'] not in NER_STATES:
        issues.append(f"State '{event_dict['state']}' not in NER states list")

    # Validate event type
    valid_types = ['landslide', 'flood', 'earthquake', 'cyclone']
    if 'event_type' in event_dict and event_dict['event_type'] not in valid_types:
        issues.append(f"event_type must be one of: {valid_types}")

    return issues


def add_event(event_dict):
    """Add a new event to the database."""

    # Validate
    issues = validate_event_entry(event_dict)
    if issues:
        print("Validation errors:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    # Load existing database or create new
    if DATABASE_FILE.exists():
        df = pd.read_csv(DATABASE_FILE)
    else:
        df = pd.DataFrame()

    # Check for duplicates
    if 'event_id' in df.columns:
        if event_dict['event_id'] in df['event_id'].values:
            print(f"Event {event_dict['event_id']} already exists in database")
            return False

    # Append
    df = pd.concat([df, pd.DataFrame([event_dict])], ignore_index=True)

    # Save
    df.to_csv(DATABASE_FILE, index=False)
    print(f"[OK] Event {event_dict['event_id']} added to database")
    print(f"Total events: {len(df)}")

    return True


def main():
    print_collection_workflow()
    return 0


if __name__ == "__main__":
    sys.exit(main())
