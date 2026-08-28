#!/usr/bin/env python3
"""
NER-Nav: Multi-Source Hazard Event Search Script

Purpose:
    Search public disaster databases for road-affecting hazard events
    across all 8 North Eastern Region states.

Target:
    Acquire 50-100 events with:
    - Date
    - Location (lat/lon or place name)
    - Event type (landslide/flood/earthquake)
    - Affected roads (if available)

Author: Senior ML Engineer
Priority: P0 (blocks training)
Date: 2026-08-27
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# NER States
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

# Event types
EVENT_TYPES = [
    "landslide",
    "flood",
    "earthquake",
    "road damage",
    "road closure",
    "bridge collapse"
]

def print_search_guide():
    """Print search guide for finding NER hazard events."""

    print("=" * 70)
    print("NER-Wide Hazard Event Search Guide")
    print("=" * 70)
    print()

    print("TARGET: 50-100 road-affecting hazard events across NER")
    print()

    print("=" * 70)
    print("SEARCH SOURCES")
    print("=" * 70)
    print()

    print("1. EM-DAT International Disaster Database")
    print("-" * 70)
    print("URL: https://www.emdat.be/")
    print()
    print("Search strategy:")
    for state in NER_STATES:
        print(f"  - Country: India, Location: {state}")
    print()
    print("Filter: Disaster type = Flood, Landslide, Earthquake")
    print("Period: 2015-2024")
    print()

    print("2. GDACS Global Disaster Alert System")
    print("-" * 70)
    print("URL: https://www.gdacs.org/")
    print()
    print("Search strategy:")
    print("  - Browse by region: Asia -> India -> Northeast")
    print("  - Filter: Floods, Earthquakes")
    print("  - Export event list with coordinates")
    print()

    print("3. ReliefWeb Situation Reports")
    print("-" * 70)
    print("URL: https://reliefweb.int/")
    print()
    print("Search queries:")
    for state in NER_STATES:
        print(f"  - \"{state} flood\" OR \"{state} landslide\"")
    print()

    print("4. News Archives (Google News, local papers)")
    print("-" * 70)
    print("Search queries per state:")
    for state in NER_STATES:
        print(f"  - \"{state} road blocked landslide\" 2015..2024")
        print(f"  - \"{state} highway flood damage\" 2015..2024")
    print()

    print("5. Government Reports")
    print("-" * 70)
    print("Sources:")
    print("  - NRSC Landslide Atlas (institutional access required)")
    print("  - State PWD annual reports")
    print("  - BRO (Border Roads Organization) bulletins")
    print()

    print("=" * 70)
    print("DATA COLLECTION TEMPLATE")
    print("=" * 70)
    print()

    print("For each event, record:")
    print()
    print("event_id: ner_[state]_[type]_YYYY_MM_DD")
    print("event_date: YYYY-MM-DD")
    print("event_type: landslide | flood | earthquake")
    print("state: [NER state name]")
    print("location_name: [Village/District name]")
    print("latitude: [decimal degrees]")
    print("longitude: [decimal degrees]")
    print("affected_road: [Road name if known]")
    print("source: [URL or report citation]")
    print("severity: low | moderate | high | severe")
    print()

    print("Save to: data/raw/hazards/ner_hazard_events_YYYY.csv")
    print()

    print("=" * 70)
    print("EXAMPLE EVENT ENTRIES")
    print("=" * 70)
    print()

    print("Example 1 - Assam Flood:")
    print("  event_id: ner_assam_flood_2020_07_15")
    print("  event_date: 2020-07-15")
    print("  event_type: flood")
    print("  state: Assam")
    print("  location_name: Kaziranga, Golaghat District")
    print("  latitude: 26.5775")
    print("  longitude: 93.1711")
    print("  affected_road: NH-37")
    print("  source: https://reliefweb.int/report/...")
    print("  severity: high")
    print()

    print("Example 2 - Meghalaya Landslide:")
    print("  event_id: ner_meghalaya_landslide_2019_06_10")
    print("  event_date: 2019-06-10")
    print("  event_type: landslide")
    print("  state: Meghalaya")
    print("  location_name: Mawsynram")
    print("  latitude: 25.2969")
    print("  longitude: 91.5881")
    print("  affected_road: Shillong-Cherrapunji Road")
    print("  source: Local news report")
    print("  severity: moderate")
    print()

    print("=" * 70)
    print("STATE-SPECIFIC EVENT PRIORITIES")
    print("=" * 70)
    print()

    print("Arunachal Pradesh:")
    print("  Primary: Landslides (monsoon season)")
    print("  Secondary: Earthquakes")
    print("  High-risk roads: Itanagar-Tawang highway")
    print()

    print("Assam:")
    print("  Primary: Floods (Brahmaputra basin)")
    print("  Secondary: Riverbank erosion")
    print("  High-risk roads: NH-37, NH-52")
    print()

    print("Manipur:")
    print("  Primary: Landslides (hill areas)")
    print("  Secondary: Floods (Imphal valley)")
    print("  High-risk roads: Imphal-Dimapur highway")
    print()

    print("Meghalaya:")
    print("  Primary: Landslides (highest rainfall in India)")
    print("  Secondary: Flash floods")
    print("  High-risk roads: Shillong-Cherrapunji, Guwahati-Shillong")
    print()

    print("Mizoram:")
    print("  Primary: Landslides (steep terrain)")
    print("  Secondary: Cyclone impacts")
    print("  High-risk roads: Aizawl-Lunglei highway")
    print()

    print("Nagaland:")
    print("  Primary: Landslides")
    print("  Secondary: Earthquakes")
    print("  High-risk roads: Dimapur-Kohima highway")
    print()

    print("Sikkim:")
    print("  Primary: Landslides, earthquakes")
    print("  Secondary: Glacial lake outburst floods (GLOFs)")
    print("  High-risk roads: Gangtok-Nathula, Mangan-Chungthang")
    print()

    print("Tripura:")
    print("  Primary: Floods")
    print("  Secondary: Landslides (Jampui Hills)")
    print("  High-risk roads: Agartala-Dharmanagar highway")
    print()

    print("=" * 70)
    print("EXPECTED TIMELINE")
    print("=" * 70)
    print()
    print("Manual search effort: 2-5 days")
    print("Expected events found: 50-150 (depending on sources)")
    print("Minimum required: 50 events")
    print("Optimal: 100+ events")
    print()

    print("NEXT STEPS:")
    print("1. Search databases listed above")
    print("2. Create CSV with event data")
    print("3. Run: python scripts/resolve_real_hazard_road_identity.py")
    print("4. Build training dataset with confirmed events")
    print()
    print("=" * 70)


def main():
    print_search_guide()
    return 0


if __name__ == "__main__":
    sys.exit(main())
