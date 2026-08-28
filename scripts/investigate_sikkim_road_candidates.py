#!/usr/bin/env python3
"""
NER-Nav: Sikkim Landslide Road Candidate Investigation

Purpose:
    Investigate the 35 road candidates for the Sikkim Mantam 2016 landslide
    and identify roads that match the report description for manual confirmation.

Action:
    1. Load candidate roads from processed hazard data
    2. Check for named matches ("Passingdang", "Mantam")
    3. Rank candidates by distance and road characteristics
    4. Generate investigation report for manual review

Author: Senior ML Engineer
Priority: P0 (blocks real-data training)
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input files
CANDIDATES_FILE = (
    PROJECT_ROOT / "data" / "processed" / "hazards"
    / "nrsc_sikkim_mantam_2016_road_candidates.parquet"
)

EVENT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "hazards"
    / "nrsc_sikkim_mantam_2016_event.json"
)

# Output files
CONFIRMED_FILE = (
    PROJECT_ROOT / "data" / "processed" / "hazards"
    / "nrsc_sikkim_mantam_2016_confirmed_road.parquet"
)

INVESTIGATION_REPORT = (
    PROJECT_ROOT / "data" / "processed" / "hazards"
    / "nrsc_sikkim_mantam_2016_investigation_report.json"
)


def load_event_details() -> dict:
    """Load event metadata."""
    with EVENT_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def check_name_matches(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Identify roads with names matching the report."""

    # Keywords from NRSC report
    keywords = ["passingdang", "mantam", "kanka"]

    # Check name field
    candidates_df["name_lower"] = (
        candidates_df["name"].fillna("").str.lower()
    )

    candidates_df["name_match"] = candidates_df["name_lower"].apply(
        lambda x: any(keyword in x for keyword in keywords)
    )

    # Check ref field
    if "ref" in candidates_df.columns:
        candidates_df["ref_lower"] = (
            candidates_df["ref"].fillna("").str.lower()
        )
        candidates_df["ref_match"] = candidates_df["ref_lower"].apply(
            lambda x: any(keyword in x for keyword in keywords)
        )
    else:
        candidates_df["ref_match"] = False

    # Overall match flag
    candidates_df["has_named_match"] = (
        candidates_df["name_match"] | candidates_df["ref_match"]
    )

    return candidates_df


def rank_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Rank candidates by likelihood of being the affected road."""

    # Priority scoring
    candidates_df["priority_score"] = 0.0

    # +100 points for named match
    candidates_df.loc[
        candidates_df["has_named_match"], "priority_score"
    ] += 100.0

    # +50 points for bridge flag (report mentions Kanka bridge)
    if "bridge_flag" in candidates_df.columns:
        candidates_df.loc[
            candidates_df["bridge_flag"] == 1, "priority_score"
        ] += 50.0

    # +30 points for major road types
    if "highway" in candidates_df.columns:
        major_types = ["trunk", "primary", "secondary", "tertiary"]
        candidates_df.loc[
            candidates_df["highway"].isin(major_types), "priority_score"
        ] += 30.0

    # -10 points per km distance (penalty for distance)
    if "distance_m" in candidates_df.columns:
        candidates_df["priority_score"] -= (
            candidates_df["distance_m"] / 1000.0 * 10.0
        )

    # Sort by priority score (descending)
    candidates_df = candidates_df.sort_values(
        "priority_score", ascending=False
    ).reset_index(drop=True)

    return candidates_df


def generate_investigation_report(
    candidates_df: pd.DataFrame,
    event_details: dict
) -> dict:
    """Generate human-readable investigation report."""

    named_matches = candidates_df[candidates_df["has_named_match"]]

    report = {
        "event_id": event_details["event_id"],
        "event_date": event_details["event_date"],
        "event_location": {
            "lat": event_details["latitude"],
            "lon": event_details["longitude"]
        },
        "reported_road_name": event_details.get("nearby_road", "Unknown"),
        "total_candidates": int(len(candidates_df)),
        "named_matches": int(len(named_matches)),
        "investigation_status": "PENDING_MANUAL_REVIEW",
        "top_candidates": [],
        "recommendation": ""
    }

    # Top 10 candidates for review
    top_10 = candidates_df.head(10)

    for _, row in top_10.iterrows():
        candidate_info = {
            "osm_id": int(row["osm_id"]) if "osm_id" in row else None,
            "name": str(row["name"]) if pd.notna(row.get("name")) else None,
            "highway": str(row["highway"]) if "highway" in row else None,
            "distance_m": float(row["distance_m"]) if "distance_m" in row else None,
            "priority_score": float(row["priority_score"]),
            "has_named_match": bool(row["has_named_match"]),
            "bridge_flag": int(row.get("bridge_flag", 0)) if "bridge_flag" in row else None,
        }
        report["top_candidates"].append(candidate_info)

    # Generate recommendation
    if len(named_matches) > 0:
        report["recommendation"] = (
            f"PRIORITY: {len(named_matches)} road(s) have named matches. "
            "These should be confirmed first through:\n"
            "1. Visual inspection on OpenStreetMap at the event coordinates\n"
            "2. Satellite imagery review (2016-08-13 ± 7 days)\n"
            "3. Cross-reference with published paper coordinates\n"
            "Mark as source_supported=true AND confirmed_affected=true if evidence confirms."
        )
    else:
        report["recommendation"] = (
            "NO named matches found. Top candidates ranked by proximity and road type.\n"
            "Required actions:\n"
            "1. Open OpenStreetMap at 27.5397°N, 88.5007°E\n"
            "2. Identify road network topology connecting Passingdang-Mantam\n"
            "3. Acquire satellite imagery for visual damage confirmation\n"
            "4. Contact NRSC researchers or Sikkim PWD for additional information\n"
            "DO NOT confirm based on proximity alone."
        )

    return report


def main() -> None:
    print("=" * 70)
    print("Sikkim Landslide Road Candidate Investigation")
    print("=" * 70)
    print()

    # Load data
    print("Loading event details...")
    event_details = load_event_details()
    print(f"Event: {event_details['event_name']}")
    print(f"Date: {event_details['event_date']}")
    print(f"Location: {event_details['latitude']:.4f}°N, {event_details['longitude']:.4f}°E")
    print()

    print("Loading road candidates...")
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"Candidates file not found: {CANDIDATES_FILE}")

    candidates_df = pd.read_parquet(CANDIDATES_FILE)
    print(f"Total candidates: {len(candidates_df)}")
    print()

    # Check for named matches
    print("Checking for named matches...")
    print("-" * 70)
    candidates_df = check_name_matches(candidates_df)

    named_matches = candidates_df[candidates_df["has_named_match"]]
    print(f"Named matches found: {len(named_matches)}")

    if len(named_matches) > 0:
        print()
        print("NAMED MATCHES:")
        for _, row in named_matches.iterrows():
            print(f"  OSM ID: {row['osm_id']}")
            print(f"    Name: {row.get('name', 'N/A')}")
            print(f"    Highway: {row.get('highway', 'N/A')}")
            print(f"    Distance: {row.get('distance_m', 0):.0f}m")
            print()

    # Rank all candidates
    print("Ranking candidates by priority...")
    candidates_df = rank_candidates(candidates_df)

    print()
    print("TOP 5 PRIORITY CANDIDATES:")
    print("-" * 70)
    for idx, row in candidates_df.head(5).iterrows():
        print(f"{idx + 1}. OSM ID: {row['osm_id']}")
        print(f"   Name: {row.get('name', 'N/A')}")
        print(f"   Highway: {row.get('highway', 'N/A')}")
        print(f"   Distance: {row.get('distance_m', 0):.0f}m")
        print(f"   Priority Score: {row['priority_score']:.1f}")
        print(f"   Named Match: {row['has_named_match']}")
        print()

    # Generate investigation report
    print("Generating investigation report...")
    report = generate_investigation_report(candidates_df, event_details)

    with INVESTIGATION_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved: {INVESTIGATION_REPORT}")
    print()

    # Save updated candidates with ranking
    candidates_df.to_parquet(CONFIRMED_FILE)
    print(f"Updated candidates saved: {CONFIRMED_FILE}")
    print()

    # Print recommendation
    print("=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print(report["recommendation"])
    print()

    print("=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review top candidates manually")
    print("2. Open OpenStreetMap at event coordinates")
    print("3. Acquire satellite imagery (Sentinel-2 or Google Earth)")
    print("4. Update confirmation fields:")
    print("   - source_supported = true/false")
    print("   - confirmed_affected = true/false")
    print("   - confirmation_evidence = 'reason'")
    print("5. Re-run training gate audit")
    print()
    print("STATUS: INVESTIGATION COMPLETE, AWAITING MANUAL CONFIRMATION")
    print("=" * 70)


if __name__ == "__main__":
    main()
