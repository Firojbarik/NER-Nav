#!/usr/bin/env python3
"""
Add a new source-confirmed real hazard event to temporal_design.py.

Usage:
    python scripts/add_confirmed_event.py \
        --event_id assam_boko_2024_07_20 \
        --osm_id 44884963 \
        --ref NH27 \
        --event_date 2024-07-20 \
        --apply

The --apply flag writes the change to ml/features/temporal_design.py.
Without it, the script only validates and prints what it would do.

After adding, run:
    python scripts/rebuild_pipeline.py
to download new CHIRPS data, rebuild the dataset, and retrain.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

TEMPORAL_DESIGN = Path("ml/features/temporal_design.py")
CONFIRMED_EVENTS_PATH = TEMPORAL_DESIGN


def load_events() -> list[tuple[str, int, str, str]]:
    """Parse CONFIRMED_EVENTS from temporal_design.py (returns list of tuples)."""
    content = CONFIRMED_EVENTS_PATH.read_text(encoding="utf-8")
    # Match the actual variable assignment line, not mentions in docstrings
    m = re.search(r"CONFIRMED_EVENTS:\s*List.*?=\s*\[(.*?)\]", content, re.DOTALL)
    if not m:
        raise RuntimeError("Cannot parse CONFIRMED_EVENTS from temporal_design.py")
    block = m.group(1)
    events = []
    for line in block.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("("):
            continue
        parts = [p.strip().strip('"').strip("'") for p in line.strip("()").split(",")]
        if len(parts) == 4:
            eid, osm_str, ref, ed = parts
            events.append((eid, int(osm_str), ref, ed))
    return events


def write_events(events: list[tuple[str, int, str, str]]) -> None:
    """Rewrite CONFIRMED_EVENTS in temporal_design.py, sorted by date."""
    content = CONFIRMED_EVENTS_PATH.read_text(encoding="utf-8")
    # sort by date
    events.sort(key=lambda e: e[3])
    lines = []
    for eid, osm, ref, ed in events:
        lines.append(f'    ("{eid}", {osm}, "{ref}", "{ed}"),')
    block = "CONFIRMED_EVENTS: List[Tuple[str, int, str, str]] = [\n" + "\n".join(lines) + "\n]"
    new_content = re.sub(
        r"CONFIRMED_EVENTS:\s*List.*?=\s*\[.*?\]",
        block,
        content,
        flags=re.DOTALL,
    )
    CONFIRMED_EVENTS_PATH.write_text(new_content, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--event_id", required=True, help="Unique event identifier, e.g. assam_boko_2024_07_20")
    ap.add_argument("--osm_id", type=int, required=True, help="OSM way/relation ID of the confirmed affected road")
    ap.add_argument("--ref", required=True, help="NH or SH reference, e.g. NH27, NH37, SH5")
    ap.add_argument("--event_date", required=True, help="ISO date of the event, e.g. 2024-07-20")
    ap.add_argument("--apply", action="store_true", help="Write the change to temporal_design.py")
    args = ap.parse_args()

    events = load_events()

    # --- VALIDATION ---
    errors = []

    # event_id uniqueness
    existing_ids = {e[0] for e in events}
    if args.event_id in existing_ids:
        errors.append(f"event_id '{args.event_id}' already exists")

    # osm_id uniqueness (same road may appear in multiple events, warn but allow)
    existing_osm = {e[1] for e in events}
    if args.osm_id in existing_osm:
        dup_eid = [e[0] for e in events if e[1] == args.osm_id]
        print(f"WARNING: osm_id {args.osm_id} already in events {dup_eid} (same road in multiple events - allowed)")

    # date format
    try:
        ed = date.fromisoformat(args.event_date)
    except ValueError:
        errors.append(f"event_date '{args.event_date}' is not ISO format (YYYY-MM-DD)")

    # date range
    if ed < date(2017, 1, 1):
        errors.append(f"event_date {ed} < 2017-01-01 (CHIRPS coverage starts 2015, but real events start 2017)")

    # ref format
    if not re.match(r"^(NH|SH|NH\s|SH\s)\d+[A-Z]?$", args.ref, re.IGNORECASE):
        errors.append(f"ref '{args.ref}' does not match expected NH/SH pattern")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # --- DRY RUN ---
    new_event = (args.event_id, args.osm_id, args.ref, args.event_date)
    updated = sorted(events + [new_event], key=lambda e: e[3])
    print(f"Event validates OK: {new_event}")
    print(f"Would add to CONFIRMED_EVENTS (now {len(updated)} events):")
    for eid, osm, ref, ed in updated:
        marker = " <-- NEW" if eid == args.event_id else ""
        print(f"  {ed}  {eid:45s}  osm={osm:12d}  {ref}{marker}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write.")
        return 0

    write_events(updated)
    print(f"\nWrote {len(updated)} events to {CONFIRMED_EVENTS_PATH}")
    print("Next: run  python scripts/rebuild_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
