#!/usr/bin/env python3
"""
NER-Nav: ReliefWeb API event-collection helper.

Pulls curated disaster reports from the ReliefWeb API (v2) for India,
filters to North Eastern Region road-affecting candidates, and writes
them so they can be confirmed and added to CONFIRMED_EVENTS.

WHY RELIEFWEB
-------------
ReliefWeb (OCHA) curates reports from ~4000 humanitarian/professional
sources and tags each with country, disaster type, and date. Each report
carries an `origin` (the original source URL), which is exactly the kind
of verifiable provenance this project's event registry requires.

API GATE (read before running)
------------------------------
From 2025-11-01 the ReliefWeb API requires a PRE-APPROVED `appname`.
A plain request returns HTTP 403. To get one:
  1. Fill the short form:
     https://apidoc.reliefweb.int/parameters#appname
     (org name + purpose + random chars; ReliefWeb emails approval).
  2. Supply it to this script via --appname or the RELIEFWEB_APPNAME
     environment variable. Do NOT commit a real appname to git.

Usage:
    python scripts/collect_reliefweb_events.py \
        --appname <your-approved-appname> \
        --out data/raw/hazards/reliefweb/reports.jsonl

Without a valid appname the script fails fast with this API's 403 message.

OUTPUT
------
* Raw API pages -> --out (JSON Lines, one report object per line).
* A filtered candidate registry -> <--out>.candidates.csv with one row per
  report that mentions a NER state AND a road-blockage keyword. Each row
  still needs a human to (a) confirm it is a real, on-the-ground road
  disruption, (b) resolve the affected road's OSM id via Overpass, then
  (c) run scripts/add_confirmed_event.py --apply and rebuild.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "hazards" / "reliefweb" / "reports.jsonl"

API_BASE = "https://api.reliefweb.int/v2"
FORM_URL = "https://apidoc.reliefweb.int/parameters#appname"

NER_STATES = [
    "Arunachal Pradesh",
    "Assam",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Sikkim",
    "Tripura",
]

# GLIDE hazard codes used by ReliefWeb's disaster_type.code for reports.
DISASTER_CODES = ["LS", "FF", "FL"]  # Land Slide, Flash Flood, Flood

# Keywords in title/body that suggest a road was actually disrupted.
ROAD_KEYWORDS = [
    "road", "highway", "NH", "nh-", "blocked", "blockade", "washed away",
    "washout", "cut off", "landslide", "landslides", "flood", "floods",
    "bridge", "roadway", "arterial",
]

NER_STATE_KEYWORDS = [
    "Arunachal", "Assam", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Sikkim", "Tripura",
]


def build_payload(appname: str, offset: int, limit: int, date_start: str, date_end: str) -> dict:
    """Return the request body (owned by this app) for one page of India reports."""
    return {
        "query": {
            "value": " OR ".join(NER_STATE_KEYWORDS),
            "fields": ["body", "title"],
        },
        "filter": {
            "operator": "AND",
            "conditions": [
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "primary_country", "value": "India"},
                        {"field": "country", "value": "India"},
                    ],
                },
                {
                    "operator": "OR",
                    "conditions": [
                        {"field": "disaster_type.code", "value": c} for c in DISASTER_CODES
                    ],
                },
                {"field": "date.created", "value": {
                    "from": f"{date_start}T00:00:00+00:00",
                    "to": f"{date_end}T23:59:59+00:00",
                }},
            ],
        },
        "fields": {
            "include": [
                "title",
                "headline.title",
                "headline.summary",
                "date.created",
                "date.original",
                "country.iso3",
                "country.name",
                "primary_country.name",
                "disaster.name",
                "disaster_type.name",
                "source.name",
                "source.shortname",
                "origin",
                "url",
                "body",
            ]
        },
        "limit": limit,
        "offset": offset,
        "sort": ["date.created:desc"],
    }


def fetch_page(appname: str, payload: dict) -> dict:
    """POST one page of a query to the reports endpoint.

    The ReliefWeb API's complex query/filter/fields parameters are sent as a
    JSON POST body (the documented approach for non-trivial queries); the
    appname travels in the URL.
    """
    url = f"{API_BASE}/reports?appname={urllib.parse.quote(appname)}"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _ner_relevant(report: dict) -> bool:
    blob = " ".join(
        str(report.get(k, "")) for k in ("title", "body")
    ).lower()
    has_state = any(kw.lower() in blob for kw in NER_STATE_KEYWORDS)
    has_road = any(kw.lower() in blob for kw in ROAD_KEYWORDS)
    return has_state and has_road


def _report_row(report: dict, rw_id: int) -> list:
    src = report.get("source") or {}
    dis = report.get("disaster") or {}
    ds_types = (report.get("disaster_type") or [{}])
    dtypes = ",".join({d.get("name", "") for d in ds_types if isinstance(d, dict)})
    origin = report.get("origin") or report.get("url") or ""
    state = None
    for kw in NER_STATE_KEYWORDS:
        if kw.lower() in str(report.get("title", "")).lower() or kw.lower() in str(
            report.get("body", "")
        ).lower():
            state = kw if kw != "Sikkim" else "Sikkim"
            break
    date_created = (report.get("date") or {}).get("created") or report.get("date.created")
    return [
        f"reliefweb_{rw_id}",
        (report.get("date") or {}).get("original") or date_created,
        ",".join(dtypes) or "unknown",
        state or "",
        "",
        "",
        "",
        origin,
        "reliefweb",
        "candidate_not_yet_confirmed",
        report.get("title", "").replace("\n", " ")[:200],
    ]


CANDIDATE_HEADER = [
    "event_id", "event_date", "event_type", "state", "location_name",
    "latitude", "longitude", "affected_road", "source", "severity", "notes",
]


def write_candidates(rows: list[list], out_csv: Path) -> None:
    import csv

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CANDIDATE_HEADER)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} candidate rows -> {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--appname", default=os.environ.get("RELIEFWEB_APPNAME"),
                    help=f"Pre-approved ReliefWeb appname (or set RELIEFWEB_APPNAME). See {FORM_URL}")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="JSONL output path for raw reports")
    ap.add_argument("--date-start", default="2017-01-01")
    ap.add_argument("--date-end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    ap.add_argument("--pages", type=int, default=10, help="max pages (50 reports each)")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    if not args.appname:
        print("ERROR: an approved ReliefWeb appname is required.", file=sys.stderr)
        print(f"       Request one at {FORM_URL} then pass --appname or set RELIEFWEB_APPNAME.", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    reports = []
    for page in range(args.pages):
        payload = build_payload(args.appname, page * args.limit, args.limit, args.date_start, args.date_end)
        try:
            data = fetch_page(args.appname, payload)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print("ERROR: ReliefWeb returned HTTP 403 — the appname is not approved.", file=sys.stderr)
                print(f"       Request one at {FORM_URL} and re-run with --appname.", file=sys.stderr)
                return 2
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            print(f"ERROR: HTTP {e.code} on the reports endpoint", file=sys.stderr)
            if detail:
                print(detail, file=sys.stderr)
            return 2
        items = (data.get("data") or {}).get("items") or []
        reports.extend(items)
        total = (data.get("totalCount") or 0)
        if page * args.limit + len(items) >= total or not items:
            break

    with out_path.open("w", encoding="utf-8") as fh:
        for rpt in reports:
            fh.write(json.dumps(rpt, ensure_ascii=False) + "\n")
    print(f"Fetched {len(reports)} India disaster reports -> {out_path}")

    candidates = [r for r in reports if _ner_relevant(r)]
    rows = [_report_row(r, (r.get("id") or i)) for i, r in enumerate(candidates)]
    write_candidates(rows, out_path.with_suffix(".candidates.csv"))

    print("\nNEXT STEPS")
    print("  1. Review the .candidates.csv rows; keep only on-the-ground road")
    print("     disruptions you can verify.")
    print("  2. For each, resolve the affected road's OSM id via Overpass")
    print("     (docs/EVENT_COLLECTION_WORKFLOW.md, STEP 2).")
    print("  3. Run scripts/add_confirmed_event.py --apply for each, then")
    print("     python scripts/rebuild_pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
