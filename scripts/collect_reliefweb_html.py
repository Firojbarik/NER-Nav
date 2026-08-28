#!/usr/bin/env python3
"""
NER-Nav: ReliefWeb event-collection helper (no-auth, public HTML listing).

Companion to collect_reliefweb_events.py (which needs an approved API
appname). This script scrapes ReliefWeb's PUBLIC server-rendered India
"updates" listing pages - no API key, no registration - and filters the
reports to North Eastern Region road-affecting hazard candidates.

WHY / WHEN
----------
Use this to start collecting real, source-verifiable candidate events
immediately while a ReliefWeb API appname is approved. Each candidate
carries the report's `origin` URL (verifiable provenance) exactly like the
API flow.

USAGE
-----
    python scripts/collect_reliefweb_html.py --pages 5

Arguments:
    --pages     number of listing pages to scan (default 5; ~20 reports/page,
                newest first, going back in time ~1 page per 10 days of posts)
    --out       JSONL path for raw report cards (default
                data/raw/hazards/reliefweb/html_reports.jsonl)

OUTPUT
------
* Raw report cards -> --out (JSONL).
* Filtered NER road-hazard candidates -> <--out>.candidates.csv (same schema
  as ner_hazard_events_database.csv and collect_reliefweb_events.py).

Each candidate still needs a human to (a) confirm it is a real on-the-ground
road disruption, (b) resolve the affected road's OSM id via Overpass, then
(c) run scripts/add_confirmed_event.py --apply and rebuild.

Data is copyrighted by the original sources; cite the origin URL when reusing.
"""

from __future__ import annotations

import argparse
import csv
import html as _html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data" / "raw" / "hazards" / "reliefweb" / "html_reports.jsonl"

LISTING = "https://reliefweb.int/updates"
ADVANCED_SEARCH = "(PC119)"  # primary_country = India

NER_STATES = [
    "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Sikkim", "Tripura",
]
NER_STATE_KEYWORDS = [
    "Arunachal", "Assam", "Manipur", "Meghalaya",
    "Mizoram", "Nagaland", "Sikkim", "Tripura",
]
# Road / disruption keywords that indicate an actual road impact.
ROAD_KEYWORDS = [
    "road", "highway", "nh-", "nh ", "nh27", "nh37", "nh29", "nh13", "nh6",
    "nh2", "nh10", "blocked", "blockade", "washed away", "washout", "cut off",
    "landslide", "flood", "bridge", "roadway", "arterial",
]
# Noise: these source/title patterns are humanitarian programme briefs, not
# road-disruption events. Skip them.
NOISE_TITLE = re.compile(
    r"(country brief|factsheet|fact sheet|situation report.*(funding snapshot|response and funding)"
    r"|appeal|early recovery|rehabilitation|strategic roadmap|gender-based|protection monitoring"
    r"|food security|country strategy|annual report)", re.IGNORECASE
)
NOISE_SOURCE = re.compile(r"(WFP|UNHCR|UN Women|OCHA|IFRC|Amnesty International)", re.IGNORECASE)

REPORT_URL = re.compile(r"rw-river-article__title\"[^>]*><a href=\"([^\"]+)\">(.*?)</a>", re.DOTALL)
ARTICLE_IDS = re.compile(r"<article class=\"rw-river-article[^\"]*\" data-id=\"(\d+)\"")
SUMMARY = re.compile(r"rw-river-article__content\" lang=\"en\">.*?<p>(.*?)</p>", re.DOTALL)
SOURCE_TAG = re.compile(r"rw-entity-meta__tag-value--source.*?</ul>", re.DOTALL)
SOURCE_LINK = re.compile(r"/organization/[^\"]+\">([^<]+)</a>")
PUBLISHED = re.compile(
    r"rw-entity-meta__tag-value--published rw-entity-meta__tag-value--date "
    r"rw-entity-meta__tag-value--last\">\s*<time datetime=\"([^\"]+)\""
)
POSTED = re.compile(r"rw-entity-meta__tag-value--posted rw-entity-meta__tag-value--date\">\s*<time datetime=\"([^\"]+)\"")


def fetch_listing(page: int) -> str:
    q = urllib.parse.urlencode({
        "advanced-search": ADVANCED_SEARCH,
        "page": page,
    })
    url = f"{LISTING}?{q}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NER-Nav event scraper)"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8", "replace")


def parse_cards(html: str) -> list[dict]:
    ids = ARTICLE_IDS.findall(html)
    urls = REPORT_URL.findall(html)
    summaries = SUMMARY.findall(html)
    sources = []
    for m in SOURCE_TAG.findall(html):
        names = SOURCE_LINK.findall(m)
        sources.append([n.strip() for n in names])
    published = PUBLISHED.findall(html)
    posted = POSTED.findall(html)

    cards = []
    for i in range(max(len(ids), len(urls))):
        rw_id = int(ids[i]) if i < len(ids) else None
        url, title = (urls[i] if i < len(urls) else ("", ""))
        cards.append({
            "id": rw_id,
            "title": _html.unescape(re.sub(r"\s+", " ", title)).strip(),
            "url": url,
            "summary": _html.unescape(re.sub(r"\s+", " ", summaries[i])).strip() if i < len(summaries) else "",
            "source": sources[i] if i < len(sources) else [],
            "date_published": published[i] if i < len(published) else "",
            "date_posted": posted[i] if i < len(posted) else "",
        })
    return cards


def _is_ner_road_report(card: dict) -> bool:
    blob = " ".join([card["title"], card["summary"], " ".join(card["source"])]).lower()
    if NOISE_TITLE.search(blob):
        return False
    if NOISE_SOURCE.search(card["url"]):
        # Very generic briefs; still allow if title/summary clearly names an NER road event.
        if not any(k in blob for k in ("landslide", "flood", "washed", "blocked", "nh-", " nh")):
            return False
    has_state = any(k.lower() in blob for k in NER_STATE_KEYWORDS)
    has_road = any(k in blob for k in ROAD_KEYWORDS)
    return has_state and has_road


def candidate_row(card: dict) -> list:
    blob = (card["title"] + " " + card["summary"] + " " + " ".join(card["source"]))
    state = next((s for s in NER_STATES if s.split()[0].lower() in blob.lower()), "")
    date_iso = (card["date_published"] or card["date_posted"] or "")[:10]
    rw_id = card["id"] if card["id"] else hash(card["url"]) & 0xFFFFFFF
    return [
        f"reliefweb_{rw_id}",
        date_iso,
        "landslide/flood",
        state,
        "",
        "",
        "",
        card["url"],
        "|".join(card["source"]) or "reliefweb",
        "candidate_not_yet_confirmed",
        card["title"],
    ]


CANDIDATE_HEADER = [
    "event_id", "event_date", "event_type", "state", "location_name",
    "latitude", "longitude", "affected_road", "source", "severity", "notes",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pages", type=int, default=5, help="listing pages to scan (newest first)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--dry-scan", action="store_true", help="print NER road candidates without writing files")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_cards: list[dict] = []
    for page in range(args.pages):
        try:
            html = fetch_listing(page)
        except Exception as e:
            print(f"WARN: page {page} failed ({e})", file=sys.stderr)
            break
        cards = parse_cards(html)
        print(f"page {page}: {len(cards)} cards")
        all_cards.extend(cards)
        time.sleep(args.delay)

    cands = [c for c in all_cards if _is_ner_road_report(c)]

    if args.dry_scan:
        print(f"\n{len(cands)} NER road-hazard candidates found across {len(all_cards)} reports:")
        for c in cands:
            print(f"  {c['date_published'][:10]} {c['title']}")
            print(f"      {c['url']}")
        return 0

    with out_path.open("w", encoding="utf-8") as fh:
        for c in all_cards:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_cards)} raw report cards -> {out_path}")

    rows = [candidate_row(c) for c in cands]
    with out_path.with_suffix(".candidates.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CANDIDATE_HEADER)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} NER road-hazard candidates -> {out_path.with_suffix('.candidates.csv')}")

    print("\nNEXT STEPS")
    print("  1. Review .candidates.csv; keep only on-the-ground road disruptions.")
    print("  2. Resolve the affected road's OSM id via Overpass (EVENT_COLLECTION_WORKFLOW STEP 2).")
    print("  3. Run scripts/add_confirmed_event.py --apply, then rebuild_pipeline.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
