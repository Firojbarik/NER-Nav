#!/usr/bin/env python3
"""Screen ReliefWeb India situation reports for specific NH-road incidents.

Fetches a report page, strips HTML, and prints snippets around mentions of
national-highway codes and road-disruption vocabulary so a human (or this
agent) can pick out specific on-the-ground incidents to confirm via OSM.

Usage:
    python scripts/screen_reliefweb_report.py <report-url> ...
"""

from __future__ import annotations

import html as H
import re
import sys
import urllib.request

NH_PAT = re.compile(r"(NH\s?[-–]?\s?\d+[A-Z]?|National\s+Highway\s*\d+[A-Z]?)", re.IGNORECASE)
ROAD_PAT = re.compile(
    r"(highway|road|washed (away|out)|cut off|blocked|breach|landslide|slide|flash flood|inundat|submerg)",
    re.IGNORECASE,
)

def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (NER-Nav screen)"})
    h = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    t = re.sub(r"<script.*?</script>", " ", h, flags=re.S)
    t = re.sub(r"<style.*?</style>", " ", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return H.unescape(re.sub(r"\s+", " ", t))


def snippet(t: str, m: re.Match, radius: int = 110) -> str:
    s = re.sub(r"\s+", " ", t)
    return s[max(0, m.start() - radius): m.end() + radius].strip()


def main() -> int:
    for url in sys.argv[1:]:
        print("=" * 80)
        print(url)
        print("=" * 80)
        try:
            t = fetch(url)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        seen_nh = set()
        for m in NH_PAT.finditer(t):
            code = m.group(0).strip()
            if code in seen_nh:
                continue
            seen_nh.add(code)
            print(f"\n[NH] {code}")
            print("   >", H.unescape(snippet(t, m)))
        hits = 0
        for m in ROAD_PAT.finditer(t):
            print(f"[ROAD] ...{snippet(t, m)}")
            hits += 1
            if hits >= 25:
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
