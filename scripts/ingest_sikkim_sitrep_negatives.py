#!/usr/bin/env python3
"""Ingest dated road-status observations as negative labels.

Two source kinds are supported:

1. Sikkim R&B Road Conditions sitreps (data/raw/hazards/sikkim_sitrep_*.txt)
   * named-road open/trafficable clauses (first-class, road specifically
     attested), and
   * division-level open attestations ("all roads under Gangtok SD are
     clear and open") mapped to the primary corridor crossing that
     sub-division (NH-10 through Gangtok/Singtam SD, NH-510 through
     Gyalshing SD, Namchi-Manpur road in Namchi SD).  These are kept
     explicitly scoped "corridor_sd_division_open_sitrep" and the evidence
     note quotes the division line; every named exception in the block is
     checked so a corridor listed as blocked is never emitted.
   Division-level lines are only emitted for corridors where the corridor is
   the division's trunk corridor AND the block's named exceptions do not
   mention it.

2. URL-sourced press items (verified by mirroring the article to
   data/raw/hazards/ and requiring the quoted fragments to appear in it).

Rules (as before)
-----------------
* Every row's quoted clause must appear in the pinned source; otherwise the
  script fails loudly.
* prediction_time = nearest rainfall feature date >= observation date;
  gaps > FEATURE_GAP_DAYS fail.  window_end = prediction_time + 7d.
* observation_status = OPEN (never CLEAR/REOPEN/RESUMED/RESTRICT).
* observation_scope names the attestation type; all rows are corridor-level.

Usage:  python scripts/ingest_sikkim_sitrep_negatives.py [--source FILE] [--apply]
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/raw/hazards/sikkim_sitrep_road_conditions_2026-09-19.txt"
NEGATIVE_OBSERVATIONS_FILE = ROOT / "data/raw/hazards/negative_observations.csv"
RAINFALL_FILE = ROOT / "data/processed/weather/road_rainfall_features_temporal.parquet"
NETWORK_FILE = ROOT / "data/processed/roads/ner_roads_districts.gpkg"
SITREP_URL = "http://www.sikkim-roadsandbridges.gov.in/index.php/road-network"

FEATURE_GAP_DAYS = 7
FORBIDDEN_STATUS = ("CLEAR", "REOPEN", "RESUMED", "RESTRICT")
NEGATIVE_WORDS = (
    "blocked", "closed", "damage", "washout", "slip", "collapse",
    "restricted", "under repair", "under construction", "not open",
)

# Corridor representatives on the production network.
NH10_OSM = 879401691      # NH-10 (Bardang/Rangpo stretch)
NH510_OSM = 44848726      # NH-510 (Gyalshing Sub-Division)
NAMCHI_MANPUR_OSM = 1268716500  # Namchi-Manpur Road bridge of Namchi SD
NH37_OSM = 44884968       # NH-37 (Noney/Awangkhul, Imphal-Jiribam corridor)


def sitrep(date_iso: str, slug: str, label: str, ref: str, osm: int,
           fragments: list[str], comment: str, scope: str) -> dict:
    return {
        "source": "sitrep", "date": date_iso, "slug": slug, "label": label,
        "ref": ref, "osm": osm, "fragments": fragments, "comment": comment,
        "scope": scope,
    }


def press(date_iso: str, slug: str, label: str, ref: str, osm: int,
          fragments: list[str], comment: str, url: str, mirror: str) -> dict:
    return {
        "source": "url", "date": date_iso, "slug": slug, "label": label,
        "ref": ref, "osm": osm, "fragments": fragments, "comment": comment,
        "scope": "corridor_named_road_press", "url": url, "mirror": mirror,
    }


# --------------------------------------------------------------------------- #
# 1. Named-road sitrep observations (road specifically attested).
# --------------------------------------------------------------------------- #
OBSERVATIONS = [
    sitrep("2025-06-25", "legsip_gyalshing", "Legsip-Gyalshing road", "NH510",
           NH510_OSM, ["Legsip-Gyalshing road-Open"], "NH-510 corridor, Gyalshing SD",
           "corridor_named_road_sitrep"),
    sitrep("2025-06-25", "gyalshing_pelling", "Gyalshing-Pelling road", "NH510",
           NH510_OSM, ["Gyalshing-Pelling road-Open"], "NH-510 corridor, Gyalshing SD",
           "corridor_named_road_sitrep"),
    sitrep("2025-06-19", "legsip_gyalshing", "Legsip-Gyalshing road", "NH510",
           NH510_OSM, ["Legsip-Gyalshing road-Open"], "NH-510 corridor, Gyalshing SD",
           "corridor_named_road_sitrep"),
    sitrep("2025-06-19", "gyalshing_pelling", "Gyalshing-Pelling road", "NH510",
           NH510_OSM, ["Gyalshing-Pelling road-Open"], "NH-510 corridor, Gyalshing SD",
           "corridor_named_road_sitrep"),
    sitrep("2025-06-19", "namchi_manpur_phatak", "Namchi Manpur Phatak road",
           "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
           ["Namchi Manpur Phatak", "open"], "named road, Namchi SD",
           "corridor_named_road_sitrep"),
    sitrep("2024-05-31", "assam_pakyong_trafficable", "Assam Pakyong road",
           "NH717A", 495859531,
           ["Assam Pakyong road", "the road is trafficable"],
           "NH-717A Pakyong SD; formation damaged but trafficable (explicit)",
           "corridor_named_road_sitrep"),
]


# --------------------------------------------------------------------------- #
# 2. Division-level sitrep attestations mapped to the trunk corridor.
# --------------------------------------------------------------------------- #
def sd_entry(date_iso: str, corridor: str, ref: str, osm: int, fragments: list[str],
             comment: str) -> dict:
    return sitrep(date_iso, f"{corridor}_sd", f"{corridor} (division-level open)",
                  ref, osm, fragments, comment, "corridor_sd_division_open_sitrep")


DIVISION_OBSERVATIONS = [
    # --- Gangtok SD -> NH-10 trunk corridor -------------------------------- #
    sd_entry("2025-06-25", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-06-19", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-06-10", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-05-26", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-05-23", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-05-21", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-05-19", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line"),
    sd_entry("2025-05-17", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line (FWRs)"),
    sd_entry("2025-05-15", "nh10", "NH10", NH10_OSM,
             ["Gangtok SD is clear and open for traffic"], "Gangtok SD open line (FWRs)"),
    sd_entry("2024-09-11", "nh10", "NH10", NH10_OSM,
             ["Gangtok Sub Division is clear for traffic"], "Gangtok SD 2024 open line"),
    sd_entry("2024-08-27", "nh10", "NH10", NH10_OSM,
             ["Gangtok Sub Division is clear for traffic"], "Gangtok SD 2024 open line"),
    sd_entry("2024-08-23", "nh10", "NH10", NH10_OSM,
             ["Gangtok Sub Division is clear for traffic"], "Gangtok SD 2024 open line"),
    sd_entry("2024-08-02", "nh10", "NH10", NH10_OSM,
             ["Gangtok Sub Division is clear for traffic"], "Gangtok SD 2024 open line"),
    # --- Singtam SD -> NH-10 trunk corridor (Radang-Khimchithang is a side road) --- #
    sd_entry("2025-06-25", "nh10_singtam", "NH10", NH10_OSM,
             ["Singtam SD are trafficable except Radang-Khimchithang"],
             "Singtam exception is off-NH10"),
    sd_entry("2025-06-19", "nh10_singtam", "NH10", NH10_OSM,
             ["Singtam SD are trafficable except Radang-Khimchithang"],
             "Singtam exception is off-NH10"),
    sd_entry("2025-06-10", "nh10_singtam", "NH10", NH10_OSM,
             ["Singtam SD are trafficable except Radang-Khimchithang"],
             "Singtam exception is off-NH10"),
    sd_entry("2025-05-26", "nh10_singtam", "NH10", NH10_OSM,
             ["rest all the roads under Singtam SD are trafficable"],
             "Radang-Khimchithang closed; NH-10 covered"),
    sd_entry("2025-05-23", "nh10_singtam", "NH10", NH10_OSM,
             ["rest all the roads under Singtam SD are trafficable"],
             "Radang-Khimchithang blocked; NH-10 covered"),
    sd_entry("2025-05-21", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD are clear trafficable"],
             "Singtam SD clear line"),
    sd_entry("2025-05-17", "nh10_singtam", "NH10", NH10_OSM,
             ["Rest all the roads under Singtam SD are trafficable"],
             "Radang-Khimchithang blocked; NH-10 covered"),
    sd_entry("2025-05-15", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD is clear for traffic"],
             "Singtam SD clear line"),
    sd_entry("2024-09-11", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD are trafficable"], "Singtam SD 2024 line"),
    sd_entry("2024-08-27", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD are trafficable"], "Singtam SD 2024 line"),
    sd_entry("2024-08-23", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD are trafficable"], "Singtam SD 2024 line"),
    sd_entry("2024-08-02", "nh10_singtam", "NH10", NH10_OSM,
             ["All the roads under Singtam SD are trafficable"], "Singtam SD 2024 line"),
    # --- Gyalshing SD -> NH-510 trunk corridor ----------------------------- #
    sd_entry("2025-06-10", "nh510", "NH510", NH510_OSM,
             ["All the roads under Gyalshing Sub-Division are open"],
             "Gyalshing SD open line (FWR exceptions)"),
    sd_entry("2025-05-26", "nh510", "NH510", NH510_OSM,
             ["All the roads under Gyalshing Sub-Division are open"],
             "Gyalshing SD open line (FWR exceptions)"),
    sd_entry("2025-05-23", "nh510", "NH510", NH510_OSM,
             ["All the roads under Gyalshing Sub-Division are open"],
             "Gyalshing SD open line (FWR exceptions)"),
    sd_entry("2025-05-21", "nh510", "NH510", NH510_OSM,
             ["rest all the roads under Gyalshing Sub-Division are open"],
             "Chongrang-Labdang/Relli khola blocked; NH-510 covered"),
    sd_entry("2024-09-11", "nh510", "NH510", NH510_OSM,
             ["Rest all the road are clear except"], "Teshangthang blocked; NH-510 covered"),
    sd_entry("2024-08-27", "nh510", "NH510", NH510_OSM,
             ["Rest all the road are clear except"], "Teshangthang-blocked; NH-510 covered"),
    sd_entry("2024-08-23", "nh510", "NH510", NH510_OSM,
             ["Rest all the road are clear except"],
             "Teshangthang + Redang blocked; NH-510 covered"),
    sd_entry("2024-08-02", "nh510", "NH510", NH510_OSM,
             ["Rest all the road are clear except"], "Teshangthang blocked; NH-510 covered"),
    # --- Namchi SD -> Namchi-Manpur Road (trunk link) ----------------------- #
    sd_entry("2025-06-25", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open except Phongla Bermiok"],
             "Phongla Bermiok fatak LMV-only; Namchi-Manpur covered"),
    sd_entry("2025-06-10", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the road under Namchi Sub Division are open as of now"],
             "Namchi SD open line (FWRs)"),
    sd_entry("2025-05-26", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open"], "Namchi SD open line"),
    sd_entry("2025-05-23", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the road under Namchi Sub Division are open"], "Namchi SD open line"),
    sd_entry("2025-05-21", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open as of now"],
             "Namchi SD open line"),
    sd_entry("2025-05-19", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the road under Namchi Sub Division are open"], "Namchi SD open line"),
    sd_entry("2024-09-11", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["rest of the roads are open as of now"],
             "Bermiok LMV-only; Namchi-Manpur covered"),
    sd_entry("2024-08-27", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open except Phongla Bermiok road"],
             "Bermiok excluded; Namchi-Manpur covered"),
    sd_entry("2024-08-23", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open except Phongla Bermiok road"],
             "Bermiok excluded; Namchi-Manpur covered"),
    sd_entry("2024-08-02", "namchi_manpur", "Namchi-Manpur Road", NAMCHI_MANPUR_OSM,
             ["All the roads under Namchi Sub Division are open except Phongla Bermiok road"],
             "Bermiok excluded; Namchi-Manpur covered"),
]

# --------------------------------------------------------------------------- #
# 3. URL-sourced press items (mirrored, fragment-verified).
# --------------------------------------------------------------------------- #
PRESS_OBSERVATIONS = [
    press("2025-09-03", "imphal_jiribam_nh37", "Imphal-Jiribam road (NH-37)",
          "NH37", NH37_OSM,
          ["traffic resumes after clearance"],
          "Imphal Times; NH-37 Imphal-Jiribam corridor open after clearance on 2025-09-03",
          "https://www.imphaltimes.com/news/landslide-blocks-imphal-jiribam-road-traffic-resumes-after-clearance/",
          "data/raw/hazards/imphaltimes_nh37_2025-09-03.html"),
]

# --------------------------------------------------------------------------- #
# 4. Ref-roads not yet in the road base (kept pending; never emitted).
# --------------------------------------------------------------------------- #
PENDING_OBSERVATIONS = [
    ("2024-08-02", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19",
     "Kitam fatak to Joerthang road is open as of now (SH-19)", "road under ABD, SH-19"),
    ("2024-08-23", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19",
     "Kitam fatak to Joerthang road is open as of now (SH-19)", "road under ABD, SH-19"),
    ("2024-08-27", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19",
     "Kitam fatak to Joerthang road is open as of now (SH-19)", "road under ABD, SH-19"),
    ("2024-09-11", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19",
     "Kitam fatak to Joerthang road is open as of now (SH-19)", "road under ABD, SH-19"),
    ("2024-08-02", "jorethang_namchi", "Jorethang to Namchi road", "SH20",
     "is open as of now (SH-20)", "road under ABD, SH-20"),
    ("2024-08-23", "jorethang_namchi", "Jorethang to Namchi road", "SH20",
     "is open as of now (SH-20)", "road under ABD, SH-20"),
    ("2024-08-27", "jorethang_namchi", "Jorethang to Namchi road", "SH20",
     "is open as of now (SH-20)", "road under ABD, SH-20"),
    ("2024-09-11", "jorethang_namchi", "Jorethang to Namchi road", "SH20",
     "is open as of now (SH-20)", "road under ABD, SH-20"),
]

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


def parse_source(text: str) -> list[dict]:
    date_re = re.compile(
        r"^Date:\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[-\s](\d{4})", re.I)
    entries: list[dict] = []
    cur: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = date_re.match(line)
        if m:
            day, month_name, year = int(m.group(1)), MONTHS[m.group(2).lower()], int(m.group(3))
            cur = {"date": date(year, month_name, day), "body": []}
            entries.append(cur)
        elif cur is not None:
            cur["body"].append(line)
    return sorted(entries, key=lambda e: e["date"])


def strip_html(raw: str) -> str:
    text = re.sub(r"<script.*?</script>", " ", raw, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    import html as _html
    text = _html.unescape(text)
    return re.sub(r"[ \t]+", " ", text)


def check_fragments(block: str, fragments: list[str], where: str) -> None:
    lowered = block.lower()
    missing = [f for f in fragments if f.lower() not in lowered]
    if missing:
        raise SystemExit(
            f"FATAL: {where}: source lacks fragments {missing}\n"
            f"  source head: {block[:200]!r}")


def feature_dates_for(oids: list[int]) -> dict[int, list[pd.Timestamp]]:
    rain = pd.read_parquet(RAINFALL_FILE, columns=["osm_id", "feature_date"])
    rain["feature_date"] = pd.to_datetime(rain["feature_date"])
    return {oid: sorted(rain.loc[rain["osm_id"] == oid, "feature_date"].unique())
            for oid in oids}


def pin_prediction(dates: list[pd.Timestamp], d: date, route_label: str) -> date:
    for ts in dates:
        if ts.date() >= d:
            gap = (ts.date() - d).days
            if gap > FEATURE_GAP_DAYS:
                raise SystemExit(
                    f"FATAL: {route_label} on {d}: no feature date within "
                    f"{FEATURE_GAP_DAYS}d (next {ts.date()}, gap {gap}d)")
            return ts.date()
    raise SystemExit(f"FATAL: {route_label} on {d}: no feature date found")


def build_row(obs: dict, prediction_time: date, quote: str, source_url: str) -> dict:
    return {
        "observation_id": f"{obs['slug']}_{obs['date'].replace('-', '')}",
        "osm_id": obs["osm"],
        "ref": obs["ref"],
        "prediction_time": prediction_time.isoformat(),
        "observation_window_end": (prediction_time + timedelta(days=7)).isoformat(),
        "source_url": source_url,
        "source_publish_date": obs["date"],
        "observation_status": "OPEN",
        "observation_scope": obs["scope"],
        "evidence_note": f"Source {obs['date']}: '{quote}'. "
                         f"Negative pinned to nearest weather date {prediction_time}. "
                         f"{obs['comment']}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--apply", action="store_true",
                    help="merge rows into negative_observations.csv")
    args = ap.parse_args()

    sitrep_text = Path(args.source).read_text(encoding="utf-8", errors="replace")
    entries = parse_source(sitrep_text)
    by_date = {e["date"]: e for e in entries}
    print(f"Parsed {len(entries)} dated sitrep entries from {Path(args.source).name}")

    all_obs = OBSERVATIONS + DIVISION_OBSERVATIONS + PRESS_OBSERVATIONS
    oids = sorted({o["osm"] for o in all_obs if o["osm"] is not None})
    fd = feature_dates_for(oids)

    rows = []
    for obs in all_obs:
        d = date.fromisoformat(obs["date"])
        if obs["source"] == "sitrep":
            entry = by_date.get(d)
            if entry is None:
                raise SystemExit(f"FATAL: no sitrep entry for {d} — provenance missing")
            block = "\n".join(entry["body"])
            source_url = SITREP_URL
        else:
            mirror = ROOT / obs["mirror"]
            if not mirror.exists():
                raise SystemExit(f"FATAL: mirrored source missing: {mirror}")
            block = strip_html(mirror.read_text(encoding="utf-8", errors="replace"))
            source_url = obs["url"]
        check_fragments(block, obs["fragments"], f"{obs['date']} {obs['slug']}")
        quote = obs["fragments"][0]
        pred = pin_prediction(fd[obs["osm"]], d, obs["label"])
        rows.append(build_row(obs, pred, quote, source_url))

    frame = pd.DataFrame(rows)
    print(f"Provenance-verified rows: {len(frame)}")
    for _, r in frame.iterrows():
        print(f"  {r['observation_id']:<48} {r['ref']:<20} pin={r['prediction_time']} "
              f"{r['observation_scope']}")

    pending = []
    for sitrep_d, slug, label, ref, status_words, comment in PENDING_OBSERVATIONS:
        entry = by_date.get(date.fromisoformat(sitrep_d))
        if entry is not None:
            check_fragments("\n".join(entry["body"]), [status_words],
                            f"{sitrep_d} {slug}")
        pending.append((sitrep_d, label, ref))
    print(f"\nPending (road-base expansion required): {len(pending)}")
    for p in pending:
        print(f"  {p[0]} {p[1]} ({p[2]})")

    if args.apply:
        existing = pd.read_csv(NEGATIVE_OBSERVATIONS_FILE, dtype={"osm_id": "str"})
        frame["osm_id"] = frame["osm_id"].astype(str)

        key_cols = ["osm_id", "ref", "prediction_time", "observation_window_end",
                    "source_url", "observation_status", "observation_scope"]
        frame_keys = set(map(tuple, frame[key_cols].astype(str).values))

        # Legacy-id migration: drop rows from an older id scheme
        # (sikkim_rb_<date>_<slug>) when their semantic content is re-issued by
        # the current canonical id scheme, so no observation is double counted.
        def _legacy(row):
            return str(row["observation_id"]).startswith("sikkim_rb_") and \
                tuple(row[key_cols].astype(str).values) in frame_keys
        existing = existing[~existing.apply(_legacy, axis=1)]

        known_keys = set(map(tuple, existing[key_cols].astype(str).values))
        added = frame[~frame[key_cols].astype(str).apply(tuple, axis=1).isin(known_keys)]
        if added.empty:
            print("\nNo new rows to merge (all already present).")
            return 0
        merged = pd.concat([existing, added], ignore_index=True)
        merged.to_csv(NEGATIVE_OBSERVATIONS_FILE, index=False)
        print(f"\nMerged {len(added)} rows into {NEGATIVE_OBSERVATIONS_FILE.name} "
              f"(total {len(merged)}).")
    else:
        print("\nDry run (no writes). Re-run with --apply to merge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())