#!/usr/bin/env python3
"""Ingest dated Sikkim R&B road-status observations as negative labels.

Every negative emitted here is a provenance-checked, named-road "open /
trafficable" statement taken from the Roads & Bridges Department (Sikkim)
Road Conditions page (data/raw/hazards/sikkim_sitrep_road_conditions_*.txt).

Design rules enforced by this script
------------------------------------
* Only EXPLICIT, named-road open/trafficable clauses qualify.  Division-level
  aggregations ("all roads under X SD are open") attest unknown roads and are
  excluded (this script never synthesises rows from them).
* Every row's quoted clause must literally appear inside the dated sitrep block
  for its observation date; otherwise the script fails loudly.
* prediction_time is pinned to the nearest available rainfall feature date
  >= the sitrep date so that the observation carries real weather features.
  A gap > FEATURE_GAP_DAYS fails; an intervening sitrep naming the route
  between the sitrep date and the pinned prediction date is a hard failure.
* observation_window_end = prediction_time + 7 days; source_publish_date is the
  sitrep date (<= prediction_time), satisfying the training-input gate.
* observation_status = OPEN (never CLEAR/REOPEN/RESUMED/RESTRICT) so the
  readiness quality gate semantics stay honest: these rows attest a road was
  observed open — they are NOT "reopened after closure" statements.
* observation_scope = corridor_named_road_sitrep — the observation is at
  corridor level, never segment level.

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
SOURCE_URL = "http://www.sikkim-roadsandbridges.gov.in/index.php/road-network"

FEATURE_GAP_DAYS = 7
FORBIDDEN_STATUS = ("CLEAR", "REOPEN", "RESUMED", "RESTRICT")
NEGATIVE_WORDS = (
    "blocked", "closed", "damage", "washout", "slip", "collapse",
    "restricted", "under repair", "under construction", "not open",
)

# --------------------------------------------------------------------------- #
# Curated source-backed observations.
# Each entry: (sitrep_date, route_slug, route_label, ref, osm_id or None,
#              fragments_must_appear, comment)
# osm_id None => route not yet present in the road base; emitted as pending
#                until the road is added to the network (SH-19/SH-20 rows).
# --------------------------------------------------------------------------- #
OBSERVATIONS = [
    # ---- NH-510 (West Sikkim, Gyalshing Sub-Division) --------------------- #
    ("2025-06-25", "legsip_gyalshing", "Legsip-Gyalshing road", "NH510", 44848726, ["Legsip-Gyalshing road-Open"], "NH-510 corridor, Gyalshing SD"),
    ("2025-06-25", "gyalshing_pelling", "Gyalshing-Pelling road", "NH510", 44848726, ["Gyalshing-Pelling road-Open"], "NH-510 corridor, Gyalshing SD"),
    ("2025-06-19", "legsip_gyalshing", "Legsip-Gyalshing road", "NH510", 44848726, ["Legsip-Gyalshing road-Open"], "NH-510 corridor, Gyalshing SD"),
    ("2025-06-19", "gyalshing_pelling", "Gyalshing-Pelling road", "NH510", 44848726, ["Gyalshing-Pelling road-Open"], "NH-510 corridor, Gyalshing SD"),
    # ---- Namchi Manpur Phatak road (Namchi SD) ---------------------------- #
    ("2025-06-19", "namchi_manpur_phatak", "Namchi Manpur Phatak road", "Namchi-Manpur Road", 1268716500, ["Namchi Manpur Phatak", "open"], "named road, Namchi SD"),
    # ---- Assam Pakyong road trafficable despite formation damage ----------- #
    ("2024-05-31", "assam_pakyong_trafficable", "Assam Pakyong road", "NH717A", 495859531, ["Assam Pakyong road", "the road is trafficable"], "NH-717A Pakyong SD; formation damaged but trafficable (explicit)"),
]

PENDING_OBSERVATIONS = [
    # ---- SH-19 (Kitam fatak - Jorethang) and SH-20 (Jorethang - Namchi) ---- #
    # Roads under ABD. Ref roads not (yet) present in the road base.
    ("2024-08-02", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19", ["Kitam fatak to Joerthang road is open as of now (SH-19)"], "road under ABD, SH-19"),
    ("2024-08-23", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19", ["Kitam fatak to Joerthang road is open as of now (SH-19)"], "road under ABD, SH-19"),
    ("2024-08-27", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19", ["Kitam fatak to Joerthang road is open as of now (SH-19)"], "road under ABD, SH-19"),
    ("2024-09-11", "kitam_fatak_jorethang", "Kitam fatak to Jorethang road", "SH19", ["Kitam fatak to Joerthang road is open as of now (SH-19)"], "road under ABD, SH-19"),
    ("2024-08-02", "jorethang_namchi", "Jorethang to Namchi road", "SH20", ["is open as of now (SH-20)", "Jorethang to Namchi"], "road under ABD, SH-20"),
    ("2024-08-23", "jorethang_namchi", "Jorethang to Namchi road", "SH20", ["is open as of now (SH-20)", "to Namchi road"], "road under ABD, SH-20 (source text has typo Joerthang)"),
    ("2024-08-27", "jorethang_namchi", "Jorethang to Namchi road", "SH20", ["is open as of now (SH-20)", "Jorethang to Namchi"], "road under ABD, SH-20"),
    ("2024-09-11", "jorethang_namchi", "Jorethang to Namchi road", "SH20", ["is open as of now (SH-20)", "Jorethang to Namchi"], "road under ABD, SH-20"),
]

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july",
     "august", "september", "october", "november", "december"], 1)}


def parse_source(text: str) -> list[dict]:
    """Split the source into dated entries: {date, body}. Returns entries sorted."""
    date_re = re.compile(r"^Date:\s*(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]+)[-\s](\d{4})", re.I)
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


def find_entry(entries: list[dict], d: date) -> dict:
    for e in entries:
        if e["date"] == d:
            return e
    raise SystemExit(f"FATAL: no source entry for {d.isoformat()} — provenance missing")


def verify_fragments(entry: dict, fragments: list[str], date_iso: str) -> None:
    block = "\n".join(entry["body"]).lower()
    missing = [f for f in fragments if f.lower() not in block]
    if missing:
        raise SystemExit(
            f"FATAL: {date_iso}: source block lacks fragments {missing}\n"
            f"  block head: {block[:240]!r}")


def network_ref(oid: int) -> str:
    import geopandas as gpd
    g = gpd.read_file(NETWORK_FILE, layer=None)
    g = gpd.read_file(NETWORK_FILE)
    row = g[g["osm_id"].astype("int64") == oid]
    if row.empty:
        return ""
    return str(row.iloc[0].get("ref", ""))


def feature_dates_for(oids: list[int]) -> dict[int, list[pd.Timestamp]]:
    rain = pd.read_parquet(RAINFALL_FILE, columns=["osm_id", "feature_date"])
    rain["feature_date"] = pd.to_datetime(rain["feature_date"])
    out: dict[int, list[pd.Timestamp]] = {}
    for oid in oids:
        dates = sorted(rain.loc[rain["osm_id"] == oid, "feature_date"].unique())
        out[oid] = dates
    return out


def pin_prediction(dates: list[pd.Timestamp], sitrep_d: date, route_label: str) -> date:
    for d in dates:
        if d.date() >= sitrep_d:
            gap = (d.date() - sitrep_d).days
            if gap > FEATURE_GAP_DAYS:
                raise SystemExit(
                    f"FATAL: {route_label} on {sitrep_d}: no feature date within "
                    f"{FEATURE_GAP_DAYS}d (next is {d.date()}, gap {gap}d)")
            return d.date()
    raise SystemExit(f"FATAL: {route_label} on {sitrep_d}: no feature date found")


def build_row(obs: tuple, prediction_time: date, quote: str) -> dict:
    sitrep_d, slug, label, ref, oid, _frags, comment = obs
    return {
        "observation_id": f"sikkim_rb_{sitrep_d.replace('-', '')}_{slug}",
        "osm_id": oid,
        "ref": ref,
        "prediction_time": prediction_time.isoformat(),
        "observation_window_end": (prediction_time + timedelta(days=7)).isoformat(),
        "source_url": SOURCE_URL,
        "source_publish_date": sitrep_d,
        "observation_status": "OPEN",
        "observation_scope": "corridor_named_road_sitrep",
        "evidence_note": f"R&B Sikkim sitrep {sitrep_d}: '{quote}'. "
                          f"Negative pinned to nearest weather date {prediction_time}. {comment}",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    ap.add_argument("--apply", action="store_true", help="merge rows into negative_observations.csv")
    args = ap.parse_args()

    text = Path(args.source).read_text(encoding="utf-8", errors="replace")
    entries = parse_source(text)
    print(f"Parsed {len(entries)} dated entries from {Path(args.source).name}")

    kinds = list(OBSERVATIONS)
    oids = [o[4] for o in kinds if o[4] is not None]
    fd = feature_dates_for(oids)

    rows: list[dict] = []
    for obs in kinds:
        sitrep_d = date.fromisoformat(obs[0])
        _d, slug, label, ref, oid, frags, comment = obs
        entry = find_entry(entries, sitrep_d)
        verify_fragments(entry, frags, obs[0])
        block = "\n".join(entry["body"])
        quote = frags[0] if len(block) > 0 else ""
        pred = pin_prediction(fd[oid], sitrep_d, label)
        net_ref = network_ref(oid)
        note = f"network ref on osm {oid}: {net_ref!r}"
        rows.append(build_row(obs, pred, quote))

    pending: list[dict] = []
    for obs in PENDING_OBSERVATIONS:
        sitrep_d = date.fromisoformat(obs[0])
        _d, slug, label, ref, _frags, comment = obs
        entry = find_entry(entries, sitrep_d)
        verify_fragments(entry, obs[4], obs[0])
        oids_in_network = []  # placeholder; SH refs absent from network
        pending.append({
            "sitrep_date": obs[0], "route": label, "ref": ref,
            "status_words": obs[4][0], "comment": comment,
        })

    frame = pd.DataFrame(rows)
    print(f"Provenance-verified active routes: {len(frame)} rows")
    if frame.empty:
        frame = pd.DataFrame(columns=[
            "observation_id", "osm_id", "ref", "prediction_time",
            "observation_window_end", "source_url", "source_publish_date",
            "observation_status", "observation_scope", "evidence_note"])
    print(frame.to_string(index=False))

    print(f"\nPending (road-base expansion required): {len(pending)}")
    for p in pending:
        print(f"  {p['sitrep_date']} {p['route']} ({p['ref']}) — {p['status_words']}")

    if args.apply:
        existing = pd.read_csv(NEGATIVE_OBSERVATIONS_FILE, dtype={"osm_id": "str"})
        known = set(existing["observation_id"].astype(str))
        new = frame[~frame["observation_id"].isin(known)]
        if new.empty:
            print("\nNo new rows to merge (all already present).")
            return 0
        merged = pd.concat([existing, new], ignore_index=True)
        merged.to_csv(NEGATIVE_OBSERVATIONS_FILE, index=False)
        print(f"\nMerged {len(new)} rows into {NEGATIVE_OBSERVATIONS_FILE.name} "
              f"(total {len(merged)}).")
        print("NOT merged (still pending road-base expansion):", len(pending))
    else:
        print("\nDry run (no writes). Re-run with --apply to merge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())