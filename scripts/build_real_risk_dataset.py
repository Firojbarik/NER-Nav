"""Construct a real (static-feature) risk training dataset.

DESIGN / PROVENANCE
-------------------
Positives are the source-confirmed, source-supported CONFIRMED_AFFECTED OSM
road segments from the real hazard-event pilot (single positive per event,
gathered from *_confirmed_road.parquet artifacts; currently 12).

Negatives are sampled from the OSM trunk network that shares the same NH `ref`
corridor as a positive, EXCLUDING all confirmed-affected segments and any
CANDIDATE-only segments. Because no source confirms these negative segments as
"unaffected", they are conservatively labelled ASSUMED-unaffected
(`label_source='assumed_unaffected_real_pool'`). The dataset is therefore an
EXPLORATORY risk model input, NOT a definitive hazard classifier.

Features are STATIC only (terrain + road class + structure flags). Rainfall is
excluded: the CHIRPS window covers only 2015 while all confirmed events are
2016-2024, so any rainfall join would be out-of-window / non-informative and
we do not fabricate temporal features. With static features and a geographic
(held-out-state) split there is no cross-contamination from prediction-time
lookahead.

OUTPUTS
-------
data/processed/ml/real_risk_dataset.parquet
data/processed/ml/real_risk_dataset_qa.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROAD_FILE = Path("data/processed/roads/ner_roads_districts.gpkg")
TERRAIN_FILE = Path("data/processed/terrain/road_terrain_features.parquet")
CONFIRMED_GLOB = "data/processed/hazards/*_confirmed_road.parquet"
OUTPUT_DIR = Path("data/processed/ml")
OUTPUT_FILE = OUTPUT_DIR / "real_risk_dataset.parquet"
QA_FILE = OUTPUT_DIR / "real_risk_dataset_qa.json"

NEGATIVES_PER_REF = 4  # stratified negative draws per positive's ref corridor
RANDOM_SEED = 2026

# Reuse the established highway prior from build_risk_training_dataset.py
HIGHWAY_SCORES = {
    "motorway": 0.05, "trunk": 0.10, "primary": 0.15, "secondary": 0.20,
    "tertiary": 0.25, "tertiary_link": 0.25, "secondary_link": 0.20,
    "primary_link": 0.15, "residential": 0.30, "living_street": 0.35,
    "service": 0.40, "unclassified": 0.45, "road": 0.45, "track": 0.60,
}


def highway_score(hw):
    return HIGHWAY_SCORES.get(str(hw).lower(), 0.40)


def main() -> None:
    print("=" * 70)
    print("NER-Nav - Real Risk Dataset Construction (static features)")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Load confirmed positives
    # ------------------------------------------------------------------ #
    import glob
    files = sorted(glob.glob(CONFIRMED_GLOB))
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # keep candidate_rank 0 == the resolved/confirmed row per event
        df = df[df["review_status"] == "CONFIRMED_AFFECTED"]
        frames.append(df)
    conf = pd.concat(frames, ignore_index=True)
    conf = conf.drop_duplicates(subset=["osm_id"])
    pos_ids = conf["osm_id"].astype("int64").tolist()
    print(f"Confirmed positives: {len(pos_ids)}")
    if len(pos_ids) < 10:
        print("[WARN] fewer than 10 confirmed positives found")

    # ------------------------------------------------------------------ #
    # Load roads + terrain
    # ------------------------------------------------------------------ #
    roads = gpd.read_file(ROAD_FILE)
    roads["osm_id"] = roads["osm_id"].astype("int64")
    terrain = pd.read_parquet(TERRAIN_FILE)
    terrain["osm_id"] = terrain["osm_id"].astype("int64")

    road_cols = ["osm_id", "ref", "highway", "district", "state", "bridge"]
    rd = roads[[c for c in road_cols if c in roads.columns]].copy()
    rd = rd.merge(terrain, on="osm_id", how="left")

    # ------------------------------------------------------------------ #
    # Positives
    # ------------------------------------------------------------------ #
    pos = rd[rd["osm_id"].isin(pos_ids)].copy()
    pos["is_affected"] = 1
    pos["label_source"] = "real_confirmed"
    pos["confirmed_confidence"] = conf.set_index("osm_id")["confidence_level"].astype(str).reindex(pos_ids).to_numpy()
    pos = pos.assign(confirmed_confidence=lambda d: d["confirmed_confidence"].map(str).fillna(""))

    # ------------------------------------------------------------------ #
    # Negatives (stratified by ref, same NH trunk network, unaffected pool)
    # ------------------------------------------------------------------ #
    refs = pos["ref"].dropna().tolist()
    neg_pool = rd[
        (~rd["osm_id"].isin(pos_ids))
        & (rd["ref"].isin(refs))
        & (rd["highway"] == "trunk")
    ].copy()

    rng = np.random.default_rng(RANDOM_SEED)
    neg_frames = []
    used = set()
    for ref in refs:
        corridor = neg_pool[neg_pool["ref"] == ref]
        corridor = corridor[~corridor["osm_id"].isin(used)]
        n = min(NEGATIVES_PER_REF, len(corridor))
        pick = corridor.sample(n=n, random_state=rng.integers(0, 2**31)).copy()
        used.update(pick["osm_id"].tolist())
        neg_frames.append(pick)
    neg = pd.concat(neg_frames, ignore_index=True) if neg_frames else (
        neg_pool.iloc[0:0].copy()
    )

    if neg.empty:
        # fallback: sample from whole trunk network
        all_pool = rd[(~rd["osm_id"].isin(pos_ids)) & (rd["highway"] == "trunk")]
        neg = all_pool.sample(n=40, random_state=RANDOM_SEED).copy()

    neg["is_affected"] = 0
    neg["label_source"] = "assumed_unaffected_real_pool"
    neg["confirmed_confidence"] = ""

    # ------------------------------------------------------------------ #
    # Assemble + featurise
    # ------------------------------------------------------------------ #
    ds = pd.concat([pos, neg], ignore_index=True)
    ds = ds.reset_index(drop=True)

    ds["highway_prior"] = ds["highway"].map(highway_score)
    ds["bridge_flag"] = (ds["bridge"].fillna("").astype(str).str.lower() != "no").astype(int)

    feature_cols = [
        "osm_id", "ref", "highway", "district", "state",
        "elevation_m", "slope_degrees", "aspect_category",
        "highway_prior", "bridge_flag",
        "is_affected", "label_source", "confirmed_confidence",
    ]
    ds = ds[feature_cols].copy()

    # ------------------------------------------------------------------ #
    # QA
    # ------------------------------------------------------------------ #
    pos_count = int((ds["is_affected"] == 1).sum())
    neg_count = int((ds["is_affected"] == 0).sum())
    terrain_coverage_pos = float(ds.loc[ds["is_affected"] == 1, "slope_degrees"].notna().mean())
    terrain_coverage_neg = float(ds.loc[ds["is_affected"] == 0, "slope_degrees"].notna().mean())
    ref_balance = Counter(zip(ds["label_source"].map(lambda s: "pos" if s == "real_confirmed" else "neg"), ds["ref"]))
    ref_balance = {f"{src}:{ref}": c for (src, ref), c in sorted(ref_balance.items())}

    qa = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "design": "exploratory static-feature risk model; random negatives are ASSUMED-unaffected (no source confirmation)",
        "n_positive_labels": pos_count,
        "n_negative_labels": neg_count,
        "positive_ids": sorted(pos_ids),
        "negative_label_source": "assumed_unaffected_real_pool",
        "terrain_coverage_pos_frac": round(terrain_coverage_pos, 4),
        "terrain_coverage_neg_frac": round(terrain_coverage_neg, 4),
        "features": feature_cols,
        "ref_balance": ref_balance,
        "excluded": "rainfall excluded (2015-only window; all events 2016-2024)",
    }

    ds.to_parquet(OUTPUT_FILE, index=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_FILE if False else QA_FILE).parent.mkdir(parents=True, exist_ok=True)
    QA_FILE.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print(f"Positives: {pos_count}")
    print(f"Negatives: {neg_count}")
    print(f"Total:     {len(ds)}")
    print(f"Terrain coverage (slope): pos={terrain_coverage_pos:.3f} neg={terrain_coverage_neg:.3f}")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {QA_FILE}")


if __name__ == "__main__":
    main()
