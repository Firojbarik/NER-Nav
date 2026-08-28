"""Construct the REAL temporal risk-training dataset.

DESIGN / PROVENANCE
-------------------
Real-data only. Sample design lives in `ml/features/temporal_design.py`
(PROVENANCE docstring there). Briefly:

* **Positives** (label=1): for each source-confirmed event in the versioned
  event design, the
  confirmed affected road sampled at prediction times event-1 / -3 / -7 days.
  In every case the real event falls strictly inside (prediction_time,
  prediction_time + 7 days] -> label 1 (temporal rule from ml/labels/generate.py).
* **Same-road controls** (label=0): the same confirmed road sampled at
  event-14 days; the event is +14 > 7 days out -> label 0.
* **Corridor negatives** (label=0): N unaffected trunk roads of the same NH
  `ref`, sampled at event-3 days, EXCLUDING all confirmed-affected roads
  (`label_source='assumed_unaffected_real_pool'`, as in the static dataset).

Features are REAL: 30-day rainfall lookback (date-anchored, strictly before
predictions; no future leakage by construction) + static terrain + highway
prior + bridge flag.

Evaluation uses grouped leave-one-corridor-out CV (the same honest protocol as
the static model); rainfall is a time-varying feature but every training anchor
predates its own test positives, so no temporal leakage across folds for static
use. Corridor groups are disjoint -> no segment appears in both train and test.

OUTPUTS
-------
data/processed/ml/real_temporal_risk_dataset.parquet
data/processed/ml/real_temporal_risk_dataset_qa.json
"""

from __future__ import annotations

import json
import hashlib
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.temporal_design import (  # noqa: E402
    CONFIRMED_EVENTS,
    NEGATIVES_PER_REF,
    NEGATIVE_CORRIDOR_OFFSET,
    NEGATIVE_SAME_ROAD_OFFSETS,
    POSITIVE_OFFSETS,
)

ROAD_FILE = Path("data/processed/roads/ner_roads_districts.gpkg")
TERRAIN_FILE = Path("data/processed/terrain/road_terrain_features.parquet")
RAINFALL_FILE = Path("data/processed/weather/road_rainfall_features_temporal.parquet")
OUTPUT_DIR = Path("data/processed/ml")
OUTPUT_FILE = OUTPUT_DIR / "real_temporal_risk_dataset.parquet"
QA_FILE = OUTPUT_DIR / "real_temporal_risk_dataset_qa.json"

RANDOM_SEED = 2026

RAINFALL_COLS = ["rainfall_1day", "rainfall_3day", "rainfall_7day",
                 "rainfall_14day", "rainfall_30day"]

HIGHWAY_SCORES = {
    "motorway": 0.05, "trunk": 0.10, "primary": 0.15, "secondary": 0.20,
    "tertiary": 0.25, "tertiary_link": 0.25, "secondary_link": 0.20,
    "primary_link": 0.15, "residential": 0.30, "living_street": 0.35,
    "service": 0.40, "unclassified": 0.45, "road": 0.45, "track": 0.60,
}


def highway_score(hw):
    return HIGHWAY_SCORES.get(str(hw).lower(), 0.40)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(ds):
    """Reject structurally invalid active ML data before persistence."""
    errors = []
    key = ["event_id", "osm_id", "prediction_time"]
    if ds[key].isna().any().any():
        errors.append("sample identity contains null values")
    if ds.duplicated(key).any():
        errors.append("duplicate event/road/prediction samples detected")
    if not set(ds["label"].dropna().unique()) <= {0, 1}:
        errors.append("labels must be binary")
    if (ds[RAINFALL_COLS] < 0).any().any():
        errors.append("negative rainfall values detected")
    if ((ds["slope_degrees"] < 0) | (ds["slope_degrees"] > 90)).any():
        errors.append("slope outside [0, 90] degrees")
    if errors:
        raise ValueError("dataset validation failed: " + "; ".join(errors))
    return {
        "status": "PASS",
        "checks": ["identity_non_null", "sample_key_unique", "binary_labels",
                   "rainfall_non_negative", "slope_physical_range"],
    }


def main() -> None:
    print("=" * 70)
    print("NER-Nav - Real Temporal Risk Dataset Construction")
    print("=" * 70)

    # ------------------------------------------------------------------ #
    # Positives + same-road controls
    # ------------------------------------------------------------------ #
    pos_rows = []
    for eid, osm, ref, ed in CONFIRMED_EVENTS:
        E = date.fromisoformat(ed)
        for off in POSITIVE_OFFSETS:
            pos_rows.append({
                "event_id": eid, "osm_id": osm, "ref": ref,
                "prediction_time": str(E - timedelta(days=off)),
                "label": 1, "sample_kind": "positive",
            })
        for off in NEGATIVE_SAME_ROAD_OFFSETS:
            pos_rows.append({
                "event_id": eid, "osm_id": osm, "ref": ref,
                "prediction_time": str(E - timedelta(days=off)),
                "label": 0, "sample_kind": "same_road_control",
            })
    pos_ids = list({r["osm_id"] for r in pos_rows})
    print(f"Confirmed roads: {len(pos_ids)}; pos+same_road_control samples: {len(pos_rows)}")

    # ------------------------------------------------------------------ #
    # Load road + terrain + rainfall
    # ------------------------------------------------------------------ #
    roads = gpd.read_file(ROAD_FILE)
    roads["osm_id"] = roads["osm_id"].astype("int64")
    terrain = pd.read_parquet(TERRAIN_FILE)
    terrain["osm_id"] = terrain["osm_id"].astype("int64")
    rainfall = pd.read_parquet(RAINFALL_FILE)
    rainfall["osm_id"] = rainfall["osm_id"].astype("int64")
    rainfall["feature_date"] = pd.to_datetime(rainfall["feature_date"])

    road_cols = ["osm_id", "ref", "highway", "district", "state", "bridge"]
    rd = roads[[c for c in road_cols if c in roads.columns]].copy()
    rd = rd.merge(terrain, on="osm_id", how="left")

    # ------------------------------------------------------------------ #
    # Corridor negatives (unaffected same-NH trunk roads) per event at E-3
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(RANDOM_SEED)
    neg_rows = []
    used = set(pos_ids)
    conf_src = "assumed_unaffected_real_pool"
    for eid, osm, ref, ed in CONFIRMED_EVENTS:
        E = date.fromisoformat(ed)
        pt = str(E - timedelta(days=NEGATIVE_CORRIDOR_OFFSET))
        corridor = rd[
            (rd["ref"] == ref)
            & (~rd["osm_id"].isin(used))
            & (rd["highway"] == "trunk")
        ]
        corridor = corridor[~corridor["osm_id"].isin(used)]
        n = min(NEGATIVES_PER_REF, len(corridor))
        if n == 0:
            continue
        pick = corridor.sample(n=n, random_state=rng.integers(0, 2**31))
        for _, r in pick.iterrows():
            neg_rows.append({
                "event_id": eid, "osm_id": int(r["osm_id"]), "ref": ref,
                "prediction_time": pt, "label": 0,
                "sample_kind": "corridor_negative",
            })
            used.add(int(r["osm_id"]))
    print(f"Corridor negatives: {len(neg_rows)}")

    samples = pd.DataFrame(pos_rows + neg_rows)
    samples["prediction_time"] = pd.to_datetime(samples["prediction_time"])
    samples["osm_id"] = samples["osm_id"].astype("int64")

    # ------------------------------------------------------------------ #
    # Join static + rainfall
    # ------------------------------------------------------------------ #
    ds = samples.merge(rd.drop(columns=["ref"]), on="osm_id", how="left")
    rain = rainfall.rename(columns={"feature_date": "prediction_time"})
    ds = ds.merge(rain, on=["osm_id", "prediction_time"], how="left")

    ds["highway_prior"] = ds["highway"].map(highway_score)
    ds["bridge_flag"] = (ds["bridge"].fillna("").astype(str).str.lower() != "no").astype(int)

    # Derived rainfall intensity features — no new data, just ratios.
    # These capture whether current rain is unusually concentrated/intense
    # for this road, which is more predictive than raw amounts.
    _eps = 0.01  # avoid division by zero
    ds["rain_intensity_1_vs_7"] = ds["rainfall_1day"] / ds["rainfall_7day"].clip(lower=_eps)
    ds["rain_intensity_3_vs_14"] = ds["rainfall_3day"] / ds["rainfall_14day"].clip(lower=_eps)
    ds["rain_concentration_3_in_7"] = ds["rainfall_3day"] / ds["rainfall_7day"].clip(lower=_eps)
    ds["rain_trend_1_vs_3"] = ds["rainfall_1day"] / ds["rainfall_3day"].clip(lower=_eps)
    ds["rain_cumul_ratio_7_vs_30"] = ds["rainfall_7day"] / ds["rainfall_30day"].clip(lower=_eps)
    # NaN indicator for terrain — lets model learn "terrain unknown" as a signal
    ds["terrain_known"] = ds["elevation_m"].notna().astype(int)

    # Slope-rainfall interactions: the physical mechanism for landslides is
    # rain on steep slopes. These features let XGBoost see this directly
    # without needing to learn the interaction through tree splits.
    _slope = ds["slope_degrees"].fillna(0)
    _elev = ds["elevation_m"].fillna(0)
    ds["slope_x_rain7"] = _slope * ds["rainfall_7day"].fillna(0)
    ds["slope_x_rain30"] = _slope * ds["rainfall_30day"].fillna(0)
    ds["elev_x_slope"] = _elev * _slope

    ds["label_source"] = np.where(
        ds["label"] == 1,
        "real_confirmed_temporal",
        np.where(ds["sample_kind"] == "same_road_control",
                 "real_same_road_control", conf_src),
    )

    feature_cols = [
        "event_id", "osm_id", "ref", "highway", "district", "state",
        "prediction_time", "label", "sample_kind", "label_source",
        "elevation_m", "slope_degrees",
    ] + RAINFALL_COLS + ["rainfall_days_available", "highway_prior", "bridge_flag",
                          "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
                          "rain_concentration_3_in_7", "rain_trend_1_vs_3",
                          "rain_cumul_ratio_7_vs_30", "terrain_known",
                          "slope_x_rain7", "slope_x_rain30", "elev_x_slope"]
    ds = ds[feature_cols].copy()
    validation = validate_dataset(ds)

    # ------------------------------------------------------------------ #
    # QA
    # ------------------------------------------------------------------ #
    pos_count = int((ds["label"] == 1).sum())
    neg_count = int((ds["label"] == 0).sum())
    ref_balance = Counter(zip(ds["label"].map(lambda l: "pos" if l == 1 else "neg"), ds["ref"]))
    ref_balance = {f"{s}:{r}": c for (s, r), c in sorted(ref_balance.items())}
    rain_cov = ds[RAINFALL_COLS].notna().mean().to_dict()
    rain_cov = {k: round(v, 4) for k, v in rain_cov.items()}

    qa = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "design": (
            f"real temporal: {len(CONFIRMED_EVENTS)} confirmed events; positives at event-1/-3/-7 "
            "(event within 7-day horizon), same-road controls at event-14, "
            "corridor negatives (assumed-unaffected) sampled at event-3"
        ),
        "horizon_days": 7,
        "n_positive_labels": pos_count,
        "n_negative_labels": neg_count,
        "positive_ids": sorted(pos_ids),
        "rainfall_lookback_days": 30,
        "rainfall_coverage": rain_cov,
        "terrain_slope_coverage_pos": round(float(ds.loc[ds["label"] == 1, "slope_degrees"].notna().mean()), 4),
        "ref_balance": ref_balance,
        "features": feature_cols,
        "validation": validation,
        "input_lineage": {
            "roads_sha256": sha256_file(ROAD_FILE),
            "terrain_sha256": sha256_file(TERRAIN_FILE),
            "rainfall_sha256": sha256_file(RAINFALL_FILE),
        },
    }

    ds.to_parquet(OUTPUT_FILE, index=False)
    qa["dataset_sha256"] = sha256_file(OUTPUT_FILE)
    QA_FILE.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print(f"Positive samples: {pos_count}")
    print(f"Negative samples: {neg_count}")
    print(f"Total:            {len(ds)}")
    print(f"Rainfall coverage: {rain_cov}")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {QA_FILE}")


if __name__ == "__main__":
    main()
