#!/usr/bin/env python3
"""Build the production-constrained ML training dataset (admissible labels only).

Unlike the demo builder, this script admits only source-confirmed and
observation-backed negatives.  Same-road offsets and assumed-unaffected
corridor pools are deliberately excluded — absence from an incident feed
is not evidence of non-occurrence.

Positives:  label=1, label_source=real_confirmed_temporal
Negatives:  label=0, label_source=real_observed_unaffected
              (source-backed road-status observations from
              data/raw/hazards/negative_observations.csv)

OUTPUTS
-------
data/processed/ml/real_temporal_production_dataset.parquet
data/processed/ml/real_temporal_production_dataset_qa.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.features.seasonal import (  # noqa: E402
    SEASONAL_FEATURES,
    days_into_monsoon,
    monsoon_active,
    month_feature,
    pre_monsoon_active,
    season,
    seasonal_cos,
    seasonal_sin,
)
from ml.features.temporal_design import (  # noqa: E402
    CONFIRMED_EVENTS,
    POSITIVE_OFFSETS,
)

ROOT = Path(__file__).resolve().parents[1]
ROAD_FILE = ROOT / "data/processed/roads/ner_roads_districts.gpkg"
ROAD_IDENTITY_FILE = ROOT / "data/processed/roads/road_segment_identity.parquet"
TERRAIN_FILE = ROOT / "data/processed/terrain/road_terrain_features.parquet"
RAINFALL_FILE = ROOT / "data/processed/weather/road_rainfall_features_temporal.parquet"
NEGATIVE_OBSERVATIONS_FILE = ROOT / "data/raw/hazards/negative_observations.csv"
OUTPUT_DIR = ROOT / "data/processed/ml"
OUTPUT_FILE = OUTPUT_DIR / "real_temporal_production_dataset.parquet"
QA_FILE = OUTPUT_DIR / "real_temporal_production_dataset_qa.json"

HIGHWAY_SCORES = {
    "motorway": 0.05, "trunk": 0.10, "primary": 0.15, "secondary": 0.20,
    "tertiary": 0.25, "tertiary_link": 0.25, "secondary_link": 0.20,
    "primary_link": 0.15, "residential": 0.30, "living_street": 0.35,
    "service": 0.40, "unclassified": 0.45, "road": 0.45, "track": 0.60,
}
POSITIVE_LABEL_SOURCE = "real_confirmed_temporal"
NEGATIVE_LABEL_SOURCE = "real_observed_unaffected"
RAINFALL_COLS = ["rainfall_1day", "rainfall_3day", "rainfall_7day",
                 "rainfall_14day", "rainfall_30day"]


def highway_score(hw: str) -> float:
    return HIGHWAY_SCORES.get(str(hw).lower(), 0.40)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(ds: pd.DataFrame) -> dict:
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
    allowed_sources = {POSITIVE_LABEL_SOURCE, NEGATIVE_LABEL_SOURCE}
    bad = ds[~ds["label_source"].isin(allowed_sources)]
    if len(bad):
        errors.append("unsupported label_source values: " +
                      ", ".join(sorted(bad["label_source"].unique())))
    if errors:
        raise ValueError("production dataset validation failed: " + "; ".join(errors))
    return {"status": "PASS", "checks": [
        "identity_non_null", "sample_key_unique", "binary_labels",
        "rainfall_non_negative", "slope_physical_range",
        "label_sources_admissible",
    ]}


def main() -> int:
    print("=" * 60)
    print("NER-Nav — Production Dataset (admissible labels only)")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Positives (confirmed events at event-1, -3, -7 days)
    # ------------------------------------------------------------------ #
    pos_rows: list[dict] = []
    pos_ids: list[int] = []
    for eid, osm, ref, ed in CONFIRMED_EVENTS:
        E = date.fromisoformat(ed)
        pos_ids.append(osm)
        for off in POSITIVE_OFFSETS:
            pos_rows.append({
                "event_id": eid,
                "osm_id": osm,
                "ref": ref,
                "prediction_time": str(E - timedelta(days=off)),
                "label": 1,
                "sample_kind": "positive",
                "label_source": POSITIVE_LABEL_SOURCE,
            })
    pos_ids_unique = sorted(set(pos_ids))
    print(f"Confirmed roads: {len(pos_ids_unique)}; positive samples: {len(pos_rows)}")

    # ------------------------------------------------------------------ #
    # Observation-backed negatives (source-verified, full-horizon coverage)
    # ------------------------------------------------------------------ #
    obs_rows: list[dict] = []
    obs_osmids: set[int] = set()
    if NEGATIVE_OBSERVATIONS_FILE.exists():
        obs = pd.read_csv(
            NEGATIVE_OBSERVATIONS_FILE,
            dtype={"osm_id": "int64", "ref": str, "prediction_time": str},
        )
        for _, r in obs.iterrows():
            oid = int(r["osm_id"])
            obs_rows.append({
                "event_id": r["observation_id"],
                "osm_id": oid,
                "ref": r["ref"],
                "prediction_time": str(r["prediction_time"]),
                "label": 0,
                "sample_kind": "observed_open_negative",
                "label_source": NEGATIVE_LABEL_SOURCE,
            })
            obs_osmids.add(oid)
    print(f"Observed-open negatives: {len(obs_rows)} ({len(obs_osmids)} unique roads)")

    samples = pd.DataFrame(pos_rows + obs_rows)
    samples["prediction_time"] = pd.to_datetime(samples["prediction_time"])
    samples["osm_id"] = samples["osm_id"].astype("int64")

    # ------------------------------------------------------------------ #
    # Load features
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

    identity = pd.read_parquet(ROAD_IDENTITY_FILE)
    identity["osm_id"] = pd.to_numeric(
        identity["source_osm_id"], errors="coerce").astype("Int64")
    rd = rd.merge(identity[["osm_id", "road_segment_id"]], on="osm_id", how="left")
    rd["road_segment_id"] = rd["road_segment_id"].astype("object")

    ds = samples.merge(rd.drop(columns=["ref"]), on="osm_id", how="left")
    rain = rainfall.rename(columns={"feature_date": "prediction_time"})
    ds = ds.merge(rain, on=["osm_id", "prediction_time"], how="left")

    ds["highway_prior"] = ds["highway"].map(highway_score)
    ds["bridge_flag"] = (ds["bridge"].fillna("").astype(str).str.lower() != "no").astype(int)

    _dates = [d.date() for d in ds["prediction_time"]]
    ds["season"] = [season(d) for d in _dates]
    ds["month"] = [month_feature(d) for d in _dates]
    ds["seasonal_sin"] = [seasonal_sin(d) for d in _dates]
    ds["seasonal_cos"] = [seasonal_cos(d) for d in _dates]
    ds["monsoon_active"] = [monsoon_active(d) for d in _dates]
    ds["pre_monsoon"] = [pre_monsoon_active(d) for d in _dates]
    ds["days_into_monsoon"] = [days_into_monsoon(d) for d in _dates]

    _eps = 0.01
    ds["rain_intensity_1_vs_7"] = ds["rainfall_1day"] / ds["rainfall_7day"].clip(lower=_eps)
    ds["rain_intensity_3_vs_14"] = ds["rainfall_3day"] / ds["rainfall_14day"].clip(lower=_eps)
    ds["rain_concentration_3_in_7"] = ds["rainfall_3day"] / ds["rainfall_7day"].clip(lower=_eps)
    ds["rain_trend_1_vs_3"] = ds["rainfall_1day"] / ds["rainfall_3day"].clip(lower=_eps)
    ds["rain_cumul_ratio_7_vs_30"] = ds["rainfall_7day"] / ds["rainfall_30day"].clip(lower=_eps)
    ds["terrain_known"] = ds["elevation_m"].notna().astype(int)

    _slope = ds["slope_degrees"].fillna(0)
    _elev = ds["elevation_m"].fillna(0)
    ds["slope_x_rain7"] = _slope * ds["rainfall_7day"].fillna(0)
    ds["slope_x_rain30"] = _slope * ds["rainfall_30day"].fillna(0)
    ds["elev_x_slope"] = _elev * _slope

    feature_cols = [
        "event_id", "osm_id", "road_segment_id", "ref", "highway",
        "district", "state", "prediction_time", "label", "sample_kind",
        "label_source",
        "elevation_m", "slope_degrees",
    ] + RAINFALL_COLS + [
        "rainfall_days_available", "highway_prior", "bridge_flag",
        "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
        "rain_concentration_3_in_7", "rain_trend_1_vs_3",
        "rain_cumul_ratio_7_vs_30", "terrain_known",
        "slope_x_rain7", "slope_x_rain30", "elev_x_slope",
    ] + SEASONAL_FEATURES
    ds = ds[feature_cols].copy()
    validation = validate_dataset(ds)

    # ------------------------------------------------------------------ #
    # QA
    # ------------------------------------------------------------------ #
    pos_count = int((ds["label"] == 1).sum())
    neg_count = int((ds["label"] == 0).sum())
    rain_cov = ds[RAINFALL_COLS].notna().mean().to_dict()
    rain_cov = {k: round(float(v), 4) for k, v in rain_cov.items()}

    qa = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "design": (
            "production dataset: positives from "
            f"{len(CONFIRMED_EVENTS)} confirmed events at event-1/-3/-7; "
            f"negatives from {len(obs_rows)} source-backed observations "
            "(negative_observations.csv); no assumed-unaffected or "
            "same-road-offset negatives."
        ),
        "n_samples": len(ds),
        "n_positive_labels": pos_count,
        "n_negative_labels": neg_count,
        "n_observed_open_negatives": len(obs_rows),
        "positive_ids": sorted(pos_ids_unique),
        "rainfall_lookback_days": 30,
        "rainfall_coverage": rain_cov,
        "terrain_coverage_pos": round(float(
            ds.loc[ds["label"] == 1, "slope_degrees"].notna().mean()), 4),
        "negative_label_sources": sorted(
            ds.loc[ds["label"] == 0, "label_source"].dropna().unique().tolist()),
        "positive_label_sources": sorted(
            ds.loc[ds["label"] == 1, "label_source"].dropna().unique().tolist()),
        "features": feature_cols,
        "validation": validation,
        "input_lineage": {
            "roads_sha256": sha256_file(ROAD_FILE),
            "terrain_sha256": sha256_file(TERRAIN_FILE),
            "rainfall_sha256": sha256_file(RAINFALL_FILE),
            "negative_observations_sha256": sha256_file(NEGATIVE_OBSERVATIONS_FILE),
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(OUTPUT_FILE, index=False)
    qa["dataset_sha256"] = sha256_file(OUTPUT_FILE)
    QA_FILE.write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print(f"Positive samples: {pos_count}")
    print(f"Negative samples: {neg_count}")
    print(f"Total:            {len(ds)}")
    print(f"Rainfall coverage: {rain_cov}")
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Wrote {QA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
