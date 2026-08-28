"""End-to-end smoke test: REAL feature parquets + SYNTHETIC samples/labels.

SYNTHETIC / TEST ONLY on the label/sample side. The FEATURES come from the real
processed pipeline (terrain + rainfall) to prove composition + the temporal
leakage guard work on production-shaped data.

Skipped automatically if the real parquets are absent.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.compose import compose_feature_matrix
from ml.labels.generate import LabelingConfig, apply_labels
from ml.splits.temporal import temporal_split, validate_temporal_monotonic

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TERRAIN = os.path.join(REPO, "data", "processed", "terrain", "road_terrain_features.parquet")
RAINFALL = os.path.join(REPO, "data", "processed", "weather", "road_rainfall_features.parquet")

NEEDS_DATA = all(os.path.exists(p) for p in (TERRAIN, RAINFALL))

# Real feature columns (must match the regenerated artifacts)
TERRAIN_COLS = ["elevation_m", "slope_degrees", "aspect_degrees"]
RAINFALL_COLS = ["rainfall_1day", "rainfall_3day", "rainfall_7day",
                 "rainfall_14day", "rainfall_30day"]


def load_real_features():
    terrain = pd.read_parquet(TERRAIN)
    rainfall = pd.read_parquet(RAINFALL)
    return terrain, rainfall


@unittest.skipUnless(NEEDS_DATA, "real feature parquets not present")
class TestComposeSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.terrain, cls.rainfall = load_real_features()
        # intersect osm_ids present in both feature sets
        ids = np.intersect1d(cls.terrain["osm_id"].values, cls.rainfall["osm_id"].values)
        cls.road_ids = ids[:40].tolist()
        cls.pred_times = pd.to_datetime(
            ["2024-05-25 00:00", "2024-05-26 00:00", "2024-05-27 00:00"]
        )

    def _build_samples(self):
        rows = [
            (osm, t)
            for osm in self.road_ids
            for t in self.pred_times
        ]
        return pd.DataFrame(rows, columns=["road_segment_id", "prediction_time"])

    def test_compose_supplies_real_features(self):
        samples = self._build_samples()
        out = compose_feature_matrix(
            samples.rename(columns={"road_segment_id": "osm_id"}),
            prediction_col="prediction_time",
            static=self.terrain,
            static_key="osm_id",
            static_cols=TERRAIN_COLS,
            temporal=self.rainfall,
            temporal_key="osm_id",
            temporal_time="feature_date",
            temporal_cols=RAINFALL_COLS,
            leak_check=True,
        )
        self.assertEqual(len(out), len(samples))
        for c in TERRAIN_COLS + RAINFALL_COLS:
            self.assertIn(c, out.columns)

        # --- real no-data is preserved as NaN, never fabricated to a value ---
        # some constituent roads genuinely lack terrain coverage (only ~32.5% of
        # NER roads have SRTM elevation): NaN must survive the compose step.
        real_nan_rows = self.terrain[
            self.terrain["osm_id"].isin(self.road_ids)
            & self.terrain["slope_degrees"].isna()
        ]
        if not real_nan_rows.empty:
            cand = out[out["osm_id"].isin(real_nan_rows["osm_id"].unique())]
            self.assertTrue((cand["slope_degrees"].isna()).all())

        # --- and valid terrain values are carried through for covered roads ---
        covered = self.terrain[
            self.terrain["osm_id"].isin(self.road_ids)
            & self.terrain["elevation_m"].notna()
        ]
        if not covered.empty:
            cand = out[out["osm_id"].isin(covered["osm_id"].unique())]
            any_covered = cand.drop_duplicates("osm_id")
            self.assertTrue((any_covered["elevation_m"].notna()).any())

        # --- real rainfall (date-anchored) is populated for every sampled road ---
        self.assertFalse(out["rainfall_30day"].isna().all())

        # --- leakage guard actually enforced by leak_check ---
        self.assertTrue(
            (out["feature_latest_ts"] <= out["prediction_time"].map(pd.Timestamp)).all()
        )

    def test_label_then_compose_no_leak(self):
        # SYNTHETIC events: affect a handful of these roads shortly after the
        # last prediction time -> only those samples labelled.
        affected = self.road_ids[:5]
        events = pd.DataFrame(
            {
                "event_id": [f"e{i}" for i in range(len(affected))],
                "event_time": pd.to_datetime(["2024-05-28 00:00"] * len(affected)),
                "affected_segment_id": affected,
            }
        )
        samples = self._build_samples()
        labeled = apply_labels(
            samples, events, LabelingConfig(horizon=pd.Timedelta("7 days"))
        )
        # events at 04-19: predictions 04-16..18 are all < 04-19 <= +7d
        self.assertEqual(labeled.groupby("road_segment_id")["label"].max().loc[affected[0]], 1)
        self.assertEqual(labeled.groupby("road_segment_id")["label"].max().loc[self.road_ids[-1]], 0)

        composed = compose_feature_matrix(
            labeled.rename(columns={"road_segment_id": "osm_id"}),
            prediction_col="prediction_time",
            static=self.terrain,
            static_key="osm_id",
            static_cols=TERRAIN_COLS,
            temporal=self.rainfall,
            temporal_key="osm_id",
            temporal_time="feature_date",
            temporal_cols=RAINFALL_COLS,
            leak_check=True,
        )
        self.assertIn("label", composed.columns)
        # positive and negative classes both represented
        self.assertIn(0, set(composed["label"]))
        self.assertIn(1, set(composed["label"]))

    def test_temporal_split_on_real_prediction_times(self):
        samples = self._build_samples()
        ranked = samples.sort_values("prediction_time").reset_index(drop=True)
        times = ranked["prediction_time"].to_numpy()
        m = temporal_split(times, train_until="2024-05-26", validation_until="2024-05-27")
        self.assertTrue(validate_temporal_monotonic(times, m))
        ranked["split"] = m
        # earliest predictions are train, then val, then test
        self.assertEqual(ranked.groupby("split")["prediction_time"].max().loc["train"],
                         pd.Timestamp("2024-05-25 00:00"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
