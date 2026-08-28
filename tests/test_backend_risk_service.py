"""Tests for the backend demo prediction service (risk_model + features)."""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from backend.app.services.risk_model import (
    _calibrate,
    cached_bundle,
    demo_authorised_tags,
    load_bundle,
    risk_level,
)
from backend.app.services.features import derive_features

TAG = "2026-08-28_151030_b7486dea"
MODEL_DIR = Path("data/models")


@unittest.skipUnless((MODEL_DIR / f"prod_report_{TAG}.json").exists(),
                     "demo model artifacts absent")
class TestDemoBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = cached_bundle(TAG)

    def test_demo_tag_authorised(self):
        self.assertIn(TAG, demo_authorised_tags())

    def test_bundle_status_is_demo_ready(self):
        self.assertEqual(self.bundle["status"], "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE")

    def test_bundle_features_are_frozen(self):
        self.assertIn("rainfall_1day", self.bundle["features"])
        self.assertIn("elevation_m", self.bundle["features"])

    def test_unauthorised_tag_rejected(self):
        with self.assertRaises(ValueError):
            load_bundle("2026-08-28_222351_3322fca5")

    def test_calibration_interp_in_range(self):
        y = _calibrate(0.3, self.bundle["calibration"])
        self.assertGreaterEqual(y, 0.0)
        self.assertLessEqual(y, 1.0)

    def test_risk_level_bands(self):
        policy = {"low_below": 0.4, "high_at_or_above": 0.7}
        self.assertEqual(risk_level(0.2, policy), "LOW")
        self.assertEqual(risk_level(0.5, policy), "MEDIUM")
        self.assertEqual(risk_level(0.8, policy), "HIGH")

    def test_derive_features_matches_predict_input(self):
        values = {
            "rainfall_1day": 21.12, "rainfall_3day": 21.12,
            "rainfall_7day": 21.12, "rainfall_14day": 221.88,
            "rainfall_30day": 384.94,
            "elevation_m": 1863.0, "slope_degrees": 24.15,
        }
        prepared, flags = derive_features(values, "2026-08-28T17:00:00+05:30")
        self.assertIn("slope_x_rain7", prepared)
        self.assertIn("monsoon_active", prepared)
        self.assertEqual(prepared["terrain_known"], 1)

    def test_derive_features_requires_rainfall(self):
        with self.assertRaises(ValueError):
            derive_features({"elevation_m": 100.0}, "2026-08-28T17:00:00+05:30")

    def test_wet_vs_dry_separation(self):
        wet = self.bundle["model"].predict_proba(np.asarray([[
            40.0, 90.0, 140.0, 180.0, 220.0, 1500.0, 25.0,
            40.0 / max(140, 0.01), 90.0 / max(180, 0.01),
            90.0 / max(140, 0.01), 40.0 / max(90, 0.01),
            140.0 / max(220, 0.01), 1.0, 25.0 * 140.0,
            25.0 * 220.0, 1500.0 * 25.0,
        ]]))[:, 1][0]
        dry = self.bundle["model"].predict_proba(np.asarray([[
            0.0, 0.0, 0.0, 5.0, 20.0, 100.0, 5.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 500.0,
        ]]))[:, 1][0]
        self.assertGreater(wet, dry)


@unittest.skipUnless((Path("data/predictions/demo_risk_feed.json")).exists(),
                     "demo risk feed absent")
class TestRiskFeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO))
        from backend.app.api.routes import risk_feed
        cls.feed = risk_feed()

    def test_feed_has_segments(self):
        self.assertGreaterEqual(len(self.feed.segments), 1)

    def test_feed_uses_demo_model(self):
        self.assertEqual(self.feed.model_version, TAG)

    def test_segment_fields_mapped(self):
        seg = next(s for s in self.feed.segments if s.risk_level in ("HIGH", "MEDIUM", "LOW"))
        self.assertIsInstance(seg.osm_id, int)
        self.assertIsInstance(seg.latitude, float)
        self.assertIsInstance(seg.longitude, float)
        self.assertIsInstance(seg.disruption_probability, float)
        self.assertIn(seg.risk_level, ("LOW", "MEDIUM", "HIGH"))
        self.assertIn(seg.operating_decision, ("ALERT", "NO_ALERT"))
        self.assertEqual(seg.rainfall.one_day >= 0, True)

    def test_high_risk_segments_flagged(self):
        high = [s for s in self.feed.segments if s.risk_level == "HIGH"]
        self.assertGreaterEqual(len(high), 1)
        for s in high:
            self.assertEqual(s.operating_decision, "ALERT")


if __name__ == "__main__":
    unittest.main(verbosity=2)