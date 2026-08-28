"""Tests for production inference validation, freshness, OOD, and lineage."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestProductionInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inference = load_module(
            "production_inference", "scripts/predict_real_temporal.py")
        cls.training = load_module(
            "production_training", "scripts/train_production_risk_model.py")

    def valid_values(self):
        return {
            "rainfall_1day": 10,
            "rainfall_3day": 25,
            "rainfall_7day": 50,
            "rainfall_14day": 80,
            "rainfall_30day": 120,
            "elevation_m": 1000,
            "slope_degrees": 20,
        }

    def test_derived_features_are_computed_not_trusted(self):
        values = self.valid_values()
        values["slope_x_rain7"] = -999999
        prepared, quality = self.inference.validate_and_prepare(
            values, self.training.FEATURES, "2026-08-27T12:00:00+05:30",
            "2026-08-27T00:00:00+05:30")
        self.assertEqual(prepared["slope_x_rain7"], 1000)
        self.assertEqual(prepared["terrain_known"], 1)
        self.assertEqual(quality["weather_freshness"]["status"], "FRESH")

    def test_seasonal_features_are_derived_from_prediction_time(self):
        # A monsoon-season date (Jul 15) must be flagged by monsoon features.
        values = self.valid_values()
        prepared, _ = self.inference.validate_and_prepare(
            values, self.training.FEATURES, "2026-07-15T12:00:00+05:30")
        self.assertEqual(prepared["monsoon_active"], 1)
        self.assertEqual(prepared["month"], 7)
        self.assertGreater(prepared["days_into_monsoon"], 0)

    def test_winter_date_seasonal_features(self):
        values = self.valid_values()
        prepared, _ = self.inference.validate_and_prepare(
            values, self.training.FEATURES, "2026-01-15T12:00:00+05:30")
        self.assertEqual(prepared["monsoon_active"], 0)
        self.assertEqual(prepared["month"], 1)
        self.assertLess(prepared["days_into_monsoon"], 0)

    def test_negative_rainfall_is_rejected(self):
        values = self.valid_values()
        values["rainfall_7day"] = -1
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

    def test_malformed_metadata_is_rejected(self):
        values = self.valid_values()
        values["latitude"] = 91
        with self.assertRaisesRegex(ValueError, "latitude"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

        values = self.valid_values()
        values["longitude"] = 181
        with self.assertRaisesRegex(ValueError, "longitude"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

        values = self.valid_values()
        values["highway"] = "teleporter"
        with self.assertRaisesRegex(ValueError, "supported road category"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

        values = self.valid_values()
        values["unexpected"] = 1
        with self.assertRaisesRegex(ValueError, "unknown input fields"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

    def test_extreme_rainfall_is_rejected(self):
        values = self.valid_values()
        values["rainfall_30day"] = 1e12
        with self.assertRaisesRegex(ValueError, "physical maximum"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

        values = self.valid_values()
        values["rainfall_7day"] = {"not": "scalar"}
        with self.assertRaisesRegex(ValueError, "required and must be finite"):
            self.inference.validate_and_prepare(
                values, self.training.FEATURES,
                "2026-08-27T12:00:00+05:30")

    def test_road_segment_id_validation(self):
        with self.assertRaisesRegex(ValueError, "scalar"):
            self.inference.validate_road_segment_id({"bad": "id"})
        self.assertEqual(self.inference.validate_road_segment_id("test_segment_123"),
                         "test_segment_123")

    def test_model_tag_cannot_escape_artifact_directory(self):
        with self.assertRaisesRegex(ValueError, "model tag"):
            self.inference.load_model_bundle("..\\..\\outside")

    def test_future_prediction_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be in the future"):
            self.inference.validate_and_prepare(
                self.valid_values(), self.training.FEATURES,
                "2099-01-01T00:00:00+00:00")

    def test_stale_weather_and_ood_are_flagged(self):
        profile = {
            "rainfall_30day": {"p01": 1.0, "p99": 100.0},
        }
        prepared, quality = self.inference.validate_and_prepare(
            self.valid_values(), self.training.FEATURES,
            "2026-08-27T12:00:00+05:30",
            "2026-08-19T00:00:00+05:30", profile)
        self.assertEqual(prepared["rainfall_30day"], 120)
        self.assertIn("WEATHER_EXPIRED", quality["confidence_flags"])
        self.assertIn("OUT_OF_DISTRIBUTION", quality["confidence_flags"])
        self.assertTrue(quality["low_confidence"])

    def test_latest_bundle_loads_and_hashes_verify(self):
        bundle = self.inference.load_model_bundle("latest")
        self.assertEqual(bundle["tag"], bundle["report"]["model_version"])
        self.assertIn("training_feature_profile", (
            Path("data/models")
            / f"prod_real_temporal_{bundle['tag']}_features.json"
        ).read_text(encoding="utf-8"))

    def test_legacy_demo_artifact_is_rejected(self):
        legacy = self.inference.MODEL_DIR / "prod_report_2026-08-28_133546_b7486dea.json"
        if not legacy.exists():
            self.skipTest("historical demo artifact absent")
        with self.assertRaises(ValueError):
            self.inference.load_model_bundle("2026-08-28_133546_b7486dea")

    def test_calibration_matches_saved_float32_semantics(self):
        bundle = self.inference.load_model_bundle("latest")
        calibration = bundle["calibration"]
        raw_values = [0.08598612248897552, 0.4244638681411743,
                      0.6244120597839355]
        x = np.asarray(calibration["x_thresholds"], dtype=np.float32)
        y = np.asarray(calibration["y_thresholds"], dtype=np.float32)
        fn = self.inference.interpolate.interp1d(
            x, y, kind="linear", bounds_error=False,
            fill_value=(y[0], y[-1]))
        for raw in raw_values:
            value = np.asarray([raw], dtype=x.dtype)
            value = np.clip(value, x[0], x[-1])
            want = float(fn(value)[0])
            self.assertEqual(self.inference._calibrate(raw, calibration), want)


class TestReadinessAudit(unittest.TestCase):
    def test_active_dataset_has_observation_backed_negatives(self):
        audit = load_module(
            "production_audit", "scripts/audit_ml_production_readiness.py")
        ds = pd.read_parquet(audit.DATASET)
        result = audit.audit_dataset(ds)
        self.assertEqual(result["labels"]["assumed_negative_count"], 0)
        self.assertEqual(result["labels"]["confirmed_negative_count"], 24)
        self.assertEqual(result["quality"]["duplicate_sample_keys"], 0)


if __name__ == "__main__":
    unittest.main()
