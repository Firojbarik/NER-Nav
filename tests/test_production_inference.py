"""Tests for production inference validation, freshness, OOD, and lineage."""

import importlib.util
import unittest
from pathlib import Path

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


class TestReadinessAudit(unittest.TestCase):
    def test_active_dataset_exposes_negative_label_gap(self):
        audit = load_module(
            "production_audit", "scripts/audit_ml_production_readiness.py")
        ds = pd.read_parquet(audit.DATASET)
        result = audit.audit_dataset(ds)
        self.assertGreater(result["labels"]["assumed_negative_count"], 0)
        self.assertEqual(result["labels"]["confirmed_negative_count"], 0)
        self.assertEqual(result["quality"]["duplicate_sample_keys"], 0)


if __name__ == "__main__":
    unittest.main()
