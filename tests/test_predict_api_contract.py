"""Tests for the backend /predict API contract (freshness, explanations,
risk bands, threshold provenance, and timestamp integrity)."""

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from backend.app.api.routes import predict  # noqa: E402
from backend.app.schemas.predict import PredictRequest  # noqa: E402
from backend.app.services import risk_model  # noqa: E402
from backend.app.services.features import derive_features  # noqa: E402

TAG = (json.loads(Path("data/models/demo_latest.json").read_text(encoding="utf-8"))["model_version"]
       if Path("data/models/demo_latest.json").exists() else "2026-09-20_002229_48b0a9ff")
MODEL_DIR = Path("data/models")


def valid_features():
    return {
        "rainfall_1day": 21.12, "rainfall_3day": 21.12,
        "rainfall_7day": 21.12, "rainfall_14day": 221.88,
        "rainfall_30day": 384.94,
        "elevation_m": 1863.0, "slope_degrees": 24.15,
    }


@unittest.skipUnless((MODEL_DIR / f"prod_report_{TAG}.json").exists(),
                     "demo model artifacts absent")
class TestPredictApiContract(unittest.TestCase):
    def test_predict_includes_explanation_disclaimer(self):
        response = predict(PredictRequest(
            road_segment_id="662679409",
            ref="NH206",
            features=valid_features(),
        ))
        self.assertIsNotNone(response.explanation_disclaimer)
        self.assertIn("not causal", response.explanation_disclaimer)

    def test_predict_always_flags_unknown_weather_freshness(self):
        # The API request has no weather_observed_at input, so weather
        # freshness can never be verified there: it must stay UNKNOWN even
        # when a prediction_timestamp is supplied.
        response = predict(PredictRequest(
            road_segment_id="662679409",
            features=valid_features(),
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.assertIn("WEATHER_FRESHNESS_UNKNOWN",
                      response.data_quality["confidence_flags"])

    def test_predict_rejects_future_timestamp(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        with self.assertRaises(Exception):
            predict(PredictRequest(
                road_segment_id="662679409",
                features=valid_features(),
                prediction_timestamp=future,
            ))

    def test_predict_rejects_naive_timestamp(self):
        with self.assertRaises(Exception):
            predict(PredictRequest(
                road_segment_id="662679409",
                features=valid_features(),
                prediction_timestamp="2026-08-28T17:00:00",
            ))

    def test_predict_threshold_matches_report(self):
        report = json.loads(
            (MODEL_DIR / f"prod_report_{TAG}.json").read_text(encoding="utf-8"))
        response = predict(PredictRequest(
            road_segment_id="662679409",
            features=valid_features(),
        ))
        self.assertEqual(response.operating_threshold,
                         float(report["test_threshold"]))
        self.assertEqual(response.operating_threshold,
                         risk_model.cached_bundle(TAG)["threshold"])

    def test_predict_decision_from_threshold(self):
        response = predict(PredictRequest(
            road_segment_id="662679409",
            features=valid_features(),
        ))
        expected = ("ALERT" if response.calibrated_probability >= response.operating_threshold
                    else "NO_ALERT")
        self.assertEqual(response.operating_decision, expected)

    def test_predict_probabilities_in_unit_interval(self):
        response = predict(PredictRequest(
            road_segment_id="662679409",
            features=valid_features(),
        ))
        for attr in ("disruption_probability", "raw_probability",
                     "calibrated_probability"):
            self.assertGreaterEqual(getattr(response, attr), 0.0)
            self.assertLessEqual(getattr(response, attr), 1.0)

    def test_predict_model_status_matches_report(self):
        report = json.loads(
            (MODEL_DIR / f"prod_report_{TAG}.json").read_text(encoding="utf-8"))
        response = predict(PredictRequest(
            road_segment_id="662679409",
            features=valid_features(),
        ))
        self.assertEqual(response.model.status, report["status"])
        self.assertEqual(response.model.status,
                         "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE")


@unittest.skipUnless((MODEL_DIR / f"prod_report_{TAG}.json").exists(),
                     "demo model artifacts absent")
class TestBackendRiskBands(unittest.TestCase):
    def test_risk_level_exact_rule(self):
        policy = {"low_below": 0.4, "high_at_or_above": 0.7}
        self.assertEqual(risk_model.risk_level(0.39, policy), "LOW")
        self.assertEqual(risk_model.risk_level(0.4, policy), "MEDIUM")
        self.assertEqual(risk_model.risk_level(0.69, policy), "MEDIUM")
        self.assertEqual(risk_model.risk_level(0.7, policy), "HIGH")
        self.assertEqual(risk_model.risk_level(0.9, policy), "HIGH")

    def test_derive_features_rejects_future_date(self):
        future = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=30)).isoformat()
        with self.assertRaisesRegex(ValueError, "cannot be in the future"):
            derive_features(valid_features(), future)


if __name__ == "__main__":
    unittest.main(verbosity=2)