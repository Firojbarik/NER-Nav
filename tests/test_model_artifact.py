"""Unit tests for the saved production model artifact (round-trip).

Guards that the persisted model + frozen feature spec load and reproduce the
same predictions. Skipped if the artifacts are absent.
"""

import json
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = Path("data/models")
DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
LATEST = MODEL_DIR / "prod_latest.json"
TAG = (json.loads(LATEST.read_text(encoding="utf-8"))["model_version"]
       if LATEST.exists() else "missing")

ARTIFACTS = [
    f"prod_real_temporal_{TAG}.ubj",
    f"prod_real_temporal_{TAG}.json",
    f"prod_real_temporal_{TAG}_features.json",
    f"prod_report_{TAG}.json",
]


@unittest.skipUnless(DATASET.exists(), "real temporal dataset absent")
@unittest.skipUnless(all((MODEL_DIR / a).exists() for a in ARTIFACTS),
                     "saved model artifacts absent")
class TestModelArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = pd.read_parquet(DATASET)
        with open(MODEL_DIR / f"prod_real_temporal_{TAG}_features.json", encoding="utf-8") as f:
            meta = json.load(f)
        cls.frozen_features = meta["features"]
        cls.frozen_numeric = meta.get("numeric", meta["features"])
        cls.X = ds[cls.frozen_features].copy()
        for c in cls.frozen_numeric:
            cls.X[c] = pd.to_numeric(cls.X[c], errors="coerce")

    def test_frozen_feature_list_matches_report(self):
        with open(MODEL_DIR / f"prod_report_{TAG}.json", encoding="utf-8") as f:
            report = json.load(f)
        self.assertEqual(self.frozen_features, report["features"])

    def test_reloaded_model_ubj_json_identical(self):
        import xgboost as xgb
        m_ubj = xgb.XGBClassifier()
        m_ubj.load_model(MODEL_DIR / f"prod_real_temporal_{TAG}.ubj")
        p_ubj = m_ubj.predict_proba(self.X)[:, 1]

        m_json = xgb.XGBClassifier()
        m_json.load_model(MODEL_DIR / f"prod_real_temporal_{TAG}.json")
        p_json = m_json.predict_proba(self.X)[:, 1]

        np.testing.assert_allclose(p_ubj, p_json, atol=1e-9)
        self.assertTrue(((p_ubj >= 0.0) & (p_ubj <= 1.0)).all())
        self.assertTrue(p_ubj.std() > 0)

    def test_calibration_artifact_loads(self):
        calib_path = MODEL_DIR / f"prod_real_temporal_{TAG}_calib.json"
        self.assertTrue(calib_path.exists(), "calibration artifact missing")
        calibrator = json.loads(calib_path.read_text(encoding="utf-8"))
        raw = np.array([0.3, 0.5, 0.7, 0.9])
        cal = np.interp(raw, calibrator["x_thresholds"], calibrator["y_thresholds"])
        self.assertEqual(len(cal), 4)
        self.assertTrue((cal >= 0.0).all() and (cal <= 1.0).all())

    def test_report_hashes_match_artifacts(self):
        import hashlib
        report = json.loads(
            (MODEL_DIR / f"prod_report_{TAG}.json").read_text(encoding="utf-8"))
        model_path = MODEL_DIR / f"prod_real_temporal_{TAG}.ubj"
        feature_path = MODEL_DIR / f"prod_real_temporal_{TAG}_features.json"
        self.assertEqual(hashlib.sha256(model_path.read_bytes()).hexdigest(),
                         report["model_sha256"])
        self.assertEqual(hashlib.sha256(feature_path.read_bytes()).hexdigest(),
                         report["feature_spec_sha256"])

    def test_report_status_field(self):
        with open(MODEL_DIR / f"prod_report_{TAG}.json", encoding="utf-8") as f:
            report = json.load(f)
        self.assertIn(report["status"],
                      ["PRODUCTION_READY_EVIDENCE_SUPPORTED",
                       "HACKATHON_DEMO_READY_LIMITED_CONFIDENCE",
                       "GATED_LOW_CONFIDENCE"])
        self.assertIn("test_metrics", report)
        self.assertIn("split", report)
        self.assertIn("demo", report["gates"])
        self.assertIn("production", report["gates"])
        if report["split"]["n_test_events"] < 30:
            self.assertFalse(report["production_ready"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
