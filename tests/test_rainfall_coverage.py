"""Real-data rainfall-coverage integrity guard.

Guards against the pipeline regression where feature extraction wrote to a
different rainfall parquet than the dataset builder consumed (silently
dropping rainfall coverage to ~70%). Iterates over the real dataset's positive
samples (bound to confirmed events) and asserts every required rainfall window
is present. Skipped if the real temporal dataset is absent.
"""

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, Path(__file__).resolve().parent.parent)

DATASET = Path("data/processed/ml/real_temporal_risk_dataset.parquet")
QA = Path("data/processed/ml/real_temporal_risk_dataset_qa.json")

RAINFALL_WINDOWS = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day",
    "rainfall_14day", "rainfall_30day",
]


@unittest.skipUnless(DATASET.exists(), "real temporal dataset absent")
@unittest.skipUnless(QA.exists(), "real temporal dataset QA absent")
class TestRainfallCoverageIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = pd.read_parquet(DATASET)
        with QA.open(encoding="utf-8") as f:
            cls.qa = json.load(f)

    def test_positive_samples_have_complete_rainfall(self):
        pos = self.ds[self.ds["label"] == 1]
        self.assertGreater(len(pos), 0)
        for col in RAINFALL_WINDOWS:
            self.assertEqual(
                pos[col].notna().sum(), len(pos),
                f"positive-sample rainfall missing for {col}; "
                "run scripts/rebuild_pipeline.py to regenerate coverage")

    def test_all_samples_rainfall_coverage_is_full(self):
        for col in RAINFALL_WINDOWS:
            self.assertEqual(
                self.ds[col].notna().mean(), 1.0,
                f"dataset rainfall coverage for {col} < 100%")

    def test_qa_report_coverage_is_full(self):
        cov = self.qa.get("rainfall_coverage", {})
        for col in RAINFALL_WINDOWS:
            self.assertAlmostEqual(
                cov.get(col), 1.0,
                msg=f"QA rainfall coverage for {col} != 1.0")

    def test_every_confirmed_event_has_rainfall_at_its_anchors(self):
        pos = self.ds[self.ds["label"] == 1]
        missing_events = pos.loc[
            pos[RAINFALL_WINDOWS].isna().any(axis=1), "event_id"].unique()
        self.assertEqual(
            missing_events.tolist(), [],
            f"confirmed events lacking rainfall at anchors: {missing_events}")


if __name__ == "__main__":
    unittest.main()
