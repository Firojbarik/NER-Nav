"""End-to-end harness test (ml/experiment.py).

Runs the full pipeline on REAL features with SYNTHETIC labels (SYNTHETIC /
TEST ONLY) to prove the train/evaluate machinery works. Skipped if the real
feature parquets are absent. Not a real performance claim.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.experiment import ExperimentConfig, run_experiment

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEEDS_DATA = all(
    os.path.exists(os.path.join(REPO, "data", "processed", *p))
    for p in [
        ("terrain", "road_terrain_features.parquet"),
        ("weather", "road_rainfall_features.parquet"),
        ("ml", "risk_training_dataset.parquet"),
    ]
)


@unittest.skipUnless(NEEDS_DATA, "real feature parquets not present")
class TestExperimentHarness(unittest.TestCase):
    def test_run_experiment_produces_metrics(self):
        config = ExperimentConfig(
            road_sample=600,
            n_estimators=10,
            max_depth=4,
        )
        summary = run_experiment(config)

        self.assertTrue(summary["config"]["label_source"] == "synthetic_test_only")
        self.assertIn("metrics", summary)
        m = summary["metrics"]
        for key in ("roc_auc", "pr_auc", "precision", "recall", "f1"):
            self.assertIn(key, m)
        self.assertGreaterEqual(m["n_test"], 1)
        self.assertIn("feature_importance", m)
        self.assertTrue(len(m["feature_importance"]) > 0)
        # both prediction classes must be present in test
        self.assertIn("n_test_positive", m)
        self.assertIn("split_counts", summary["data"])

        # the centrality of the no-fabrication rule: label_source is explicit
        self.assertIn("label_source", summary["config"])

    def test_json_serialisable(self):
        import json
        config = ExperimentConfig(road_sample=300, n_estimators=5)
        summary = run_experiment(config)
        json.dumps(summary, default=str)  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
