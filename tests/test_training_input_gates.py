"""Regression tests for fail-closed real-data training gates."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from ml.data.training_gate import (
    validate_event_registry,
    validate_negative_observation_file,
    validate_negative_observation_quality,
    validate_negative_sources,
)


class TestTrainingInputGates(unittest.TestCase):
    def test_current_event_registry_is_complete_and_verified(self):
        result = validate_event_registry()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["design_event_count"], 32)
        self.assertEqual(result["verified_event_count"], 32)
        self.assertFalse(result["errors"])

    def test_missing_negative_observation_file_is_a_data_gap(self):
        result = validate_negative_observation_file(
            Path("this-file-does-not-exist.csv"))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("missing", result["errors"][0])

    def test_assumed_and_temporal_controls_are_rejected(self):
        frame = pd.DataFrame({
            "label": [0, 0, 1],
            "label_source": ["assumed_unaffected_real_pool",
                             "real_same_road_control",
                             "real_confirmed_temporal"],
        })
        result = validate_negative_sources(frame)
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["assumed_unaffected_count"], 1)

    def test_source_backed_negative_is_accepted_by_label_gate(self):
        frame = pd.DataFrame({
            "label": [0],
            "label_source": ["source_confirmed_unaffected"],
        })
        self.assertEqual(validate_negative_sources(frame)["status"], "PASS")

    def test_current_controls_are_not_production_segment_negatives(self):
        result = validate_negative_observation_quality()
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["segment_level_rows"], 0)
        self.assertEqual(result["corridor_only_rows"], 24)
        self.assertGreater(result["clearance_or_reopening_rows"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
