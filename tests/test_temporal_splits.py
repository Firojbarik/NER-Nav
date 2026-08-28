"""Unit tests for ml/splits/temporal.py.

SYNTHETIC / TEST ONLY: timestamps/groups are hand-built to exercise the
mechanics, never real labels.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.splits.temporal import (
    TEST,
    TRAIN,
    VALIDATION,
    assert_features_not_future,
    geographic_split,
    temporal_split,
    validate_temporal_monotonic,
)


class TestTemporalSplit(unittest.TestCase):
    def test_basic_chronological_order(self):
        t = np.array(
            ["2020-01-01", "2020-01-11", "2020-01-21"], dtype="datetime64[D]"
        )
        m = temporal_split(t, train_until="2020-01-10", validation_until="2020-01-20")
        self.assertEqual(m[0], TRAIN)
        self.assertEqual(m[1], VALIDATION)
        self.assertEqual(m[2], TEST)

    def test_gap_period_excluded(self):
        t = np.array(
            ["2020-01-01", "2020-01-11", "2020-01-16", "2020-01-21"],
            dtype="datetime64[D]",
        )
        m = temporal_split(
            t,
            train_until="2020-01-10",
            validation_until="2020-01-15",
            gap_after_validation="2020-01-20",
        )
        self.assertEqual(list(m), [TRAIN, VALIDATION, "gap", TEST])

    def test_monotonic_valid(self):
        rng = np.random.default_rng(0)
        t = np.arange("2020-01-01", "2021-01-01", dtype="datetime64[D]")[::7]
        m = temporal_split(t, train_until="2020-04-01", validation_until="2020-08-01")
        self.assertTrue(validate_temporal_monotonic(t, m))

    def test_reversed_order_raises(self):
        t = np.array(
            ["2020-01-01", "2020-01-11", "2020-01-21"], dtype="datetime64[D]"
        )
        # manually craft a reversed membership
        m = np.array([TEST, VALIDATION, TRAIN], dtype=object)
        with self.assertRaises(ValueError):
            validate_temporal_monotonic(t, m)

    def test_unknown_label_raises(self):
        t = np.array(["2020-01-01"], dtype="datetime64[D]")
        m = np.array(["bogus"], dtype=object)
        with self.assertRaises(ValueError):
            validate_temporal_monotonic(t, m)

    def test_requires_datetime(self):
        with self.assertRaises(TypeError):
            temporal_split(np.array([1, 2, 3]), "2020-01-01")


class TestGeographicSplit(unittest.TestCase):
    def test_disjoint_groups(self):
        groups = np.array(["AS", "ML", "MZ", "NL", "TR"])
        m = geographic_split(groups, train_groups=["AS", "ML"], test_groups=["MZ"])
        self.assertTrue(np.all(m[np.isin(groups, ["AS", "ML"])] == TRAIN))
        self.assertTrue(np.all(m[groups == "MZ"] == TEST))
        # NL, TR not nominated -> conservative HOLDOUT (excluded from both)
        self.assertTrue(np.all(np.isin(m[np.isin(groups, ["NL", "TR"])], "holdout")))

    def test_train_test_overlap_raises(self):
        groups = np.array(["AS", "ML"])
        with self.assertRaises(ValueError):
            geographic_split(groups, train_groups=["AS"], test_groups=["AS"])

    def test_holdout_excluded(self):
        groups = np.array(["AS", "NL"])
        m = geographic_split(groups, train_groups=["AS"], test_groups=[], holdout="NL")
        self.assertEqual(m[0], TRAIN)
        self.assertEqual(m[1], "holdout")


class TestNoFutureGuard(unittest.TestCase):
    def test_passes_when_not_future(self):
        pred = np.array(["2020-01-10", "2020-01-11"], dtype="datetime64[D]")
        feats = np.array(["2020-01-09", "2020-01-11"], dtype="datetime64[D]")
        self.assertTrue(assert_features_not_future(pred, feats, ["r"]))

    def test_raises_when_future(self):
        pred = np.array(["2020-01-10"], dtype="datetime64[D]")
        feats = np.array(["2020-01-12"], dtype="datetime64[D]")
        with self.assertRaises(ValueError):
            assert_features_not_future(pred, feats, ["r"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
