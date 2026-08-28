"""Unit tests for ml/features/missingness.py.

SYNTHETIC / TEST ONLY frames - only the mechanics are exercised; none of these
become real training data.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.missingness import (
    MissingnessStrategy,
    apply_missingness_strategy,
    assess_missingness,
    coverage_by_group,
    report_retention,
    testable_groups,
)


def mk_frame():
    # state A: all observed; state B: all missing (systematic -> structural gap)
    return pd.DataFrame(
        {
            "state": ["A", "A", "B", "B", "B"],
            "elev": [100.0, 200.0, np.nan, np.nan, np.nan],
            "slope": [1.0, 2.0, np.nan, np.nan, np.nan],
        }
    )


class TestAssess(unittest.TestCase):
    def test_assess_missingness_rates(self):
        df = mk_frame()
        rep = assess_missingness(df, ["elev", "slope"])
        self.assertEqual(rep.loc[rep.feature == "elev", "missing_rate"].iloc[0], 0.6)

    def test_coverage_groups_structural(self):
        df = mk_frame()
        rep = coverage_by_group(df, "state", ["elev"])
        b = rep[rep.group == "B"].iloc[0]
        self.assertEqual(b["observed_rate"], 0.0)
        a = rep[rep.group == "A"].iloc[0]
        self.assertEqual(a["observed_rate"], 1.0)

    def test_testable_groups_excludes_missing_state(self):
        df = mk_frame()
        testable, _ = testable_groups(df, "state", ["elev", "slope"])
        self.assertEqual(testable, ["A"])


class TestStrategies(unittest.TestCase):
    def test_XGB_native_keeps_nan(self):
        df = mk_frame()
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.XGB_NATIVE)
        self.assertEqual(len(out), len(df))
        self.assertTrue(out["elev"].isna().any())

    def test_drop_any_loses_missing_state(self):
        df = mk_frame()
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.DROP_ANY)
        self.assertEqual(list(out["state"]), ["A", "A"])

    def test_drop_all_drops_only_fully_missing(self):
        df = pd.DataFrame({"state": ["A", "B"], "elev": [5.0, np.nan],
                           "slope": [np.nan, np.nan]})
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.DROP_ALL)
        # A has one observed -> kept; B all-missing -> dropped
        self.assertEqual(list(out["state"]), ["A"])

    def test_flag_adds_indicators(self):
        df = mk_frame()
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.FLAG)
        self.assertIn("is_missing_elev", out.columns)
        self.assertEqual(out["is_missing_elev"].sum(), 3)

    def test_impute_in_group_leaves_missing_group_nan(self):
        # state A: one NaN (imputable within A via A mean); state B fully missing
        df = pd.DataFrame(
            {
                "state": ["A", "A", "B", "B"],
                "elev": [100.0, np.nan, np.nan, np.nan],
                "slope": [1.0, 3.0, np.nan, np.nan],
            }
        )
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.IMPUTE_IN_GROUP,
                                         group_col="state")
        # A's NaN filled with A mean (100.0); A's observed values unchanged
        self.assertAlmostEqual(out.loc[out.state == "A", "elev"].iloc[0], 100.0)
        self.assertAlmostEqual(out.loc[out.state == "A", "elev"].iloc[1], 100.0)
        # B stays NaN (no cross-boundary fabrication)
        self.assertTrue(out.loc[out.state == "B", "elev"].isna().all())

    def test_impute_requires_group_col(self):
        with self.assertRaises(ValueError):
            apply_missingness_strategy(mk_frame(), ["elev", "slope"],
                                       MissingnessStrategy.IMPUTE_IN_GROUP)


class TestRetention(unittest.TestCase):
    def test_report(self):
        df = mk_frame()
        out = apply_missingness_strategy(df, ["elev", "slope"],
                                         MissingnessStrategy.DROP_ANY)
        r = report_retention(len(df), out)
        self.assertEqual(r["retained_rows"], 2)
        self.assertAlmostEqual(r["retention_rate"], 2 / 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
