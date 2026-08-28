"""Unit tests for ml/features/rainfall aggregation logic.

SYNTHETIC / TEST ONLY
---------------------
These tests exercise the algorithm mechanics with small hand-built synthetic
arrays. They must never be used as model training data.

Verifies:
* no-data is NaN, not 0 (F-3 regression guard)
* windows are calendar-date-anchored, so gaps cannot misalign (F-7 guard)
* window sums use only days strictly before the anchor (no temporal leakage)
* strict vs partial missingness semantics
"""

import os
import sys
import unittest

import numpy as np

# Make the project importable regardless of CWD.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.rainfall import (  # noqa: E402
    aggregate_window,
    aggregate_windows,
    available_day_counts,
    clean_daily_samples,
    window_offsets,
)


class TestCleanDailySamples(unittest.TestCase):
    def test_negative_values_become_nan(self):
        raw = np.array([0.0, 1.5, -9999.0, -0.5, 12.0])
        out = clean_daily_samples(raw)
        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], 1.5)
        np.testing.assert_allclose(out[4], 12.0)
        self.assertTrue(np.isnan(out[2]))
        self.assertTrue(np.isnan(out[3]))

    def test_valid_zero_is_preserved_not_nan(self):
        # A real 0 mm day is "no rain", NOT no-data.
        out = clean_daily_samples(np.array([0.0, 0.0]))
        np.testing.assert_allclose(out, [0.0, 0.0])


class TestWindowOffsets(unittest.TestCase):
    def test_offsets_are_negative_before_anchor(self):
        np.testing.assert_array_equal(window_offsets(3), [-3, -2, -1])

    def test_never_includes_anchor_day(self):
        # day 0 (prediction timestamp) must never be in the window.
        self.assertNotIn(0, window_offsets(30).tolist())
        self.assertNotIn(0, window_offsets(1).tolist())


class TestAggregateWindowAnchored(unittest.TestCase):
    def test_sum_of_three_days(self):
        # 2 roads x 5 days offsets [-5,-4,-3,-2,-1]
        daily = np.array(
            [
                [0.0, 0.0, 1.0, 2.0, 3.0],  # closest-to-anchor first day =3
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        )
        offsets = np.array([-5, -4, -3, -2, -1])
        out = aggregate_window(daily, offsets, window_days=3)
        # window 3 = offsets -3,-2,-1 -> cols [2,3,4]
        np.testing.assert_allclose(out, [1.0 + 2.0 + 3.0, 7.0 + 8.0 + 9.0])

    def test_day_0_rainfall_is_excluded(self):
        # Anchor-day rainfall (offset 0) is excluded even if present.
        daily = np.array([[999.0, 1.0, 2.0]])
        offsets = np.array([0, -2, -1])  # one col is the anchor day itself
        # only offsets -2,-1 are used
        out = aggregate_window(daily, offsets, window_days=5)
        np.testing.assert_allclose(out, [1.0 + 2.0])


class TestAggregateWindowMissingSemantics(unittest.TestCase):
    def test_strict_returns_nan_when_any_day_missing(self):
        daily = np.array(
            [
                [1.0, np.nan, 3.0],  # middle day missing -> NaN
                [1.0, 2.0, 3.0],  # complete -> 6
            ]
        )
        offsets = np.array([-3, -2, -1])
        out = aggregate_window(daily, offsets, window_days=3, allow_partial=False)
        self.assertTrue(np.isnan(out[0]))
        np.testing.assert_allclose(out[1], 6.0)

    def test_partial_sums_valid_days_only(self):
        daily = np.array([[1.0, np.nan, 3.0]])
        offsets = np.array([-3, -2, -1])
        out = aggregate_window(daily, offsets, window_days=3, allow_partial=True)
        np.testing.assert_allclose(out, [4.0])

    def test_all_missing_stays_nan_even_partial(self):
        daily = np.array([[np.nan, np.nan, np.nan]])
        offsets = np.array([-3, -2, -1])
        out = aggregate_window(daily, offsets, window_days=3, allow_partial=True)
        self.assertTrue(np.isnan(out[0]))

    def test_smaller_window_shifts_anchor(self):
        # window 1 should use only offset -1
        daily = np.array([[0.0, 0.0, 5.0]])
        offsets = np.array([-3, -2, -1])
        out = aggregate_window(daily, offsets, window_days=1)
        np.testing.assert_allclose(out, [5.0])


class TestGapDoesNotMisalign(unittest.TestCase):
    def test_gap_moves_values_to_correct_calendar_slot(self):
        # Calendar days -4..-1. Day -2 is missing (gap).
        # Daily values are anchored by offset array, so the gap yields NaN at
        # the correct slot and the 7-day window stays NaN (strict).
        offsets = np.array([-4, -3, -2, -1])
        daily = np.array([[1.0, 2.0, np.nan, 4.0]])
        out = aggregate_window(daily, offsets, window_days=7)
        self.assertTrue(np.isnan(out[0]))
        # partial: only the 3 present days are summed
        partial = aggregate_window(daily, offsets, window_days=7, allow_partial=True)
        np.testing.assert_allclose(partial, [1.0 + 2.0 + 4.0])


class TestAggregateWindowsDict(unittest.TestCase):
    def test_returns_all_windows(self):
        daily = np.array([[1.0, 2.0, 3.0]])
        offsets = np.array([-3, -2, -1])
        out = aggregate_windows(daily, offsets, [1, 3])
        self.assertIn(1, out)
        self.assertIn(3, out)
        np.testing.assert_allclose(out[1], [3.0])
        np.testing.assert_allclose(out[3], [6.0])


class TestOffsetSignContract(unittest.TestCase):
    def test_positive_offsets_select_nothing(self):
        # The core is documented to consume NEGATIVE offsets (days before the
        # anchor). Positive offsets indicate a caller bug; nothing should be
        # selected, so result is all-NaN (rather than silently summing wrong).
        daily = np.array([[1.0, 2.0, 3.0]])
        offsets = np.array([1, 2, 3])  # wrong sign - caller bug
        out = aggregate_window(daily, offsets, window_days=7)
        self.assertTrue(np.isnan(out[0]))

    def test_zero_offset_excluded_from_window(self):
        # A 0 offset (= anchor day) must be excluded even for large windows.
        daily = np.array([[999.0, 1.0, 2.0]])
        offsets = np.array([0, -2, -1])
        out = aggregate_window(daily, offsets, window_days=3)
        np.testing.assert_allclose(out, [3.0])


class TestAvailableDayCounts(unittest.TestCase):
    def test_counts_valid_days(self):
        daily = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 3.0]])
        np.testing.assert_array_equal(available_day_counts(daily), [2, 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
