"""Unit tests for leakage-safe seasonal / monsoon feature derivation.

SYNTHETIC dates are used here to verify algorithm mechanics only - they never
enter training data.
"""

import unittest
from datetime import date

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from ml.features.seasonal import (  # noqa: E402
    SEASONAL_FEATURES,
    days_into_monsoon,
    is_monsoon,
    monsoon_active,
    month_feature,
    pre_monsoon_active,
    recommended_features,
    season,
    seasonal_cos,
    seasonal_sin,
)


class TestSeasonalDeterminism(unittest.TestCase):
    def test_monsoon_window(self):
        self.assertEqual(monsoon_active(date(2024, 6, 1)), 1)
        self.assertEqual(monsoon_active(date(2024, 9, 30)), 1)
        self.assertEqual(monsoon_active(date(2024, 5, 31)), 0)
        self.assertEqual(monsoon_active(date(2024, 10, 1)), 0)
        self.assertTrue(is_monsoon(date(2024, 7, 1)))
        self.assertFalse(is_monsoon(date(2024, 1, 1)))

    def test_season_labels(self):
        self.assertEqual(season(date(2024, 1, 15)), "winter")
        self.assertEqual(season(date(2024, 4, 1)), "pre_monsoon")
        self.assertEqual(season(date(2024, 7, 1)), "monsoon")
        self.assertEqual(season(date(2024, 11, 1)), "post_monsoon")

    def test_pre_monsoon(self):
        self.assertEqual(pre_monsoon_active(date(2024, 3, 1)), 1)
        self.assertEqual(pre_monsoon_active(date(2024, 5, 31)), 1)
        self.assertEqual(pre_monsoon_active(date(2024, 6, 1)), 0)

    def test_month_and_cyclical_bounds(self):
        self.assertEqual(month_feature(date(2024, 12, 31)), 12)
        self.assertGreaterEqual(seasonal_sin(date(2024, 6, 1)), -1)
        self.assertLessEqual(seasonal_sin(date(2024, 6, 1)), 1)
        self.assertGreaterEqual(seasonal_cos(date(2024, 6, 1)), -1)
        self.assertLessEqual(seasonal_cos(date(2024, 6, 1)), 1)

    def test_days_into_monsoon(self):
        # Jun 1 -> 0; May 31 -> -1; Aug 1 -> 61
        self.assertEqual(days_into_monsoon(date(2024, 6, 1)), 0)
        self.assertEqual(days_into_monsoon(date(2024, 5, 31)), -1)
        self.assertEqual(days_into_monsoon(date(2024, 8, 1)), 61)

    def test_recommended_features_are_deterministic_and_complete(self):
        a = recommended_features(date(2024, 7, 15))
        b = recommended_features(date(2024, 7, 15))
        self.assertEqual(a, b)
        self.assertEqual(
            set(SEASONAL_FEATURES),
            {"month", "seasonal_sin", "seasonal_cos",
             "monsoon_active", "days_into_monsoon"})
        for name in SEASONAL_FEATURES:
            self.assertIn(name, a)

    def test_temp_date_validation_not_finite_internals_only(self):
        # month/seasonal features are all real numerics, never NaN.
        import math
        f = recommended_features(date(2024, 2, 29))
        for name, value in f.items():
            self.assertFalse(math.isnan(value))

    def test_year_independence(self):
        # Leap vs common year must not change month/day-derived features.
        monsoon_day_2019 = days_into_monsoon(date(2019, 7, 1))
        monsoon_day_2024 = days_into_monsoon(date(2024, 7, 1))
        self.assertEqual(monsoon_day_2019, monsoon_day_2024)


if __name__ == "__main__":
    unittest.main()
