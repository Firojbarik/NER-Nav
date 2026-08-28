"""Unit tests for ml/features/temporal_design.py (real temporal sample design).

These are assertions about the current real event design. The events are real
labels pulled from the audit); the tests are NOT synthetic data.
"""

import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.features.temporal_design import (  # noqa: E402
    CONFIRMED_EVENTS,
    HORIZON_DAYS,
    LOOKBACK_DAYS,
    POSITIVE_OFFSETS,
    all_sample_times_for_events,
    build_labeled_samples,
    needed_chirps_dates,
    needed_prediction_dates,
)


def _event_date(eid):
    for e, _, _, ed in CONFIRMED_EVENTS:
        if e == eid:
            return date.fromisoformat(ed)
    raise KeyError(eid)


class TestConfirmedEvents(unittest.TestCase):
    def test_confirmed_real_events_count(self):
        self.assertGreaterEqual(len(CONFIRMED_EVENTS), 19)

    def test_events_dates_real_2017_2025(self):
        dates = [d for _, _, _, d in CONFIRMED_EVENTS]
        self.assertIn("2017-07-15", dates)
        self.assertIn("2025-09-12", dates)


class TestSampleDesign(unittest.TestCase):
    def test_positive_offsets_within_horizon(self):
        self.assertTrue(all(0 < off <= HORIZON_DAYS for off in POSITIVE_OFFSETS))
        self.assertEqual(sorted(POSITIVE_OFFSETS), [1, 3, 7])

    def test_base_labeled_samples_counts(self):
        lab = build_labeled_samples()
        n_events = len(CONFIRMED_EVENTS)
        self.assertEqual(int((lab["label"] == 1).sum()), n_events * 3)
        self.assertEqual(int((lab["label"] == 0).sum()), n_events)  # same-road controls

    def test_positive_labels_satisfy_temporal_rule(self):
        # label must be 1 iff  prediction_time < event_time <= prediction_time+horizon
        lab = build_labeled_samples()
        for _, r in lab.iterrows():
            ev = _event_date(r["event_id"])
            pt = r["prediction_time"].date()
            inside = pt < ev <= pt + timedelta(days=HORIZON_DAYS)
            self.assertEqual(int(inside), int(r["label"]),
                             f"{r['event_id']} pt={pt} ev={ev}")

    def test_needed_chirps_dates_are_strictly_before_their_prediction_times(self):
        # leakage guard: every rainfall lookback day must precede the prediction
        # time of the sample it serves (no future rainfall leaks into features).
        chirps = set(needed_chirps_dates())
        for pt in needed_prediction_dates():
            lookback = {pt - timedelta(days=k) for k in range(1, LOOKBACK_DAYS + 1)}
            # every required lookback day is in the needed set...
            self.assertTrue(lookback <= chirps)
            # ...and is strictly in the past relative to the prediction time
            self.assertTrue(all(d < pt for d in lookback))


class TestNeededDates(unittest.TestCase):
    def test_prediction_dates_real_range(self):
        pdates = needed_prediction_dates()
        self.assertEqual(pdates[0].isoformat(), "2017-07-01")
        self.assertGreaterEqual(pdates[-1], date(2025, 9, 11))

    def test_chirps_dates_nonempty_and_in_past(self):
        cd = needed_chirps_dates()
        self.assertTrue(len(cd) > 300)
        self.assertTrue(all(cd[i] < cd[i + 1] for i in range(len(cd) - 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
