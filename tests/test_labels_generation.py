"""Unit tests for ml/labels/generate.py.

SYNTHETIC / TEST ONLY: event/sample frames here are hand-built to exercise the
matching mechanics. They must never become real training labels.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.labels.generate import LabelingConfig, apply_labels


def mk_sample(roads, times):
    return pd.DataFrame(
        {"road_segment_id": roads, "prediction_time": pd.to_datetime(times)}
    )


def mk_events(segments, times, etypes=None):
    n = len(segments)
    df = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(n)],
            "event_time": pd.to_datetime(times),
            "affected_segment_id": segments,
        }
    )
    if etypes is not None:
        df["event_type"] = etypes
    return df


H = pd.Timedelta("1 days")


class TestApplyLabelsExplicit(unittest.TestCase):
    def test_event_inside_horizon_is_positive(self):
        samples = mk_sample([100, 200], ["2020-01-01 00:00", "2020-01-01 00:00"])
        events = mk_events([200], ["2020-01-02 00:00"])  # +1 day, inside 1-day horizon
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        self.assertEqual(out.loc[out.road_segment_id == 100, "label"].iloc[0], 0)
        self.assertEqual(out.loc[out.road_segment_id == 200, "label"].iloc[0], 1)

    def test_labels_are_sample_specific_not_segment_broadcast(self):
        # Same segment, two prediction times: the event falls inside the future
        # window for t1 but is already past for t2. Each (segment, time) sample
        # must get its own label (regression: labels were once aggregated per
        # segment only and broadcast onto every prediction_time).
        samples = mk_sample([200, 200], ["2020-01-01 00:00", "2020-01-05 00:00"])
        events = mk_events([200], ["2020-01-02 00:00"])
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        lbl = dict(zip(out.prediction_time, out.label))
        self.assertEqual(lbl[pd.Timestamp("2020-01-01 00:00")], 1)
        self.assertEqual(lbl[pd.Timestamp("2020-01-05 00:00")], 0)
        self.assertIs(pd.isna(out.loc[out.prediction_time == "2020-01-05", "lead_time"].iloc[0]), True)

    def test_event_before_prediction_is_not_future(self):
        samples = mk_sample([200], ["2020-01-10 00:00"])
        events = mk_events([200], ["2020-01-01 00:00"])  # past event
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        self.assertEqual(out["label"].iloc[0], 0)

    def test_event_exactly_at_prediction_time_not_future(self):
        samples = mk_sample([200], ["2020-01-05 00:00"])
        events = mk_events([200], ["2020-01-05 00:00"])  # == prediction_time
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        self.assertEqual(out["label"].iloc[0], 0)

    def test_event_at_horizon_end_is_included(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = mk_events([200], ["2020-01-02 00:00"])  # +1 day == horizon end
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        self.assertEqual(out["label"].iloc[0], 1)

    def test_event_after_horizon_excluded(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = mk_events([200], ["2020-01-03 00:00"])  # +2 days > horizon
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        self.assertEqual(out["label"].iloc[0], 0)


class TestLeadTime(unittest.TestCase):
    def test_lead_time_is_first_event_minus_prediction(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = mk_events(
            [200, 200], ["2020-01-01 06:00", "2020-01-02 00:00"]
        )
        out = apply_labels(samples, events, LabelingConfig(horizon=H))
        row = out.loc[out.road_segment_id == 200].iloc[0]
        self.assertEqual(row["label"], 1)
        self.assertEqual(row["matching_event_count"], 2)
        self.assertEqual(row["earliest_event_time"], pd.Timestamp("2020-01-01 06:00"))
        self.assertEqual(row["lead_time"], pd.Timedelta("6 hours"))


class TestEventTypeFilter(unittest.TestCase):
    def test_filters_qualifying_types(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = mk_events(
            [200, 200],
            ["2020-01-02 00:00", "2020-01-02 00:00"],
            etypes=["landslide", "flood"],
        )
        out = apply_labels(
            samples, events, LabelingConfig(horizon=H, include_event_types={"landslide"})
        )
        # two qualifying events of the same included type both count
        self.assertEqual(out["label"].iloc[0], 1)
        self.assertEqual(out["matching_event_count"].iloc[0], 1)


class TestValidationErrors(unittest.TestCase):
    def test_requires_affected_segment_in_explicit_mode(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = pd.DataFrame(
            {
                "event_id": ["e0"],
                "event_time": pd.to_datetime(["2020-01-02 00:00"]),
            }
        )
        with self.assertRaises(ValueError):
            apply_labels(samples, events, LabelingConfig(horizon=H))

    def test_buffer_mode_not_implemented(self):
        samples = mk_sample([200], ["2020-01-01 00:00"])
        events = mk_events([200], ["2020-01-02 00:00"])
        with self.assertRaises(NotImplementedError):
            apply_labels(
                samples,
                events,
                LabelingConfig(horizon=H, spatial_mode="buffer", buffer_m=500.0),
            )

    def test_zero_horizon_rejected(self):
        with self.assertRaises(ValueError):
            LabelingConfig(horizon=pd.Timedelta(0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
