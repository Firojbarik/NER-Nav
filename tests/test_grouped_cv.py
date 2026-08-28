"""Tests for grouped expanding-window CV (robust generalization metric).

SYNTHETIC / TEST ONLY (never used as training data).
Verifies the grouped-CV helper produces a stable pooled estimate that:
* uses every event group as a future test block at least once (n_folds > 0)
* separates strong-signal synthetic data (high pooled ROC-AUC) from
  no-signal synthetic data (pooled ROC-AUC near 0.5)
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def _import_harness():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tprm", REPO / "scripts" / "train_production_risk_model.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


m = _import_harness()


def build_synthetic(seed=0, n_events_train=10, n_events_val=3, n_events_test=4,
                    strong=True):
    rng = np.random.default_rng(seed)
    rows = []
    ev_id = 0
    base_time = np.datetime64("2016-01-01")
    event_dates = []
    all_ids = []
    for year_i, n_ev in enumerate([n_events_train, n_events_val, n_events_test]):
        for e in range(n_ev):
            event_date = base_time + np.timedelta64(year_i * 365 + e * 45, "D")
            event_dates.append(event_date)
            event_id = f"ev_{ev_id}"
            all_ids.append(event_id)
            ev_id += 1
    order = np.argsort(np.array(event_dates, dtype="datetime64[D]"))
    ordered_ids = [all_ids[i] for i in order]
    for k, eid in enumerate(ordered_ids):
        for sample in range(8):
            pos = sample < 3
            if strong:
                r30 = rng.uniform(180, 300) if pos else rng.uniform(5, 60)
            else:
                r30 = rng.uniform(0, 300)
            r1 = r30 * rng.uniform(0.05, 0.2)
            r3 = r30 * rng.uniform(0.2, 0.45)
            r7 = r30 * rng.uniform(0.4, 0.75)
            r14 = r30 * rng.uniform(0.6, 0.9)
            elev = rng.uniform(50, 3000)
            slope = rng.uniform(0, 30)
            rows.append({
                "event_id": eid, "osm_id": f"{eid}_{sample}", "ref": "TEST",
                "state": "TEST",
                "prediction_time": np.datetime64(event_dates[k]) - np.timedelta64(7, "D"),
                "label": int(pos), "sample_kind": "positive" if pos else "control",
                "label_source": "synthetic_unit_test",
                "rainfall_1day": r1, "rainfall_3day": r3, "rainfall_7day": r7,
                "rainfall_14day": r14, "rainfall_30day": r30,
                "rainfall_days_available": 30.0,
                "elevation_m": elev, "slope_degrees": slope,
                "highway_prior": rng.uniform(0, 1), "bridge_flag": float(rng.integers(0, 2)),
                "rain_intensity_1_vs_7": r1 / max(r7, 0.01),
                "rain_intensity_3_vs_14": r3 / max(r14, 0.01),
                "rain_concentration_3_in_7": r3 / max(r7, 0.01),
                "rain_trend_1_vs_3": r1 / max(r3, 0.01),
                "rain_cumul_ratio_7_vs_30": r7 / max(r30, 0.01),
                "terrain_known": 1,
                "slope_x_rain7": slope * r7,
                "slope_x_rain30": slope * r30,
                "elev_x_slope": elev * slope,
            })
    frame = pd.DataFrame(rows)
    # Seasonal/monsoon features derived from the (synthetic) prediction time,
    # matching the training-feature contract. Synthetic/test only.
    from datetime import datetime, timezone
    from ml.features.seasonal import (
        days_into_monsoon, monsoon_active, month_feature,
        seasonal_cos, seasonal_sin,
    )
    dates = [datetime.fromtimestamp(ts / 1e9, timezone.utc).date()
             for ts in frame["prediction_time"].astype("int64") / 1e9]
    frame["month"] = [float(month_feature(d)) for d in dates]
    frame["seasonal_sin"] = [seasonal_sin(d) for d in dates]
    frame["seasonal_cos"] = [seasonal_cos(d) for d in dates]
    frame["monsoon_active"] = [float(monsoon_active(d)) for d in dates]
    frame["days_into_monsoon"] = [float(days_into_monsoon(d)) for d in dates]
    return frame


class TestGroupedCV(unittest.TestCase):
    def test_runs_and_produces_pooled_metrics(self):
        ds = build_synthetic(strong=True)
        cv = m.grouped_cv_evaluate(ds)
        self.assertGreater(cv["n_folds"], 0)
        self.assertIsNotNone(cv["pooled_roc_auc"])
        self.assertIsNotNone(cv["pooled_avg_precision"])
        self.assertLessEqual(cv["pooled_roc_auc_max"], 1.0)
        self.assertLessEqual(cv["pooled_roc_auc_min"], cv["pooled_roc_auc_max"])

    def test_separates_strong_from_no_signal(self):
        strong = m.grouped_cv_evaluate(build_synthetic(seed=1, strong=True))
        weak = m.grouped_cv_evaluate(build_synthetic(seed=1, strong=False))
        self.assertGreater(strong["pooled_roc_auc"],
                           weak["pooled_roc_auc"] + 0.05)
        self.assertGreater(strong["pooled_roc_auc"], 0.5)

    def test_folds_use_different_test_events(self):
        ds = build_synthetic(strong=True)
        cv = m.grouped_cv_evaluate(ds)
        tested = set()
        for fold in cv["folds"]:
            for eid in fold["test_events"]:
                self.assertNotIn(eid, tested)
                tested.add(eid)
        # every event (except the first training block) should appear in a test
        # fold at least once across the expanding window
        n_events = ds["event_id"].nunique()
        self.assertGreaterEqual(len(tested), n_events - 5)


if __name__ == "__main__":
    unittest.main()
