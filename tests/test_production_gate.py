"""Tests for chronological splitting, recall thresholding, and evidence gates."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

MODULE_KW = {}


def _import_harness():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tprm", REPO / "scripts" / "train_production_risk_model.py")
    m = importlib.util.module_from_spec(spec)
    # patch module constants before exec so it works on a temp dataset
    spec.loader.exec_module(m)
    return m


FEATURES = ["rainfall_1day", "rainfall_3day", "rainfall_7day", "rainfall_14day",
            "rainfall_30day",
            "elevation_m", "slope_degrees",
            "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
            "rain_concentration_3_in_7", "rain_trend_1_vs_3",
            "rain_cumul_ratio_7_vs_30", "terrain_known",
            "slope_x_rain7", "slope_x_rain30", "elev_x_slope"]


def build_synthetic(seed=0, n_events_train=10, n_events_val=3, n_events_test=4,
                    strong=True):
    """Synthetic-only dataset (for unit test) with a strong separable signal."""
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

    # event -> positive probability depends on rainfall_30day (the signal)
    for k, eid in enumerate(ordered_ids):
        for sample in range(8):
            pos = sample < 3
            if strong:
                # rain signal strongly separates; positives have heavy 30-day rain
                if pos:
                    r30 = rng.uniform(180, 300)
                else:
                    r30 = rng.uniform(5, 60)
            else:
                r30 = rng.uniform(0, 300)
            r1 = r30 * rng.uniform(0.05, 0.2)
            r3 = r30 * rng.uniform(0.2, 0.45)
            r7 = r30 * rng.uniform(0.4, 0.75)
            r14 = r30 * rng.uniform(0.6, 0.9)
            elev = rng.uniform(50, 3000)
            slope = rng.uniform(0, 30)
            rows.append({
                "event_id": eid,
                "osm_id": f"{eid}_{sample}",
                "ref": "TEST",
                "state": "TEST",
                "prediction_time": np.datetime64(event_dates[k]) - np.timedelta64(7, "D"),
                "label": int(pos),
                "sample_kind": "positive" if pos else "control",
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


def run_pipeline(m, ds):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ds.parquet"
        ds.to_parquet(p, index=False)
        # monkeypatch the module's DATASET/report dir to the temp dir
        m.DATASET = p
        m.MODEL_DIR = Path(td) / "models"
        m.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        train, val, test = m.split_by_events(ds, n_val_events=3, n_test_events=4)

        def mat(df):
            X = df[m.FEATURES].copy()
            for c in m.NUMERIC:
                X[c] = pd.to_numeric(X[c], errors="coerce")
            return X.values, df["label"].to_numpy()
        Xtr, ytr = mat(train); Xva, yva = mat(val); Xte, yte = mat(test)
        import xgboost as xgb
        model = xgb.XGBClassifier(**m.HYPERPARAMS).fit(Xtr, ytr)
        pa = model.predict_proba(Xva)[:, 1]
        pv = model.predict_proba(Xte)[:, 1]
        from sklearn.calibration import IsotonicRegression
        if yva.sum() and (yva == 0).sum():
            iso = IsotonicRegression(out_of_bounds="clip").fit(pa, yva)
            pv = iso.predict(pv)
        from sklearn.metrics import roc_auc_score
        return {
            "n_train_events": train["event_id"].nunique(),
            "n_test_events": test["event_id"].nunique(),
            "n_test_pos": int(yte.sum()),
            "roc_auc": float(roc_auc_score(yte, pv)),
        }


class TestProductionGate(unittest.TestCase):
    def test_default_real_test_split_has_two_classes(self):
        m = _import_harness()
        dataset = REPO / "data/processed/ml/real_temporal_risk_dataset.parquet"
        if not dataset.exists():
            self.skipTest("real temporal dataset absent")
        ds = pd.read_parquet(dataset)
        _, val, test = m.split_by_events(ds)
        self.assertEqual(set(val["label"].unique()), {0, 1})
        self.assertEqual(set(test["label"].unique()), {0, 1})
        self.assertIsNotNone(m._safe_auc(test["label"], np.zeros(len(test))))

    def test_single_class_test_split_is_rejected(self):
        m = _import_harness()
        test = pd.DataFrame({"label": [1, 1, 1]})
        with self.assertRaises(ValueError):
            m.require_two_class_test_split(test)

    def test_split_is_chronological_and_event_intact(self):
        m = _import_harness()
        ds = build_synthetic(strong=True)
        train, val, test = m.split_by_events(ds, n_val_events=3, n_test_events=4)
        self.assertGreaterEqual(
            test["prediction_time"].min(), val["prediction_time"].max())
        self.assertGreater(
            val["prediction_time"].min(), train["prediction_time"].max())
        # no event straddles two splits
        all_ids = set(train["event_id"]) | set(val["event_id"]) | set(test["event_id"])
        self.assertEqual(len(all_ids), train["event_id"].nunique()
                         + val["event_id"].nunique() + test["event_id"].nunique())
        # every event keeps exactly its samples
        self.assertEqual(train["event_id"].nunique() * 8, len(train))

    def test_demo_and_production_gates_are_independent(self):
        m = _import_harness()
        metrics = {"roc_auc": 0.70, "avg_precision": 0.55, "recall": 0.80}
        gates = m.assess_gates(3, metrics, chronological=True)
        self.assertTrue(gates["demo"]["passed"])
        self.assertFalse(gates["production"]["passed"])
        self.assertIn("test events 3 < 30", gates["production"]["reasons"])

    def test_three_future_events_can_never_claim_production_ready(self):
        m = _import_harness()
        perfect = {"roc_auc": 1.0, "avg_precision": 1.0, "recall": 1.0}
        self.assertFalse(m.assess_gates(3, perfect)["production"]["passed"])

    def test_threshold_selection_prioritizes_target_recall(self):
        m = _import_harness()
        y = np.array([1, 1, 1, 0, 0, 0])
        scores = np.array([0.90, 0.60, 0.40, 0.70, 0.30, 0.10])
        threshold, details = m.select_recall_threshold(
            y, scores, target_recall=2 / 3, precision_floor=0.30)
        pred = scores >= threshold
        recall = (pred & (y == 1)).sum() / (y == 1).sum()
        self.assertGreaterEqual(recall, 2 / 3)
        self.assertEqual(details["strategy"], "target_recall_met_max_precision")

    def test_threshold_selection_respects_validation_alert_budget(self):
        m = _import_harness()
        y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.95, 0.80, 0.70, 0.90, 0.60,
                           0.50, 0.40, 0.30, 0.20, 0.10])
        threshold, details = m.select_recall_threshold(
            y, scores, target_recall=0.70, precision_floor=0.30,
            max_alert_rate=0.30)
        self.assertLessEqual((scores >= threshold).mean(), 0.30)
        self.assertEqual(details["max_alert_rate"], 0.30)
        self.assertIn("alert_budget", details["strategy"])

    def test_top_k_threshold_is_label_free_and_budgeted(self):
        m = _import_harness()
        scores = np.array([0.95, 0.80, 0.70, 0.60, 0.50, 0.40,
                           0.30, 0.20, 0.10, 0.05])
        threshold, details = m.select_top_k_threshold(scores, 0.30)
        self.assertEqual(details["strategy"], "top_k_validation_only")
        self.assertEqual(details["validation_selected"], 3)
        self.assertLessEqual((scores >= threshold).mean(), 0.30)

    def test_caveat_uses_dataframe_counts(self):
        m = _import_harness()
        ds = pd.DataFrame({
            "event_id": ["a", "a", "b", "c", "c"],
            "label": [1, 0, 1, 0, 0],
        })
        caveat = m.build_dataset_caveat(ds, n_test_events=1)
        self.assertIn("5 samples", caveat)
        self.assertIn("3 events", caveat)
        self.assertIn("2 positives", caveat)
        self.assertIn("3 negatives", caveat)
        self.assertNotIn("36 positives", caveat)

    def test_operational_metrics_include_error_counts(self):
        m = _import_harness()
        metrics = m.eval_metrics(
            np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.8, 0.1]), 0.5)
        self.assertEqual(metrics["false_negatives"], 1)
        self.assertEqual(metrics["false_positives"], 1)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)
        self.assertIn("brier_score", metrics)


if __name__ == "__main__":
    unittest.main(verbosity=2)
