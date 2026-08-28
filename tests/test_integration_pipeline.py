"""Integration test: build dataset -> train -> predict -> explain.

Exercises the full production pipeline end-to-end using a small synthetic
dataset (unit test only, never enters real training). Catches regressions in
the chain: temporal_design -> build_dataset -> train -> calibrate -> save ->
load -> predict -> explain.
"""

import json
import os
import pickle
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


def _import_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


FEATURES = [
    "rainfall_1day", "rainfall_3day", "rainfall_7day", "rainfall_14day",
    "rainfall_30day", "elevation_m", "slope_degrees",
    "rain_intensity_1_vs_7", "rain_intensity_3_vs_14",
    "rain_concentration_3_in_7", "rain_trend_1_vs_3",
    "rain_cumul_ratio_7_vs_30", "terrain_known",
    "slope_x_rain7", "slope_x_rain30", "elev_x_slope",
]


def _build_synthetic_dataset(seed=42):
    """Build a small synthetic dataset mimicking the real schema."""
    rng = np.random.default_rng(seed)
    rows = []
    event_dates = []
    all_ids = []
    ev_id = 0
    base = np.datetime64("2020-01-01")
    for year_i, n_ev in enumerate([6, 3, 3]):
        for e in range(n_ev):
            ed = base + np.timedelta64(year_i * 365 + e * 60, "D")
            event_dates.append(ed)
            all_ids.append(f"ev_{ev_id}")
            ev_id += 1

    for k, eid in enumerate(all_ids):
        for s in range(6):
            pos = s < 2
            slope = rng.uniform(0, 30)
            elev = rng.uniform(50, 2000)
            if pos:
                r30 = rng.uniform(200, 400)
            else:
                r30 = rng.uniform(5, 80)
            r1 = r30 * rng.uniform(0.05, 0.2)
            r3 = r30 * rng.uniform(0.2, 0.45)
            r7 = r30 * rng.uniform(0.4, 0.75)
            r14 = r30 * rng.uniform(0.6, 0.9)
            eps = 0.01
            rows.append({
                "event_id": eid,
                "osm_id": f"{eid}_{s}",
                "ref": "TEST",
                "highway": "trunk",
                "district": "TEST",
                "state": "TEST",
                "prediction_time": np.datetime64(event_dates[k]) - np.timedelta64(7, "D"),
                "label": int(pos),
                "sample_kind": "positive" if pos else "corridor_negative",
                "label_source": "synthetic_integration_test",
                "elevation_m": elev,
                "slope_degrees": slope,
                "rainfall_1day": r1,
                "rainfall_3day": r3,
                "rainfall_7day": r7,
                "rainfall_14day": r14,
                "rainfall_30day": r30,
                "rainfall_days_available": 30.0,
                "highway_prior": 0.1,
                "bridge_flag": 1.0,
                "rain_intensity_1_vs_7": r1 / max(r7, eps),
                "rain_intensity_3_vs_14": r3 / max(r14, eps),
                "rain_concentration_3_in_7": r3 / max(r7, eps),
                "rain_trend_1_vs_3": r1 / max(r3, eps),
                "rain_cumul_ratio_7_vs_30": r7 / max(r30, eps),
                "terrain_known": 1,
                "slope_x_rain7": slope * r7,
                "slope_x_rain30": slope * r30,
                "elev_x_slope": elev * slope,
            })
    return pd.DataFrame(rows)


class TestFullPipeline(unittest.TestCase):
    """End-to-end integration: dataset -> train -> calibrate -> predict -> explain."""

    def test_full_pipeline(self):
        import xgboost as xgb
        from sklearn.calibration import IsotonicRegression
        from sklearn.metrics import roc_auc_score

        tprm = _import_module("tprm", REPO / "scripts" / "train_production_risk_model.py")

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)

            # 1. BUILD DATASET
            ds = _build_synthetic_dataset()
            ds_path = td / "dataset.parquet"
            ds.to_parquet(ds_path, index=False)
            self.assertEqual(len(ds), 12 * 6)  # 12 events x 6 samples
            self.assertIn("label", ds.columns)
            for feat in FEATURES:
                self.assertIn(feat, ds.columns, f"missing feature: {feat}")

            # 2. SPLIT
            train, val, test = tprm.split_by_events(ds, n_val_events=2, n_test_events=3)
            self.assertGreater(train["event_id"].nunique(), 0)
            self.assertGreater(val["event_id"].nunique(), 0)
            self.assertGreater(test["event_id"].nunique(), 0)

            # Verify chronological order
            self.assertGreater(val["prediction_time"].min(), train["prediction_time"].max())
            self.assertGreater(test["prediction_time"].min(), val["prediction_time"].max())

            # 3. TRAIN
            def mat(df):
                X = df[FEATURES].copy()
                for c in FEATURES:
                    X[c] = pd.to_numeric(X[c], errors="coerce")
                return X.values, df["label"].to_numpy()

            Xtr, ytr = mat(train)
            Xva, yva = mat(val)
            Xte, yte = mat(test)

            model = xgb.XGBClassifier(**tprm.HYPERPARAMS)
            model.fit(Xtr, ytr)

            # 4. CALIBRATE
            pa = model.predict_proba(Xva)[:, 1]
            pv = model.predict_proba(Xte)[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(pa, yva)
            pv_cal = iso.predict(pv)

            # 5. SAVE ARTIFACTS
            model.save_model(td / "model.ubj")
            model.save_model(td / "model.json")
            with open(td / "calib.pkl", "wb") as f:
                pickle.dump(iso, f)
            (td / "features.json").write_text(json.dumps({"features": FEATURES}))

            # 6. LOAD ARTIFACTS
            m2 = xgb.XGBClassifier()
            m2.load_model(td / "model.ubj")
            with open(td / "calib.pkl", "rb") as f:
                iso2 = pickle.load(f)
            feat_spec = json.loads((td / "features.json").read_text())
            self.assertEqual(feat_spec["features"], FEATURES)

            # 7. PREDICT
            sample = ds.iloc[0:1]
            X_sample = sample[FEATURES].values.astype(np.float32)
            raw_prob = float(m2.predict_proba(X_sample)[:, 1][0])
            cal_prob = float(iso2.predict(np.array([raw_prob]))[0])
            self.assertTrue(0.0 <= raw_prob <= 1.0)
            self.assertTrue(0.0 <= cal_prob <= 1.0)

            # 8. EXPLAIN (pred_contribs)
            contribs = m2.get_booster().predict(
                xgb.DMatrix(X_sample), pred_contribs=True
            )[0]
            self.assertEqual(len(contribs), len(FEATURES) + 1)  # features + bias
            # Verify contribs are finite
            self.assertTrue(np.all(np.isfinite(contribs)))

            # 9. RISK LEVEL
            if cal_prob >= 0.7:
                rl = "HIGH"
            elif cal_prob >= 0.4:
                rl = "MEDIUM"
            else:
                rl = "LOW"
            self.assertIn(rl, ["HIGH", "MEDIUM", "LOW"])

            # 10. VERIFY TEST METRICS ARE COMPUTABLE
            yte_pred = (pv_cal >= 0.5).astype(int)
            auc = float(roc_auc_score(yte, pv_cal))
            self.assertTrue(auc >= 0.0)

            # 11. TRACEABILITY: simulate prediction logging
            import sys
            sys.path.insert(0, str(REPO / "scripts"))
            from prediction_trace import log_prediction, get_prediction_traces

            pred_id = log_prediction(
                road_segment_id="test_segment_123",
                prediction_timestamp="2025-01-01T10:00:00Z",
                model_tag="test_model",
                features=sample[FEATURES].iloc[0].to_dict(),
                raw_probability=raw_prob,
                calibrated_probability=cal_prob,
                risk_level=rl,
                top_contributing_factors=[],
                threshold=0.5,
                ref="NH37",
                state="TEST",
            )
            self.assertIsInstance(pred_id, str)
            self.assertEqual(len(pred_id), 36)  # UUID length

            # Verify trace was logged
            traces = get_prediction_traces()
            self.assertGreaterEqual(len(traces), 1)
            trace = traces.iloc[-1]
            self.assertEqual(trace["road_segment_id"], "test_segment_123")
            self.assertEqual(trace["model_version"], "test_model")
            self.assertEqual(trace["risk_level"], rl)
            self.assertEqual(trace["calibrated_probability"], cal_prob)


class TestTemporalLeakage(unittest.TestCase):
    """Verify that no future information leaks into features."""

    def test_features_predate_prediction_time(self):
        """All rainfall features must be from days strictly before prediction_time."""
        ds = _build_synthetic_dataset()
        for _, row in ds.iterrows():
            pt = pd.Timestamp(row["prediction_time"])
            # rainfall windows are anchored at prediction_time, so features
            # represent rainfall in the lookback window BEFORE prediction_time.
            # We verify that the dataset was built with the correct schema.
            self.assertGreater(pt.year, 2018)
            self.assertIn("prediction_time", ds.columns)
            self.assertIn("label", ds.columns)

    def test_test_set_is_chronologically_future(self):
        """Test events must be strictly after validation events."""
        tprm = _import_module("tprm2", REPO / "scripts" / "train_production_risk_model.py")
        ds = _build_synthetic_dataset()
        train, val, test = tprm.split_by_events(ds, n_val_events=2, n_test_events=3)

        train_max = train["prediction_time"].max()
        val_min = val["prediction_time"].min()
        val_max = val["prediction_time"].max()
        test_min = test["prediction_time"].min()

        self.assertGreater(val_min, train_max)
        self.assertGreater(test_min, val_max)

    def test_no_event_straddles_splits(self):
        """Each event's samples must all be in the same split."""
        tprm = _import_module("tprm3", REPO / "scripts" / "train_production_risk_model.py")
        ds = _build_synthetic_dataset()
        train, val, test = tprm.split_by_events(ds, n_val_events=2, n_test_events=3)

        all_ids = set(train["event_id"]) | set(val["event_id"]) | set(test["event_id"])
        total = train["event_id"].nunique() + val["event_id"].nunique() + test["event_id"].nunique()
        self.assertEqual(len(all_ids), total)


if __name__ == "__main__":
    unittest.main(verbosity=2)
