# NER-Nav ML Completion / Pre-Integration Verification Report

**Date:** 2026-09-20
**Scope:** Final ML acceptance + pre-integration verification of the production-risk
model (`2026-09-20_002229_48b0a9ff`) after the AUC-collapse fix, the addition of
100 source-backed observed-unaffected negatives, the grouped-CV reporting correction,
and a re-run against the locked production dataset.
**Decision:** **BLOCKED BY DATA GAP** - the model is demo-ready (honest demo gate
PASS) but NOT PRODUCTION READY FOR INTEGRATION. It must not be integrated into
routing, emergency response, or public safety flows.

Every claim below was re-verified live during this audit (commands re-run, hashes
recomputed, splits re-derived, predictions regenerated). The prior 2026-08-28 report
(`BLOCKED BY DATA GAP` at `GATED_LOW_CONFIDENCE`) is superseded: this run materially
improved negatives (0 -> 100 source-backed), events (32 -> 58), the demo gate
(FAIL -> honest PASS), and reproducibility.

---

## 1. Model under audit

| Item | Value |
|---|---|
| Active pointer | `data/models/prod_latest.json` -> `2026-09-20_002229_48b0a9ff` |
| Demo pointer | `data/models/demo_latest.json` -> `2026-09-20_002229_48b0a9ff` |
| Dataset | `real_temporal_production_dataset.parquet` (274 rows; 174 pos / 100 neg) |
| Dataset SHA256 | `48b0a9ffb5da403ca9449d2e65a24a8f4ec0eb25376d3e5526d5ebc61e1d9225` (verified matches disk) |
| Events / roads / states | 58 confirmed events / 77 roads / 8 states |
| Time range | prediction_time 2017-07-03 -> 2026-07-19 |
| Algorithm | XGBoost, 200 trees, depth 2, lr 0.03, scale_pos_weight auto, monotone constraints on 13 rainfall/monsoon features |
| Code version (report) | git `cb9a79e`; tree is clean today at HEAD `ae0f73b` (report recorded working_tree_dirty=true) |
| Status | `HACKATHON_DEMO_READY_LIMITED_CONFIDENCE` |

**Test metrics (frozen threshold 0.820):** ROC-AUC 0.659, AP 0.922, precision 1.0,
recall 0.187, F1 0.315, balanced accuracy 0.593, Brier 0.155, confusion
[[12,0],[61,14]], 0 FPs, 61 FNs. Train-val generalization gap 0.013 (< 0.20).
Grouped-CV: 45 folds, only 4 two-class folds; two-class fold mean ROC-AUC 0.858
[0.5, 0.933, 1.0, 1.0]; pooled metrics are dominated by single-class folds and are
NOT the headline signal.

---

## 2. Final scorecard (25 phases)

Legend: **PASS** = verified and satisfies the phase gate; **PARTIAL** = mechanism
exists but incomplete/unproven; **FAIL** = does not satisfy the gate; **N/A** = out
of scope for this verification-only gate.

| # | Phase | Verdict | Evidence / note |
|---|---|---|---|
| 1 | Data realness & provenance | **PASS** | All 58 positive events map 1:1 to `event_label_registry.json` (source_url_status=VERIFIED, URL+date present); roads on valid CRS (EPSG:4326, sane bounds); full 30-day rainfall coverage on all 274 rows. **Finding:** 48 of 274 rows (21 (osm,time) groups) are feature-identical duplicates - distinct sitrep dates pin to the same nearest-weather prediction_time; the builder uniqueness key (event_id, osm_id, prediction_time) does not catch them, so those negatives are double-counted in training (weight inflation). |
| 2 | Absence of synthetic data in training | **PASS** | Production dataset label_source in {real_confirmed_temporal, real_observed_unaffected} only. All synthetic_* labels are tagged SYNTHETIC / TEST ONLY and confined to tests/legacy dev paths (ml/labels/generate.py:6, ml/experiment.py:13, ml/__init__.py:4). No training label is fabricated. |
| 3 | Label target & horizon integrity | **PASS** | Live check: 0/174 positive violations of prediction_time < event_time <= prediction_time+7d; positive lead times exactly {1,3,7}. Negative semantics are observation-anchored, horizon-imputed: sources report "road open" with publish date ~pt-1.8d but observation_window_end = pt+7d is computed, not independently observed. 99/100 negatives are corridor/named-road scope; only 1 is segment_observed_open. Documented in NEGATIVE_OBSERVATION_PROVENANCE.md and the ingest docstring. |
| 4 | Temporal leakage | **PASS** | ml/features/rainfall.py aggregates only strictly-negative offsets (day 0/future excluded); process_chirps_features.py is calendar-date-anchored and never fabricates "no rain" from missing data. Seasonal/monsoon features derive from the prediction timestamp at build and inference. Covered by test_temporal_splits.py, test_rainfall_aggregation.py. |
| 5 | Spatial leakage | **PARTIAL** | 8 roads overlap train/test, 4 train/val, 5 val/test. Expected (same road, different prediction times; features are time-anchored rainfall) so not hard label leakage, but static terrain features (77 unique slope/elev across 77 roads) are shared identities and the tiny two-class CV surface (4 folds) cannot isolate road-level memorization. No road-grouped rejection test. |
| 6 | Train/validation/test separation | **PASS** | split_by_events: chronological by event min-time, event-disjoint (0 events shared), tied dates grouped, test boundary never moved, validation expanded backward only to guarantee two classes. |
| 7 | Baseline comparison | **PARTIAL** | XGBoost beats majority and logistic on ROC-AUC and AP, but the plain rainfall_7day_threshold baseline scores 0.654 vs model 0.659 (delta ~0.005, within noise on n=12 negatives). Official verdict stays "model advantage is unproven" (baseline_comparison.statement). |
| 8 | Test-set size adequacy | **FAIL (production) / PASS (demo)** | Demo: 37 test events >= 3 -> pass. Production: needs >= 30 events and a defensible 2-class test; test has 87 rows / 75 pos / 12 negatives, 10 of which sit on a single date (2025-06-25) plus NH2 2025-08-29 and NH37 2025-09-03. Negatives cannot be expanded honestly: Sikkim archive tops at 2025-05-07, IFI ends 2023, ReliefWeb lacks corridor attestations, CHIRPS tops at 2026-07-31. |
| 9 | Operational decision threshold | **PARTIAL** | Frozen threshold 0.820 (max_recall_with_precision_floor_with_alert_budget), feasible=false on val (3 pos); gives precision 1.0 / recall 0.187. Demo feed ALERT rate is 8/14 (57%) vs MAX_REVIEWABLE_ALERT_RATE=0.30 - the scale-free per-window top-k policy plus a HIGH business-band absolute floor forces alerts above the 30% budget by design; acceptable for demo, not for production where over-alerting must be controlled. |
| 10 | Calibration | **PARTIAL** | identity_no_validation_calibration (no calibrator fitted); Brier 0.155. Live binned check: pred 0.29 -> actual 0.67, pred 0.52 -> actual 0.87, pred 0.73 -> actual 0.88, pred 0.83 -> actual 0.90 - systematically miscalibrated (underestimates positive rate until the highest bins). Probabilities must not be read as true disruption probabilities. |
| 11 | Error analysis | **PARTIAL** | 61 FN / 14 TP; all 20 missed-event groups concentrated near the threshold (max p 0.496-0.812, all < 0.82); FN anchors have high rainfall (7-day mean 73 mm, 30-day mean 361 mm) the model still under-ranks. Only 12 test negatives limit per-class decomposition. |
| 12 | Genuinely unseen input handling | **PASS** | Demo feed scores 14 segments at anchor 2026-07-31 = latest CHIRPS day, after max training prediction_time 2026-07-19, so no label bleed. OOD + freshness flags emitted (WEATHER_FRESHNESS_UNKNOWN; OUT_OF_DISTRIBUTION for NH717A/NH13). |
| 13 | Repeatability of a fixed input | **PASS** | Two independent fits with the frozen seed produce identical test predictions (max abs diff 0.0); model bundle deterministic and hash-stamped. |
| 14 | Adversarial / invalid input rejection | **PASS** | Live: negative rainfall, missing 7-day rainfall, NaN, lat 999, rainfall > 10000 mm, unknown field, string rainfall, slope 91, unsupported highway all rejected. Minor: list-valued elevation raises a bare TypeError (still rejected) instead of a clean ValueError. |
| 15 | Explainability | **PASS** | Per-prediction SHAP-style pred_contribs returned with explicit non-causal framing; monotone constraints verified live (0/260 sampled violations); gain importances physically sensible (rainfall_7day 5.55, slope_x_rain7 5.46, rainfall_30day, seasonal). |
| 16 | Traceability | **FAIL** | data/predictions/traces/prediction_traces.jsonl is stale/mixed: 41 rows, 38 are test_model, and none reference the current model 2026-09-20_002229_48b0a9ff; feature_version differs. Test traces are correctly segregated (test_prediction_traces.jsonl, 56 rows, current feature_version 2cbe0e4f...) - good - but the live trace is not current. |
| 17 | Reproducibility (hard gate) | **PARTIAL** | Deterministic given the frozen pipeline (verified this session). Dataset sha256 matches report; model sha256 matches report (fdffa703...). A full clean-environment raw rebuild from the lockfile alone was NOT re-executed this session; report recorded working_tree_dirty=true at train time (tree clean today). |
| 18 | Unit tests | **PASS** | python -m unittest discover -s tests -p "test_*.py" -> 134 tests, OK (temporal splits, grouped-CV, production gate, inference, artifact hash, seasonal, missingness, labels, composition smoke). |
| 19 | Data-validation tests | **PASS** | Builder QA: identity non-null, binary labels, non-negative rainfall, slope in [0,90], admissible label_source; all raise on violation and pass on the current dataset. |
| 20 | Leakage/schema automation | **PARTIAL** | Temporal leakage guarded in code + tests. No automated spatial-leakage or no-fabrication test on live labels (manual provenance review only). |
| 21 | Observation vs prediction distinction | **PASS** | Inference reports freshness flags, future-timestamp guard (predict_real_temporal.py), and separates data_quality.confidence_flags from the probability. Weather freshness is surfaced even when unknown. |
| 22 | Integration contract | **PASS (ML side)** | docs/API_CONTRACT.json v2.0.0 implemented end-to-end: backend serves /api/v1/models, /api/v1/feed, /api/v1/predict with contract-shaped responses. Verified live through the service layer (load_bundle + derive_features + predict + risk_level); hash-integrity checks run on load. Integration beyond demo tiers is not sanctioned (decision below). |
| 23 | Performance / latency | **PARTIAL** | Single-road scoring effectively instant (XGBoost, 170 KB .ubj, 200x depth-2 trees); 14-segment feed builds in seconds. No load/batch benchmark, no SLA. |
| 24 | Security | **PASS** | Tag whitelist + path-traversal guard (load_bundle rejects /, backslash, ..), SHA256 integrity on model/feature/calibration artifacts, no secrets committed (only .env.example; .env gitignored). No auth/rate-limit on the API (backend scope, not ML). |
| 25 | Documentation completeness | **PASS** | Rich honest docs: ML_AUDIT.md, ML_CONFIDENCE_TIERS_BLOCKERS.md, ML_PRODUCTION_READINESS_REPORT.md, ML_DATA_CATALOG.md, NEGATIVE_OBSERVATION_PROVENANCE.md, ML_DATA_GAP_REMEDIATION.md, RETRAINING_STRATEGY.md, MONITORING_STRATEGY.md, API_CONTRACT.json. This report supersedes the 2026-08-28 version. |

**Tally:** PASS 14 · PARTIAL 9 · FAIL 2 (test-set adequacy for production - row 8;
trace currency - row 16).---

## 3. Gate results (exact, from prod_report_2026-09-20_002229_48b0a9ff.json)

| Gate | Required | Actual | Result |
|---|---|---|---|
| Demo: test events | >= 3 | 37 | pass |
| Demo: ROC-AUC | >= 0.65 | 0.659 | pass |
| Demo: AP | >= 0.50 | 0.922 | pass |
| Production: test events | >= 30 | 37 | pass |
| Production: ROC-AUC | >= 0.75 | 0.659 | **FAIL** |
| Production: recall | >= 0.70 | 0.187 | **FAIL** |
| Production: AP | >= 0.60 | 0.922 | pass |

production_ready = false, demo_ready = true, status = HACKATHON_DEMO_READY_LIMITED_CONFIDENCE.
Grouped-CV two-class fold mean ROC-AUC 0.858 [0.5, 0.933, 1.0, 1.0] is positive but rests
on just 4 two-class folds and cannot carry high-confidence claims.

---

## 4. Why the model is (still) not production-ready (root causes)

1. **Production performance thresholds are not met.** ROC-AUC 0.659 < 0.75 and recall
   0.187 < 0.70 are hard gate failures; the model is not demonstrably better than a plain
   7-day rainfall rule (0.659 vs 0.654).
2. **Test-window negatives are inadequate.** 12 negatives, 10 clustered on one date,
   cannot support a stable recall/precision operating point or a defensible calibration
   target. Every realistic source is exhausted (Sikkim archive <= 2025-05-07, IFI <= 2023,
   ReliefWeb without corridor attestations, CHIRPS <= 2026-07-31) - hence **data gap**,
   not a code defect.
3. **Probabilities are miscalibrated.** Identity calibration + Brier 0.155 + measured
   positive-rate under-estimation below the highest bins. Real-world "probability of
   disruption in 7 days" claims are not yet defensible.
4. **Negative-label horizon is imputed, not observed.** The observation supports "open on
   publish date ~ pt"; "no disruption in (pt, pt+7]" is the ingest design inference
   (window_end = pt + 7d), and 99/100 negatives are corridor-level. Absence-in-future is
   not a source-backed observation.
5. **Duplicate feature rows inflate negative weights** (48 feature-identical rows across
   21 (osm,time) groups) and the sample-key uniqueness check does not detect them.
6. **Live prediction trace is stale** (does not reference the frozen model version).

None of the above are ML-scope code defects fixable inside a verification-only gate;
they are data-acquisition / calibration-data / operational gaps.

---

## 5. Decision

### BLOCKED BY DATA GAP - DEMO-READY, NOT PRODUCTION READY FOR INTEGRATION

The model passes the **demo gate honestly** (ROC-AUC 0.659 >= 0.65, AP 0.922 >= 0.50,
37 test events) and is safe to serve as an explicitly-labelled, low-confidence hackathon
demo with the existing caveats and the alert-budget note. It **must not** be integrated
into production routing, emergency response, or public safety flows: the production gate
fails on ROC-AUC (0.659 < 0.75) and recall (0.187 < 0.70), the baseline advantage is
unproven, probabilities are miscalibrated, and the test-window negative base cannot
support the required operating point.

Per the audit mandate: **no integration, no promotion, no changes outside ML scope
were made in this gate.**

**Required before production re-submission (data/ops scope, not code fixes):**
1. Source-backed **segment-level** negatives/future-event attestations (PWD/DDMA closure
   logs, post-2025-05-07 sitreps, district engineer records) so the test window has a
   real, non-single-date negative base and the horizon claim is observed, not imputed
   (highest priority). Target >= 30 test events and a negative pool supporting
   recall >= 0.70 at precision >= 0.30.
2. A real calibration set (model-free residual/binomial targets) so Brier and the identity
   calibrator can be replaced by a validated calibrator.
3. Deduplicate training samples at the (osm_id, prediction_time) key (keep max evidence/ID)
   and strengthen the builder QA check to reject duplicate feature rows before training.
4. Regenerate/refresh the live prediction trace against the frozen model version.
5. >= 95% terrain coverage across all 8 states with a spatially-neutral coverage check
   (prior audit noted 4-state terrain gaps).
6. A clean-environment, locked-dependency, full raw rebuild recorded at its owning commit
   (complete the reproducibility hard gate).

**Explicitly NOT done:** integration with frontend/backend/mobile/database; promotion of
prod_latest.json beyond HACKATHON_DEMO_READY_LIMITED_CONFIDENCE; fabrication of any
metric, label, or source.

---

## 6. What was verified as genuinely good (retain)

- **Honest demo gate.** 0.659 / 0.922 / 37 events is a real, reproducible PASS on
  source-confirmed positives and observation-backed negatives - a genuine improvement over
  the 0.54-era GATED model.
- **Zero positive-label temporal violations** (leads exactly {1,3,7}); no synthetic
  training data; the no-fabrication rule is enforced in code and data.
- **No temporal leakage** - rainfall strictly pre-anchor; seasonal features derived from
  the prediction timestamp.
- **Monotone rainfall/monsoon constraints hold** (0 violations on the live model);
  predictions are explainable per-feature with non-causal framing.
- **Reproducible, hash-integrity-checked artifacts** (dataset, model, feature spec,
  calibration all SHA256-verified at load).
- **134/134 tests green; clean data-validation gates; adversarial inputs rejected.**
- **Backend /predict and /feed implement the API contract** and score hash-verified
  bundles only for demo-authorised tags.
- Duplicate-row, calibration, test-window, trace-currency, and alert-rate findings are
  surfaced honestly in this scorecard rather than masked.

*End of report.*
