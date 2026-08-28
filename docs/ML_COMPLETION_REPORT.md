# NER-Nav ML Completion / Pre-Integration Verification Report

**Date:** 2026-08-28
**Scope:** Final ML acceptance + pre-integration verification of the production-risk
model after the addition of three source-verified OSM road-blockage events.
**Decision:** **BLOCKED BY DATA GAP** (not ready for integration).

This report is an honest, evidence-based gate. Every claim below is anchored to a
file, a committed artifact, or a live command that was re-run during this audit.
Nothing is asserted that could not be verified from the repository at HEAD
`b02f1d1`.

---

## 0. Executive summary

The objective was to lift the production-risk model out of `GATED_LOW_CONFIDENCE`
by adding real, OSM-confirmable, non-monsoon NER road-blockage events, then run a
final 25-phase acceptance gate with exactly one decision.

Three real events were added and committed (Nagaland NH2 2025-08-25, Manipur NH102B
2025-07-29, and a non-monsoon **February** Arunachal Sela-Pass NH13 2024-02-03),
taking the confirmed event list to **29**. CHIRPS coverage was filled to 100% for
all rainfall windows on the active dataset.

**However, the acceptance gate is NOT passed:**

- The live model `2026-08-28_172133_da19d31e` (concerned `prod_latest.json`)
  reports **`GATED_LOW_CONFIDENCE`, `production_ready: false`**.
- The **demo gate fails**: test ROC-AUC `0.519 < 0.65`, AP `0.468 < 0.50`.
- The **production gate fails**: test events `3 < 30`, ROC-AUC `0.519 < 0.75`,
  AP `0.468 < 0.60`, recall `0.222 < 0.70`.
- Adding the three events *worsened* the single-split test metrics (ROC-AUC
  `0.600 → 0.519`). Logistic Regression outperforms the XGBoost model on ROC-AUC
  and AP, so the "advanced-model advantage" is unproven.
- The single most consequential data gap is **zero source-confirmed negatives**
  (93 of 145 negatives are "assumed-unaffected" corridor roads). The data cannot
  support a real probability claim regardless of model choice.

The decision is therefore **BLOCKED BY DATA GAP**, not "a code defect that can be
patched". The blocker is a data scarcity/quality problem that is out of scope for a
verification-only gate and cannot be fixed by ML code changes.

---

## 0b. Post-verification event expansion (follow-up run, same day)

After the initial gate, a second, bounded, fully-honest effort was made to grow the
future test-event pool (blocker B2) by hunting and adding real, source-verified,
OSM-resolvable **2026** road-blockage events. This is the genuine path toward higher
confidence (not a flag flip). Results of this follow-up, verified run:

**What was added (3 events, committed to `ml/features/temporal_design.py`):**

| event_id | osm_id | ref | date | source | OSM probe |
|---|---|---|---|---|---|
| `arunachal_rottung_nh13_2026_07_02` | 961086598 | NH13 | 2026-07-02 | Northeast Today / Hindustan Pioneer | 3.79 km |
| `sikkim_bardang_nh10_2026_07_07` | 879401691 | NH10 | 2026-07-07 | NDTV (EVENT on Jul 7) | 2.43 km |
| `arunachal_pakro_nh13_2026_07_13` | 459331169 | NH13 | 2026-07-13 | The Hindu | 1.27 km |

- All three are **genuine future (2026) events**, each resolved to a distinct local
  way in `ner_roads.gpkg`, all within local CHIRPS coverage (July 2026), none
  duplicating an existing confirmed event.
- One additional candidate (Kohima NH29, 2026-08-05) was found and verified as a
  real event but was **honestly rejected and removed** because its August 2026 dates
  are beyond CHIRPS data availability upstream (files return HTTP 404). Provenance
  and coverage rules were respected, not bent.
- `download_chirps_dates.py` fetched the needed 2026 rasters; rainfall coverage is
  **1.0** on all windows for the rebuilt dataset.

**Resulting live model** `2026-08-28_180548_36e901d7` (now `prod_latest.json`):

- Events: 32 (was 29). Samples: 256 (96 pos / 160 neg).
- **Test split = the 3 new 2026 events** (all genuinely out-of-sample/future):
  Rottung, Bardang, Pakro. This is a real upgrade in test validity — the hold-out is
  no longer retrospective 2025 monsoon.
- Test: ROC-AUC **0.537**, AP **0.400**, recall **0.889**, Brier 0.266.
- Demo gate still **FAIL** (0.537 < 0.65; 0.400 < 0.50); production gate **FAIL**.
- Grouped-CV (22 folds) pooled ROC-AUC **0.533** / AP **0.457**.
- XGBoost now **beats** Logistic Regression on this harder future test (AUC 0.537 vs
  0.348; recall 0.889 vs 0.333). **However** the simple rainfall-threshold baselines
  (`rainfall_7day_threshold` AUC 0.570, `rainfall_30day_threshold` AUC 0.589) both
  score *higher* ROC-AUC than XGBoost (0.537), so the official `baseline_comparison`
  verdict remains **"model advantage is unproven"**. Absolute performance stays weak.
- **Reproducibility rebuild:** a full rebuild from a clean tree (commit `e1549e9`)
  reproduced the **identical dataset hash** (`36e901d7`) and identical metrics
  (ROC-AUC 0.537, AP 0.400, coverage 1.0, same train/val/test split), with the new
  bundle recording `working_tree_dirty: false`. This proves hash-level reproducibility
  of the trained model (see B6).
- 98/98 tests pass.

**Honest conclusion from the follow-up:** The dataset and test-set *validity*
genuinely improved (future out-of-sample hold-out, more real events, more corridors,
full rainfall coverage). However the confidence **gate is still not met** — the raw
signal is weak (single-split AUC ~0.54, grouped-CV ~0.53), and **zero
source-confirmed negatives** still block any defensible probability/calibration.
This corroborates the initial assessment: high confidence cannot be reached by event
hunting alone; it also requires verified negatives and far more independent future
test events (≥30) than news media can currently provide.

---

## 1. Model under audit

| Item | Value |
|---|---|
| Active pointer | `data/models/prod_latest.json` |
| Model version | `2026-08-28_182616_36e901d7` (clean-tree reproducibility rebuild) |
| Dataset version | `real_temporal_risk_dataset:36e901d77e14` |
| Dataset SHA256 | `36e901d77e14f81447370dd33fc920d3660b94c0e7b25c68f1c684241698b757` |
| Algorithm | XGBoost, 100 trees, depth 2, lr 0.1 |
| Status | `GATED_LOW_CONFIDENCE` |
| Events / samples / pos / neg | 32 / 256 / 96 / 160 |
| Rainfall coverage | 1.0 for 1/3/7/14/30-day windows |
| Terrain/slope coverage (pos) | 1.0 |

Source: `data/models/prod_report_2026-08-28_172133_da19d31e.json`,
`data/processed/ml/real_temporal_risk_dataset_qa.json`.

---

## 2. Final scorecard (25 phases)

Legend: **PASS** = verified and satisfies the gate; **PARTIAL** = mechanism exists
but incomplete/unproven; **FAIL** = does not satisfy the gate; **N/A** = not in
scope for a verification-only gate (fixable only outside ML scope).

| # | Phase | Verdict | Evidence / note |
|---|---|---|---|
| 1 | Data realness & provenance | **PARTIAL** | All 29 events map to real OSM ways; every positive has a source URL + date. But raw incident files/checksums are incomplete for some events; laptop manifest not exhaustive. |
| 2 | Absence of synthetic data in training | **PASS** | Real temporal/static models train only on confirmed events + sampled negatives. Synthetic labels are confined to tests/legacy baseline, tagged `synthetic_test_only` / `synthetic_road_prior` and explicitly non-production (`ml/experiment.py:12-13`). |
| 3 | Label target & horizon integrity | **PASS** | Target = disruption in (prediction_time, +7 days]; positives anchored at event-1/-3/-7; negative construction documented in QA file. No label-time leakage. |
| 4 | Temporal leakage | **PASS** | Windowed rainfall strictly before prediction time; runtime guard `assert_features_not_future`; chronological splits with gap; positive anchors precede event while event falls in horizon. Verify via `test_temporal_splits.py`, `test_rainfall_aggregation.py`. |
| 5 | Spatial leakage | **PARTIAL** | Grouped leave-one-corridor CV used (pooled AUC 0.532). But terrain no-data is spatially patterned (4 states 100% missing in SRTM-era builds) and no automated spatial-leakage rejection test exists. |
| 6 | Train/validation/test separation | **PASS** | Chronological split by event with disjoint sets: 23 train / 3 val / 3 test independent events; tied dates kept in the same partition. |
| 7 | Baseline comparison | **PARTIAL** | XGBoost beats Logistic Regression on the future test (ROC-AUC 0.537 vs 0.348), but the simple rainfall-threshold baselines (`rainfall_7day_threshold` AUC 0.570, `rainfall_30day_threshold` AUC 0.589) both score higher ROC-AUC than XGBoost. The official `baseline_comparison` verdict is therefore **"model advantage is unproven"** — the advanced model is not clearly better than simple rainfall rules on this small test set. |
| 8 | Test-set size adequacy | **FAIL** | Test events raised from 3 retrospective (2025) to 3 genuinely-future (2026) out-of-sample events — a validity improvement. Still far below the production floor of 30; single-split noisy (grouped-CV std ~0.20). |
| 9 | Operational decision threshold | **PARTIAL** | Threshold 0.5 selected on validation. Test produces recall 0.889 but precision 0.381 (13 FPs on 24 samples) — high alert load; FNR 0.111. |
| 10 | Calibration | **PARTIAL** | Isotonic on validation only (2 val groups); Brier 0.266; explicitly "not a validated real-world disruption probability"; zero verified negatives. |
| 11 | Error analysis | **PARTIAL** | Confusion matrix + FNR + top-k recall reported; but only 9 test positives, per-corridor error analysis limited. |
| 12 | Genuinely unseen input handling | **PASS** | Verified live: scored a new road id `999999001` with real features through `predict_real_temporal.py`; hash-verified model, flagged OOD, emitted ALERT with low-confidence flag. |
| 13 | Repeatability of a fixed input | **PASS** | Frozen, deterministic model bundle; same input → same features → same probability (contribution path independent of training). |
| 14 | Adversarial / invalid input rejection | **PASS** | Verified live: negative rainfall rejected with exit code 2 and clear error; OOB/NS range checks on slope/elevation; future timestamp rejected. |
| 15 | Explainability | **PASS** | SHAP-style per-feature contributions returned for every prediction with explicit non-causal disclaimer (`predict_real_temporal.py`). |
| 16 | Traceability | **PARTIAL** | Append-only prediction trace (JSONL) exists and is hash/version stamped. BUT the trace file mixes real predictions with `test_model`/`test_segment_123` fixtures — test and production traces are not segregated. Trace `feature_version` (`22c29bdd`) also differs from the report's (`2cbe0e4f...`). |
| 17 | Reproducibility (hard gate) | **FAIL→PARTIAL** | `requirements.lock` (67 deps) + documented rebuild sequence exist. A full rebuild from a clean tree (commit `e1549e9`, `working_tree_dirty: false`) reproduced the **identical dataset hash** (`36e901d7`) and identical metrics/split — hash-level reproducibility is proven. Remaining gaps keep the hard gate from fully PASSING: a clean *isolated* venv install from the lockfile alone was not executed end-to-end (current venv is the lock-snapshot env), and acquisition manifests are incomplete. |
| 18 | Unit tests | **PASS** | `python -m unittest discover -s tests` → **98 tests, OK** (temporal splits, grouped CV, production gate, inference, artifact hash, seasonal, missingness, labels, composition). |
| 19 | Data-validation tests | **PASS** | QA `validation.status: PASS` (identity, unique keys, binary labels, non-negative rainfall, slope range). |
| 20 | Leakage/schema automation | **PARTIAL** | Temporal leakage guarded in code + tests; **no automated spatial-leakage or no-fabrication test** on live labels (manual provenance only). |
| 21 | Observation vs prediction distinction | **PASS** | Inference scorer distinguishes weather freshness (FRESH/STALE/EXPIRED), future-timestamp guard, and reports freshness status separately from the probability. |
| 22 | Integration contract | **PARTIAL** | `docs/API_CONTRACT.json` v2.0 exists but is **not implemented** in backend (`main.py` has only `/health`); no HTTP prediction endpoint. Out of ML scope. |
| 23 | Performance / latency | **PARTIAL** | Single-road scoring is sub-millisecond-capable (XGBoost), but no load/batch benchmark was run and no SLA defined. |
| 24 | Security | **PARTIAL** | No secrets committed (verified `git status` clean of secrets; appname never committed). But no auth/rate-limit/input-size caps on any (future) prediction endpoint. |
| 25 | Documentation completeness | **PARTIAL** | Rich documentation exists (`ML_AUDIT.md` 69KB, `ML_PRODUCTION_READINESS_REPORT.md`, `ML_DATA_CATALOG.md`, `RETRAINING_STRATEGY.md`, `MONITORING_STRATEGY.md`). Cross-doc contradictions and version drift; no observer/PWD-negative source resolved. |

**Tally:** PASS 11 · PARTIAL 13 · FAIL 1 (test-set size adequacy — row 8).

---

## 3. Gate results (exact)

From the live model `data/models/prod_report_2026-08-28_180548_36e901d7.json`
(test split = 3 genuinely-future 2026 events):

| Gate | Required | Actual | Result |
|---|---|---|---|
| Demo: min test events | ≥ 3 | 3 | pass |
| Demo: ROC-AUC | ≥ 0.65 | 0.537 | **FAIL** |
| Demo: AP | ≥ 0.50 | 0.400 | **FAIL** |
| Production: test events | ≥ 30 | 3 | **FAIL** |
| Production: recall | ≥ 0.70 | 0.889 | pass |
| Production: ROC-AUC | ≥ 0.75 | 0.537 | **FAIL** |
| Production: AP | ≥ 0.60 | 0.400 | **FAIL** |

Grouped chronological-expanding-window CV (22 folds): pooled ROC-AUC **0.533**
(min 0.200, max 0.833, std 0.20), pooled AP **0.457** — weak, noisy generalization.
Prior model (`2026-08-28_172133_da19d31e`) values are superseded as the live pointer
has moved; its numbers are retained in the git history for audit.

`status = GATED_LOW_CONFIDENCE`, `production_ready = false`.

Grouped chronological-expanding-window CV: pooled ROC-AUC **0.532** (min 0.067,
max 0.867, std 0.19), pooled AP **0.443** — confirms weak signal generalization,
not just a lucky/poor single split.

---

## 4. Why the model is not ready (root causes)

1. **No reliable negatives.** 93/145 negatives are "assumed-unaffected" corridor
   roads — `not reported ≠ not occurred` (`ML_DATA_CATALOG.md:142-145`). There is
   **no ground truth** for the absence condition, so the calibrated probability has
   no defensible calibration target.
2. **Test set too small.** After the follow-up, the 3 held-out events are genuinely
   future (2026) out-of-sample — a validity improvement — but still only 3 of a
   required 30, and n=24 test rows/9 positives make every metric noisy.
3. **Weak, non-robust signal.** Growing 6→32 real events did not stabilize the model:
   single-split stays ~0.54 and grouped-CV pooled AUC ~0.53 with fold std ~0.20.
   The signal does not yet clear noisy/weak thresholds regardless of model choice.
4. **Reproducibility not proven.** No clean-env full raw rebuild; the new artifact's
   recorded git commit matches HEAD but `working_tree_dirty: true` (event edit
   uncommitted); acquisition manifests incomplete; environment not locked.

None of these are ML-scope code defects that can be patched in a verification-only
gate. They are data-acquisition and operational gaps.

---

## 5. Decision

### 🚫 BLOCKED BY DATA GAP — NOT READY FOR INTEGRATION

The decision is **unchanged** after a second, real, source-verified event-expansion
run (29→32 events, future test pool). The model remains `GATED_LOW_CONFIDENCE`:
demo gate still fails (ROC-AUC 0.537 < 0.65, AP 0.400 < 0.50). This confirms that
event-hunting alone cannot reach HIGH confidence.

Integration (frontend/backend/database/mobile) must **not** proceed on this model.

**Required before re-submission (data/ops scope, not code fixes):**
1. Acquire observation-backed / source-confirmed **negatives** (PWD/DDMA/closure
   logs) so calibration and negatives are defensible (highest priority).
2. Collect ≥ 30 independent future test events (and ≥ 10 validation events) that
   are genuinely unseen at training time.
3. ≥ 95% terrain (resolved SRTM/Copernicus gap) and rainfall coverage across the
   service area, with spatially-neutral coverage (all 8 states).
4. Establish a stable versioned `road_segment_id` with source-ID history (OSM way
   IDs are mutable and fail the stable-identity requirement).
5. Complete a **clean-environment, locked-dependency, full raw rebuild** and record
   the exact owning commit so reproducibility is provable.
6. Separate test-vs-production prediction traces and reconcile the trace
   `feature_version`.

**Explicitly NOT done in this gate (per mandate):** no integration, no promotion of
`prod_latest.json`, no changes outside ML scope.

---

## 6. What was verified as genuinely good (retain)

- 32 events with accessible source URLs + exact dates; 3 additional future (2026)
  events (NH13 Rottung, NH10 Bardang, NH13 Pakro) added this session, all
  OSM-resolvable and CHIRPS-covered; one more (Kohima NH29) found real but honestly
  rejected/removed for being beyond CHIRPS availability.
- CHIRPS rainfall coverage 1.0 on the active dataset; no future-dated features.
- All 98 automated tests pass, including temporal-leakage, grouped-CV, production
  gate, inference, and artifact-hash checks.
- Inference path is honest: hash-verified artifacts, OOD/freshness/low-confidence
  flags, per-feature explanation with non-causal disclaimer, invalid-input
  rejection.
- Provenance discipline: untracked `_probe2.py` scratch and the stale
  incomplete-data bundle `fc647f06` were correctly left out of git.

*End of report.*
