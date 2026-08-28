# NER-Nav — Confidence Tiers & Blockers (What Separates Us From HIGH)

**Date:** 2026-08-28
**Purpose:** Precise, evidence-backed definition of the exact gaps between the current
state (GATED_LOW_CONFIDENCE) and each target confidence tier. This document answers
"what exactly must change for the model to be honestly gated as HIGH confidence."

The tiers are defined by the model's own gates (`scripts/train_production_risk_model.py:53-60`)
plus the honesty floor required to make a probability claim defensible.

---

## 0. Where we are now (verified)

Live artifact `data/models/prod_latest.json` → `2026-08-28_172133_da19d31e`:

| Metric | Value | Gate needs |
|---|---|---|
| Status | `GATED_LOW_CONFIDENCE` | — |
| Test events | 3 | ≥ 30 (production) |
| Test ROC-AUC | 0.519 | ≥ 0.65 (demo) / ≥ 0.75 (prod) |
| Test AP | 0.468 | ≥ 0.50 (demo) / ≥ 0.60 (prod) |
| Test recall | 0.222 | ≥ 0.70 (prod) |
| Grouped-CV pooled AUC | 0.532 | robust ≥ 0.65+ |
| Source-confirmed negatives | **0** | needs verified negatives |
| Terrain coverage (network) | patterned (some states 0%) | ≥ 95% service area |
| Clean-env reproducibility | not proven | proven |

---

## 1. Tier definitions

### DEMO_READY_LIMITED_CONFIDENCE (achievable target short-term)
- Test events ≥ 3; test ROC-AUC ≥ 0.65; AP ≥ 0.50.
- *This is a research/demo posture, explicitly NOT production-safe.*

### PRODUCTION_READY_EVIDENCE_SUPPORTED (production target)
- Test events ≥ 30; recall ≥ 0.70; ROC-AUC ≥ 0.75; AP ≥ 0.60.
- Plus honesty floor: verified negatives, ≥95% terrain coverage, stable road identity,
  reproducible training.

### HIGH_CONFIDENCE (the user's requested "full high confidence")
- A production-ready model with **demonstrated, robust** performance:
  - Performance held on multiple held-out blocks (low grouped-CV variance, not a
    lucky split).
  - **Verified negatives** so both the probability and its calibration are defensible.
  - Spatial coverage across all 8 states / 10 corridors (no 0%-coverage states).
  - Forward-looking validation (real future pollution, not only retrospective media
    positives constructed after known events).
  - Fully reproducible locked-environment rebuild with exact owning commit.
  - Advanced-model advantage proven (currently logistic regression ≥ XGBoost).
  - Stable, versioned `road_segment_id` (OSM way ids are mutable).

---

## 2. Blockers by category (exact gaps)

### B1 — Label / ground-truth quality (HARDEST, most decisive)
- **0 source-confirmed negatives.** 145 negative rows = 23 "same-road, out-of-horizon"
  controls + 122 "assumed-unaffected" corridor roads. `not reported ≠ did not occur`.
  Calibration (Brier 0.307) and probability have no defensible negative ground truth.
- Required: observation-backed negatives from an operational source
  (PWD / DDMA / NHIDCL road-closure logs, or an approved ReliefWeb stream with
  negative/reopened windows). This is an **external data acquisition** — not fixable
  by ML code or event hunting alone.

### B2 — Test-set size & forward-looking validity
- Only **3 future test events** (2025-08/09) vs production floor of 30.
- All positives are retrospective (constructed after media-reported events). No
  forward validation on genuinely-unseen future disturbance.
- Every genuinely-new *future* event (2026) that resolves locally and falls in CHIRPS
  coverage would directly grow the "future test event" pool.

### B3 — Weak, non-robust signal
- Adding events regressed the model: 6→29 events moved single-split AUC 0.600→0.519.
- Grouped-CV pooled AUC ~0.53 with fold std 0.19 → no stable signal.
- Logistic Regression beats XGBoost (AUC 0.696 vs 0.519; AP 0.678 vs 0.468) →
  "advanced-model advantage is unproven."

### B4 — Non-monsoon temporal gap
- Jan (0), Apr (0), Nov (0), Dec (0) events at OSM granularity are the documented
  hard gaps. Feb has exactly 1 (Sela 2024).
- Prior attempts: Sikkim GLOF NH10 (Nov 2023) rejected (nearest local way 18+ km);
  NH29 Jan / NH6 Nov-Dec returned only monsoon events.

### B5 — Feature coverage gaps
- Terrain/slope missingness is patterned; some states have 0% terrain coverage in
  SRTM-era builds (Copernicus fixed training corridors but not the full network).
- Active dataset rainfall coverage is now 1.0 (good) but was historically ~23%
  missing and prone to drift.

### B6 — Reproducibility (hard gate, currently FAIL)
- A frozen dependency lockfile (`requirements.lock`, 67 pinned deps) and a documented
  clean-env rebuild sequence (`docs/RETRAINING_STRATEGY.md` §Reproducibility) now exist.
  This closes part of the "no environment lockfile" gap.
- Still FAILING: a clean-environment, dependency-locked, full-raw rebuild has NOT been
  executed and validated end-to-end.
- The live artifact records `git_commit: 7c96d338` with `working_tree_dirty: true`,
  but is managed under commit `b02f1d1` → owning-commit mismatch.
- Acquisition manifests (raw incident files, checksums, URLs, timestamps, usage
  terms) incomplete. A per-event label registry scaffold now exists
  (`data/processed/ml/event_label_registry.json`), but only 5 of 32 events have
  recorded source URLs; the rest are honestly marked UNRECORDED (not fabricated).

### B7 — Identity & ops
- Uses mutable OSM way id as the segment identity → fails the stable-identity
  requirement (a documented limitation in `ML_PRODUCTION_READINESS_REPORT.md:355`).
- No monitoring owner, no SLA, no loss-feedback loop.

---

## 3. What will NOT move us to HIGH confidence (do not mistake for progress)

- Adding *more monsoon, retrospective* positives alone — historically made metrics
  *noisier* (B3).
- Raising gate thresholds or relabeling — fabrication, prohibited.
- More baseline/threshold tuning on a weak signal — overfitting to 3-9 holdout points.
- A larger model / more features — the limitation is ground truth (B1), not capacity.

## 4. What WOULD move us (in priority order)

1. **Acquire verified negatives** (operational source / appname) — unblocks B1, the
   single biggest step.
2. **Add genuinely-future (2026) local, source-verified events** that resolve in the
   local OSM network and fall in CHIRPS coverage — grows the future-test pool (B2).
3. **Prove reproducibility** — run and validate the locked clean-env rebuild at the
   owning commit (lockfile exists; the rebuild has not yet been executed) (B6).
4. **For every confirmed event**, complete the label registry (raw file, checksum,
   URL, timestamp, usage terms) (B6/B7).
5. **Terrain coverage** across all corridors/service area (B5).

*Target state when done: production gates pass, grouped-CV robust, verified negatives
present, reproducible lockstep, spatial coverage complete → then and only then HIGH
confidence is honest.*
