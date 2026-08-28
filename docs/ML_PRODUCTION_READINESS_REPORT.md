# NER-Nav ML Production Readiness Report

Assessment date: 2026-08-28

## 1. Executive Summary

The repository contains a functioning real-data research pipeline for
seven-day road-disruption risk. It has event-disjoint chronological splitting,
past-only rainfall windows, model persistence, validation-only threshold
selection, simple baselines, calibration metrics, artifact hashing, input
validation, explanation output, and prediction tracing.

It is not capable of supporting real-world NER logistics decisions today.
The principal blockers are zero source-confirmed negative road/windows, only
23 positive event groups, only four independent future test events, 70.11%
terrain missingness, 23.37% rainfall missingness, mutable OSM way identity,
incomplete source manifests, weak model performance, and no operational
monitoring service.

After expanding from 19 to 23 real event groups, the chronological XGBoost
ROC-AUC fell from 0.652 to 0.550 and average precision from 0.535 to 0.481.
This regression is retained and reported.

## 2. Data Sources

The active system uses:

- OpenStreetMap/Geofabrik road geometry and tags under ODbL.
- Survey of India state/district boundaries with incomplete local license and
  acquisition records.
- CHIRPS v2 daily 0.05-degree rainfall. The provider describes CHIRPS as
  public-domain data from 1981 to near-present:
  https://www.chc.ucsb.edu/data/chirps
- Six SRTM v4.1 terrain tiles.
- Twenty-three manually curated disruption event/road tuples from mixed
  government, scientific, and media evidence.

The detailed inventory is in docs/ML_DATA_CATALOG.md.

## 3. Data Quality

The active modelling table has 184 rows, no duplicate
event/road/prediction keys, no negative rainfall, no invalid slopes, and all
required columns. It has material coverage failures:

| Measure | Result |
|---|---:|
| Rainfall coverage | 76.63% |
| Terrain/elevation coverage | 29.89% |
| Terrain missingness | 70.11% |
| Rainfall missingness | 23.37% |
| Source-confirmed negative windows | 0 |

The active builder now rejects null sample identity, duplicate sample keys,
non-binary labels, negative rainfall, and slopes outside 0-90 degrees. A future
rebuild records hashes for roads, terrain, rainfall, and the output dataset.

## 4. Geographic Coverage

The road foundation covers eight NER states and 289,841 OSM ways. Model
evidence covers only 23 events, 115 source road IDs, and nine NH references.
Tripura, Sikkim, Meghalaya, and Mizoram have especially few independent
events. Dataset rows do not represent full NER road-class, terrain, district,
or accessibility variation.

District attribution reports 100% assignment, but source names contain encoding
artifacts. The model uses an OSM way ID rather than a durable, versioned
road_segment_id. This fails the stable-identity requirement.

## 5. Temporal Coverage

Prediction anchors span 2017-07-01 to 2025-09-13. Coverage is sparse event
windows, not a continuous surveillance history. The latest split is:

| Partition | Independent events | Prediction period |
|---|---:|---|
| Train | 15 | 2017-07-01 to 2025-03-14 |
| Validation | 4 | 2025-05-18 to 2025-06-23 |
| Test | 4 | 2025-07-02 to 2025-09-13 |

Tied event dates remain in the same split, which expanded nominal 2/3 event
targets to actual 4/4 validation/test groups.

## 6. Target Definition

- Unit: an OSM road way at a prediction date.
- Target: whether a recorded logistics-disrupting event affects that mapped
  road in the next seven days.
- Positive condition: prediction_time < event_time <= prediction_time + 7 days.
- Negative condition in current data: named event outside the horizon or no
  event recorded for a sampled corridor road.

The negative condition is not reliable enough for a real probability claim.

## 7. Label Quality

There are 69 positive rows derived from three anchors for each of 23 events,
23 same-road controls, and 92 assumed-unaffected corridor rows. Three positive
rows for one event are correlated observations, not three independent events.

Only 13 processed event JSON files were found for the 23-event code list.
Several processed records reference research notes/media and omit raw local
files, checksums, acquisition timestamps, exact source URLs, or cleared usage
terms. The label registry is incomplete.

## 8. Feature Catalogue

The frozen model has 16 features:

- Rainfall: 1, 3, 7, 14, and 30-day totals.
- Terrain: elevation and slope.
- Ratios: 1/7, 3/14, 3/7, 1/3, and 7/30 rainfall ratios.
- Missingness: terrain_known.
- Interactions: slope x rain7, slope x rain30, elevation x slope.

Inference now recomputes derived features from validated base observations
instead of trusting caller-supplied interaction values.

Road class, bridge, state, district, season, historical event frequency, and
nearby incidents are not in the final model. They should not be added until
availability, leakage, and sample-size constraints are resolved.

## 9. Leakage Analysis

Rainfall windows are anchored strictly before prediction time. Positive event
times are after prediction time and within the seven-day horizon. Events do not
straddle chronological partitions. Threshold and isotonic calibration are fit
on validation only.

Remaining risks:

- Positive samples are constructed retrospectively around known events, while
  background non-event time is poorly observed.
- Corridor evaluation is geographic but not a pure forward-time validation.
- Manual event selection may encode reporting and availability bias.

No evidence of direct feature-time leakage was found in the active feature
contract.

## 10. Validation Strategy

The production harness uses chronological event groups with tied dates kept
together. The final test has four events, far below the production floor of 30.
Grouped leave-one-corridor-out evaluation tests spatial separation across nine
corridors, but not future deployment behavior.

Rolling forward validation is not statistically useful yet because there are
only 23 event groups. Random splitting is rejected.

## 11. Baselines

Chronological test metrics:

| Model | ROC-AUC | Avg precision | Precision | Recall | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| Majority/no disruption | 0.500 | 0.375 | 0.000 | 0.000 | 0.000 | 0.375 |
| Rainfall 7-day | 0.454 | 0.346 | 0.375 | 1.000 | 0.545 | 0.387 |
| Rainfall 30-day | 0.454 | 0.366 | 0.375 | 1.000 | 0.545 | 0.403 |
| Logistic Regression | 0.596 | 0.437 | 0.500 | 0.583 | 0.538 | 0.255 |
| XGBoost | 0.550 | 0.481 | 0.412 | 0.583 | 0.483 | 0.284 |

XGBoost does not beat Logistic Regression on ROC-AUC, precision, F1, or Brier
score. It has higher average precision. Advanced-model advantage is unproven.

## 12. Model Comparison

Grouped corridor evaluation on all 184 rows:

| Model | ROC-AUC | Avg precision | Recall | F1 |
|---|---:|---:|---:|---:|
| XGBoost | 0.584 | 0.425 | 0.174 | 0.245 |
| Logistic Regression | 0.489 | 0.382 | 0.609 | 0.462 |
| Rainfall 7-day | 0.512 | 0.392 | 0.420 | 0.409 |
| Rainfall 30-day | 0.483 | 0.397 | 0.290 | 0.305 |

XGBoost ranks better in this grouped evaluation but misses 57 of 69 positive
samples at threshold 0.5. Model choice cannot be finalized from these event
counts.

## 13. Final Model

Latest immutable research model:

- Model version: 2026-08-28_113307_2cccd364
- Dataset SHA256:
  2cccd364ad2b5ca1abfeefbe1c71fffb313ef546e1653eb2279e913d19f01ace
- Feature version:
  35e8d5f71ccc0505f3efe34d73b9a58a5a938ac46f9c446bf8d82b5e80aabff8
- Algorithm: XGBoost, 100 trees, depth 2, learning rate 0.1.
- Status: GATED_LOW_CONFIDENCE.

It is retained as a research artifact, not accepted as the production model.

## 14. Probability Calibration

Isotonic calibration is fitted on four validation event groups and stored as
JSON thresholds rather than pickle. Test Brier score is 0.284. Four events are
insufficient to establish calibration reliability. Calibration curves and ECE
would be misleadingly unstable at this size.

The output may be displayed as a model score for research. It must not be
represented as a validated real-world disruption probability.

## 15. Threshold

Validation selected threshold 0.3793 with target recall >=0.70. On test:

| Metric | Value |
|---|---:|
| Precision | 0.412 |
| Recall | 0.583 |
| F1 | 0.483 |
| False positives | 10 |
| False negatives | 5 |
| False-negative rate | 0.417 |
| Alert rate | 53.1% |

The threshold misses the operational recall target and exceeds a reviewable
alert load.

## 16. Error Analysis

Test confusion matrix is [[10, 10], [5, 7]]. Five of twelve positive samples
are missed. Ten of twenty negative samples alert. The grouped corridor result
has 57 false negatives and 17 false positives at threshold 0.5.

Reliable state/district/season/road-class error slices cannot be estimated with
so few independent events. Sample-level slices would overstate confidence
because the three positive anchors share one event.

## 17. Generalization

The model is not demonstrated to generalize across NER. Nine corridors are
represented, but event counts by state/corridor are too small and imbalanced.
Terrain is missing for most active samples. There is no independent evaluation
for road classes beyond selected trunk/NH corridors, dry-season behavior,
flood versus landslide mechanisms, or unseen states with adequate event counts.

## 18. Robustness

Implemented inference controls:

- Required finite, non-negative rainfall.
- Slope in 0-90 degrees and plausible elevation range.
- Timezone-aware timestamps and rejection of future prediction times.
- Weather freshness states.
- Missing-terrain flag.
- Training-profile p01/p99 OOD indicators.
- Derived-feature recomputation.
- Artifact hash and schema checks.
- Controlled JSON errors.

Not implemented end to end: external API outage simulation, queue/database
failure recovery, concurrency/load testing, malformed geospatial requests, and
network-partition behavior.

## 19. Out-of-Distribution Analysis

Training-only min, max, p01, and p99 profiles are frozen with the feature
schema. Inference emits OUT_OF_DISTRIBUTION and lists outlying features.

This is a heuristic guard, not a validated OOD detector. Geographic, seasonal,
and event-rate shift require stable segment identity, live input logging, and
larger reference data.

## 20. Explainability

Inference returns the five largest XGBoost TreeSHAP contributors with observed
values. Missing values are represented as null. Output explicitly says that
contributions explain model behavior and are not causal proof.

No explanation should claim rainfall or slope caused an event.

## 21. Inference

The frozen model was loaded in a new process and scored a real CHIRPS/SRTM/OSM
road observation whose OSM ID was not in the model dataset. It was not retrained.

| Measure | Result |
|---|---:|
| Model load | 109.108 ms |
| First inference | 10.498 ms |
| Mean warm inference, 100 runs | 7.985 ms |

The observation has no future label, so this proves technical execution only.
It cannot prove performance. Freshness metadata was absent and correctly
flagged.

## 22. Reproducibility

New model runs are immutable and include dataset, feature, code, model,
calibration, and risk-policy hashes/versions. Exploratory results are written
to immutable run files with a latest pointer.

Full clean raw-to-model reproducibility still fails because:

- The 23-event list is maintained manually in code.
- Not every incident has a raw local source and checksum.
- Boundary/terrain/weather acquisition manifests are incomplete.
- The worktree is dirty and much of the pipeline is untracked.
- A full raw rebuild was not completed in a clean isolated environment.

## 23. Monitoring

Freshness, missingness, drift, alert-load, delayed-label, calibration, and
geographic monitoring requirements are documented in
docs/MONITORING_STRATEGY.md. No scheduled monitoring service, durable metrics
store, alert integration, ownership roster, or delayed ground-truth join exists.

Monitoring acceptance fails.

## 24. Failure Modes

| Failure | Required behavior | Current status |
|---|---|---|
| Model/schema/hash mismatch | Reject inference | Implemented |
| Invalid/missing rainfall | Controlled error | Implemented |
| Future/bad timestamp | Controlled error | Implemented |
| Stale rainfall | Flag stale/expired | Implemented in inference contract |
| Missing terrain | Score with low-confidence flag | Implemented |
| Bad road geometry/identity | Reject | Not integrated with scoring CLI |
| Weather feed unavailable | Last-known age then expire | Policy only |
| Trace/database unavailable | Durable queue/retry | Missing |
| Field reports unavailable | Mark operational context unknown | Contract only |

## 25. Model Versioning

Each new bundle contains model_version, dataset_version, feature_version, Git
commit/dirty state, training timestamp, train/validation/test periods,
hyperparameters, metrics, model SHA256, feature-spec SHA256, calibration
SHA256, and risk-policy SHA256.

Legacy date-only artifacts remain for history but the latest pointer selects
the immutable content-hashed bundle.

## 26. Integration Contract

docs/API_CONTRACT.json version 2 separates:

- Calibrated model score.
- Frozen model operating threshold and ALERT/NO_ALERT decision.
- Configurable business risk policy and LOW/MEDIUM/HIGH level.
- Model, dataset, feature, and policy versions.
- Freshness, OOD, missingness, and low-confidence flags.
- Non-causal feature contributions.

Frontend, backend, and mobile integration remains intentionally deferred.

## 27. Known Limitations

- Very small independent event count.
- No verified negative surveillance windows.
- Incomplete raw incident provenance and licensing.
- Severe terrain and material rainfall missingness.
- Mutable OSM identity.
- Sparse, non-continuous temporal sampling.
- Uneven state/corridor/season/mechanism coverage.
- Weak final metrics and high false-negative/false-alert rates.
- Calibration based on four validation events.
- No live monitoring or operational owner.

## 28. Data Gaps

P0/P1 gaps:

1. Observation-backed negatives from PWD/DDMA/road closure operations.
2. Complete raw incident files, checksums, source URLs, acquisition timestamps,
   usage terms, deduplication, and verification state.
3. At least 30 independent future test events and at least 10 validation events.
4. Stable versioned road_segment_id with source-ID history.
5. At least 95% terrain and rainfall feature coverage in the intended service
   area.
6. Real travel-time/delay labels. Delay prediction remains blocked.
7. Field-report verification, deduplication, privacy, and retention workflow.

## 29. Improvement History

This assessment added:

- Independent demo and production gates.
- Recall-oriented validation thresholding and operational error counts.
- Majority, rainfall, and Logistic Regression baselines.
- Brier score, confusion matrices, FNR, and top-risk recall.
- Dynamic caveats and current counts.
- Immutable model/evaluation run IDs and content hashes.
- Safe JSON isotonic calibration loading.
- Training-only OOD profiles and freshness flags.
- Input validation and derived-feature recomputation.
- Dataset build validation and lineage hashes.
- Clean-process unseen-real-input inference verification.
- Data catalog, monitoring contract, API v2, and acceptance audit.

The expanded 23-event model was not selected as an improvement by metric:
performance fell, demo readiness was lost, and Logistic Regression beat
XGBoost on multiple chronological-test measures. The result remains preserved.

## 30. Production Decision

BLOCKED BY DATA GAP
