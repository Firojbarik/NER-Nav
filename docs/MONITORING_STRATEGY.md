# Monitoring Strategy

## Purpose
Detect when the model's inputs or outputs drift from expected behavior, so we can retrain before performance degrades.

## What to Monitor

### 1. Input Data Quality (per prediction)
- **Missing features**: count of NaN values in input features per prediction request
- **Out-of-range values**: rainfall < 0, elevation < 0, slope > 90
- **Feature drift**: compare distribution of incoming features against training distribution

### 2. Prediction Distribution
- **Probability distribution**: track mean, median, std of `disruption_probability` over time
- **Risk level distribution**: count of HIGH/MEDIUM/LOW predictions per day
- **Alert rate**: fraction of predictions that trigger alerts
- If alert rate spikes or drops suddenly, investigate

### 3. Model Performance (when ground truth arrives)
- **Post-event accuracy**: after a confirmed event, check if the model predicted HIGH risk for the affected road in the preceding 7 days
- **False positive rate**: fraction of HIGH predictions that did NOT result in an event within 7 days
- **False negative rate**: fraction of confirmed events where the model predicted LOW risk

### 4. Data Pipeline Health
- **CHIRPS data freshness**: date of most recent downloaded raster
- **Feature extraction success rate**: fraction of prediction dates with complete features
- **Road segment coverage**: fraction of road segments with terrain data

## Monitoring Thresholds

| Metric | Threshold | Action |
|--------|-----------|--------|
| NaN input rate | > 10% of requests | Alert data engineering |
| Probability mean shift | > 0.15 from training mean | Trigger drift investigation |
| Alert rate | < 1% or > 30% of requests | Review model calibration |
| CHIRPS data lag | > 7 days old | Trigger download |
| False negative | Any missed confirmed event | Post-mortem + retraining |

## Implementation
- Log every prediction request/response to a structured log (JSON)
- Daily aggregation of prediction statistics
- Weekly comparison against training baseline
- Monthly performance review against ground truth events

## Current Implementation Status

The strategy is not yet an operational monitoring service. Prediction tracing
exists, but there is no scheduled aggregation, alert delivery, delayed-label
join, drift store, or on-call owner. Production acceptance remains failed.

## Freshness Contract

| Input | Fresh | Stale | Expired / action |
|---|---|---|---|
| Daily rainfall | age <= 48 hours | 48-168 hours | >168 hours: do not present as current operational risk |
| Field report | age <= 6 hours | 6-24 hours | >24 hours: context only, require reconfirmation |
| Road closure/status | age <= 1 hour | 1-6 hours | >6 hours: status unknown |
| Static terrain | Versioned snapshot | Replacement available | Rebuild after source/version change |
| OSM road graph | <=30 days for demo | 30-90 days | >90 days: refresh before operational use |

Inference must expose FRESH, STALE, EXPIRED, or MISSING_METADATA. A stale input
must never be silently represented as fresh.

## Required Automated Jobs

| Job | Cadence | Warning | Review / retrain trigger |
|---|---|---|---|
| Source freshness and ingestion success | Each fetch | One failed fetch or stale input | Three consecutive failures |
| Schema, ranges, duplicates, CRS, geometry | Each ingestion | Any rejected rows | Repeated failures or >1% rejection |
| Missingness by feature/state/source | Daily | +5 percentage points vs training | >10 points or production floor breach |
| Feature and prediction drift | Weekly | PSI >0.10 or material KS shift | PSI >0.25 with investigation |
| Alert load | Daily | >20% of scored roads | >30% or operator capacity exceeded |
| Delayed-label performance | Monthly when labels exist | Recall <0.75 | Recall <0.70 or FNR >0.30 |
| Calibration | Quarterly when labels exist | Brier deterioration >10% | Material reliability-curve failure |
| Geographic performance | Each evaluation | Any state lacks evaluable events | State recall/FNR breaches |

PSI/KS thresholds are investigation heuristics, not automatic proof of model
failure. Performance alerts require observation-backed delayed labels.

## Failure Policy

- Weather unavailable: retain last observation with explicit age; expire after
  seven days and suppress current-risk claims.
- Terrain unavailable: score only with MISSING_TERRAIN and LOW_CONFIDENCE; do
  not claim NER-wide production coverage.
- Road identity unavailable or geometry invalid: reject the request.
- Model/hash/schema mismatch: reject inference and alert the model owner.
- Database or trace sink unavailable: do not lose the inference error; queue
  traces durably before production deployment.
- Field reports unavailable: model can run, but operational risk engine must
  display field status as unknown.

## Ownership Gap

Owners, escalation channels, retention periods, privacy controls, and service
level objectives are **DATA GAP / OPERATIONS GAP** and must be assigned before
deployment.
