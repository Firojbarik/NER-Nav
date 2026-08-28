# ML Data-Gap Remediation Runbook

This runbook is the required path from `BLOCKED BY DATA GAP` to a new
acceptance decision. It does not permit synthetic rows, guessed sources, or
manual database edits.

## P0 — source-backed labels

For production-quality negatives, a URL and seven-day window are not enough.
Each control must also identify the exact mapped road segment and document
no-disruption status for that segment. Corridor reopening/clearance reports
remain research-only controls and are rejected by the readiness audit.

1. Re-verify every entry in `ml/features/temporal_design.py` against an
   authorized source. Complete `data/processed/ml/event_label_registry.json`
   with the matching `event_id`, `osm_id`, reference, event date, source URL,
   source publication date, event type, and location. Entries without a
   verifiable source remain excluded.
2. Provide
   `data/raw/hazards/negative_observations.csv` from an authorized road-status,
   closure, field-report, or equivalent source. Required columns are:

   `observation_id, osm_id, ref, prediction_time, observation_window_end,
   source_url, source_publish_date`

   Each row must have a unique ID, a real HTTP source URL, a valid source date,
   and an observation window covering `prediction_time + 7 days`. No-event
   absence from an incident feed is not sufficient.
3. Run:

   ```text
   python scripts/validate_training_inputs.py
   ```

   It must exit successfully before any rebuild is attempted.

## P0 — road identity and provenance

Build the deterministic snapshot mapping and review source-ID history:

```text
python scripts/build_road_segment_identity.py
python scripts/create_ml_input_manifest.py
python scripts/verify_ml_input_manifest.py
```

The mapping is intentionally marked
`SNAPSHOT_ONLY_PENDING_SOURCE_ID_HISTORY` until OSM changes can be reconciled
by reviewed geometry/source-ID history. It must not be described as globally
stable before that review.

## P1 — rebuild and acceptance

After the P0 checks pass, use a clean Python 3.12 environment created from
`requirements.lock` and run:

```text
python scripts/rebuild_pipeline.py
python scripts/verify_ml_input_manifest.py
python -m unittest discover -s tests -v
```

Then rerun the final acceptance audit. The untouched future test set must be
expanded to at least 30 independent events and validation to at least 10
independent events before the production gate can pass. Existing metrics must
not be copied to the new artifact; all metrics must come from that run.

## What the code now prevents

- The rebuild stops before downloads when event provenance or negative-source
  prerequisites are missing.
- The dataset builder no longer creates temporal-control or assumed-unaffected
  corridor negatives.
- Training refuses datasets containing unsupported negative label sources.
- Every future rebuilt row carries a versioned `road_segment_id` mapping.
- Input files are checksum-manifested and recorded with the model artifact.
- New prediction traces are separated into `production` and `test` streams;
  historical mixed traces are retained as an audit archive.

Until the external source records are supplied and independently verified, the
correct decision remains `BLOCKED BY DATA GAP`.
