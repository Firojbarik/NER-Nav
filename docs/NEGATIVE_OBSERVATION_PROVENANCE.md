# Negative observation provenance

The production temporal dataset uses `data/raw/hazards/negative_observations.csv`
for label-0 controls. The 24 rows are mapped to distinct OSM road segments on
NH-2, NH-6, NH-10, NH-29, NH-37, and NH-102B and each row records a source URL,
publication date, and a seven-day observation window required by the training
input gate.

The sources are road-status reports: they document normal, reopened, or
traffic-resumed movement on the named corridor at the cited observation. They
do not establish that every metre of the corridor was hazard-free for the full
window, so these controls are observation-backed but remain weaker than a
segment-level field survey. The CSV preserves that limitation in
`observation_status`, `observation_scope`, and `evidence_note`; future curation
should replace or strengthen these rows with segment-level status logs when
available.

The production-readiness audit now treats these as research controls only:
post-clearance/reopening reports, corridor-scoped reports, and reports
published after the proposed prediction time fail the negative-observation
quality check. Passing the basic training-input gate therefore does not mean
that these controls are suitable for a production probability or alert policy.

The records are not generated from absence in an incident feed, a temporal
offset from a positive event, or an assumed-unaffected road pool.

## Web negative-source sweep (2026-09-19)

A fresh search-and-verify sweep was run against the public web to find clean,
segment-scoped, no-disruption road observations on the model's NH corridors
(executed as part of the `fix/model-collapse-and-imbalance` work; evidence in
the sweep conclusion below). Result: **the surface web still yields zero
admissible clean negatives on the loaded network.**

What was tried and what was verified:

- Multiple targeted searches for "open for traffic" / "traffic normal" /
  daily road-status bulletins for NH-2, NH-6, NH-10, NH-29, NH-37 across
  Sikkim/Nagaland/Manipur/Assam. Every contemporaneous NH-scoped record found
  was a disruption, closure, blockade, or reopening report — rejected by the
  quality gate on purpose, and correctly so.
- **Sikkim Roads & Bridges "Road Conditions" sitreps** — the strongest official
  source found. URL pattern:
  `http://www.sikkim-roadsandbridges.gov.in/index.php/road-network`.
  It publishes dated, named-road, open-status sitreps (e.g. "Legsip-Gyalshing
  road-Open", "Jorethang to Namchi road is open as of now"). These are genuine
  source-backed negatives — but two blockers apply:
  1. Nearly every clean "open" line names a **state/district road** (SH-19,
     SH-20, Gyalshing–Pelling, etc.). The SIKKIM ways loaded in
     `ner_roads_districts.gpkg` are mostly NH-10/NH-310/NH-510/NH-710/NH-717;
     a name probe found only "Namchi-Manpur Road" matches. Ingesting the sitrep
     roads therefore requires **expanding the OSM road/terrain/rainfall base**
     to include state roads — an infrastructure change, not a data-entry task.
  2. Using state-road negatives against NH positives would also let
     `highway_prior` become near-discriminative, so the base must be expanded
     deliberately, not one row at a time.
- The event label registry is fully verified: **54/54 `CONFIRMED_EVENTS` have
  `source_url_status = VERIFIED` with HTTP URL + publication date**.

Decision recorded: the verified-negative ceiling from the earlier acquisition
run is confirmed with fresh evidence. Next unblocking step (recommended) is the
road-base expansion plus an ingestion harness for official tectonic sitrep
feeds (Sikkim R&B first; Nagaland/Manipur PWD/DDMA equivalents in the same
format), OR acquiring segment-scoped open-status logs from operations/RTI.
