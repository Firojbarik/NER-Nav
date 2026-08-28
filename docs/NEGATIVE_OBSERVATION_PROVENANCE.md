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
