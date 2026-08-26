# NER-Nav — Raw Hazard Data Area

This directory stores ORIGINAL hazard/event datasets acquired from
official or explicitly permitted sources.

## Rules

1. Never modify original source files in-place.
2. Never overwrite an original download.
3. Preserve the original filename where practical.
4. Record acquisition metadata in `source_registry.json`.
5. Record:
   - source
   - URL
   - acquisition date
   - dataset name
   - coverage
   - temporal coverage
   - spatial resolution/geometry
   - license/access terms
   - checksum
6. Do not scrape NDEM or other restricted services without explicit
   authorization.
7. Do not fabricate hazard events.
8. Do not convert synthetic road-risk labels into "real incidents".
9. Do not train the final disruption model until the source has passed
   schema, spatial and temporal validation.

## Expected future structure

```text
data/raw/hazards/
    README.md
    source_registry.json

    nrsc_landslide_atlas/
        ORIGINAL_FILES_HERE

    ndem/
        ORIGINAL_FILES_HERE

    other_permitted_sources/
        ORIGINAL_FILES_HERE