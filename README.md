# Amazon ML Challenge 2026 — Business Entity Resolution

Matches business records from Source 1 against Source 2/Source 3 across
US, India, and France, scored by macro-averaged F0.5. Solution code lives in
[`code/business_entity_resolution/`](code/business_entity_resolution/).

## Where things are

| What you want | Where |
|---|---|
| The submission writeup, with comprehensive explanation of the current solution | [`Documentation_template.md`](Documentation_template.md) |
| The actual pipeline code | [`code/business_entity_resolution/src/`](code/business_entity_resolution/src/) |
| Internal engineering docs (why things are the way they are) | [`documents/`](documents/) — start with [`documents/README.md`](documents/README.md) |
| Current pipeline architecture | [`documents/architecture.md`](documents/architecture.md) |
| Measured validation/leaderboard numbers | [`documents/validation_results.md`](documents/validation_results.md) |
| Chronological log of what was tried, what worked, what didn't | [`documents/engineering_log.md`](documents/engineering_log.md) |
| Raw competition data | [`csv_data/`](csv_data/) — **CSV-converted copy of the official data, what the pipeline actually reads.** The original TSV files from the competition are in [`student_resource/`](student_resource/) (official scripts/rules included) but are not used directly by any code path. |

## Running it

The recommended entry point is `pipeline_duckdb.py` — see
[`code/business_entity_resolution/README.md`](code/business_entity_resolution/README.md)
for the full architecture and reproduction steps:

```bash
cd code/business_entity_resolution
python3 -m src.pipeline_duckdb --rebuild-index   # full test-set inference
```

To validate a change before spending a full ~3hr run on it, use the
validation harness (runs the real pipeline against a held-out split scored
the same way the leaderboard scores submissions):

```bash
python3 -m src.validate_pipeline --mini   # fast 50K-entity smoke test
python3 -m src.validate_pipeline          # full 441K-entity validation split
```

See [`documents/architecture.md`](documents/architecture.md) for the full
pipeline breakdown (blocking → screener → TF specialist → decode → dedupe)
and [`documents/README.md`](documents/README.md) for a guided orientation
to the rest of the internal docs.

## Current status

Public leaderboard: 0.893 (confirmed). Validation macro F0.5 since improved
further to 0.9035 in a second round of work (blocking-key tuning +
LightGBM hyperparameter search) — leaderboard result for that round not
yet confirmed. See [`documents/validation_results.md`](documents/validation_results.md)
for the full breakdown, caveats, and what's still open.
