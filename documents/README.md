# Internal Documentation

This folder is internal engineering documentation — how the pipeline actually
works today and how it got here. It is **not** the competition's required
submission document; that is [`Documentation_template.md`](../Documentation_template.md)
at the repo root, which describes an earlier round of fixes (the SQLite-only
pipeline, public leaderboard 0.65 → 0.8177 validation) and has not yet been
updated for the work described here (DuckDB rewrite, Indic transliteration,
per-entity decoding). **Update that file from [`validation_results.md`](validation_results.md)
before the next submission package is assembled** — it's the one graders read.

## Contents

- [`engineering_log.md`](engineering_log.md) — narrative of the work in this
  round: why it happened, what was tried, what was rejected and why, in
  roughly chronological order.
- [`architecture.md`](architecture.md) — the pipeline as it exists today,
  covering data flow and design rationale in more depth than
  [`../code/business_entity_resolution/README.md`](../code/business_entity_resolution/README.md)
  (which now describes the same current DuckDB pipeline, at reproduction-guide
  level of detail — the original single-process `pipeline.py` it used to
  document is superseded, see `engineering_log.md` §1 and §8 for that history).
- [`validation_results.md`](validation_results.md) — every measured number
  from this round, with exactly what configuration produced each one. The
  raw `*.log` files these numbers came from were run-specific temp output
  in the repo root and are not preserved long-term — the numbers themselves
  are what's durable, recorded here as they were confirmed.

## Quick orientation if you're new to this round of changes

The short version: the original pipeline's bottleneck wasn't inference, it
was index-build time and a since-fixed multiprocessing regression (see
`engineering_log.md` §1). Once that was sorted out, the actual score gap
(0.863 public leaderboard vs. 0.95+ for top teams) turned out to concentrate
almost entirely in one place — Indian business names/addresses written in a
non-Latin script, where blocking recall measured **0.195** instead of the
~0.90 achieved elsewhere. Fixing that, plus replacing a single global
decision threshold with a decision computed per S1 entity, moved validation
macro F0.5 from 0.8693 → 0.8980 (leaderboard confirmed at 0.893). A second
round (blocking-key density + LightGBM hyperparameter tuning, engineering_log.md
§8-§14) moved validation to **0.9035**; a France-specific false-positive
pattern was found via hand-labeling (§12) and partially addressed via
hyperparameter search, though the resulting `name_sim_floor` gate is not
yet wired into real inference (see `validation_results.md` §7 for the
known gap). Leaderboard score for this second round not yet confirmed.
