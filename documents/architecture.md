# Pipeline Architecture (current)

Goes into more data-flow and design-rationale depth than
[`../code/business_entity_resolution/README.md`](../code/business_entity_resolution/README.md),
which covers the same current DuckDB pipeline at reproduction-guide level.
Recommended entry point is `run_full_inference_duckdb.sh` →
`src/pipeline_duckdb.py`. See [`engineering_log.md`](engineering_log.md)
for why each change was made.

**Data note:** the pipeline reads from `csv_data/` (a CSV-converted copy of
the official competition data), not the original TSV files in
`student_resource/dataset/`. All paths below (`TRAIN_SOURCE1_PATH` etc. in
`config.py`) point at `csv_data/`.

## Stage 0: Blocking index build (`src/duckdb_blocking.py`)

`DuckDBBlockingEngine.build_index(s2_path, s3_path, translit_dict=...)`

1. Ingest S2/S3 candidate records in chunks (pandas/Arrow, columnar bulk
   append, not row-by-row).
2. **Transliteration** (new): each candidate's raw name/address is passed
   through `transliterate_text()` (`src/transliterate.py`) before
   normalization — dictionary lookup first
   (`data_split/translit_dict.json`, learned from training ground truth),
   then `transliterate_brahmic_token()` (`normalize.py`) as a rule-based
   fallback for any Indic word the dictionary hasn't seen. A no-op for
   non-Indic text. Controlled by `use_translit` in `pipeline_duckdb.py` /
   `--no-translit` on the CLI.
3. Normalize (`normalize_business_name`, `normalize_address`) and generate
   blocking keys (`build_blocking_keys()`, defined in `sqlite_blocking.py`
   and imported directly from there so key generation can't silently
   diverge between engines): exact name, compressed name, distinctive name
   tokens, address-number anchors, address-token bigrams, and (India only)
   phonetic-skeleton keys.
4. Cap over-frequent keys (`MAX_TOKEN_POSTINGS`, `config.py`) via an
   `ANTI JOIN` against a materialized "too-common" table.

Output: a single `.duckdb` file (`output/candidates_index.duckdb` for test,
`data_split/train_candidates_index.duckdb` for validation).

## Stage 1: Batched candidate retrieval (`src/duckdb_blocking.py`)

`DuckDBBlockingEngine.retrieve_candidates_batch(s1_chunk)` — takes a whole
chunk of S1 entities (default `chunk_size=4000`) and runs each of the 3
lookup passes (exact name / compressed name / blocking keys) as one
set-based JOIN across the entire chunk, instead of one query per entity.
Reranks the top `RERANK_POOL_SIZE` (by summed key weight) using rapidfuzz
text similarity, truncates to `MAX_CANDIDATES_PER_S1` (35).

## Stage 2+3: Scoring cascade (`src/pipeline_duckdb.py`)

Unchanged from the original design: LightGBM screener
(`models/screener_lgbm.pkl`) scores every retrieved pair; the ambiguous
middle band (`0.20 ≤ p < 0.88`) is re-scored by the TensorFlow specialist
(`models/tf_specialist_model.keras`) and blended
(`0.70·P_lgbm + 0.30·P_tf`). Every candidate's final probability is now
written to `match_scores.tsv` — not just the ones over a threshold (see
Stage 4).

## Stage 4: Decoding (`src/decode.py`)

Replaces a single global threshold. For each S1 entity, sort its
candidates by probability and pick the top-k (k = 0..n) that maximizes an
approximation of that entity's own expected F0.5 (see
`decode.py`'s module docstring and `engineering_log.md` §5 for the exact
formula and its derivation). A confidence floor
(`MIN_TOP_PROB_FOR_ANY_MATCH = 0.70`) requires the best candidate to clear
0.70 before any non-empty match set is considered, to avoid over-triggering
on entities with several mediocre — but no individually convincing —
candidates. Controlled by `decode_mode` in `pipeline_duckdb.py` /
`--decode-mode {expected_f05,threshold}` on the CLI; `threshold` preserves
the original global-cutoff behavior for comparison.

## Stage 5: Global dedupe (`src/dedupe_matches.py`, unchanged)

No S2/S3 id belongs to more than one S1 entity in the ground truth; this
pass keeps a contested candidate only under whichever S1 scored it highest.
Runs identically regardless of which decode mode produced
`matching_results.tsv`.

## Training (`src/train_screener.py`, `src/train_tf_specialist.py`, `src/tune_screener.py`)

Both training scripts now retrieve candidates through `DuckDBBlockingEngine`
against `data_split/train_candidates_index.duckdb` (batched, same
`RETRIEVAL_CHUNK_SIZE=4000` pattern as inference), not `DiskBlockingEngine`
- see `engineering_log.md` §8. `train_screener.py` accepts
`--train-sample-size` and `--lgbm-params-json` (loads a winning
hyperparameter config produced by `tune_screener.py` instead of
`config.py`'s static `LGBM_PARAMS`). `tune_screener.py` runs a random search
over LightGBM hyperparameters plus a post-hoc `name_sim_floor` gate
(rejects a candidate outright when `name_token_set` is below the floor,
regardless of the model's own probability - see `engineering_log.md` §12-13
for why), building the train/val feature matrices once and reusing them
across every trial rather than re-extracting per trial.

## Validation harness (`src/validate_pipeline.py`)

Runs the exact same `run_pipeline_duckdb()` function used for real
inference, but pointed at the training S2/S3 pool and the held-out
validation S1 split (`data_split/val_s1_ids.txt`, the full 441,364 entities
— not a subsample), then scores the result with `evaluate.py`'s
`evaluate_predictions()` instead of running the submission-format
validator, broken down by country. This is what closes the loop between
"the model changed" and "here's a leaderboard-comparable number" — see
`engineering_log.md` §2 for why this didn't already exist and
`validation_results.md` for what it's measured so far.

```bash
python3 -m src.validate_pipeline --build-index          # full run, rebuilds index
python3 -m src.validate_pipeline                         # reuses existing index
python3 -m src.validate_pipeline --mini                  # fast 50K-entity smoke test
python3 -m src.validate_pipeline --no-translit            # baseline, translit off
python3 -m src.validate_pipeline --decode-mode threshold  # baseline, old decoder
```

## Files not on this path

- `src/pipeline.py` — original single-process pipeline, superseded by
  `pipeline_duckdb.py` (see `engineering_log.md` §1 and §8 for that
  history). Still correct, but no longer documented in
  `code/business_entity_resolution/README.md` (which now describes only
  the current DuckDB pipeline) and not used by the current recommended run
  script. `train_screener.py`/`train_tf_specialist.py` were migrated off
  `DiskBlockingEngine` to `DuckDBBlockingEngine` (`engineering_log.md` §8),
  so `pipeline.py` no longer has a load-bearing reason to stay - it appears
  to have no remaining callers, but hasn't been deleted without more
  certainty on that. `DiskBlockingEngine` itself (`src/sqlite_blocking.py`)
  is still load-bearing regardless: its key-generation functions
  (`build_blocking_keys`, `key_hash`, `KEY_WEIGHTS`) are imported directly
  by `duckdb_blocking.py`.

The multiprocessing sharded version (`src/pipeline_parallel.py`,
`run_full_inference_fast.sh`) and the abandoned LLM-specialist data
prep (`src/llm_dataset.py`) were removed during the cleanup in
`engineering_log.md` §7 — both were confirmed unused by anything, and the
former's own measured conclusion was "don't use this" (page-cache
fragmentation across worker DuckDB connections made it slower, not
faster, than the single-process path). The reasoning that led to
rejecting multiprocessing is preserved in `engineering_log.md` §1, in more
useful form than the code itself.
