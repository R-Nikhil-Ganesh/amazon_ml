# Business Entity Resolution — Solution Package

High-performance, 3-stage cascaded hybrid Entity Resolution pipeline for Amazon ML Challenge 2026.
Blocking is DuckDB-based (batched, vectorized joins), scoring is a LightGBM +
TensorFlow cascade, and decoding is per-entity expected-F0.5 optimization
rather than a single global threshold. See
[`../../documents/architecture.md`](../../documents/architecture.md) for the
full breakdown and [`../../documents/engineering_log.md`](../../documents/engineering_log.md)
for why each piece is built the way it is.

## System Architecture

1. **Stage 0: Blocking index build & retrieval (`duckdb_blocking.py`)**
   - `DuckDBBlockingEngine.build_index()` ingests S2/S3 candidate records in
     bulk (pandas/Arrow, columnar append) into a single `.duckdb` file.
   - Each candidate's raw name/address passes through transliteration
     (`transliterate.py` — dictionary lookup, then a rule-based Brahmic
     fallback for Indic-script text; a no-op otherwise) before normalization.
   - Generates blocking keys (`build_blocking_keys()`): exact normalized
     name, space-compressed name, distinctive name tokens (plus pairs of
     them), address-number anchors, unordered address-token bigrams, and
     (India only) cross-script phonetic-skeleton keys.
   - Enforces strict country isolation (cross-border candidate pruning) and
     caps over-frequent keys (`MAX_TOKEN_POSTINGS`) so near-stopword tokens
     don't blow out precision.
   - `retrieve_candidates_batch()` takes a whole chunk of S1 entities
     (default 4,000) and runs every lookup pass as one set-based JOIN across
     the chunk, then reranks the pool by rapidfuzz text similarity and
     truncates to `MAX_CANDIDATES_PER_S1`.

2. **Stage 2: LightGBM Screener (`train_screener.py`)**
   - Dense similarity features across names, addresses, numbers, and source
     indicators (`features.py`).
   - Fast gradient boosted tree inference separating high-confidence matches
     from obvious negatives; supports a `--lgbm-params-json` flag to load a
     tuned hyperparameter config from `tune_screener.py`'s output.
   - Active mining of hard false positives (near-miss street numbers,
     identical names in different locations) for Stage 3.

3. **Stage 3: Deep Residual Neural Specialist (`train_tf_specialist.py`)**
   - TensorFlow / Keras 3 ResNet-MLP architecture with Swish activations and
     Batch Normalization.
   - Trained on true positive pairs + mined active hard negatives to resolve
     ambiguous borderline cases the screener alone can't confidently call.

4. **Cascaded Inference (`pipeline_duckdb.py`)**
   - High-confidence LightGBM matches are auto-accepted; obvious non-matches
     are auto-rejected; the ambiguous middle band is re-scored by the
     TensorFlow specialist and blended with the LightGBM probability.
   - Every retrieved candidate's final probability is written to
     `output/match_scores.tsv`, not just ones over a threshold — decoding
     (next stage) needs the full per-entity probability set.

5. **Stage 4: Per-entity decoding (`decode.py`)**
   - Replaces a single global probability threshold: for each S1 entity,
     picks the top-k candidates (k = 0..n) that maximize an approximation of
     that entity's own expected F0.5, rather than thresholding every
     candidate identically regardless of its entity's other candidates.
   - A confidence floor (`MIN_TOP_PROB_FOR_ANY_MATCH`) requires the best
     candidate to clear a minimum probability before any non-empty match set
     is considered, to avoid over-triggering on entities with several
     mediocre-but-unconvincing candidates.

6. **Stage 5: Global Duplicate Resolution (`dedupe_matches.py`, run
   automatically at the end of `pipeline_duckdb.py`)**
   - No Source 2/Source 3 id ever belongs to more than one Source 1 entity in
     the ground truth, but the cascade scores each S1 independently, so the
     same candidate can be accepted under more than one S1 (e.g. two
     businesses sharing an address).
   - This pass keeps a contested candidate only under the S1 that scored it
     highest and drops it everywhere else, using the per-pair probabilities
     in `output/match_scores.tsv`. Since F$_{0.5}$ is precision-heavy, an
     uncontested duplicate is a guaranteed false positive for every other S1
     it's attached to.

---

## Directory Structure

```
business_entity_resolution/
├── README.md                  # This reproduction guide
├── requirements.txt           # Python dependencies
└── src/
    ├── __init__.py            # Package exports and metadata
    ├── config.py               # Global paths, thresholds, and hyper-parameters
    ├── normalize.py             # Multilingual text, address, and numeric cleaners
    ├── transliterate.py         # Indic-script transliteration (dictionary + rule-based fallback)
    ├── blocking.py               # Shared entity-CSV reading & recall evaluator
    ├── duckdb_blocking.py         # DuckDB blocking engine: index build & batched retrieval
    ├── features.py                 # Pairwise feature extraction
    ├── evaluate.py                  # Competition Macro F_0.5 metric evaluation
    ├── decode.py                     # Per-entity expected-F0.5 decoding
    ├── dedupe_matches.py              # Global one-S1-per-candidate resolution
    ├── split_data.py                   # Disjoint S1 entity-level train/validation splitter
    ├── train_screener.py                # Stage 2 LightGBM training & hard negative mining
    ├── train_tf_specialist.py            # Stage 3 TensorFlow ResNet-MLP training
    ├── tune_screener.py                   # Random-search LightGBM hyperparameter tuning
    ├── pipeline_duckdb.py                  # Full cascaded test inference pipeline (recommended)
    ├── validate_pipeline.py                 # Validation harness: real pipeline vs. held-out split
    ├── eval_blocking_recall.py               # Standalone blocking-recall diagnostic
    └── test_blocking.py                       # Unit tests for blocking and recall evaluation
```

`pipeline.py` and `sqlite_blocking.py` (the original single-process
SQLite-backed implementation this package started from) are still present
in `src/` but superseded — see
[`../../documents/engineering_log.md`](../../documents/engineering_log.md)
for that history if you need it.

---

## Installation

Ensure Python 3.10+ is available:

```bash
pip install -r requirements.txt
```

---

## Reproduction Guide

All commands should be executed from the project root directory with `PYTHONPATH=code/business_entity_resolution`.

### 1. Run Unit Tests
```bash
PYTHONPATH=code/business_entity_resolution python3 -m unittest src.test_blocking
```

### 2. Prepare Train/Val Split (Optional if already generated)
```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.split_data
```

### 3. Train LightGBM Screener & Mine Hard Negatives
```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.train_screener
```

### 4. Train TensorFlow Specialist
```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.train_tf_specialist
```

### 5. (Optional) Tune LightGBM Hyperparameters
```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.tune_screener --trials 15
```

### 6. Validate Before a Full Run
```bash
PYTHONPATH=code/business_entity_resolution python3 -m src.validate_pipeline --mini
```

### 7. Run Full Test Inference
```bash
./run_full_inference_duckdb.sh
# or directly:
PYTHONPATH=code/business_entity_resolution python3 -u -m src.pipeline_duckdb --rebuild-index
```
This also runs Stage 5 (`dedupe_matches.py`) and Stage 4 decoding
automatically at the end, writing the final `output/matching_results.tsv`
and `output/candidate_pairs.tsv`. `output/match_scores.tsv` holds the
per-pair probabilities used for decoding and dedupe ownership.

### 8. Validate & Package Submission Archive
```bash
./package_submission.sh
```
