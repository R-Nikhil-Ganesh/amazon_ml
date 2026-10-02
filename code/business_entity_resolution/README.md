# Business Entity Resolution — Solution Package

High-performance, 3-stage cascaded hybrid Entity Resolution pipeline for Amazon ML Challenge 2026.
Blocking is DuckDB-based (batched, vectorized joins), scoring is a LightGBM +
TensorFlow cascade, and decoding is per-entity expected-F0.5 optimization
rather than a single global threshold. 

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
<submission root>/
├── output/                     # matching_results.tsv, candidate_pairs.tsv (final submission files)
├── models/                     # trained screener_lgbm.pkl + tf_specialist_model.keras
├── student_resource/dataset/   # NOT in the zip - the provided competition data goes here
│   ├── train/  train_source{1,2,3}.tsv, train_ground_truth.tsv
│   └── test/   test_source{1,2,3}.tsv
└── code/business_entity_resolution/
    ├── README.md
    ├── requirements.txt
    └── src/
        ├── config.py              # Paths, thresholds, hyper-parameters
        ├── normalize.py           # Multilingual text, address, numeric cleaners
        ├── transliterate.py       # Indic-script transliteration (learned dictionary + rule-based fallback)
        ├── blocking.py            # Entity file reader & recall evaluator
        ├── sqlite_blocking.py     # Blocking-key generation shared by the DuckDB engine
        ├── duckdb_blocking.py     # DuckDB blocking engine: index build & batched retrieval
        ├── features.py            # Pairwise feature extraction
        ├── evaluate.py            # Macro F_0.5 metric
        ├── decode.py              # Per-entity expected-F0.5 decoding
        ├── dedupe_matches.py      # Global one-S1-per-candidate resolution
        ├── split_data.py          # Disjoint S1 train/validation split
        ├── train_screener.py      # Stage 2 LightGBM training & hard-negative mining
        ├── train_tf_specialist.py # Stage 3 TensorFlow ResNet-MLP training
        └── pipeline_duckdb.py     # Full cascaded test inference pipeline
```

The input data is read directly from the original `.tsv` files; no conversion
is needed. The code expects the layout above, i.e. this folder placed at
`<root>/code/business_entity_resolution/`. To run from another location, set
`ER_PROJECT_ROOT=<root>`.

---

## Installation

Python 3.10 (tested with 3.10.12):

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## Reproduction Guide

Run every command from `<root>` with `PYTHONPATH=code/business_entity_resolution`.
Steps 1-4 are only needed to retrain; the trained models are already in
`models/`, so step 5 alone regenerates the submission files.

```bash
export PYTHONPATH=code/business_entity_resolution
```

### 1. Train/validation split
```bash
python3 -m src.split_data          # writes data_split/{val,mini_val}_s1_ids.txt and *_ground_truth.tsv
```

### 2. Build the Indic-script transliteration dictionary
```bash
python3 -m src.transliterate       # writes data_split/translit_dict.json
```

### 3. Train the LightGBM screener & mine hard negatives
```bash
python3 -m src.train_screener      # writes models/screener_lgbm.pkl, data_split/active_hard_negatives.csv
```
The first run builds a disk-backed DuckDB candidate index over the full
training S2/S3 pool (~6 GB, `data_split/train_candidates_index.duckdb`).

### 4. Train the TensorFlow specialist
```bash
python3 -m src.train_tf_specialist # writes models/tf_specialist_model.keras
```

### 5. Full test inference
```bash
python3 -u -m src.pipeline_duckdb --threshold 0.75 --rebuild-index
```
Builds the test candidate index (`output/candidates_index.duckdb`, ~6 GB), then
runs blocking, the LightGBM/TensorFlow cascade, per-entity decoding and global
duplicate resolution, writing `output/matching_results.tsv` and
`output/candidate_pairs.tsv`. `output/match_scores.tsv` holds the per-pair
probabilities used for decoding and dedupe ownership. Use `--max-entities N`
for a quick smoke test.

### 6. Validate & package
```bash
./package_submission.sh
```
