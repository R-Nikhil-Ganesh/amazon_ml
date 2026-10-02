# ML Challenge 2026: Business Entity Resolution Solution Methodology

**Team Name:** Neural Collapse
**Problem Track:** Business Entity Resolution (ER)
**Submission Date:** October 2026

---

## 1. Executive Summary
We built a scalable, cascaded hybrid Entity Resolution pipeline for large-scale
multi-source commercial entity matching: a DuckDB-backed multi-pass blocking
engine with a learned cross-script transliteration layer, feeding an ensemble
of a LightGBM gradient-boosted screener and a TensorFlow/Keras 3 deep residual
specialist trained on actively mined hard negatives, decoded per-entity via an
expected-F0.5 approximation (not a single global threshold), and finished with
a global one-Source-1-per-candidate duplicate resolution pass.

**Confirmed public leaderboard score: 0.893**, on the full 1,732,544-entity
test set. On the full 441,364-entity held-out validation split (scored with
the same macro F0.5 the leaderboard uses), the pipeline measures **0.8980**,
within 0.005 of the leaderboard result — the tightest validation/leaderboard
alignment this project has achieved, and the basis for trusting the
validation harness (`src/validate_pipeline.py`) for everything reported below
it. A further round of blocking-key tuning and LightGBM hyperparameter search
(Section 5) raised a fast 50K-entity validation subsample to **0.9035**; that
change has been retrained and run across the full test set, but a fresh
leaderboard confirmation for it was not yet in hand as of this document.

---

## 2. Methodology

### 2.1 Problem Analysis
Three S1-vs-S2/S3 noise patterns dominate the matching difficulty, plus one
structural blind spot in the training data itself:
1. **Multilingual Transliteration & Script Shift:** Indian businesses
   frequently appear with the same name in Latin script on one source and
   Devanagari/other Brahmic script on another (e.g. *Shiva Exports* ↔ *शिवा
   एक्सपोर्ट्स*). Measured directly: blocking recall for India candidates
   written in Brahmic script was **0.195** before any fix — vs. 0.900 for
   Latin-script India candidates and 0.922 for US — making this the single
   largest lever in the whole pipeline (Section 3).
2. **Domain Names & Website Prefixes:** Many Source 2/3 records carry web
   artifacts (`davisdecker.com` vs. `Davis & Decker`).
3. **Address Anchoring & Street Discrepancies:** Addresses share colony
   names, postal codes, or landmarks while suffering component reordering;
   distinct street/house numbers on the same street (e.g. `1424` vs. `1433
   Cottage View Ln`) must disambiguate genuinely different entities.
4. **Unseen Country in Test (France):** Training data (`csv_data/train/`)
   contains **zero** France entities — 1,323,633 US + 883,188 India only —
   while the test set is **15% France** (259,452 of 1,732,544 entities),
   weighted equally per-entity in the macro F0.5 average. No labeled France
   data exists anywhere to validate against. Hand-labeling a random France
   sample (Section 5) surfaced a specific, non-France-exclusive failure mode
   rather than a general "France doesn't work" conclusion.

### 2.2 Solution Strategy
**Approach Type:** Multi-pass DuckDB-backed blocking → LightGBM screener →
TensorFlow deep residual specialist → per-entity expected-F0.5 decoding →
global duplicate resolution.

**Core design choices:**
- **Learned cross-script transliteration**, applied once at index-build
  time rather than as a runtime feature: a word-level dictionary learned
  directly from `train_ground_truth.tsv` matched pairs (1,378 words, e.g.
  `प्राइवेट`→`private`), with a Unicode-block-arithmetic rule-based fallback
  for any Indic word the dictionary hasn't seen. Both are purely derived from
  the competition's own training data and offline script-structure rules —
  no external corpus or pretrained model, keeping this inside the
  competition's no-external-data-augmentation rule.
- **Active hard-negative mining:** the LightGBM screener's own
  false-positive predictions on non-matching, lexically-similar pairs are
  fed back in as hard negatives for the TensorFlow specialist, rather than
  training it on trivial random negatives.
- **DuckDB-backed multi-pass blocking:** batched set-based retrieval
  (`retrieve_candidates_batch()`) processes a whole chunk of Source 1
  entities per query instead of one row at a time, bringing full-test-set
  index build down from a 25+-minute SQLite build (still incomplete when
  killed) to ~24 minutes, and real inference throughput to ~287 entities/sec.
- **Per-entity decoding, not a single global threshold:** because the
  competition metric is macro-averaged *per S1 entity*, the same candidate
  probability should be accepted for one entity and rejected for another
  depending on that entity's other candidates — a single global cutoff is
  provably suboptimal against this metric.

---

## 3. Candidate Generation (Blocking)
To avoid the quadratic $O(N \times M)$ comparison space:
- **Country strict partitioning:** entities never match across borders;
  candidate generation partitions by `country` throughout.
- **Transliteration before normalization:** every Source 2/3 name and
  address is passed through the learned dictionary, then the rule-based
  phonetic fallback, before any blocking key is generated — so every
  downstream key and feature sees comparable text regardless of script.
- **Multi-pass blocking keys**, generated by one shared, order-invariant key
  generator used identically at index-build time and query time:
  1. *Exact Normalized Name:* legal-suffix stripping (`Pvt Ltd`, `Inc`,
     `LLC`, `SARL`, dotted acronyms like `S.A.S.`), accent folding
     (`Fédération` = `Federation`).
  2. *Compressed Name:* whitespace removed (`davis decker` ↔ `davisdecker`).
  3. *Distinctive Name Tokens:* non-stopword tokens, with keys posted by
     more than `MAX_TOKEN_POSTINGS` (150) records per country dropped after
     ingestion.
  4. *Address Anchors:* every street number paired with the top rare
     address tokens (tuned this round — see below), every postal/PIN-coded
     number alone, plus address-token bigrams, so a reordered address still
     blocks together.
  5. *(India only) Phonetic-skeleton keys:* a lossy Brahmic-to-Latin
     consonant fold, used as a secondary cross-script bridge alongside the
     transliteration layer above.
- **Blocking-key density was re-tuned this round.** A much earlier version
  of this pipeline had trimmed address-anchor key density hard to fit a
  7.4GB RAM ceiling that no longer applies (now 11GB, with keys stored as
  compact int64 hashes rather than raw text). A full revert of both
  address-anchor key types was attempted and OOM-killed during index build;
  a partial revert (one of the two key types back to its richer form) built
  successfully: final test-set index 10,320,219 candidates, 52,933,024
  keys, 6.08GB on disk.
- **Candidate recall ceiling** (best possible Macro F0.5 with a perfect
  matcher, same blocking output, mini-validation split): **0.9614** overall
  (India 0.9471, US 0.9709) — the gap between this ceiling and the
  cascade's achieved score (Section 5) is larger than blocking's own
  remaining gap, i.e. the matching model, not blocking, is now the larger
  lever.

---

## 4. Matching Model

**Features engineered (20 dense tabular features) include:**
- **Name similarity:** Levenshtein ratio, partial ratio, token-sort ratio
  (word-order invariant), token-set ratio (substring invariant),
  Jaro-Winkler distance, a phonetic-skeleton token-set ratio (bridges
  Latin/Brahmic spelling even when the raw strings don't match), first-token
  exact-match flag, length discrepancy.
- **Address similarity:** token-set ratio, Jaccard similarity, partial
  ratio, length difference.
- **Numerical disambiguation:** exact number-set match, Jaccard overlap,
  overlap count, conflict flag (distinct house numbers on the same street).
- **Source indicator:** binary S2-vs-S3 flag.

A group-relative feature set (each candidate's rank/gap relative to its own
entity's other candidates) was also tried this round — it measurably
improved non-singleton F0.5 but hurt singleton accuracy badly enough that it
was reverted rather than shipped (Section 5).

**Model architecture:**
- **Stage 1 Screener:** LightGBM gradient-boosted trees, trained via
  `src/train_screener.py` against the full DuckDB-retrieved candidate pool
  for 150,000 Source 1 entities. Hyperparameters were tuned this round via
  a random search (`src/tune_screener.py`, 12 trials over `num_leaves`,
  `learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`,
  `min_child_samples`) rather than left at their original untuned values —
  best trial: `num_leaves=95`, `min_child_samples=10`.
- **Stage 2 Specialist:** TensorFlow/Keras 3 deep residual network
  (ResNet-MLP, Swish activations, batch normalization, dropout), trained on
  the same pool plus actively-mined hard negatives. Epoch budget corrected
  this round: validation AUC plateaus by epoch 3-4 and only oscillates
  within noise afterward, so the training loop (previously a 12-epoch cap
  that `EarlyStopping` never actually triggered, because sub-0.0001 AUC
  noise kept resetting its improvement counter) now hard-caps at 6 epochs
  with an explicit `min_delta` on `EarlyStopping`, cutting wasted wall-clock
  without touching model quality (`restore_best_weights=True` already kept
  the real best epoch either way).
- **Ensemble:** $P_{\text{final}} = 0.70 \times P_{\text{LGBM}} + 0.30
  \times P_{\text{TF}}$ in the ambiguous probability band; every retrieved
  candidate's final probability is persisted (`match_scores.tsv`), not just
  the ones that clear a threshold, since Stage 3 needs every candidate's
  score to decode per entity.
- **Stage 3 — Per-entity expected-F0.5 decoding (`src/decode.py`):**
  replaces a single global cutoff. For each S1 entity, candidates are sorted
  by probability and the top-k (k = 0..n) maximizing a first-order
  plug-in approximation of that entity's own expected F0.5 is selected
  (`E[F_beta(k)] ≈ (1+β²)·S_k / (β²·S_n + k)`, with `k=0` handled exactly
  via the competition's own singleton rule). A confidence floor
  (`MIN_TOP_PROB_FOR_ANY_MATCH = 0.70`) requires the best candidate to clear
  0.70 before any non-empty match set is considered at all, specifically to
  avoid over-triggering on entities with several mediocre candidates and
  none individually convincing — tuned conservative rather than at the
  raw validation peak, since France (15% of the real test set) has no
  training representation to check decoder calibration against.
- **Stage 4 — Global Duplicate Resolution (`src/dedupe_matches.py`):** no
  Source 2/3 id belongs to more than one Source 1 entity in the ground
  truth, but Stages 1-3 score each S1 entity independently. A candidate
  accepted by more than one S1 entity is kept only under whichever one
  scored it highest. On the real full test-set run, this removed 179,535
  (3.42%) contested matches.

---

## 5. Results & Error Analysis

**Full 441,364-entity held-out validation split** (`src/validate_pipeline.py`,
same macro F0.5 the leaderboard computes, full production path incl. dedupe):

| Configuration | Macro F0.5 | Singleton Acc. | Non-singleton F0.5 |
| --- | --- | --- | --- |
| Baseline (pre-transliteration, global threshold) | 0.8693 | 92.87% | 0.8658 |
| + Transliteration + per-entity decoding (shipped, leaderboard 0.893) | **0.8980** | 90.74% | 0.8974 |

Country breakdown for the baseline row: US 0.9102 / India 0.8081 — a ~10
point gap that motivated the transliteration fix (Section 3). On the real
test-set run, the India-specific before/after (no ground truth exists for
test, so this is a before/after count, not a recall measurement) showed
matches/entity rising 2.534→2.852 and empty-match rate dropping 11.7%→8.2%,
while US and France (the controls) barely moved — consistent with the fix
being specific to the targeted segment.

**Leaderboard confirmation:** **0.893**, up from 0.863 before this round's
transliteration/decoding fixes, and within 0.005 of the validation harness's
prediction — the closest val/leaderboard alignment this project has
measured.

**Second round (blocking-key density + LightGBM hyperparameter tuning),
measured on a fast 50K-entity mini-validation subsample:**

| | Previous best | This round |
| --- | --- | --- |
| Macro F0.5 | 0.8980 | **0.9035** (+0.0055) |
| Singleton accuracy | 90.99% | 85.41% |
| Non-singleton F0.5 | 0.8972 | 0.9064 |

Net positive, but singleton accuracy dropped ~5.6 points and has not yet
been re-swept against this model's own probability distribution (the
existing 0.70 confidence floor was tuned against the previous model). This
configuration has been retrained and run across the full test set
(1,732,544 entities, 1,612,294 with at least one predicted match), but its
leaderboard score was not yet confirmed as of this document.

Two experiments were tried this round and reverted rather than shipped,
recorded so they aren't retried the same way:
- **Group-relative candidate features** (each candidate's rank/gap within
  its own entity's candidate pool): improved non-singleton F0.5
  (0.8972→0.8985) but dropped singleton accuracy badly (90.99%→84.71%);
  best achievable even after re-sweeping the confidence floor (0.8946 @
  floor=0.80) still fell short of the no-relative-features baseline.
- **Full blocking-key-density revert:** OOM-killed mid-index-build at the
  11GB RAM ceiling; a partial revert (one of the two key types) succeeded
  and is what shipped above.

**France hand-labeling** (60 random France test entities, predictions
checked by hand against the actual candidate name/address text, since no
France ground truth exists): most predictions were correct clean variants,
but a specific, recurring false-positive pattern emerged — a candidate with
**no name similarity at all** accepted purely on a strong address/house-number
match (e.g. `"UG Club"` matched to `"epicuriens unissons club
participations sas"` at the same street number). Roughly 4% of 184 sampled
predicted pairs showed this pattern. This is folded into the hyperparameter
search above as a `name_sim_floor` post-hoc gate (force a candidate's
probability to 0 when its name-similarity feature is below a threshold,
independent of the model's own prediction) — validated inside
`tune_screener.py`'s own scoring loop (best floor: 0.25) but **not yet wired
into real inference** (`decode.py`/`pipeline_duckdb.py`); likely remaining
headroom.

**Candidate recall ceiling vs. achieved:** 0.9614 overall ceiling vs. 0.9035
achieved on the same mini-validation candidates — this gap (the matching
model's own scoring, not blocking) is now the larger remaining lever than
blocking recall itself.

---

## 6. Conclusion
Pairing a DuckDB-backed multi-pass blocking engine (with a learned
cross-script transliteration layer) with an active-hard-negative-trained
LightGBM + TensorFlow ensemble, a per-entity expected-F0.5 decoder, and
global duplicate resolution, raised the confirmed public leaderboard score
from 0.863 to **0.893**. A further round of blocking-key density tuning and
LightGBM hyperparameter search measured a further gain to 0.9035 on a fast
validation subsample and has been run across the full test set; its
leaderboard confirmation and the known remaining gaps (the `name_sim_floor`
gate not yet wired into inference, the confidence floor not yet re-swept for
the new model, France's blind spot) are the explicitly tracked next steps.
All code is self-contained and compliant with open-source MIT/Apache 2.0
licensing constraints, with no external data lookups or APIs used anywhere
in the pipeline.

---

## Appendix: Code Artefacts
- **Runnable Source Package:** `code/business_entity_resolution/src/`
- **Reproduction Guide:** `code/business_entity_resolution/README.md`
- **Dependencies:** `code/business_entity_resolution/requirements.txt`
- **Outputs Produced:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`
- **Compliance Validator:** Validated with `student_resource/utils/validate_submission.py`
- **Internal engineering docs** (full chronological detail behind every
  claim above): `documents/engineering_log.md`, `documents/validation_results.md`,
  `documents/architecture.md`
