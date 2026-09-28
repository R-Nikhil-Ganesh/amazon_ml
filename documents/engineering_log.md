# Engineering Log — DuckDB Rewrite, Transliteration, Per-Entity Decoding

Chronological account of one round of work on top of the pipeline described
in [`Documentation_template.md`](../Documentation_template.md) (public
leaderboard 0.803 → 0.863 by the start of this round). Each section names
the files touched; line numbers are omitted deliberately since they drift —
search the named function instead.

## 1. Runtime: from ~6 hours to ~2 hours

**Symptom:** end-to-end runs took ~6 hours; public leaderboard submissions
from other teams came in much faster.

**Diagnosis:** the original `DiskBlockingEngine` (`src/sqlite_blocking.py`)
issued 3+ SQLite round trips per S1 entity — one row at a time, ~5M+ queries
total at full test scale — and index *build* time (not inference) was the
dominant cost, not the cascade scoring itself.

**Fix:** `src/duckdb_blocking.py`'s `DuckDBBlockingEngine` reuses the exact
same key generator (`build_blocking_keys`, `key_hash`, `KEY_WEIGHTS` — all
imported from `sqlite_blocking.py`, not duplicated, so the two engines can't
silently diverge) but batches retrieval: `retrieve_candidates_batch()` takes
a whole chunk of S1 entities and runs each of the 3 lookup passes (exact
name / compressed name / blocking keys) as one set-based JOIN across the
chunk, instead of one query per entity. Index build for the full ~9.97M-record
test candidate pool: **~24 minutes** (vs. the old index still mid-build after
25+ minutes when it was killed).

**A multiprocessing detour that didn't pay off.** The obvious next step —
shard the S1 stream across worker processes — was built (`src/pipeline_parallel.py`)
and two real bugs in it were found and fixed along the way:
- `_merge_shard_files()` skipped the first line of every shard part file
  assuming a header that workers never wrote, silently dropping one entity
  per shard.
- `multiprocessing.Pool.starmap()` silently respawns a replacement worker
  when one dies (e.g. OOM-killed) but does not resubmit the lost task, so a
  worker death meant the whole run hung forever with no error. Replaced with
  explicit `Process`+`Queue` and a timeout-based liveness check that fails
  loudly instead.

Even after both fixes, a real 30K-entity benchmark measured **~82 ent/s**
with 2 workers against the full index — no better than the single-threaded
baseline. Root cause: each worker opens its own DuckDB connection to a
~5.6GB index file; on a RAM-constrained box, multiple connections fragment
the page cache instead of sharing it. A same-scale single-process run
(`src/pipeline_duckdb.py`) measured **245-270 ent/s** — matching the
historically-best ~255 ent/s this pipeline had achieved before. **Decision:
single-process, not multiprocessing**, on this hardware profile.
`run_full_inference_duckdb.sh` is the recommended entry point;
`pipeline_parallel.py` / `run_full_inference_fast.sh` are kept but not used.

**Separately: available RAM was capped below the host's real capacity.**
WSL2 defaults to min(50% of host RAM, 8GB) when no `.wslconfig` exists; this
box was seeing 7.4GB out of a real 16GB host. Added
`C:\Users\<user>\.wslconfig` with `memory=12GB`, `swap=4GB`. This is a
Windows-side change and needs `wsl --shutdown` + reopen to take effect — it
does not persist through this repo, only through the Windows user profile.

## 2. Trustworthiness of the validation split

**Prompt for this:** a much earlier submission (before this round) had
scored 90% on validation but 65% on the leaderboard — a red flag that
validation numbers in this project have not always meant what they appeared
to mean.

**Finding:** at the time this was raised, the models actually deployed
(`models/screener_lgbm.pkl`, `models/tf_specialist_model.keras`) were newer
than every training log on disk — none of `retrain_screener_v4.log` (0.8172),
`retrain_tf_specialist.log` (0.8591), or `retrain_tf_v2.log` (0.8112) came
from the exact model that was live. Separately, no script in the repo ran
the *full* production path (blocking → LightGBM → TF specialist → dedupe)
against the held-out split and scored it the way the leaderboard scores
`matching_results.tsv` — `train_screener.py` and `train_tf_specialist.py`
each report their own number, but neither includes the dedupe step, and
both evaluate a subsample (8,000-50,000 of the real 441,364-entity
`val_s1_ids.txt`).

**Fix:** new script `src/validate_pipeline.py` — runs the real
`pipeline_duckdb.run_pipeline_duckdb()` over the *full* validation split
against a DuckDB index built from the *train* S2/S3 pool, then scores with
`evaluate.py`'s `evaluate_predictions()` (the same macro F0.5 the
leaderboard computes), broken down by country.

**Result:** current deployed model, unmodified, full 441K-entity split:
**macro F0.5 = 0.8693**, vs. **0.863** on the actual leaderboard — within
0.6 points. The validation split is a trustworthy proxy for this pipeline
as it stands today; whatever caused the earlier 90%→65% gap isn't present
now. See `validation_results.md` for the full table. This harness is what
every number in the rest of this log was measured with.

## 3. The real score gap: Indic-script blocking recall

With a trustworthy validation harness in hand, blocking recall was measured
split by country and by script (`_is_brahmic_char` from `normalize.py`
detects Devanagari/Telugu/Tamil/Bengali/Gujarati/Gurmukhi/Kannada/Malayalam —
about 23% of India S2/S3 candidate names/addresses use one of these):

| Segment | Blocking recall |
|---|---|
| US | 0.922 |
| India, Latin-script candidate | 0.900 |
| India, Indic-script candidate | **0.195** |

India is ~47% of the test set. This is the dominant lever. The existing
`phonetic_skeleton()` (`normalize.py`) — a lossy Brahmic-to-Latin-consonant
fold used as one blocking key and one feature — was the only cross-script
bridge in the pipeline before this round, and it wasn't enough on its own.

## 4. Fix: learned transliteration dictionary + rule-based fallback

**Dictionary (`src/transliterate.py`, `build_transliteration_dict()`).**
Built from `train_ground_truth.tsv` matched pairs only — not external data.
Restricting to India pairs where the candidate name has Indic-script
characters, 99.998% of pairs have equal token counts between the S1 (Latin)
and candidate (Indic) name, so a word-level positional alignment learns a
near-complete word→word map. Result: **1,378 distinct words** (e.g.
`प्राइवेट`→`private`, `లిమిటెడ్`→`limited`), persisted to
`data_split/translit_dict.json`. Applied to candidate name/address at
index-build time, in both `DiskBlockingEngine.build_index()` and
`DuckDBBlockingEngine.build_index()` (before normalization, so every
downstream blocking key and feature sees the transliterated text) —
`sqlite_blocking.py` and `duckdb_blocking.py` share this logic to avoid the
two engines diverging.

**Measured effect (via a full rebuild of the train index + the same
per-script recall measurement as §3):**

| Segment | Before | After (dictionary only) |
|---|---|---|
| India, Indic-script | 0.195 | **0.781** |
| India, Latin-script | 0.900 | 0.889 (unchanged, within noise) |

The gain is specific to the targeted segment, not a general shift — the
Latin-script number staying flat is the control.

**Rule-based fallback (`transliterate_brahmic_token()` in `normalize.py`).**
The dictionary is exact-word-only: any Indic word absent from the ~1,400
training-derived vocabulary still passes through untranslated. Added a
second, independent mechanism — a full phonetic transliteration using only
Unicode block arithmetic (all 8 scripts share Devanagari's relative layout,
an ISCII legacy), not any learned model or external corpus: every consonant
carries an implicit "a" vowel unless a vowel sign replaces it or a virama
suppresses it, plus schwa deletion on a word's final consonant (so
`ग्लोबल`→`global`, not `globala` — these are English loanwords, and this is
how Hindi/Indic phonology actually drops that trailing vowel in speech).
Deliberately kept in a **separate table** from `phonetic_skeleton()`'s
existing consonant map, so this doesn't shift the already-trained LightGBM
screener's `name_skeleton_ratio` feature. Used only as a fallback — the
dictionary is checked first.

Two related options were considered and rejected:
- **IndicXlit (AI4Bharat)** — MIT-licensed transliteration *model*, but its
  weights are trained on Aksharantar, a CC-BY external corpus. The
  competition's rules prohibit "external data augmentation from internet
  sources" on pain of disqualification; a model whose knowledge comes from
  an external corpus is a plausible target for that rule even run offline.
  Rejected on compliance risk, not technical merit.
- **`indic_transliteration` (sanscript)** — purely rule-based (character
  mapping tables, no learned weights), so lower compliance risk than
  IndicXlit, but its output is scholarly romanization (IAST/ITRANS
  conventions used for Sanskrit texts), not business spelling — would still
  need the same phonetic-folding layer on top to be useful, for the cost of
  a new third-party dependency in a submission package that gets audited.
  Rejected as not worth the added review surface over just extending
  `phonetic_skeleton()`'s existing machinery directly.

**Status:** the fallback is implemented and unit-verified at the word level
(2 of 6 spot-checked words transliterate to an exact match; 4 of 6 produce
an identical `phonetic_skeleton()` fold to the true English word, meaning
the existing blocking/feature layer bridges them even without an exact
string match). A quick attempt to verify its effect on blocking recall
without a full rebuild came back **inconclusive** (see
`validation_results.md` §"A test that didn't work" for why) — real
verification requires the full index rebuild, which the pending full
test-set run (§6) includes.

## 5. Fix: per-entity expected-F0.5 decoding

**Problem.** The pipeline used one global probability threshold (0.75),
chosen once by sweeping validation, applied identically to every S1 entity.
Since the competition metric is F0.5 macro-averaged *per entity*, this is
provably suboptimal: the same candidate probability should be accepted for
one entity and rejected for another depending on that entity's other
candidates. `match_scores.tsv` also only retained probabilities for
candidates that already passed the threshold, so this couldn't even be
fixed as a post-process without first capturing every candidate's score.

**Fix (`src/decode.py`).** `pipeline_duckdb.py` now writes every retrieved
candidate's probability, not just accepted ones. `decode.decide_matches_for_entity()`
picks, per entity, the top-k candidates (by probability) that maximize an
approximation of that entity's expected F0.5. The literature's exact
computation (Ye et al., ICML 2012) requires a joint Poisson-binomial
distribution and is O(n²)-O(n³) per entity — too slow at 1.73M entities in
pure Python. Used instead: a first-order plug-in approximation,
`E[F_beta(k)] ≈ (1+β²)·S_k / (β²·S_n + k)` where `S_k` is the sum of the
top-k probabilities — O(n) per entity. `k=0` (predict no match) is handled
exactly via the competition's own singleton rule:
`E[F_beta(0)] = P(every candidate is a non-match) = Π(1-p_i)`.

**First measurement exposed a real problem.** Without any safeguard, the
approximation is weakest exactly when an entity's probabilities are all
mediocre and none stand out — precisely the profile of a true singleton
where blocking still returned a few so-so candidates. Net effect: macro
F0.5 improved only marginally (+0.0032) while singleton accuracy collapsed
from 92.87% to 74.34% (see `validation_results.md`).

**Fix: a confidence floor.** `decode.py`'s `MIN_TOP_PROB_FOR_ANY_MATCH`
requires the single best candidate to clear a floor before any non-empty
set is considered at all — directly targets the failure mode (several weak
candidates, none individually convincing) without touching the k≥1
selection logic, which was producing a real, separately-measured gain
(+0.0143 non-singleton F0.5). The floor value was tuned by sweeping against
already-computed probabilities from a prior run (no pipeline rerun needed —
`resolve_winners()`'s candidate-ownership resolution is independent of the
floor, computed once and reused across the sweep). **0.70** was chosen:
recovers singleton accuracy to within 2 points of the pre-decoder baseline
(90.74% vs. 92.87%) while keeping nearly all of the macro gain. A more
conservative floor was deliberately preferred over the sweep's raw peak
(0.60, marginally higher macro but 84.6% singleton accuracy) specifically
because the France segment (15% of the test set, zero training
representation) has no ground truth to check calibration against before a
real submission — safer to lean conservative there.

## 6. Real test-set run and leaderboard result

The full 1.73M-entity test-set run (`full_run_duckdb_v2.log`) completed:
6,042s (100.7 min) at ~287 ent/s, validator PASS, 5,073,697 final matches
after dedupe. **Leaderboard: 0.893**, up from 0.863 before this round, and
within 0.005 of the validation harness's 0.8980 prediction — the tightest
val/leaderboard alignment this project has seen, confirming
`validate_pipeline.py` (§2) generalizes as a trustworthy proxy beyond the
specific run it was first checked against. Full numbers, including a
country-level before/after on the real test output, are in
`validation_results.md` §6.

**Open item:** the rule-based transliteration fallback (§4) still hasn't
had its blocking-recall effect measured properly — the one attempt (§4's
closing paragraph) used a shortcut that turned out to be insensitive to
the real bottleneck. A rebuild of the train validation index with current
code, re-run through the same per-script recall check as §3, is needed to
close this out.

## 7. Repository cleanup

Once the fixes above were validated, three categories of dead weight were
removed from the package a grader would actually audit:

- **LLM remains:** `src/llm_dataset.py` (self-documented as unused —
  produces `data_split/llm_train_data.jsonl`, which nothing consumes; the
  committed version in git history was empty, so nothing of value was
  lost) and that orphaned JSONL output.
- **The rejected multiprocessing pipeline:** `src/pipeline_parallel.py` and
  `run_full_inference_fast.sh` — §1's own measured conclusion was "don't
  use this," so keeping a working-but-rejected full pipeline sitting in
  the graded `src/` folder, indistinguishable from a real option, was a
  liability rather than useful history. That history is preserved here in
  §1 instead, in more useful form than the code itself.
- **Debris from the killed early multiprocessing test run:** `output/shards/`,
  `output/*.part{0,1,2}.tsv` (several 0 bytes), and the stale SQLite test
  index `output/candidates_index.db` (5.47 GB, predated every fix in this
  log, superseded by `candidates_index.duckdb`; `pipeline.py` — still kept,
  see below — auto-rebuilds it if that path is ever exercised).
- Added `/pytorch` to `.gitignore` — an unrelated, unused 5.4 GB virtual
  environment (`torch`/`transformers`/`accelerate`, not referenced
  anywhere, not in `requirements.txt`) sitting at the repo root with no
  ignore rule, meaning a `git add -A` would have staged it. Not deleted
  (not confirmed whether it's wanted for something else), but no longer a
  silent commit risk.

**Update (§8): the training/inference engine mismatch flagged here was
resolved later the same session** — `train_screener.py` and
`train_tf_specialist.py` were migrated to `DuckDBBlockingEngine`. See §8.
`src/pipeline.py` and `DiskBlockingEngine` (`sqlite_blocking.py`) itself
were still not deleted — `DiskBlockingEngine`'s key-generation functions
(`build_blocking_keys`, `key_hash`, `KEY_WEIGHTS`) are imported directly by
`duckdb_blocking.py` and are genuinely load-bearing; only the
SQLite-specific query/storage code is now dead. `pipeline.py` itself
appears to have no remaining callers after §8, but was left in place rather
than deleted without more certainty.

## 8. Training/inference engine consistency: migrating to DuckDB

**The debt from §7:** `train_screener.py`/`train_tf_specialist.py` still
built their training pairs via `DiskBlockingEngine` (SQLite,
`data_split/train_candidates_index.db`) while real inference
(`pipeline_duckdb.py`) used `DuckDBBlockingEngine`. The SQLite training
index was stale (predated transliteration), so a retrain would have trained
against a candidate universe the deployed pipeline didn't actually search.

**Fix:** both scripts now use `DuckDBBlockingEngine` against
`TRAIN_CANDIDATE_INDEX_DUCKDB_PATH` (`config.py`), with a genuine rewrite of
their candidate-retrieval loop, not just a mechanical swap: both now
retrieve in `RETRIEVAL_CHUNK_SIZE=4000`-entity batches via
`retrieve_candidates_batch()` (matching `pipeline_duckdb.py`'s own
batching) instead of one entity at a time — this made later work (bumping
`train_sample_size`, the hyperparameter search in §13) cheap enough to run
repeatedly. `DuckDBBlockingEngine.get_by_id()` was added to match
`DiskBlockingEngine`'s existing method (same signature/shape), needed by
both scripts' missed-positive-injection and hard-negative-mining code.

## 9. Relative/contextual features: tried, measured, reverted

**Hypothesis:** the ceiling-vs-achieved measurement (candidate recall
ceiling minus what the deployed cascade actually achieves within those same
candidates — overall gap 0.0661, India 0.0726, US 0.0618 on the mini-val
split) showed the LightGBM+TF cascade's own scoring, not blocking recall,
as the larger remaining lever. `extract_pair_features()` scores every
(S1, candidate) pair in isolation, with no signal for how that candidate
ranks among the same entity's other retrieved candidates — a plausible gap
per prior deep-research findings on entity-resolution feature design.

**Implementation:** `add_group_relative_features()` in `features.py` added
four features computed once per entity's full candidate group:
`rel_composite` (a cheap blended proxy score from features already
computed), `rel_rank_frac` (this candidate's rank within its group, 0=best),
`rel_gap_to_top` (gap to the group's best score), `group_size_norm`. Wired
into all three feature-matrix builders (`train_screener.py`,
`train_tf_specialist.py`, `pipeline_duckdb.py`), computed on the *full*
retrieved group before any negative-subsampling (subsampling would corrupt
what "best in group" means).

**Result — net negative, reverted:**

| | No relative features (baseline) | With relative features |
|---|---|---|
| Macro F0.5 | 0.8980 | 0.8955 (best floor: 0.8946 @ 0.80) |
| Singleton accuracy | 90.99% | 84.71% (best floor: 91.16% @ 0.80) |
| Non-singleton F0.5 | 0.8972 | 0.8985 |

Non-singleton F0.5 genuinely improved, confirming the hypothesis had real
signal — but singleton accuracy dropped hard. Mechanism: `rel_rank_frac=0`
("best in its group") gets assigned to *some* candidate for every entity
with any candidates at all, including true singletons where none of them
are a real match — the model likely learned "best of group" as a positive
signal, which actively misleads it specifically on singletons. Re-sweeping
`decode.py`'s confidence floor (the same lever that fixed an unrelated
singleton-accuracy problem in `validation_results.md` §4) recovered some of
this but never closed the gap to the no-relative-features baseline at any
floor tested (0.70-0.95). **Reverted** — `add_group_relative_features()`
now a documented no-op kept as a stable import target.

## 10. Blocking-key density revisited: `num_`/`bg_` trim, partially reverted

A much earlier session had trimmed `build_blocking_keys()`'s `num_` and
`bg_` key generation (in `sqlite_blocking.py`, shared by both engines) to
fit the then-7.4GB WSL2 RAM cap: `num_` keys went from pairing every street
number with the top-2 rarest address tokens down to just 1 token (measured
at the time as 37.1% of all `cand_keys` rows); `bg_` went from full
`combinations()` over the top-4 rarest tokens down to a single pair. That
trim cost real recall at the time (validation pair recall 86.4%→77.1%,
ensemble macro F0.5 0.8665→0.8177) but was accepted given the RAM
constraint.

Since then, the RAM cap was raised to 11GB and blocking-key storage moved
from raw `TEXT` postings to compact int64 hashes (`key_hash()`) — worth
re-testing whether the original trim still needs to be that aggressive.

**Full revert (both `num_`→2 tokens and `bg_`→top-4 combinations)**: OOM-killed
during index build — `cand_keys` had already reached ~7GB with `CREATE
INDEX` (the most memory-hungry step) still ahead of it. Real evidence, not
estimate: reimplementing both generators over a 50K-record sample predicted
~1.6x more total keys (`bg_` alone ~4.2x), and the actual attempt confirmed
that doesn't fit even at 11GB.

**Partial revert (`num_` alone, `bg_` left at its original single-pair trim)**:
succeeded — final index 10,320,219 candidates, 52,933,024 keys (5.13
keys/candidate), 6.08 GB on disk (vs. ~5.1GB before). Combined with §13's
hyperparameter tuning (both retrained together, not isolated — see below
for why), full pipeline mini-val result: **macro F0.5 0.9035** (vs. 0.8980
baseline), singleton accuracy 85.41%, non-singleton F0.5 0.9064.

**Known gap:** because this was retrained together with the tuned LightGBM
hyperparameters (§13) rather than as an isolated A/B, the individual
contribution of the `num_` key revert alone is not separately measured.

## 11. TF specialist: epoch budget was 3x too long

Epoch-by-epoch `val_auc` from a real training run: 0.9960 (ep1) → 0.9962
(ep2) → 0.9971 (ep3) → oscillates 0.9966-0.9975 for the remaining 9 epochs
(net gain ~0.0004). The `EarlyStopping(patience=4)` callback never actually
triggered, because sub-0.0001 float fluctuations (invisible at the printed
4-decimal precision) kept resetting its improvement counter, so every run
burned its full 12-epoch budget (~18 min) for essentially no gain past
epoch 3-4. Fixed two ways: `min_delta=0.0005` on `EarlyStopping` so noise
can't reset it, and the hard `epochs` cap in `train_tf_specialist.py`
lowered from 12 (later raised to 6 per explicit request, still well under
the original 12) — `restore_best_weights=True` already guarantees the
actual best epoch is kept either way, so this only cuts wasted wall-clock,
not model quality.

## 12. France: a complete blind spot, hand-labeled to check

`csv_data/train/` (both `source1.csv` and `ground_truth.csv`) contains
**zero France entities** — 1,323,633 US + 883,188 India only. The real test
set is 15% France (259,452 of 1,732,544). No validation ground truth for
France exists anywhere, so every other measurement in this log and in
`validation_results.md` is implicitly US/India-only.

**Aggregate stats first** (from the real deployed full-test-set output, no
ground truth needed — just volume/confidence sanity checks): blocking looks
fine (avg 30.2 candidates/entity for France vs. 28.5 India / 29.5 US), but
France's average top-candidate probability (0.9643) was *higher* than both
US (0.9419) and India (0.9368) despite zero training representation —
ambiguous on its own (could mean genuinely cleaner French records, or
uncalibrated overconfidence).

**Hand-labeled sample** (60 random France test entities, predicted matches
read against actual candidate name/address text): most predictions were
correct clean variants (typos, case, legal-suffix differences), but a
specific, recurring false-positive pattern emerged: candidates with **no
name similarity at all** being accepted purely on a strong address/number
match. Examples: `"UG Club"` → `"epicuriens unissons club participations
sas"` (same street number, unrelated name); `"Professionnels Ecole SAS"` →
`"Quofayeyuma"` and `"Vantagearia"` (both same address, both semantically
unrelated names); `"Boules Club SAS"` → a candidate at a **different city**
entirely. ~7 of 184 sampled predicted pairs (~4%) showed this pattern,
concentrated in ~12% of entities (the ones with many co-located
candidates) rather than spread evenly.

**Conclusion:** not "France is broken" — it's a specific, identifiable
LightGBM screener weakness (under-weighting name dissimilarity when
address/number features are strong) that plausibly affects US/India too,
just with no labeled France sample to compare against directly. Folded into
§13 as a `name_sim_floor` gate rather than built as a separate fix.

## 13. LightGBM random-search hyperparameter tuning + name-similarity floor

`LGBM_PARAMS` (`config.py`) had been unchanged since the first commit —
never tuned. New `src/tune_screener.py`: builds the train/val feature
matrices **once** (the expensive part, unrelated to hyperparameters) and
reuses them across every trial, unlike naively re-running
`train_screener.py` N times. Random search over `num_leaves`,
`learning_rate`, `max_depth`, `feature_fraction`, `bagging_fraction`,
`min_child_samples`; each trial additionally swept against 4
`name_sim_floor` values (0, 0.15, 0.25, 0.35 — the §12 fix: force a
candidate's probability to 0 when `name_token_set` falls below the floor,
independent of what the model itself predicts, the same style of gate as
`decode.py`'s `MIN_TOP_PROB_FOR_ANY_MATCH`).

**Bug hit and fixed:** reusing the same `lgb.Dataset` objects across trials
with varying `min_child_samples` breaks LightGBM's feature pre-filtering
(locked in from the first trial) — `LightGBMError: Reducing
min_data_in_leaf with feature_pre_filter=true...`. Fixed by setting
`feature_pre_filter: False` in every trial's params. Verified with a smoke
test (2 trials, 3K samples) before re-running the full search.

**Result:** best trial (12 trials × 4 floors = 48 evaluations, 150K training
sample) scored macro F0.5 **0.9080** vs. 0.8980 baseline, on
`tune_screener.py`'s own quick val split — `num_leaves=95` (up from 63),
`min_child_samples=10` (down from 20), `name_sim_floor=0.25`. `train_screener.py`
gained a `--lgbm-params-json` flag to retrain with a winning config directly
from `tune_screener.py`'s output.

**Known gap:** the `name_sim_floor` gate was only evaluated inside
`tune_screener.py`'s own scoring loop — it was never wired into
`decode.py`/`pipeline_duckdb.py` for real inference, so the real deployed
pipeline does not currently apply it. The §10/§13-combined retrain that
produced the real 0.9035 mini-val result (§10) used only the tuned LGBM
hyperparameters, not the floor — there may be more headroom left
unclaimed here.

## 14. Final retrain and real submission run (this session)

Screener and TF specialist retrained together on the `num_`-key-reverted
index (§10) with the tuned hyperparameters (§13, floor not applied — see
§13's known gap), 150K training samples (reduced from the originally
planned 300K after a live OOM near-miss — `build_pair_dataset()` accumulates
every candidate pair's features as Python lists before the final
`np.array()` conversion, more memory-hungry per row than the old lower-key-density
index needed; a real architectural inefficiency worth fixing if training
sample size needs to go back up). Confirmed via `validate_pipeline.py --mini`
before the real run: **macro F0.5 0.9035** (vs. 0.8980 previous best),
singleton accuracy 85.41%, non-singleton F0.5 0.9064.

Real full-scale inference (`pipeline_duckdb.py --rebuild-index`, rebuilding
`output/candidates_index.duckdb` with the `num_`-reverted keys) completed:
1,732,544 entities, 1,612,294 with at least one predicted match. Leaderboard
score not yet confirmed as of this log entry.
