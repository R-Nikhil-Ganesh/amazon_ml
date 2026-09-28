# Validation Results Log

All numbers below are macro F0.5 on the **full 441,364-entity validation
split** (`data_split/val_s1_ids.txt`), scored via `evaluate.py`'s
`evaluate_predictions()` — the same metric the leaderboard computes —
through `src/validate_pipeline.py` (see `architecture.md`). Raw logs are in
the repo root (`validate_*.log`). Every run here used the same deployed
LightGBM + TF specialist models; nothing below retrains either model.

## 1. Baseline: does validation track the leaderboard at all?

The concern motivating this check: an earlier submission (before this
round) scored 90% on validation but 65% on the leaderboard.

| | Score |
|---|---|
| Leaderboard (real submission, before this round's changes) | 0.863 |
| This validation run (`validate_baseline.log`) — no transliteration, global 0.75 threshold, full production path incl. dedupe | **0.8693** |

Within 0.6 points — validation is a trustworthy proxy for this pipeline as
it stands. Country breakdown, same run:

| Country | n | Macro F0.5 | Singleton acc. |
|---|---|---|---|
| US | 264,631 | 0.9102 | 96.50% |
| India | 176,733 | 0.8081 | 87.48% |

The ~10-point India/US gap here is what led to §3.

## 2. Decoding alone (no transliteration)

`validate_decode_gain.log` — same index as §1 (no transliteration), decode
mode switched to per-entity expected-F0.5 top-k, **no confidence floor yet**
(that came after this run, see §4).

| | Macro F0.5 | Singleton acc. | Non-singleton F0.5 |
|---|---|---|---|
| §1 baseline | 0.8693 | 92.87% | 0.8658 |
| Decode, no floor | 0.8725 (+0.0032) | **74.34%** (−18.5 pts) | 0.8801 (+0.0143) |

Net positive but a bad trade — see §4 for the fix. Country breakdown:
India 0.8122 (singleton 67.68%), US 0.9127 (singleton 78.81%).

## 3. India blocking recall, before and after transliteration

Not from `validate_pipeline.py` — a direct measurement of
`retrieve_candidates_batch()` against a sample of 1,500 India validation
entities, split by whether the true-match candidate's name is Indic-script
or Latin-script (script detected via `_is_brahmic_char`).

| Segment | Before | After (learned dictionary only, 1,378 words) |
|---|---|---|
| India, Indic-script | **0.195** | **0.781** |
| India, Latin-script | 0.900 | 0.889 (unchanged, within noise) |
| India, overall | 0.776 | 0.870 |

The Latin-script number staying flat while only the targeted segment moves
is the control confirming this is the transliteration's effect, not noise
or an unrelated index change.

## 4. Both fixes together, floor sweep, and the final choice

`validate_both_fixes.log` — transliteration (dictionary only, rule-based
fallback added later, see `engineering_log.md` §4) + decode with an
un-tuned floor guess of 0.50.

| | Macro F0.5 | Singleton acc. | Non-singleton F0.5 |
|---|---|---|---|
| §1 baseline | 0.8693 | 92.87% | 0.8658 |
| Both fixes, floor=0.50 | 0.8986 (+0.0293) | 79.85% | 0.9045 |

Country breakdown: India **0.8773** (+0.0692 vs. §1's 0.8081 — the dominant
contributor), US 0.9129 (+0.0027).

The 0.50 floor was a guess. Swept against the *same run's* already-computed
`match_scores.tsv` (no pipeline rerun — `resolve_winners()`'s global
candidate-ownership resolution doesn't depend on the floor, so it was
computed once and reused; full sweep took 150s):

| Floor | Macro F0.5 | Singleton acc. | Non-singleton F0.5 |
|---|---|---|---|
| 0.00 | 0.8980 | 74.71% | 0.9069 |
| 0.50 | 0.8986 | 79.85% | 0.9045 |
| 0.55 | 0.8989 | 82.44% | 0.9033 |
| **0.60** | **0.8989 (peak)** | 84.56% | 0.9020 |
| **0.70 (shipped)** | 0.8980 | **90.74%** | 0.8974 |
| 0.75 | 0.8967 | 93.54% (exceeds §1 baseline) | 0.8944 |
| 0.85 | 0.8938 | 95.62% | 0.8901 |

**0.70 was chosen over the raw peak (0.60)**: macro F0.5 is nearly flat
across 0.50-0.70 (±0.0009) while singleton accuracy swings by 11 points
across that same range — 0.70 gets back to within 2 points of the original
92.87% while keeping nearly all of the macro gain. Preferred conservative
over optimal-on-val specifically because the France segment (15% of the
real test set) has zero training representation, so its calibration can't
be checked before a real submission.

Confirmed via the actual `decode.py` module (not the sweep script's
reimplementation) at floor=0.70: **macro F0.5 = 0.8980, singleton accuracy
90.74%, non-singleton F0.5 = 0.8974** — no drift between the sweep and the
shipped code.

## 5. A test that didn't work (recorded so it isn't tried again the same way)

After adding the rule-based transliteration fallback (`engineering_log.md`
§4), tried to check its effect without a full index rebuild by computing
raw blocking-key overlap between S1 and candidate records directly (same
`build_blocking_keys()` function, no database). Result: **both dictionary-only
and dictionary+fallback scored ~97-99.9%** — the test is not sensitive to
the actual bottleneck. The real index applies two things this shortcut
skipped: `MAX_TOKEN_POSTINGS` (any key shared by >150 candidates is dropped
entirely — likely eating common words like "private"/"limited") and
`RERANK_POOL_SIZE`/`MAX_CANDIDATES_PER_S1` truncation to the top 35. A
"does any key match, in principle" check can't distinguish a real
improvement from an already-saturated signal. **Any future check on
blocking recall needs to go through the real built index and
`retrieve_candidates_batch()`, not a standalone key-overlap check.**

## 6. Real test-set run and leaderboard result

`full_run_duckdb_v2.log` — full 1,732,544-entity test set, transliteration
(dictionary + rule-based fallback) + decode floor=0.70, index rebuilt from
scratch (5.29 GB, 39,950,822 keys). Completed in 6,042s (100.7 min);
inference alone ran at ~287 ent/s. Validator: **PASS**, no blocking issues.
5,253,232 matches decoded, 179,535 (3.42%) removed by dedupe as
duplicate-claimed, leaving 5,073,697 final matches across 1,732,544
entities (133,834 predicted as singletons).

**Leaderboard result: 0.893** (up from 0.863 before this round). Within
0.005 of the validation harness's 0.8980 prediction (§4) — the closest
val/leaderboard alignment seen in this project's history, and a strong
confirmation that `validate_pipeline.py` (§ "Baseline" above) is a
trustworthy proxy going forward, not just for this specific change.

**Corroborating signal on the real test set** (no ground truth exists for
test, so this isn't a recall measurement — it's a country-level before/after
using the same candidate/match-count breakdown run on the pre-fix test
output earlier in this project):

| Country | Matches/entity (before → after) | Empty-match rate (before → after) |
|---|---|---|
| India | 2.534 → **2.852** | 11.7% → **8.2%** |
| US | 2.852 → 2.921 | 8.3% → 7.9% |
| France | 3.093 → 3.187 | 6.3% → 5.8% |

India moved the most by a wide margin (closing a 0.318 matches/entity gap
to US down to 0.069), while US and France — the control — barely moved.
Consistent with the fix being specific to the targeted segment. Candidates
per India entity stayed roughly flat (28.7 → 28.5) while accepted matches
rose, consistent with the same ~28-slot candidate budget now more often
containing the true match in comparable (transliterated) text, rather than
blocking retrieving more candidates overall.

**Remaining open item:** whether the rule-based fallback (on top of the
dictionary) measurably improves Indic-script blocking recall specifically
— §5's key-overlap shortcut couldn't answer this, and no validation run
since the fallback was added has re-measured it properly. A rebuild of
`data_split/train_candidates_index.duckdb` with current code, re-run
against the same 1,500-entity India validation sample as §3, will give
that number.

**Still needed before the next submission package:** update
`Documentation_template.md` with this round's methodology and results —
it currently describes the pre-DuckDB, pre-transliteration state.

## 7. Second round: key-density revert + hyperparameter tuning

See `engineering_log.md` §8-§14 for full detail on each change. Two
experiments were tried and reverted before landing on the combination
below — recorded so they aren't retried the same way:

- **Relative/contextual features** (§9): non-singleton F0.5 improved
  (0.8972→0.8985) but singleton accuracy dropped badly (90.99%→84.71%);
  best achievable even after re-sweeping the floor (0.8946 @ floor=0.80)
  still fell short of baseline. Reverted.
- **Full `bg_`/`num_` blocking-key revert** (§10): OOM-killed during index
  build (~7GB, `CREATE INDEX` still ahead of it). `num_` alone (partial
  revert) succeeded.

**What shipped:** `num_` key revert (§10) + LightGBM hyperparameter tuning
via random search (§13, `num_leaves=95`, `min_child_samples=10`, both
retrained together, not isolated) + TF specialist epoch budget fix (§11,
6-epoch cap instead of a noise-defeated 12-epoch early-stop). Confirmed via
`validate_pipeline.py --mini` (50K entities, real candidate retrieval +
dedupe):

| | Previous best | This round |
|---|---|---|
| Macro F0.5 | 0.8980 | **0.9035** (+0.0055) |
| Singleton accuracy | 90.99% | 85.41% |
| Non-singleton F0.5 | 0.8972 | 0.9064 |

Net positive, but singleton accuracy dropped ~5.6 points — not yet
re-swept for this model (the floor tuning in §4 was done against the
*previous* model's probability distribution, not this one). **The
`name_sim_floor` gate from §13 (found via France hand-labeling, §12) was
never wired into the real pipeline** — the 0.9035 result does not include
it, so there is likely unclaimed headroom (and probably some singleton-
accuracy recovery) left on the table from both a floor re-sweep and
actually applying the name-similarity gate.

**Real full-scale run completed** (`pipeline_duckdb.py --rebuild-index`,
rebuilding the test index with `num_`-reverted keys): 1,732,544 entities,
1,612,294 with at least one predicted match. **Leaderboard score not yet
confirmed** as of this entry — update this section once known.
