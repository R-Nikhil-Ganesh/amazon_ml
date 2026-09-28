#!/usr/bin/env python3
"""
Per-entity expected-F0.5 decoding, replacing a single global probability
threshold (e.g. 0.75) applied identically to every S1 entity.

Why: the competition metric is F_0.5 computed per S1 entity, then averaged
(see evaluate.py). Under that macro-average, the same candidate probability
should be accepted for one entity and rejected for another depending on
that entity's other candidates - a fixed global threshold cannot express
this. This is a known result (Waegeman et al., JMLR 2014; Dembczynski et
al., NeurIPS 2011): for a set of candidates with independent match
probabilities, F_beta is maximized by choosing, for each entity separately,
the top-k candidates by probability for whichever k (0..n) maximizes that
entity's own expected F_beta - not by thresholding each candidate in
isolation.

Exact computation of E[F_beta(k)] requires the joint distribution of (a)
true positives within the top-k set and (b) true positives among the
remaining n-k candidates (both Poisson-binomial, since each candidate's
true-match label is an independent Bernoulli(p_i)) - an O(n^2)-O(n^3) exact
DP (Ye et al., ICML 2012). At this pipeline's scale (~1.73M entities) that
is too slow in pure Python. Instead this uses the standard identity

    F_beta(k) = (1 + beta^2) * TP / (beta^2 * T + k)

(T = total true matches among all n candidates, TP = true matches within
the top-k) and a first-order (plug-in) approximation E[TP/(...)] ~=
E[TP]/E[...], giving an O(n) per-entity computation:

    E[F_beta(k)] ~= (1 + beta^2) * S_k / (beta^2 * S_n + k)

where S_k is the sum of the top-k probabilities and S_n is the sum of all n
probabilities for that entity. This is an approximation, not the exact
expectation - it should be close whenever an entity's probabilities are
well-separated (near 0 or 1), which is the cascade's design goal (see
THRESHOLD_AUTO_ACCEPT/REJECT in config.py), and is still a strictly better
decision rule than one global threshold. k=0 (predict no match) is handled
exactly via the competition's own singleton rule (see evaluate.py): its
expected score is P(every candidate is a non-match) = prod(1 - p_i).
"""

import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple

BETA2 = 0.25  # beta^2 for F_0.5

# Measured on the real 441K-entity validation split: without this floor, the
# plug-in approximation (see module docstring) pushed singleton accuracy
# from 92.9% down to 74.3% - several mediocre-probability candidates (none
# individually convincing) summed enough plug-in "expected TP" to beat the
# k=0 empty-set score, when the honest answer was usually "none of these".
# The approximation is weakest exactly when probabilities aren't
# well-separated, which is this case. Requiring the single best candidate to
# clear this floor before considering any non-empty set targets that
# failure mode directly, without touching the k>=1 selection logic itself.
#
# 0.70 was chosen by sweeping the floor against the same validation run's
# already-computed probabilities (cheap - no pipeline rerun needed):
#   floor=0.50: macro F0.5=0.8986, singleton acc=79.9%
#   floor=0.60: macro F0.5=0.8989 (peak), singleton acc=84.6%
#   floor=0.70: macro F0.5=0.8980, singleton acc=90.7%
#   floor=0.75: macro F0.5=0.8967, singleton acc=93.5% (exceeds the
#               pre-decoder baseline of 92.9%)
# Macro F0.5 is nearly flat across 0.50-0.70 (+/-0.0009) while singleton
# accuracy swings by 11 points - 0.70 gets back to within 2 points of the
# original 92.9% singleton accuracy while keeping nearly all of the macro
# gain, which matters most on the unseen France slice (no training data,
# so its calibration can't be checked before a real submission - a more
# conservative floor is the safer default there).
MIN_TOP_PROB_FOR_ANY_MATCH = 0.70


def decide_matches_for_entity(probs: List[float], beta2: float = BETA2) -> int:
    """Given one entity's candidate probabilities (any order), return the
    number of top-ranked candidates (by probability) to keep."""
    if not probs:
        return 0
    probs_sorted = sorted(probs, reverse=True)
    if probs_sorted[0] < MIN_TOP_PROB_FOR_ANY_MATCH:
        return 0
    n = len(probs_sorted)
    s_n = sum(probs_sorted)

    f0 = 1.0
    for p in probs_sorted:
        f0 *= (1.0 - p)
    best_k, best_f = 0, f0

    s = 0.0
    for k in range(1, n + 1):
        s += probs_sorted[k - 1]
        denom = beta2 * s_n + k
        f_k = ((1.0 + beta2) * s / denom) if denom > 0 else 0.0
        if f_k > best_f:
            best_k, best_f = k, f_k
    return best_k


def decode_all(scores_path: str, candidate_pairs_path: str, out_path: str, beta2: float = BETA2) -> Tuple[int, int]:
    """Read every candidate's probability from scores_path (s1_id, cand_id,
    prob - no threshold applied, every retrieved candidate present), pick
    each S1 entity's match set via decide_matches_for_entity(), and write
    matching_results.tsv. candidate_pairs_path supplies the full, ordered
    list of S1 ids (including those with zero candidates), so every S1
    entity gets exactly one output row per the submission format rules.

    Returns (total_entities, total_matches_written).
    """
    s1_probs: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    with open(scores_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 3:
                continue
            s1_id, cand_id, prob = row[0], row[1], row[2]
            s1_probs[s1_id].append((cand_id, float(prob)))

    total_entities = 0
    total_matches = 0
    with open(candidate_pairs_path, "r", encoding="utf-8", newline="") as f_in, \
         open(out_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in, delimiter="\t")
        next(reader, None)
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            total_entities += 1
            pairs = s1_probs.get(s1_id, [])
            pairs.sort(key=lambda x: x[1], reverse=True)
            k = decide_matches_for_entity([p for _, p in pairs], beta2=beta2)
            matches = [cid for cid, _ in pairs[:k]]
            f_out.write(f"{s1_id}\t{','.join(matches)}\n")
            total_matches += len(matches)

    return total_entities, total_matches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True, help="match_scores.tsv (every candidate, not just accepted ones)")
    parser.add_argument("--candidates", required=True, help="candidate_pairs.tsv (supplies the full S1 id list)")
    parser.add_argument("--out", required=True, help="Output matching_results.tsv path")
    args = parser.parse_args()
    n_entities, n_matches = decode_all(args.scores, args.candidates, args.out)
    print(f"Decoded {n_entities:,} entities, {n_matches:,} total matches -> {args.out}")
