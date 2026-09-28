#!/usr/bin/env python3
"""
Global one-S1-per-candidate resolution.

Every Source 2 / Source 3 id belongs to at most one Source 1 entity - checked
across the full training ground truth (7.6M matched ids, zero shared by two
S1 rows). pipeline.py scores each S1 entity's candidates independently, so it
can accept the same candidate id under more than one S1 (e.g. two businesses
at the same address, or several similarly-named chain outlets). Since F_0.5
is precision-heavy, an uncontested duplicate match is a guaranteed false
positive for every S1 it's attached to except at most one.

This pass keeps a candidate only under the S1 it scored highest against and
drops it from every other S1's match list, using the per-pair probabilities
pipeline.py writes to match_scores.tsv. Memory use is O(unique matched
candidate ids), not O(all candidate pairs) - safe within the ~400MB RAM
budget even at full test scale (a few million matched ids at most, well
under the singleton/non-singleton totals seen on the train ground truth).
"""

import argparse
import csv
from typing import Dict, Tuple


def resolve_winners(scores_path: str) -> Dict[str, Tuple[str, float]]:
    """cand_id -> (winning_s1_id, prob): the highest-probability S1 per candidate."""
    winners: Dict[str, Tuple[str, float]] = {}
    with open(scores_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 3:
                continue
            s1_id, cid, prob_s = row[0], row[1], row[2]
            try:
                prob = float(prob_s)
            except ValueError:
                continue
            prev = winners.get(cid)
            if prev is None or prob > prev[1]:
                winners[cid] = (s1_id, prob)
    return winners


def dedupe(matching_in: str, scores_path: str, matching_out: str) -> Tuple[int, int]:
    """Rewrite matching_in -> matching_out, dropping candidates lost to a higher-scoring S1."""
    winners = resolve_winners(scores_path)

    total_before = 0
    total_after = 0
    with open(matching_in, "r", encoding="utf-8", newline="") as fin, \
         open(matching_out, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        next(reader, None)  # header
        fout.write("source1_entity_id\tmatched_entity_ids\n")
        for row in reader:
            if not row:
                continue
            s1_id = row[0]
            match_val = row[1] if len(row) > 1 else ""
            cids = [c for c in match_val.split(",") if c]
            total_before += len(cids)
            # Default owner is this row's own s1_id so a candidate with no
            # recorded score (shouldn't happen, but stay safe) is kept.
            kept = [c for c in cids if winners.get(c, (s1_id, 0.0))[0] == s1_id]
            total_after += len(kept)
            fout.write(f"{s1_id}\t{','.join(kept)}\n")

    return total_before, total_after


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True, help="matching_results.tsv from pipeline.py")
    parser.add_argument("--scores", required=True, help="match_scores.tsv from pipeline.py")
    parser.add_argument("--out", required=True, help="Output path for the deduplicated matching_results.tsv")
    args = parser.parse_args()

    before, after = dedupe(args.matching, args.scores, args.out)
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    print(f"Matches before: {before:,}  after: {after:,}  removed as duplicate-claimed: {removed:,} ({pct:.2f}%)")


if __name__ == "__main__":
    main()
