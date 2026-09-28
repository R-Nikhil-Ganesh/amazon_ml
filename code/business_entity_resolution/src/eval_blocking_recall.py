#!/usr/bin/env python3
"""
Blocking-recall diagnostic: measures pair recall of DiskBlockingEngine against
a random sample of validation-split S1 entities with known ground truth, and
breaks down why any missed pair was missed (capped key / ranked out of the
top-N / no shared key at all).

Not part of the shipped pipeline - a measurement tool, run manually after any
blocking change to compare against the baseline recorded in the plan/docs.
"""

import argparse
import collections
import csv
import random
import time

from rapidfuzz import fuzz

from .config import TRAIN_SOURCE1_PATH, VAL_GROUND_TRUTH_PATH
from .normalize import normalize_business_name, normalize_address, extract_numbers
from .sqlite_blocking import DiskBlockingEngine, build_blocking_keys, _address_token_candidates


def _df_lookup_for(conn, country, norm_addr):
    """Fetch real addr_df rarity values for one record's address tokens -
    mirrors what DiskBlockingEngine.retrieve_candidates() does internally,
    so this diagnostic's shared_keys_all_capped/no_shared_key split reflects
    the actual keys production code would generate, not an alphabetical
    fallback."""
    tokens = _address_token_candidates(norm_addr)
    if not tokens:
        return {}
    placeholders = ",".join("?" for _ in tokens)
    rows = conn.execute(
        f"SELECT token, df FROM addr_df WHERE country = ? AND token IN ({placeholders})",
        [country, *tokens],
    ).fetchall()
    return {(country, token): df for token, df in rows}


def load_sample(n: int, seed: int):
    gt = {}
    with open(VAL_GROUND_TRUTH_PATH) as f:
        r = csv.reader(f)
        next(r)
        for s1, m in r:
            gt[s1] = [x for x in m.split(",") if x] if m else []
    pos_ids = [s for s, v in gt.items() if v]
    random.seed(seed)
    sample_ids = set(random.sample(pos_ids, min(n, len(pos_ids))))

    s1recs = {}
    with open(TRAIN_SOURCE1_PATH) as f:
        for row in csv.DictReader(f):
            if row["entity_id"] in sample_ids:
                s1recs[row["entity_id"]] = row
    return gt, s1recs


def evaluate(db_path: str, n: int, seed: int, max_candidates: int):
    gt, s1recs = load_sample(n, seed)

    engine = DiskBlockingEngine(db_path, max_candidates=max_candidates)
    engine.open()
    c = engine.conn

    cats = collections.Counter()
    by_country = collections.Counter()
    miss_country = collections.Counter()
    pool_sizes = []
    total = 0
    t0 = time.time()

    for s1, rec in s1recs.items():
        # Untruncated pool (for the ranked-out-vs-no-key distinction).
        engine.max_candidates = 10 ** 6
        full_pool = engine.retrieve_candidates(rec)
        engine.max_candidates = max_candidates
        pool_sizes.append(len(full_pool))
        full_ids = {p[0] for p in full_pool}
        top_ids = {p[0] for p in full_pool[:max_candidates]}

        norm_name = normalize_business_name(rec["business_name"])
        norm_addr = normalize_address(rec["business_address"])
        country = rec["country"]
        q_df = _df_lookup_for(c, country, norm_addr)
        q_keys = set(build_blocking_keys(norm_name, norm_addr, country, q_df))

        for cid in gt[s1]:
            total += 1
            by_country[rec["country"]] += 1
            if cid in top_ids:
                cats["HIT"] += 1
                continue
            miss_country[rec["country"]] += 1
            row = c.execute(
                "SELECT norm_name, norm_addr, country FROM candidates WHERE cid = ?",
                (cid,),
            ).fetchone()
            if row is None:
                cats["cand_not_in_index"] += 1
                continue
            c_norm_name, c_norm_addr, c_country = row
            if c_country != rec["country"]:
                cats["country_mismatch"] += 1
                continue
            if cid in full_ids:
                cats["retrieved_but_ranked_out"] += 1
                continue
            c_df = _df_lookup_for(c, c_country, c_norm_addr)
            c_keys = set(build_blocking_keys(c_norm_name, c_norm_addr, c_country, c_df))
            if q_keys & c_keys:
                cats["shared_keys_all_capped"] += 1
            else:
                cats["no_shared_key"] += 1

    dt = time.time() - t0
    print(f"Evaluated {len(s1recs)} S1 entities / {total} true pairs in {dt:.0f}s")
    for k, v in cats.most_common():
        print(f"  {k:28s} {v:6d} {v / total * 100:5.1f}%")
    print("recall by country:")
    for k in by_country:
        recall = (by_country[k] - miss_country[k]) / by_country[k] * 100
        print(f"  {k}: {recall:.1f}% of {by_country[k]}")
    ps = sorted(pool_sizes)
    if ps:
        print(
            "pool size p50/p90/p99/max:",
            ps[len(ps) // 2], ps[int(len(ps) * 0.9)], ps[int(len(ps) * 0.99)], ps[-1],
        )
    hits = cats["HIT"]
    print(f"\nOverall pair recall: {hits / total * 100:.1f}% ({hits}/{total})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="Path to a candidates_index.db")
    parser.add_argument("--n", type=int, default=2000, help="Number of positive S1 entities to sample")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-candidates", type=int, default=None, help="Override MAX_CANDIDATES_PER_S1")
    args = parser.parse_args()

    from .config import MAX_CANDIDATES_PER_S1
    max_candidates = args.max_candidates or MAX_CANDIDATES_PER_S1
    evaluate(args.index, args.n, args.seed, max_candidates)
