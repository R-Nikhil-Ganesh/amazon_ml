#!/usr/bin/env python3
"""
Random-search LightGBM hyperparameter tuning for the Stage 2 screener, plus
a post-hoc name-similarity floor sweep folded into the same search.

Why combined: the France hand-labeling check this session (see
documents/engineering_log.md) found a recurring false-positive pattern -
candidates with near-zero name similarity getting accepted purely on a
strong address/number-key match (e.g. "UG Club" matched to "epicuriens
unissons club participations sas" at the same street number). A
`name_sim_floor` gate (force probability to 0 when name_token_set is below
the floor, independent of what the model itself predicts) directly targets
that pattern, the same way decode.py's MIN_TOP_PROB_FOR_ANY_MATCH targets a
different failure mode. Sweeping it alongside LGBM_PARAMS in one search
means every trial's best floor is chosen for that trial's actual model,
not a floor tuned in isolation against a fixed model.

Feature extraction (build_pair_dataset - the expensive part) does not
depend on LGBM_PARAMS or the floor at all, so it's done ONCE here and
reused across every trial, unlike naively re-running train_screener.py N
times.
"""

import argparse
import csv
import json
import os
import random
import time

import lightgbm as lgb
import numpy as np

from .config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    MINI_VAL_GROUND_TRUTH_PATH,
    MINI_VAL_S1_IDS_PATH,
    SPLIT_DIR,
    LGBM_PARAMS,
    MAX_CANDIDATES_PER_S1,
    TRAIN_CANDIDATE_INDEX_DUCKDB_PATH,
)
from .blocking import read_entity_csv
from .duckdb_blocking import DuckDBBlockingEngine
from .features import FEATURE_NAMES
from .evaluate import evaluate_predictions, load_mapping_file
from .train_screener import build_pair_dataset

NAME_SIM_FEATURE_IDX = FEATURE_NAMES.index("name_token_set")

SEARCH_SPACE = {
    "num_leaves": [31, 63, 95, 127],
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_depth": [6, 8, 10, -1],
    "feature_fraction": [0.7, 0.8, 0.85, 0.9, 1.0],
    "bagging_fraction": [0.7, 0.8, 0.85, 0.9, 1.0],
    "min_child_samples": [10, 20, 30, 50],
}

# 0.0 = off (current behavior). Values above are the fraction of shared
# name tokens (name_token_set, a rapidfuzz token_set_ratio/100) below which
# a candidate is forced to non-match regardless of the model's probability.
NAME_SIM_FLOORS = [0.0, 0.15, 0.25, 0.35]


def sample_params(rng: random.Random) -> dict:
    params = dict(LGBM_PARAMS)
    for key, choices in SEARCH_SPACE.items():
        params[key] = rng.choice(choices)
    # dtrain/dval are built once and reused across every trial (that's the
    # whole point - feature extraction is the expensive part, not training).
    # LightGBM's Dataset pre-filters features by min_data_in_leaf on first
    # use and then locks that in; since min_child_samples varies per trial,
    # pre-filtering must be disabled or every trial after the first errors.
    params["feature_pre_filter"] = False
    return params


def sweep_thresholds(probs: np.ndarray, val_meta, val_s1_records, eval_truth):
    best_thresh, best_f05, best_stats = 0.50, 0.0, None
    for thresh in np.arange(0.40, 0.95, 0.05):
        pred_map = {r["entity_id"]: set() for r in val_s1_records}
        for i, p in enumerate(probs):
            if p >= thresh:
                pred_map[val_meta[i]["s1_id"]].add(val_meta[i]["cand_id"])
        macro_f05, stats = evaluate_predictions(eval_truth, pred_map)
        if macro_f05 > best_f05:
            best_f05, best_thresh, best_stats = macro_f05, thresh, stats
    return best_thresh, best_f05, best_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--train-sample-size", type=int, default=300000)
    parser.add_argument("--val-sample-size", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  RANDOM-SEARCH LGBM TUNING + NAME-SIMILARITY FLOOR SWEEP")
    print("=" * 60)

    val_ids = set()
    with open(MINI_VAL_S1_IDS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < args.val_sample_size:
                val_ids.add(line.strip())

    train_ids = set()
    with open(TRAIN_GROUND_TRUTH_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        s1_col = 1 if len(header) > 2 and header[1] == "source1_entity_id" else 0
        if "source1_entity_id" in header:
            s1_col = header.index("source1_entity_id")
        for row in reader:
            if not row or len(row) <= s1_col:
                continue
            sid = row[s1_col].strip()
            if sid not in val_ids:
                train_ids.add(sid)
                if len(train_ids) >= args.train_sample_size:
                    break

    print(f"Loaded {len(train_ids)} train S1 IDs and {len(val_ids)} val S1 IDs.")

    train_gt = load_mapping_file(TRAIN_GROUND_TRUTH_PATH)
    val_gt = load_mapping_file(MINI_VAL_GROUND_TRUTH_PATH)

    all_s1_records = read_entity_csv(TRAIN_SOURCE1_PATH, allowed_ids=train_ids | val_ids)
    train_s1_records = [r for r in all_s1_records if r["entity_id"] in train_ids]
    val_s1_records = [r for r in all_s1_records if r["entity_id"] in val_ids]

    engine = DuckDBBlockingEngine(db_path=TRAIN_CANDIDATE_INDEX_DUCKDB_PATH, max_candidates=MAX_CANDIDATES_PER_S1)
    print(f"Opening train candidate index: {TRAIN_CANDIDATE_INDEX_DUCKDB_PATH}")
    engine.open(read_only=True)

    print("Building training feature matrix (once, reused across all trials)...")
    t0 = time.time()
    X_train, y_train, train_meta, _ = build_pair_dataset(
        train_s1_records, engine, train_gt, max_neg_ratio=5, include_missed_positives=True,
    )
    print(f"Train: {X_train.shape[0]} pairs in {time.time()-t0:.1f}s")

    print("Building validation feature matrix (once, reused across all trials)...")
    t0 = time.time()
    X_val, y_val, val_meta, _ = build_pair_dataset(
        val_s1_records, engine, val_gt, max_neg_ratio=None, include_missed_positives=False,
    )
    print(f"Val: {X_val.shape[0]} pairs in {time.time()-t0:.1f}s")
    engine.close()

    eval_truth = {r["entity_id"]: val_gt.get(r["entity_id"], set()) for r in val_s1_records}
    name_sim = X_val[:, NAME_SIM_FEATURE_IDX]

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES, reference=dtrain)

    rng = random.Random(args.seed)
    results = []
    for trial in range(args.trials):
        params = sample_params(rng)
        t0 = time.time()
        model = lgb.train(
            params, dtrain, valid_sets=[dval],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        probs = model.predict(X_val)

        for floor in NAME_SIM_FLOORS:
            gated_probs = np.where(name_sim < floor, 0.0, probs) if floor > 0 else probs
            thresh, f05, stats = sweep_thresholds(gated_probs, val_meta, val_s1_records, eval_truth)
            results.append({
                "trial": trial, "params": params, "name_sim_floor": floor,
                "best_thresh": float(thresh), "macro_f05": float(f05),
                "singleton_acc": stats["singleton_accuracy"] if stats else None,
            })
            print(f"[trial {trial}] floor={floor:.2f} thresh={thresh:.2f} macro_f05={f05:.4f} "
                  f"singleton_acc={(stats['singleton_accuracy'] if stats else 0):.2%}")
        print(f"  trial {trial} done in {time.time()-t0:.1f}s  params={params}")

    results.sort(key=lambda r: -r["macro_f05"])
    print("\n" + "=" * 60)
    print("  TOP 5 RESULTS")
    print("=" * 60)
    for r in results[:5]:
        print(json.dumps(r, indent=2))

    out_path = os.path.join(SPLIT_DIR, "tune_screener_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll {len(results)} results saved to {out_path}")
    print(f"Best: macro_f05={results[0]['macro_f05']:.4f} (baseline: 0.8980) "
          f"floor={results[0]['name_sim_floor']} params={results[0]['params']}")


if __name__ == "__main__":
    main()
