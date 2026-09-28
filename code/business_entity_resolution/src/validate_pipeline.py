#!/usr/bin/env python3
"""
End-to-end validation harness: runs the *actual* production path (DuckDB
blocking -> LightGBM+TF cascade -> dedupe, via pipeline_duckdb.run_pipeline_duckdb)
over the held-out validation S1 entities, then scores the result against
val_ground_truth with the same macro F_0.5 metric the leaderboard uses.

Exists because no other script in this repo does this: train_screener.py and
train_tf_specialist.py each report their own macro F_0.5, but neither runs
dedupe, and both evaluate a small subsample instead of the full val split.
Use this whenever you need a number that's actually comparable to the
leaderboard score, e.g. before/after a pipeline change.

Usage:
    python3 -m src.validate_pipeline --build-index    # first run: builds
                                                        # data_split/train_candidates_index.duckdb
                                                        # from the full train S2/S3 pool (~20-25 min)
    python3 -m src.validate_pipeline                  # reuses that index
    python3 -m src.validate_pipeline --mini            # fast smoke test on
                                                         # mini_val (50K entities)
                                                         # instead of full val (441K)
"""

import argparse
import csv
import os

from .config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    SPLIT_DIR,
    VAL_S1_IDS_PATH,
    MINI_VAL_S1_IDS_PATH,
    VAL_GROUND_TRUTH_PATH,
    MINI_VAL_GROUND_TRUTH_PATH,
    TRAIN_CANDIDATE_INDEX_DUCKDB_PATH,
)
from .pipeline_duckdb import run_pipeline_duckdb

TRAIN_DUCKDB_INDEX_PATH = TRAIN_CANDIDATE_INDEX_DUCKDB_PATH


def _write_filtered_s1_csv(src_path: str, ids_path: str, out_path: str) -> int:
    """Filter TRAIN_SOURCE1_PATH down to just the given S1 ids, preserving the
    original header/column layout so stream_s1_records() reads it unchanged."""
    with open(ids_path, "r", encoding="utf-8") as f:
        wanted = {line.strip() for line in f if line.strip()}

    n = 0
    with open(src_path, "r", encoding="utf-8", errors="replace") as f_in, \
         open(out_path, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)
        header = next(reader)
        writer.writerow(header)
        id_idx = 1 if len(header) > 2 and header[1] == "entity_id" else 0
        for row in reader:
            if len(row) > id_idx and row[id_idx].strip() in wanted:
                writer.writerow(row)
                n += 1
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75, help="Decision threshold (only used if --decode-mode threshold in later phases)")
    parser.add_argument("--mini", action="store_true", help="Use the 50K mini-val split instead of the full 441K val split (fast smoke test)")
    parser.add_argument("--build-index", action="store_true", help="(Re)build the train DuckDB candidate index before scoring")
    parser.add_argument("--no-translit", action="store_true", help="Disable Indic-script transliteration when building the index (use for a pre-Part-1 baseline)")
    parser.add_argument("--decode-mode", choices=["expected_f05", "threshold"], default="expected_f05", help="expected_f05: per-entity top-k decoding (default). threshold: old global-cutoff behavior (use for a pre-Part-2 baseline).")
    parser.add_argument("--chunk-size", type=int, default=4000)
    parser.add_argument("--duckdb-memory-limit", type=str, default="3GB")
    parser.add_argument("--build-memory-limit", type=str, default="3GB")
    parser.add_argument("--out-dir", type=str, default=os.path.join(SPLIT_DIR, "val_pipeline_output"))
    args = parser.parse_args()

    ids_path = MINI_VAL_S1_IDS_PATH if args.mini else VAL_S1_IDS_PATH
    gt_path = MINI_VAL_GROUND_TRUTH_PATH if args.mini else VAL_GROUND_TRUTH_PATH

    os.makedirs(args.out_dir, exist_ok=True)
    filtered_s1_path = os.path.join(args.out_dir, "val_source1.csv")
    n = _write_filtered_s1_csv(TRAIN_SOURCE1_PATH, ids_path, filtered_s1_path)
    print(f"Filtered {n:,} val S1 entities from {TRAIN_SOURCE1_PATH} -> {filtered_s1_path}")

    run_pipeline_duckdb(
        s1_path=filtered_s1_path,
        s2_path=TRAIN_SOURCE2_PATH,
        s3_path=TRAIN_SOURCE3_PATH,
        output_dir=args.out_dir,
        decision_threshold=args.threshold,
        db_path=TRAIN_DUCKDB_INDEX_PATH,
        rebuild_index=args.build_index,
        chunk_size=args.chunk_size,
        duckdb_memory_limit=args.duckdb_memory_limit,
        build_memory_limit=args.build_memory_limit,
        ground_truth_path=gt_path,
        use_translit=not args.no_translit,
        decode_mode=args.decode_mode,
    )


if __name__ == "__main__":
    main()
