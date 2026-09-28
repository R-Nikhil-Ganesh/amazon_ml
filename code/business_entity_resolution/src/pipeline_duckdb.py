#!/usr/bin/env python3
"""
Single-Process DuckDB Cascaded Test Inference Pipeline.

Same cascade (LightGBM screener -> TensorFlow specialist on the ambiguous
middle -> threshold decision) and same output contract as pipeline.py.
Only the blocking engine changes: DiskBlockingEngine (sqlite_blocking.py)
issues 3+ SQLite round trips per S1 entity; DuckDBBlockingEngine
(duckdb_blocking.py) does the same 3 lookup passes as one batched,
vectorized JOIN per *chunk* of S1 entities.

Deliberately single-process (no multiprocessing/sharding, unlike
pipeline_parallel.py) - measured on this box (7.4GB RAM, index ~5.6GB) that
splitting into multiple worker processes made things *worse*: each worker's
own DuckDB connection + TensorFlow runtime competes for a page cache that's
already too small to hold the whole index, so 2 workers got ~50 ent/s each
(100 ent/s combined) instead of scaling up. One process gets the whole
machine's page cache to itself instead of fragmenting it across workers.

Produces the same output/matching_results.tsv and output/candidate_pairs.tsv
as pipeline.py, via the same global one-S1-per-candidate dedupe pass and the
same official validator script.
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
import time

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

from .config import (
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from .duckdb_blocking import DuckDBBlockingEngine
from .features import extract_pair_features, add_group_relative_features, FEATURE_NAMES
from .decode import decode_all
from .dedupe_matches import dedupe
from .evaluate import evaluate_predictions, load_mapping_file
from .normalize import normalize_business_name, normalize_address

TOTAL_TEST_ENTITIES = 1732544


def stream_s1_records(s1_path: str, chunk_size: int = 10000):
    """Yield chunks of S1 records."""
    chunk = []
    with open(s1_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx = 1 if len(header) > 2 and header[1] == "entity_id" else 0
        name_idx = id_idx + 1
        addr_idx = id_idx + 2
        cntry_idx = id_idx + 3

        for row in reader:
            if not row or len(row) <= cntry_idx:
                continue
            chunk.append({
                "entity_id": row[id_idx].strip(),
                "business_name": row[name_idx].strip(),
                "business_address": row[addr_idx].strip(),
                "country": row[cntry_idx].strip(),
            })
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def run_pipeline_duckdb(
    s1_path: str = TEST_SOURCE1_PATH,
    s2_path: str = TEST_SOURCE2_PATH,
    s3_path: str = TEST_SOURCE3_PATH,
    output_dir: str = OUTPUT_DIR,
    decision_threshold: float = 0.75,
    lgbm_weight: float = 0.70,
    test_dir_for_validator: str = "student_resource/dataset/test",
    max_entities: int = None,
    rebuild_index: bool = False,
    chunk_size: int = 4000,
    duckdb_memory_limit: str = "3GB",
    build_memory_limit: str = "3GB",
    db_path: str = None,
    ground_truth_path: str = None,
    use_translit: bool = True,
    decode_mode: str = "expected_f05",
):
    """
    ground_truth_path: when given (e.g. a val_ground_truth file), this is a
    validation run rather than a real test run - score the deduped output
    against it with evaluate.py's macro F0.5 (broken down by country, since
    S1 country is already on hand from stream_s1_records) instead of running
    the official test-set submission validator, which assumes every test S1
    entity is present and would reject a val-only subset.
    """
    print("=" * 65)
    print("  BUSINESS ENTITY RESOLUTION — SINGLE-PROCESS DUCKDB INFERENCE PIPELINE")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    scores_tsv = os.path.join(output_dir, "match_scores.tsv")
    if db_path is None:
        db_path = os.path.join(output_dir, "candidates_index.duckdb")

    print("Loading Stage 2 LightGBM model...")
    lgbm_path = os.path.join(MODELS_DIR, "screener_lgbm.pkl")
    with open(lgbm_path, "rb") as f:
        lgbm_model = pickle.load(f)

    tf_model = None
    if HAS_TF:
        tf_path = os.path.join(MODELS_DIR, "tf_specialist_model.keras")
        if os.path.exists(tf_path):
            print("Loading Stage 3 TensorFlow Specialist model...")
            tf_model = keras.models.load_model(tf_path)

    engine = DuckDBBlockingEngine(db_path=db_path)
    if rebuild_index or not os.path.exists(db_path):
        print(f"\nBuilding DuckDB candidate index at {db_path}... (transliteration {'ON' if use_translit else 'OFF'})")
        engine.build_index(
            s2_path, s3_path, reset=True, memory_limit=build_memory_limit,
            translit_dict=None if use_translit else {},
        )
    else:
        print(f"\nUsing existing DuckDB index: {db_path}")
        engine.open(read_only=True, memory_limit=duckdb_memory_limit)

    print("\nStreaming S1 records and predicting entity matches...")
    f_match = open(matching_tsv, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_tsv, "w", encoding="utf-8", newline="")
    f_scores = open(scores_tsv, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_scores.write("source1_entity_id\tcand_entity_id\tprob\n")

    total_s1_processed = 0
    total_matches_written = 0
    start_time = time.time()
    s1_country = {} if ground_truth_path else None

    for chunk in stream_s1_records(s1_path, chunk_size=chunk_size):
        if max_entities and total_s1_processed >= max_entities:
            break

        cand_by_s1 = engine.retrieve_candidates_batch(chunk)

        chunk_pairs = []
        chunk_s1_ids = []
        chunk_cand_strings = []

        for s1 in chunk:
            s1_id = s1["entity_id"]
            if s1_country is not None:
                s1_country[s1_id] = s1["country"]
            candidates = cand_by_s1.get(s1_id, [])
            cand_ids = [cid for cid, _, _, _, _ in candidates]

            chunk_s1_ids.append(s1_id)
            chunk_cand_strings.append(",".join(cand_ids))

            s1_norm_name = normalize_business_name(s1["business_name"])
            s1_norm_addr = normalize_address(s1["business_address"])

            for cid, cname, caddr, cnorm_name, cnorm_addr in candidates:
                chunk_pairs.append({
                    "s1_id": s1_id,
                    "cand_id": cid,
                    "s1_name": s1["business_name"],
                    "s1_addr": s1["business_address"],
                    "cand_name": cname,
                    "cand_addr": caddr,
                    "s1_norm_name": s1_norm_name,
                    "s1_norm_addr": s1_norm_addr,
                    "cand_norm_name": cnorm_name,
                    "cand_norm_addr": cnorm_addr,
                })

        for s1_id, cstr in zip(chunk_s1_ids, chunk_cand_strings):
            f_cand.write(f"{s1_id}\t{cstr}\n")

        s1_to_matches = {s1_id: [] for s1_id in chunk_s1_ids}
        if chunk_pairs:
            feats_list = [
                extract_pair_features(
                    p["s1_name"], p["s1_addr"],
                    p["cand_name"], p["cand_addr"],
                    p["cand_id"],
                    s1_norm_name=p["s1_norm_name"],
                    s1_norm_addr=p["s1_norm_addr"],
                    cand_norm_name=p["cand_norm_name"],
                    cand_norm_addr=p["cand_norm_addr"],
                )
                for p in chunk_pairs
            ]

            # chunk_pairs is already grouped by s1 (built s1-by-s1 above), so
            # group consecutive feats dicts by s1_id and add relative
            # features per group before flattening to X_chunk.
            groups = {}
            for feats, p in zip(feats_list, chunk_pairs):
                groups.setdefault(p["s1_id"], []).append(feats)
            for group_feats in groups.values():
                add_group_relative_features(group_feats)

            X_rows = [[feats[k] for k in FEATURE_NAMES] for feats in feats_list]
            X_chunk = np.array(X_rows, dtype=np.float32)
            probs_lgbm = lgbm_model.predict(X_chunk)

            final_probs = probs_lgbm.copy()
            if tf_model is not None:
                amb_mask = (probs_lgbm >= 0.20) & (probs_lgbm < 0.88)
                if np.any(amb_mask):
                    probs_tf = tf_model.predict(X_chunk[amb_mask], batch_size=1024, verbose=0).flatten()
                    final_probs[amb_mask] = lgbm_weight * probs_lgbm[amb_mask] + (1.0 - lgbm_weight) * probs_tf

            for idx, p in enumerate(chunk_pairs):
                # Every retrieved candidate's probability is kept (not just
                # ones over decision_threshold) so match_scores.tsv is a
                # complete per-candidate probability table - decode.py's
                # per-entity expected-F0.5 decoding needs every candidate's
                # score, not just the ones a fixed threshold would accept.
                s1_to_matches[p["s1_id"]].append((p["cand_id"], float(final_probs[idx])))

        for s1_id in chunk_s1_ids:
            seen = set()
            matches_over_threshold = []
            for cid, prob in s1_to_matches[s1_id]:
                if cid in seen:
                    continue
                seen.add(cid)
                f_scores.write(f"{s1_id}\t{cid}\t{prob:.6f}\n")
                if prob >= decision_threshold:
                    matches_over_threshold.append(cid)
            if decode_mode == "threshold":
                f_match.write(f"{s1_id}\t{','.join(matches_over_threshold)}\n")
            # decode_mode == "expected_f05": matching_results.tsv is left as
            # header-only here and fully (re)written by decode_all() below,
            # after every candidate's probability has been persisted.
            total_matches_written += len(matches_over_threshold)

        total_s1_processed += len(chunk)
        elapsed = time.time() - start_time
        speed = total_s1_processed / elapsed if elapsed > 0 else 0
        pct = (total_s1_processed / TOTAL_TEST_ENTITIES) * 100
        print(f"[{time.strftime('%X')}] Progress: {total_s1_processed:,} / {TOTAL_TEST_ENTITIES:,} ({pct:.1f}%) | Speed: {speed:.0f} ent/s | Matches: {total_matches_written:,}", flush=True)

    f_match.close()
    f_cand.close()
    f_scores.close()
    engine.close()

    elapsed = time.time() - start_time
    print(f"\n\nPipeline execution finished in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f"  - Total S1 Entities:     {total_s1_processed}")
    print(f"  - Total Matches Found:   {total_matches_written}")
    print(f"  - matching_results.tsv:  {matching_tsv}")
    print(f"  - candidate_pairs.tsv:   {candidate_tsv}")

    if decode_mode == "expected_f05":
        print("\n" + "=" * 50)
        print("    DECODING MATCHES (per-entity expected F_0.5, top-k)")
        print("=" * 50)
        n_entities, n_matches = decode_all(scores_tsv, candidate_tsv, matching_tsv)
        print(f"Decoded {n_entities:,} entities, {n_matches:,} matches")

    print("\n" + "=" * 50)
    print("    RESOLVING CROSS-ENTITY DUPLICATE MATCHES")
    print("=" * 50)
    dedup_tmp = matching_tsv + ".dedup_tmp"
    before, after = dedupe(matching_tsv, scores_tsv, dedup_tmp)
    os.replace(dedup_tmp, matching_tsv)
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    print(f"Matches before: {before:,}  after: {after:,}  removed as duplicate-claimed: {removed:,} ({pct:.2f}%)")

    if ground_truth_path:
        print("\n" + "=" * 50)
        print("    SCORING AGAINST GROUND TRUTH (macro F_0.5)")
        print("=" * 50)
        truth = load_mapping_file(ground_truth_path)
        preds = load_mapping_file(matching_tsv)
        macro_f05, stats = evaluate_predictions(truth, preds)
        print(f"Macro F_0.5 Score:      {macro_f05:.4f}")
        print(f"Total S1 Entities:      {stats['total_entities']}")
        print(f"Singleton Count:        {stats['singleton_count']} (Accuracy: {stats['singleton_accuracy']:.2%})")
        print(f"Non-Singleton Count:    {stats['non_singleton_count']} (F_0.5: {stats['non_singleton_f05']:.4f})")

        by_country = {}
        for s1_id, true_set in truth.items():
            country = s1_country.get(s1_id, "UNKNOWN")
            by_country.setdefault(country, {})[s1_id] = true_set
        print("\nBreakdown by country:")
        for country, truth_subset in sorted(by_country.items()):
            c_f05, c_stats = evaluate_predictions(truth_subset, preds)
            print(f"  {country:10s}  n={c_stats['total_entities']:>7,}  macro F_0.5={c_f05:.4f}  singleton_acc={c_stats['singleton_accuracy']:.2%}")
    else:
        validator_script = os.path.join(PROJECT_ROOT, "student_resource/utils/validate_submission.py")
        if os.path.exists(validator_script):
            print("\n" + "=" * 50)
            print("    RUNNING OFFICIAL SUBMISSION VALIDATOR")
            print("=" * 50)
            cmd = [
                sys.executable,
                validator_script,
                "--matching", matching_tsv,
                "--candidate", candidate_tsv,
                "--test-dir", test_dir_for_validator,
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.75, help="Decision threshold")
    parser.add_argument("--max-entities", type=int, default=None, help="Debug max entities")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild DB index even if exists")
    parser.add_argument("--chunk-size", type=int, default=4000, help="S1 records per batched-retrieval call")
    parser.add_argument("--duckdb-memory-limit", type=str, default="3GB", help="DuckDB query memory limit (single process, so can be generous)")
    parser.add_argument("--build-memory-limit", type=str, default="3GB", help="DuckDB memory limit during index build")
    parser.add_argument("--no-translit", action="store_true", help="Disable Indic-script transliteration during index build (only matters with --rebuild-index)")
    parser.add_argument("--decode-mode", choices=["expected_f05", "threshold"], default="expected_f05", help="expected_f05: per-entity top-k decoding (default). threshold: old global-cutoff behavior.")
    args = parser.parse_args()

    run_pipeline_duckdb(
        decision_threshold=args.threshold,
        max_entities=args.max_entities,
        rebuild_index=args.rebuild_index,
        chunk_size=args.chunk_size,
        duckdb_memory_limit=args.duckdb_memory_limit,
        build_memory_limit=args.build_memory_limit,
        use_translit=not args.no_translit,
        decode_mode=args.decode_mode,
    )
