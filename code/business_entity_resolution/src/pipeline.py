#!/usr/bin/env python3
"""
End-to-End Cascaded Test Inference Pipeline.

Uses DiskBlockingEngine (SQLite on disk) to keep RAM usage < 400 MB.
Ensembles LightGBM + TensorFlow Deep Residual Matcher via Cascaded Routing:
- High confidence LightGBM matches (P >= 0.88) -> Auto-Accept
- Obvious LightGBM non-matches (P < 0.20) -> Auto-Reject
- Ambiguous middle (0.20 <= P < 0.88) -> Evaluated by TensorFlow Neural Matcher

Produces output/matching_results.tsv and output/candidate_pairs.tsv.
Runs student_resource/utils/validate_submission.py to verify compliance.
"""

import argparse
import csv
import gc
import os
import pickle
import subprocess
import sys
import time
from typing import Dict, List, Set

import numpy as np

# Ensure GPU is used by TensorFlow
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
    SUBMISSION_MATCHING_PATH,
    SUBMISSION_CANDIDATE_PATH,
    MODELS_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    MAX_CANDIDATES_PER_S1,
)
from .sqlite_blocking import DiskBlockingEngine
from .features import extract_pair_features, FEATURE_NAMES


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


def run_pipeline(
    s1_path: str = TEST_SOURCE1_PATH,
    s2_path: str = TEST_SOURCE2_PATH,
    s3_path: str = TEST_SOURCE3_PATH,
    output_dir: str = OUTPUT_DIR,
    decision_threshold: float = 0.88,
    lgbm_weight: float = 0.70,
    test_dir_for_validator: str = "student_resource/dataset/test",
    max_entities: int = None,
    rebuild_index: bool = False,
):
    print("=" * 65)
    print("  BUSINESS ENTITY RESOLUTION — FULL CASCADED INFERENCE PIPELINE")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    db_path = os.path.join(output_dir, "candidates_index.db")

    # 1. Load Trained Models
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

    # 2. Build or Open SQLite Candidate Index
    engine = DiskBlockingEngine(db_path=db_path, max_candidates=MAX_CANDIDATES_PER_S1)
    if rebuild_index or not os.path.exists(db_path):
        print("\nBuilding disk-based candidate index (low RAM mode)...")
        engine.build_index(s2_path, s3_path, reset=True)
    else:
        print(f"\nUsing existing candidate index: {db_path}")
        engine.open()

    # 3. Stream S1 and Generate Predictions
    print("\nStreaming S1 records and predicting entity matches...")
    f_match = open(matching_tsv, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_tsv, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    total_s1_processed = 0
    total_matches_written = 0
    start_time = time.time()

    for chunk in stream_s1_records(s1_path, chunk_size=8000):
        if max_entities and total_s1_processed >= max_entities:
            break

        chunk_pairs = []
        chunk_s1_ids = []
        chunk_cand_strings = []

        # Retrieve candidates for chunk
        for s1 in chunk:
            s1_id = s1["entity_id"]
            candidates = engine.retrieve_candidates(s1)
            cand_ids = [cid for cid, _, _ in candidates]

            chunk_s1_ids.append(s1_id)
            chunk_cand_strings.append(",".join(cand_ids))

            for cid, cname, caddr in candidates:
                chunk_pairs.append({
                    "s1_id": s1_id,
                    "cand_id": cid,
                    "s1_name": s1["business_name"],
                    "s1_addr": s1["business_address"],
                    "cand_name": cname,
                    "cand_addr": caddr,
                })

        # Write candidate_pairs.tsv rows
        for s1_id, cstr in zip(chunk_s1_ids, chunk_cand_strings):
            f_cand.write(f"{s1_id}\t{cstr}\n")

        # Score pairs
        s1_to_matches = {s1_id: [] for s1_id in chunk_s1_ids}
        if chunk_pairs:
            X_rows = []
            for p in chunk_pairs:
                feats = extract_pair_features(
                    p["s1_name"], p["s1_addr"],
                    p["cand_name"], p["cand_addr"],
                    p["cand_id"],
                )
                X_rows.append([feats[k] for k in FEATURE_NAMES])

            X_chunk = np.array(X_rows, dtype=np.float32)
            probs_lgbm = lgbm_model.predict(X_chunk)

            # Cascaded Routing:
            # Only evaluate TensorFlow model on the ambiguous middle zone
            final_probs = probs_lgbm.copy()
            if tf_model is not None:
                amb_mask = (probs_lgbm >= 0.20) & (probs_lgbm < 0.88)
                if np.any(amb_mask):
                    probs_tf = tf_model.predict(X_chunk[amb_mask], batch_size=1024, verbose=0).flatten()
                    final_probs[amb_mask] = lgbm_weight * probs_lgbm[amb_mask] + (1.0 - lgbm_weight) * probs_tf

            for idx, p in enumerate(chunk_pairs):
                if final_probs[idx] >= decision_threshold:
                    s1_to_matches[p["s1_id"]].append(p["cand_id"])

        # Write matching_results.tsv rows
        for s1_id in chunk_s1_ids:
            matches = list(dict.fromkeys(s1_to_matches[s1_id]))
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            total_matches_written += len(matches)

        total_s1_processed += len(chunk)
        elapsed = time.time() - start_time
        speed = total_s1_processed / elapsed if elapsed > 0 else 0
        pct = (total_s1_processed / 1732544) * 100
        print(f"[{time.strftime('%X')}] Progress: {total_s1_processed:,} / 1,732,544 ({pct:.1f}%) | Speed: {speed:.0f} ent/s | Matches: {total_matches_written:,}", flush=True)


    f_match.close()
    f_cand.close()

    elapsed = time.time() - start_time
    print(f"\n\nPipeline execution finished in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f"  - Total S1 Entities:     {total_s1_processed}")
    print(f"  - Total Matches Found:   {total_matches_written}")
    print(f"  - matching_results.tsv:  {matching_tsv}")
    print(f"  - candidate_pairs.tsv:   {candidate_tsv}")

    # 4. Run Official Challenge Validator Script
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
    parser.add_argument("--threshold", type=float, default=0.88, help="Decision threshold")
    parser.add_argument("--max-entities", type=int, default=None, help="Debug max entities")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild DB index even if exists")
    args = parser.parse_args()

    run_pipeline(
        decision_threshold=args.threshold,
        max_entities=args.max_entities,
        rebuild_index=args.rebuild_index,
    )
