#!/usr/bin/env python3
"""
End-to-End Cascaded Test Inference Pipeline.

Uses DiskBlockingEngine (SQLite on disk) to keep RAM usage < 400 MB.
Ensembles LightGBM + TensorFlow Deep Residual Matcher via Cascaded Routing:
- High confidence LightGBM matches (P >= decision_threshold) -> Auto-Accept
- Obvious LightGBM non-matches (P < 0.20) -> Auto-Reject
- Ambiguous middle (0.20 <= P < 0.88) -> Evaluated by TensorFlow Neural Matcher

NOTE: decision_threshold's default below is the best threshold measured on
the real LGBM+TF ensemble (train_screener.py's own "BEST THRESHOLD" sweep
only covers LGBM alone, and can disagree with the ensemble's). Re-swept
after the blocking-recall improvements (rerank, rarity-based address
anchors, name-token pairs, cross-script skeleton key/feature): best moved
from 0.70 to 0.75 (Macro F0.5 0.8689 on the 50k-entity mini-val pool, up
from 0.8177 before that round). Re-check this any time blocking, features,
or either model change.

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
from .dedupe_matches import dedupe
from .normalize import normalize_business_name, normalize_address

# Full test set for reference in progress-percent printouts / smoke-test ETA math.
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


MIN_SAFE_ENT_PER_SEC = 100  # see run_smoke_test()


def run_smoke_test(
    engine: DiskBlockingEngine, s1_path: str, db_path: str,
    duration_s: float = 90.0, max_sample: int = 20000,
) -> float:
    """
    Time retrieve_candidates() on a real sample before committing to the
    full multi-hour streaming loop below, and abort loudly if steady-state
    throughput is far slower than expected.

    This exists because an oversized index (bigger than available RAM) only
    shows up as page-cache thrashing during actual queries - a run that
    measured a healthy index-build time still turned into 31-34 ent/s
    (a ~14-15hr full run) instead of the ~200-300 ent/s baseline, and that
    wasn't visible until 90 minutes in. retrieve_candidates() is the exact
    call the full loop makes, just on a sample instead of 1.7M records.

    Runs for up to `duration_s` seconds (or `max_sample` records, whichever
    comes first) and judges on the *second half* of that window, not the
    overall average. Right after build_index() finishes a heavy write burst
    (ingest + capping + VACUUM), the page cache is cold for reads and the
    first ~60-90s can measure far below steady state even on a perfectly
    healthy index - measured on a real 4.84 GB index: 103 ent/s at 60s,
    but climbing steadily past 490 ent/s by 10 minutes. Judging on an
    early-window average alone caused a false abort here; splitting the
    window and using only the later half's rate avoids that false negative
    while still catching a genuinely thrashing (non-recovering) index.
    """
    print(f"\nRunning retrieval throughput smoke-test (up to {duration_s:.0f}s / {max_sample} entities)...")
    sample = next(stream_s1_records(s1_path, chunk_size=max_sample), [])
    if not sample:
        print("Smoke-test: no S1 records to sample - skipping.")
        return float("inf")

    t0 = time.time()
    timestamps = []
    for s1 in sample:
        engine.retrieve_candidates(s1)
        timestamps.append(time.time())
        if timestamps[-1] - t0 >= duration_s:
            break

    n = len(timestamps)
    dt = timestamps[-1] - t0
    overall_rate = n / dt if dt > 0 else float("inf")

    half_idx = n // 2
    second_half_n = n - half_idx
    second_half_dt = timestamps[-1] - (timestamps[half_idx - 1] if half_idx > 0 else t0)
    steady_rate = second_half_n / second_half_dt if second_half_dt > 0 else overall_rate

    print(
        f"Smoke-test: {n} entities in {dt:.1f}s => {overall_rate:.0f} ent/s overall, "
        f"{steady_rate:.0f} ent/s steady-state (second half of the window)"
    )

    if steady_rate < MIN_SAFE_ENT_PER_SEC:
        eta_hours = TOTAL_TEST_ENTITIES / steady_rate / 3600
        db_size_gb = os.path.getsize(db_path) / 1e9 if os.path.exists(db_path) else float("nan")
        raise SystemExit(
            f"ABORTING: measured steady-state retrieval throughput ({steady_rate:.0f} "
            f"ent/s) is far below the ~200-300 ent/s baseline. At this rate the full "
            f"run over {TOTAL_TEST_ENTITIES:,} entities would take ~{eta_hours:.1f} "
            f"hours instead of ~100 minutes - almost certainly page-cache thrashing "
            f"because the index ({db_size_gb:.2f} GB at {db_path}) doesn't fit in "
            f"available RAM (see build_index()'s size warning above, if any). "
            f"Rebuild the index (fresh, with --rebuild-index) or free up RAM before "
            f"retrying. Pass --skip-smoke-test to override this check (not "
            f"recommended - you will get the multi-hour run this check exists to "
            f"prevent)."
        )
    return steady_rate


def run_pipeline(
    s1_path: str = TEST_SOURCE1_PATH,
    s2_path: str = TEST_SOURCE2_PATH,
    s3_path: str = TEST_SOURCE3_PATH,
    output_dir: str = OUTPUT_DIR,
    decision_threshold: float = 0.75,
    lgbm_weight: float = 0.70,
    test_dir_for_validator: str = "student_resource/dataset/test",
    max_entities: int = None,
    rebuild_index: bool = False,
    skip_smoke_test: bool = False,
):
    print("=" * 65)
    print("  BUSINESS ENTITY RESOLUTION — FULL CASCADED INFERENCE PIPELINE")
    print("=" * 65)
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")
    scores_tsv = os.path.join(output_dir, "match_scores.tsv")
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

    if not skip_smoke_test:
        run_smoke_test(engine, s1_path, db_path)

    # 3. Stream S1 and Generate Predictions
    print("\nStreaming S1 records and predicting entity matches...")
    f_match = open(matching_tsv, "w", encoding="utf-8", newline="")
    f_cand = open(candidate_tsv, "w", encoding="utf-8", newline="")
    # Per-match probability, consumed by dedupe_matches.py to resolve a
    # candidate that was accepted by more than one Source 1 entity (every
    # S2/S3 id has exactly one true owner - see dedupe_matches.py docstring).
    f_scores = open(scores_tsv, "w", encoding="utf-8", newline="")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    f_scores.write("source1_entity_id\tcand_entity_id\tprob\n")

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
            cand_ids = [cid for cid, _, _, _, _ in candidates]

            chunk_s1_ids.append(s1_id)
            chunk_cand_strings.append(",".join(cand_ids))

            # Normalize this S1's own name/address once, reused for every
            # one of its ~30 candidate pairs below instead of re-normalizing
            # the same string on each extract_pair_features() call.
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
                    s1_norm_name=p["s1_norm_name"],
                    s1_norm_addr=p["s1_norm_addr"],
                    cand_norm_name=p["cand_norm_name"],
                    cand_norm_addr=p["cand_norm_addr"],
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
                    s1_to_matches[p["s1_id"]].append((p["cand_id"], float(final_probs[idx])))

        # Write matching_results.tsv + match_scores.tsv rows
        for s1_id in chunk_s1_ids:
            seen = set()
            matches = []
            for cid, prob in s1_to_matches[s1_id]:
                if cid in seen:
                    continue
                seen.add(cid)
                matches.append(cid)
                f_scores.write(f"{s1_id}\t{cid}\t{prob:.6f}\n")
            f_match.write(f"{s1_id}\t{','.join(matches)}\n")
            total_matches_written += len(matches)

        total_s1_processed += len(chunk)
        elapsed = time.time() - start_time
        speed = total_s1_processed / elapsed if elapsed > 0 else 0
        pct = (total_s1_processed / TOTAL_TEST_ENTITIES) * 100
        print(f"[{time.strftime('%X')}] Progress: {total_s1_processed:,} / {TOTAL_TEST_ENTITIES:,} ({pct:.1f}%) | Speed: {speed:.0f} ent/s | Matches: {total_matches_written:,}", flush=True)


    f_match.close()
    f_cand.close()
    f_scores.close()

    elapsed = time.time() - start_time
    print(f"\n\nPipeline execution finished in {elapsed:.1f}s ({elapsed/60:.2f} mins)!")
    print(f"  - Total S1 Entities:     {total_s1_processed}")
    print(f"  - Total Matches Found:   {total_matches_written}")
    print(f"  - matching_results.tsv:  {matching_tsv}")
    print(f"  - candidate_pairs.tsv:   {candidate_tsv}")

    # 4. Global one-S1-per-candidate resolution (see dedupe_matches.py):
    #    each S1 was scored independently above, so a candidate accepted by
    #    more than one S1 is kept only under its highest-scoring S1.
    print("\n" + "=" * 50)
    print("    RESOLVING CROSS-ENTITY DUPLICATE MATCHES")
    print("=" * 50)
    dedup_tmp = matching_tsv + ".dedup_tmp"
    before, after = dedupe(matching_tsv, scores_tsv, dedup_tmp)
    os.replace(dedup_tmp, matching_tsv)
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    print(f"Matches before: {before:,}  after: {after:,}  removed as duplicate-claimed: {removed:,} ({pct:.2f}%)")

    # 5. Run Official Challenge Validator Script
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
    parser.add_argument(
        "--skip-smoke-test", action="store_true",
        help="Skip the retrieval throughput smoke-test (not recommended - see run_smoke_test()).",
    )
    args = parser.parse_args()

    run_pipeline(
        decision_threshold=args.threshold,
        max_entities=args.max_entities,
        rebuild_index=args.rebuild_index,
        skip_smoke_test=args.skip_smoke_test,
    )
