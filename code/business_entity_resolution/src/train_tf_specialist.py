#!/usr/bin/env python3
"""
Stage 3: Deep Neural Entity Matcher using TensorFlow & Keras 3 (Dense Residual Architecture).

Uses pre-installed TensorFlow 2.21.0 & Keras 3.12.4.
High-speed training (< 30 seconds on CPU) on tabular feature vectors with non-linear feature interactions.
Trained on Positives + Active Hard Negatives mined by LightGBM to specialize in edge cases.
"""

import argparse
import csv
import os
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import keras
from keras import layers

from .config import (
    delimiter_for,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    MINI_VAL_GROUND_TRUTH_PATH,
    MINI_VAL_S1_IDS_PATH,
    MODELS_DIR,
    SPLIT_DIR,
    MAX_CANDIDATES_PER_S1,
    TRAIN_CANDIDATE_INDEX_DUCKDB_PATH,
)
from .blocking import read_entity_csv
from .duckdb_blocking import DuckDBBlockingEngine
from .features import extract_pair_features, add_group_relative_features, FEATURE_NAMES
from .evaluate import evaluate_predictions, load_mapping_file
from .normalize import normalize_business_name, normalize_address

# Matches pipeline_duckdb.py's default and train_screener.py's
# RETRIEVAL_CHUNK_SIZE - training sees the same retrieval batching real
# inference does.
RETRIEVAL_CHUNK_SIZE = 4000


def create_deep_res_mlp(input_dim: int = len(FEATURE_NAMES)) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name="dense_features")

    x = layers.Dense(128, activation="swish")(inputs)
    x = layers.BatchNormalization()(x)

    res1 = x
    x = layers.Dense(128, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Add()([x, res1])

    res2 = x
    x = layers.Dense(128, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="swish")(x)
    x = layers.Add()([x, res2])

    x = layers.Dense(64, activation="swish")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1, activation="sigmoid", name="probability")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="Deep_Res_Matcher")
    return model


def build_pair_features(
    s1_records: List[Dict[str, str]],
    engine: DuckDBBlockingEngine,
    ground_truth_map: Dict[str, set],
    hard_negatives: List[Dict[str, str]] = None,
    max_neg_ratio: int = 5,
    include_missed_positives: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]]]:
    """
    Retrieve REAL blocking candidates from the full-scale DuckDB index (same
    engine/scale pipeline_duckdb.py uses at test time, including
    transliteration) instead of a small pre-curated candidate dict, in
    batches of RETRIEVAL_CHUNK_SIZE via retrieve_candidates_batch().
    include_missed_positives must be False for validation - see the matching
    note in train_screener.build_pair_dataset.
    """
    X_rows = []
    y_labels = []
    meta = []

    for chunk_start in range(0, len(s1_records), RETRIEVAL_CHUNK_SIZE):
        chunk = s1_records[chunk_start:chunk_start + RETRIEVAL_CHUNK_SIZE]

        # norm_name/norm_addr are already-computed at index-build time, so
        # extract_pair_features() below doesn't need to re-normalize the
        # candidate side from raw text (same pattern as train_screener.py).
        retrieved_by_s1 = engine.retrieve_candidates_batch(chunk)

        for s1 in chunk:
            s1_id = s1["entity_id"]
            true_cids = ground_truth_map.get(s1_id, set())

            retrieved = retrieved_by_s1.get(s1_id, [])
            cand_lookup = {
                cid: (name, addr, norm_name, norm_addr)
                for cid, name, addr, norm_name, norm_addr in retrieved
            }

            if include_missed_positives:
                for cid in true_cids:
                    if cid not in cand_lookup:
                        row = engine.get_by_id(cid)
                        if row is not None:
                            name, addr, _country, norm_name, norm_addr = row
                            cand_lookup[cid] = (name, addr, norm_name, norm_addr)

            s1_norm_name = normalize_business_name(s1["business_name"])
            s1_norm_addr = normalize_address(s1["business_address"])

            group = []
            for cid, (cand_name, cand_addr, cand_norm_name, cand_norm_addr) in cand_lookup.items():
                is_match = 1.0 if cid in true_cids else 0.0
                feats = extract_pair_features(
                    s1["business_name"],
                    s1["business_address"],
                    cand_name,
                    cand_addr,
                    cid,
                    s1_norm_name=s1_norm_name,
                    s1_norm_addr=s1_norm_addr,
                    cand_norm_name=cand_norm_name,
                    cand_norm_addr=cand_norm_addr,
                )
                group.append((cid, feats, is_match))

            add_group_relative_features([f for _, f, _ in group])

            neg_count = 0
            for cid, feats, is_match in group:
                if not is_match:
                    neg_count += 1
                    if max_neg_ratio and neg_count > max(len(true_cids) * max_neg_ratio, 10):
                        continue
                X_rows.append([feats[k] for k in FEATURE_NAMES])
                y_labels.append(is_match)
                meta.append({"s1_id": s1_id, "cand_id": cid})

    if hard_negatives:
        for hn in hard_negatives:
            feats = extract_pair_features(
                hn["s1_name"],
                hn["s1_addr"],
                hn["cand_name"],
                hn["cand_addr"],
                hn["cand_id"],
            )
            # hard_negatives are mined standalone pairs, not part of a
            # retrieved candidate group - treat each as its own group of one
            # (rel_rank_frac=0, rel_gap_to_top=0, group_size_norm=1/35) so it
            # gets neutral-but-valid relative feature values.
            add_group_relative_features([feats])
            X_rows.append([feats[k] for k in FEATURE_NAMES])
            y_labels.append(0.0)
            meta.append({"s1_id": hn["s1_id"], "cand_id": hn["cand_id"]})

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_labels, dtype=np.float32),
        meta,
    )


def train_tf_specialist(train_sample_size: int = 80000, val_sample_size: int = 8000):
    print("============================================================")
    print("  STAGE 3: TRAINING TENSORFLOW / KERAS DEEP NEURAL MATCHER")
    print("============================================================")

    val_ids = set()
    with open(MINI_VAL_S1_IDS_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < val_sample_size:
                val_ids.add(line.strip())

    train_ids = set()
    with open(TRAIN_GROUND_TRUTH_PATH, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter_for(TRAIN_GROUND_TRUTH_PATH))
        header = next(reader)
        s1_col = 1 if len(header) > 2 and header[1] == "source1_entity_id" else 0
        for row in reader:
            if not row or len(row) <= s1_col:
                continue
            sid = row[s1_col].strip()
            if sid not in val_ids:
                train_ids.add(sid)
                if len(train_ids) >= train_sample_size:
                    break

    train_gt = load_mapping_file(TRAIN_GROUND_TRUTH_PATH)
    val_gt = load_mapping_file(MINI_VAL_GROUND_TRUTH_PATH)

    all_s1 = read_entity_csv(TRAIN_SOURCE1_PATH, allowed_ids=(train_ids | val_ids))
    train_s1 = [r for r in all_s1 if r["entity_id"] in train_ids]
    val_s1 = [r for r in all_s1 if r["entity_id"] in val_ids]

    # Reuse the same full-scale DuckDB candidate index that train_screener.py
    # builds over the entire training S2/S3 pool, so this stage sees the
    # identical realistic candidate universe (including transliteration)
    # pipeline_duckdb.py searches at real test time (see train_screener.py
    # for why the previous 200,000-row-plus-guaranteed-positives pool was
    # misleading, and documents/engineering_log.md for why this is DuckDB
    # now rather than the original SQLite DiskBlockingEngine).
    engine = DuckDBBlockingEngine(db_path=TRAIN_CANDIDATE_INDEX_DUCKDB_PATH, max_candidates=MAX_CANDIDATES_PER_S1)
    if os.path.exists(TRAIN_CANDIDATE_INDEX_DUCKDB_PATH):
        engine.open(read_only=True)
    else:
        print(f"Building full-scale train candidate index at {TRAIN_CANDIDATE_INDEX_DUCKDB_PATH}...")
        engine.build_index(TRAIN_SOURCE2_PATH, TRAIN_SOURCE3_PATH, reset=True)

    hard_neg_path = os.path.join(SPLIT_DIR, "active_hard_negatives.csv")
    hard_negatives = []
    if os.path.exists(hard_neg_path):
        with open(hard_neg_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hard_negatives.append(row)

    print("Building training feature vectors...")
    X_tr, y_tr, m_tr = build_pair_features(
        train_s1, engine, train_gt,
        hard_negatives=hard_negatives, max_neg_ratio=4, include_missed_positives=True,
    )
    print(f"Train samples: {len(y_tr)} (Positives: {int(np.sum(y_tr))}, Negatives: {int(np.sum(y_tr == 0))})")

    print("Building validation feature vectors...")
    X_val, y_val, m_val = build_pair_features(
        val_s1, engine, val_gt,
        hard_negatives=None, max_neg_ratio=None, include_missed_positives=False,
    )
    print(f"Val samples:   {len(y_val)} (Positives: {int(np.sum(y_val))}, Negatives: {int(np.sum(y_val == 0))})")

    model = create_deep_res_mlp(input_dim=X_tr.shape[1])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )

    print("\nTraining Deep Residual Matcher...")
    # min_delta matters here: val_auc typically converges by epoch 3-4 (see
    # documents/engineering_log.md) and then oscillates within +/-0.0005 for
    # the rest of the run - without a min_delta, that noise keeps resetting
    # EarlyStopping's counter and every run burns its full epoch budget for
    # ~0.0004 AUC of real gain (~15 extra minutes on this box).
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", patience=3, min_delta=0.0005, mode="max", restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_auc", factor=0.5, patience=2, mode="max"),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=6,  # val_auc plateaus by epoch 3-4 (see documents/engineering_log.md); hard cap, no early-stop dependency
        batch_size=512,
        callbacks=callbacks,
        verbose=1,
    )

    save_path = os.path.join(MODELS_DIR, "tf_specialist_model.keras")
    model.save(save_path)
    print(f"\nModel saved to: {save_path}")

    print("\n==================================================")
    print("      EVALUATING TF SPECIALIST FOR MACRO F_0.5")
    print("==================================================")
    tf_probs = model.predict(X_val, batch_size=1024).flatten()

    eval_truth = {r["entity_id"]: val_gt.get(r["entity_id"], set()) for r in val_s1}

    best_thresh = 0.5
    best_f05 = 0.0
    for thresh in np.arange(0.4, 0.95, 0.05):
        pred_map = {r["entity_id"]: set() for r in val_s1}
        for i, p in enumerate(tf_probs):
            if p >= thresh:
                pred_map[m_val[i]["s1_id"]].add(m_val[i]["cand_id"])

        macro_f05, stats = evaluate_predictions(eval_truth, pred_map)
        print(f"Threshold: {thresh:.2f} | Macro F_0.5: {macro_f05:.4f} | Singleton Acc: {stats['singleton_accuracy']:.2%}")
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_thresh = thresh

    print("==================================================")
    print(f"TF SPECIALIST BEST THRESHOLD: {best_thresh:.2f} => MACRO F_0.5: {best_f05:.4f}")
    print("==================================================")
    engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-sample-size", type=int, default=80000)
    parser.add_argument("--val-sample-size", type=int, default=8000)
    args = parser.parse_args()
    train_tf_specialist(train_sample_size=args.train_sample_size, val_sample_size=args.val_sample_size)

