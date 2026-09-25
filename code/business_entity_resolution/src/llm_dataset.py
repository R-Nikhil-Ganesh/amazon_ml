#!/usr/bin/env python3
"""
Prepares the fine-tuning dataset for Stage 3 (Qwen2.5-3B Specialist).

Combines:
1. True positive pairs from Ground Truth.
2. Active hard negatives mined by LightGBM (near-miss street numbers, same name/diff city).
3. Formats into chat templates for Qwen2.5.
"""

import csv
import json
import os
import random
from typing import Dict, List, Tuple

from .config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    MINI_VAL_S1_IDS_PATH,
    SPLIT_DIR,
)
from .blocking import read_entity_csv
from .evaluate import load_mapping_file

SYSTEM_PROMPT = (
    "You are an expert business entity resolution system. Determine whether Entity A and "
    "Entity B refer to the exact same real-world business and location. Pay close attention "
    "to exact house/street numbers and city. Respond strictly with SAME or DIFFERENT."
)


def format_chat_prompt(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
    label: str = None,
) -> Dict:
    """Format into standard ChatML conversation structure."""
    user_content = (
        f"Entity A:\n"
        f"Name: {s1_name or 'N/A'}\n"
        f"Address: {s1_addr or 'N/A'}\n"
        f"Country: {s1_country or 'N/A'}\n\n"
        f"Entity B:\n"
        f"Name: {cand_name or 'N/A'}\n"
        f"Address: {cand_addr or 'N/A'}\n"
        f"Country: {cand_country or 'N/A'}\n\n"
        f"Decision:"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if label:
        messages.append({"role": "assistant", "content": label})

    return {"messages": messages}


def build_llm_training_data(
    num_positives: int = 4000,
    seed: int = 42,
    output_path: str = os.path.join(SPLIT_DIR, "llm_train_data.jsonl"),
):
    random.seed(seed)
    print("Building Stage 3 LLM training dataset...")

    # Load validation IDs to exclude from training
    val_ids = set()
    with open(MINI_VAL_S1_IDS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            val_ids.add(line.strip())

    # 1. Load active hard negatives mined by LightGBM
    hard_neg_path = os.path.join(SPLIT_DIR, "active_hard_negatives.csv")
    hard_negatives = []
    if os.path.exists(hard_neg_path):
        with open(hard_neg_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hard_negatives.append(row)
    print(f"Loaded {len(hard_negatives)} active hard negatives.")

    # 2. Sample true positives (disjoint from validation set)
    gt_map = load_mapping_file(TRAIN_GROUND_TRUTH_PATH)
    train_s1_with_matches = [sid for sid, cids in gt_map.items() if sid not in val_ids and len(cids) > 0]
    random.shuffle(train_s1_with_matches)
    sampled_s1 = set(train_s1_with_matches[:num_positives])

    # Load S1 records
    s1_records = {r["entity_id"]: r for r in read_entity_csv(TRAIN_SOURCE1_PATH, allowed_ids=sampled_s1)}

    # Collect needed candidate IDs
    needed_cand_ids = set()
    for sid in sampled_s1:
        needed_cand_ids |= gt_map[sid]

    cand_records = {}
    for p in [TRAIN_SOURCE2_PATH, TRAIN_SOURCE3_PATH]:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row:
                    continue
                cid = row[1].strip() if len(row) > 1 and row[1].startswith(("S2-", "S3-")) else row[0].strip()
                if cid in needed_cand_ids:
                    cand_records[cid] = {
                        "entity_id": cid,
                        "business_name": row[2].strip() if len(row) > 2 else "",
                        "business_address": row[3].strip() if len(row) > 3 else "",
                        "country": row[4].strip() if len(row) > 4 else "",
                    }

    positives = []
    for sid in sampled_s1:
        if sid not in s1_records:
            continue
        s1 = s1_records[sid]
        for cid in gt_map[sid]:
            if cid in cand_records:
                cand = cand_records[cid]
                positives.append({
                    "s1_name": s1["business_name"],
                    "s1_addr": s1["business_address"],
                    "s1_country": s1["country"],
                    "cand_name": cand["business_name"],
                    "cand_addr": cand["business_address"],
                    "cand_country": cand["country"],
                    "label": "SAME",
                })

    print(f"Collected {len(positives)} positive pairs.")

    # Combine positives and hard negatives
    all_examples = []
    for p in positives:
        all_examples.append(format_chat_prompt(
            p["s1_name"], p["s1_addr"], p["s1_country"],
            p["cand_name"], p["cand_addr"], p["cand_country"],
            label="SAME",
        ))

    for hn in hard_negatives:
        all_examples.append(format_chat_prompt(
            hn["s1_name"], hn["s1_addr"], hn["s1_country"],
            hn["cand_name"], hn["cand_addr"], hn["cand_country"],
            label="DIFFERENT",
        ))

    random.shuffle(all_examples)
    print(f"Total training examples: {len(all_examples)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Dataset successfully written to {output_path}")
    return output_path


if __name__ == "__main__":
    build_llm_training_data()
