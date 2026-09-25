#!/usr/bin/env python3
"""
Create a disjoint Entity-Level Train / Validation split.

Important:
- Splitting is performed strictly at the Source 1 entity level.
- 80% train S1 entities, 20% validation S1 entities.
- Saves entity IDs and corresponding ground truth for validation.
"""

import argparse
import os
import random
from typing import Set


def create_split(
    ground_truth_path: str,
    output_dir: str,
    val_ratio: float = 0.20,
    seed: int = 42,
    sample_val_size: int = 50000,
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    print(f"Reading ground truth from: {ground_truth_path}")
    all_s1 = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if line:
                s1_id = line.split("\t")[0].strip()
                all_s1.append(s1_id)

    total_count = len(all_s1)
    print(f"Total S1 entities in ground truth: {total_count}")

    random.shuffle(all_s1)
    val_count = int(total_count * val_ratio)
    val_ids = set(all_s1[:val_count])
    train_ids = set(all_s1[val_count:])

    print(f"Train S1 entities: {len(train_ids)} ({len(train_ids)/total_count:.1%})")
    print(f"Val S1 entities:   {len(val_ids)} ({len(val_ids)/total_count:.1%})")

    # Save full val IDs
    full_val_path = os.path.join(output_dir, "val_s1_ids.txt")
    with open(full_val_path, "w", encoding="utf-8") as f:
        for sid in sorted(val_ids):
            f.write(f"{sid}\n")

    # Also save a fast mini-val set for quick iteration during development
    mini_val_ids = set(random.sample(list(val_ids), min(sample_val_size, len(val_ids))))
    mini_val_path = os.path.join(output_dir, "mini_val_s1_ids.txt")
    with open(mini_val_path, "w", encoding="utf-8") as f:
        for sid in sorted(mini_val_ids):
            f.write(f"{sid}\n")

    # Save validation ground truth files
    val_gt_path = os.path.join(output_dir, "val_ground_truth.tsv")
    mini_val_gt_path = os.path.join(output_dir, "mini_val_ground_truth.tsv")

    with open(ground_truth_path, "r", encoding="utf-8") as f_in, \
         open(val_gt_path, "w", encoding="utf-8") as f_val, \
         open(mini_val_gt_path, "w", encoding="utf-8") as f_mini:
        
        header = f_in.readline()
        f_val.write(header)
        f_mini.write(header)

        for line in f_in:
            s1_id = line.split("\t")[0].strip()
            if s1_id in val_ids:
                f_val.write(line)
            if s1_id in mini_val_ids:
                f_mini.write(line)

    print(f"Validation splits saved to {output_dir}:")
    print(f"  - Full val: {full_val_path} ({len(val_ids)} entities)")
    print(f"  - Mini val: {mini_val_path} ({len(mini_val_ids)} entities)")


def main():
    parser = argparse.ArgumentParser(description="Create train/val entity split.")
    parser.add_argument(
        "--truth",
        default="student_resource/dataset/train/train_ground_truth.tsv",
        help="Path to train_ground_truth.tsv",
    )
    parser.add_argument(
        "--out",
        default="data_split",
        help="Output directory for split files",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation ratio (default 0.20)",
    )
    args = parser.parse_args()

    create_split(args.truth, args.out, val_ratio=args.val_ratio)


if __name__ == "__main__":
    main()
