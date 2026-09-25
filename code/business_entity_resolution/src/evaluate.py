#!/usr/bin/env python3
"""
Official Macro F_0.5 Metric Evaluator for Business Entity Resolution Challenge.

Formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

Evaluation rules:
- Macro-averaged across all S1 reference entities in the evaluation set.
- Singletons (entities with no true matches):
    - Predicting an empty set -> score = 1.0
    - Predicting any non-empty set -> score = 0.0
- Entities with true matches:
    - Predicting an empty set -> score = 0.0
    - Otherwise -> standard F_0.5 with precision weighted 2x over recall.
- Supports both CSV and TSV formats seamlessly.
"""

import argparse
import csv
import sys
from typing import Dict, Set, Tuple


def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """Compute F_0.5 for a single S1 entity."""
    n_true = len(true_matches)
    n_pred = len(pred_matches)

    # Singleton case: entity has no true matches
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0

    # True matches exist, but nothing predicted
    if n_pred == 0:
        return 0.0

    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0

    precision = tp / n_pred
    recall = tp / n_true

    denom = 0.25 * precision + recall
    if denom == 0.0:
        return 0.0

    return (1.25 * precision * recall) / denom


def evaluate_predictions(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
) -> Tuple[float, Dict[str, float]]:
    """Compute overall macro F_0.5 and breakdown statistics."""
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        scores.append(score)
        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = sum(scores) / len(scores) if scores else 0.0
    singleton_acc = sum(singleton_scores) / len(singleton_scores) if singleton_scores else 0.0
    non_singleton_f05 = sum(non_singleton_scores) / len(non_singleton_scores) if non_singleton_scores else 0.0

    stats = {
        "macro_f05": macro_f05,
        "total_entities": len(ground_truth),
        "singleton_count": len(singleton_scores),
        "singleton_accuracy": singleton_acc,
        "non_singleton_count": len(non_singleton_scores),
        "non_singleton_f05": non_singleton_f05,
    }
    return macro_f05, stats


def load_mapping_file(filepath: str) -> Dict[str, Set[str]]:
    """Load S1 -> set of matched IDs from CSV or TSV file."""
    mapping = {}
    is_csv = filepath.endswith(".csv")
    delimiter = "," if is_csv else "\t"

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header = next(reader, None)
        if not header:
            return mapping

        # Detect column indices for source1_entity_id and matched_entity_ids
        s1_col = 0
        match_col = 1
        for idx, col in enumerate(header):
            col_clean = col.strip().lower()
            if "source1" in col_clean or "entity_id" in col_clean and idx == 0:
                s1_col = idx
            elif "matched" in col_clean or "candidate" in col_clean:
                match_col = idx

        for row in reader:
            if not row or len(row) <= s1_col:
                continue
            s1_id = row[s1_col].strip()
            # If the CSV has an unnamed index column as first column:
            if s1_id.isdigit() and len(row) > s1_col + 1 and row[s1_col + 1].startswith("S1-"):
                s1_id = row[s1_col + 1].strip()
                match_val = row[s1_col + 2].strip() if len(row) > s1_col + 2 else ""
            else:
                match_val = row[match_col].strip() if len(row) > match_col else ""

            if match_val:
                match_ids = {m.strip() for m in match_val.split(",") if m.strip()}
            else:
                match_ids = set()
            mapping[s1_id] = match_ids
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate Entity Resolution predictions.")
    parser.add_argument("--truth", "-t", required=True, help="Path to ground truth CSV or TSV")
    parser.add_argument("--preds", "-p", required=True, help="Path to predicted CSV or TSV")
    args = parser.parse_args()

    truth = load_mapping_file(args.truth)
    preds = load_mapping_file(args.preds)

    macro_f05, stats = evaluate_predictions(truth, preds)

    print("=" * 50)
    print("           EVALUATION RESULTS")
    print("=" * 50)
    print(f"Macro F_0.5 Score:      {macro_f05:.4f}")
    print(f"Total S1 Entities:      {stats['total_entities']}")
    print(f"Singleton Count:        {stats['singleton_count']} (Accuracy: {stats['singleton_accuracy']:.2%})")
    print(f"Non-Singleton Count:    {stats['non_singleton_count']} (F_0.5: {stats['non_singleton_f05']:.4f})")
    print("=" * 50)


if __name__ == "__main__":
    main()
