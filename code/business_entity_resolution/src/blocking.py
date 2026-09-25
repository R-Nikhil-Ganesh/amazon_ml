#!/usr/bin/env python3
"""
High-Recall Multi-Pass Blocking & Candidate Retrieval Engine.

Processes records partitioned by country to stay within memory limits (< 3 GB RAM).
Builds multi-pass inverted indexes:
1. Exact normalized name
2. Compressed name (without spaces: e.g. 'davis decker' <-> 'davisdecker')
3. Distinctive name tokens
4. Address anchor keys (street number + street name token)
5. Distinctive address bigrams (e.g. 'safal solitaire', 'charitravan buxar', 'shalimar bagh')
   to catch non-Latin transliterations (Hindi, Gujarati) and DBA variations.
"""

import argparse
import csv
from collections import defaultdict
import os
import sys
from typing import Dict, List, Set, Tuple

from .config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    MINI_VAL_GROUND_TRUTH_PATH,
    MAX_CANDIDATES_PER_S1,
    MAX_TOKEN_POSTINGS,
    STOPWORDS,
)
from .normalize import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    clean_basic,
)

ADDRESS_STOPWORDS = {
    "road", "street", "avenue", "drive", "lane", "boulevard", "highway", "court",
    "suite", "apartment", "unit", "floor", "near", "opposite", "infront", "behind",
    "colony", "nagar", "village", "town", "district", "city", "state", "door", "no",
    "rue", "allee", "chemin", "place", "impasse", "france", "india", "delhi", "ny", "ca",
    "tx", "fl", "il", "pa", "oh", "ga", "nc", "mi", "nj", "va", "wa", "az", "ma",
}


def read_entity_csv(filepath: str, allowed_ids: Set[str] = None) -> List[Dict[str, str]]:
    """Stream and parse entity CSV into list of records."""
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        
        col_map = {}
        for idx, col in enumerate(header):
            c = col.strip().lower()
            if "entity_id" in c:
                col_map["entity_id"] = idx
            elif "business_name" in c or "name" in c:
                col_map["name"] = idx
            elif "business_address" in c or "address" in c:
                col_map["address"] = idx
            elif "country" in c:
                col_map["country"] = idx

        for row in reader:
            if not row:
                continue
            entity_id = row[col_map.get("entity_id", 1)].strip()
            if allowed_ids is not None and entity_id not in allowed_ids:
                continue

            name = row[col_map.get("name", 2)].strip() if "name" in col_map else ""
            address = row[col_map.get("address", 3)].strip() if "address" in col_map else ""
            country = row[col_map.get("country", 4)].strip() if "country" in col_map else ""

            records.append({
                "entity_id": entity_id,
                "business_name": name,
                "business_address": address,
                "country": country,
            })
    return records


class BlockingEngine:
    def __init__(self, max_candidates: int = MAX_CANDIDATES_PER_S1):
        self.max_candidates = max_candidates
        # Per-country indexes
        self.exact_name_idx = defaultdict(lambda: defaultdict(list))
        self.compressed_name_idx = defaultdict(lambda: defaultdict(list))
        self.token_idx = defaultdict(lambda: defaultdict(list))
        self.addr_anchor_idx = defaultdict(lambda: defaultdict(list))
        self.addr_bigram_idx = defaultdict(lambda: defaultdict(list))
        self.candidates_data = {}

    def index_candidates(self, candidate_records: List[Dict[str, str]]):
        """Index Source 2 and Source 3 records."""
        for rec in candidate_records:
            cid = rec["entity_id"]
            country = rec["country"]
            raw_name = rec["business_name"]
            raw_addr = rec["business_address"]

            self.candidates_data[cid] = rec

            # 1. Exact normalized name
            norm_name = normalize_business_name(raw_name)
            if norm_name:
                self.exact_name_idx[country][norm_name].append(cid)

            # 2. Compressed name (no spaces)
            no_spaces = "".join(norm_name.split())
            if len(no_spaces) >= 4:
                self.compressed_name_idx[country][no_spaces].append(cid)

            # 3. Distinctive name tokens
            tokens = [
                t for t in norm_name.split()
                if len(t) >= 3 and t not in STOPWORDS
            ]
            for token in tokens:
                postings = self.token_idx[country][token]
                if len(postings) < MAX_TOKEN_POSTINGS:
                    postings.append(cid)

            # 4. Address anchor keys (number + distinctive address token)
            norm_addr = normalize_address(raw_addr)
            nums = extract_numbers(norm_addr)
            addr_tokens = [
                w for w in norm_addr.split()
                if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
            ]
            if nums and addr_tokens:
                for num in list(nums)[:3]:
                    for atok in addr_tokens[:2]:
                        key = f"{num}_{atok}"
                        postings = self.addr_anchor_idx[country][key]
                        if len(postings) < MAX_TOKEN_POSTINGS:
                            postings.append(cid)

            # 5. Distinctive address bigrams (e.g. 'safal solitaire', 'charitravan buxar')
            if len(addr_tokens) >= 2:
                for i in range(min(len(addr_tokens) - 1, 4)):
                    bigram = f"{addr_tokens[i]}_{addr_tokens[i+1]}"
                    postings = self.addr_bigram_idx[country][bigram]
                    if len(postings) < MAX_TOKEN_POSTINGS:
                        postings.append(cid)

    def retrieve_candidates_for_s1(self, s1_rec: Dict[str, str]) -> List[str]:
        """Retrieve top candidate IDs for a single S1 record."""
        country = s1_rec["country"]
        raw_name = s1_rec["business_name"]
        raw_addr = s1_rec["business_address"]

        norm_name = normalize_business_name(raw_name)
        no_spaces = "".join(norm_name.split())
        s1_tokens = [
            t for t in norm_name.split()
            if len(t) >= 3 and t not in STOPWORDS
        ]

        candidate_scores = defaultdict(float)

        # 1. Exact name match bonus
        if norm_name in self.exact_name_idx[country]:
            for cid in self.exact_name_idx[country][norm_name]:
                candidate_scores[cid] += 15.0

        # 2. Compressed name match
        if len(no_spaces) >= 4 and no_spaces in self.compressed_name_idx[country]:
            for cid in self.compressed_name_idx[country][no_spaces]:
                candidate_scores[cid] += 12.0

        # 3. Name token overlap score
        for token in s1_tokens:
            if token in self.token_idx[country]:
                for cid in self.token_idx[country][token]:
                    candidate_scores[cid] += 2.0

        # 4. Address anchor match
        norm_addr = normalize_address(raw_addr)
        nums = extract_numbers(norm_addr)
        addr_tokens = [
            w for w in norm_addr.split()
            if len(w) >= 3 and not w.isdigit() and w not in ADDRESS_STOPWORDS
        ]
        if nums and addr_tokens:
            for num in list(nums)[:3]:
                for atok in addr_tokens[:2]:
                    key = f"{num}_{atok}"
                    if key in self.addr_anchor_idx[country]:
                        for cid in self.addr_anchor_idx[country][key]:
                            candidate_scores[cid] += 4.0

        # 5. Address bigram match
        if len(addr_tokens) >= 2:
            for i in range(min(len(addr_tokens) - 1, 4)):
                bigram = f"{addr_tokens[i]}_{addr_tokens[i+1]}"
                if bigram in self.addr_bigram_idx[country]:
                    for cid in self.addr_bigram_idx[country][bigram]:
                        candidate_scores[cid] += 5.0

        if not candidate_scores:
            return []

        # Sort by score descending and take top K
        sorted_candidates = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [cid for cid, score in sorted_candidates[:self.max_candidates]]


def evaluate_blocking_recall(
    candidates_map: Dict[str, List[str]],
    ground_truth_path: str,
) -> Tuple[float, float, int]:
    """Measure candidate recall ceiling against ground truth."""
    total_true_matches = 0
    captured_true_matches = 0
    total_s1 = 0
    total_candidates_generated = 0

    with open(ground_truth_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        s1_idx = 1 if len(header) > 2 and header[1] == "source1_entity_id" else 0
        if "source1_entity_id" in header:
            s1_idx = header.index("source1_entity_id")

        for row in reader:
            if not row or len(row) <= s1_idx:
                continue
            s1_id = row[s1_idx].strip()
            if s1_id not in candidates_map:
                continue

            total_s1 += 1
            cand_set = set(candidates_map[s1_id])
            total_candidates_generated += len(cand_set)

            match_col = s1_idx + 1
            match_str = row[match_col].strip() if len(row) > match_col else ""
            if match_str:
                true_ids = {m.strip() for m in match_str.split(",") if m.strip()}
                total_true_matches += len(true_ids)
                captured_true_matches += len(true_ids & cand_set)

    recall = captured_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    avg_cand = total_candidates_generated / total_s1 if total_s1 > 0 else 0.0

    print("=" * 50)
    print("        BLOCKING RECALL BENCHMARK")
    print("=" * 50)
    print(f"Evaluated S1 Entities:     {total_s1}")
    print(f"Total True Match Pairs:    {total_true_matches}")
    print(f"Captured Match Pairs:      {captured_true_matches}")
    print(f"Candidate Recall Ceiling:  {recall:.2%}")
    print(f"Avg Candidates per S1:     {avg_cand:.1f}")
    print("=" * 50)

    return recall, avg_cand, total_s1


if __name__ == "__main__":
    print("Blocking engine module ready.")
