#!/usr/bin/env python3
"""
High-Speed Tabular Feature Extraction for Candidate Pairs.

Computes lexical, token-based, phonetic, and numerical agreement features
between a Source 1 entity and a candidate (Source 2 or Source 3) entity.
Optimized for high throughput across millions of pairs.
"""

from typing import Dict, Any, List, Set
import numpy as np

try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from .normalize import (
    normalize_business_name,
    normalize_address,
    extract_numbers,
    clean_basic,
    phonetic_skeleton,
)


def _name_skeleton(norm_name: str) -> str:
    """Space-joined phonetic skeleton of a normalized name's tokens - a
    script-agnostic form so a Latin/Brahmic spelling mismatch still gets
    credit here (see phonetic_skeleton()'s docstring), even for pairs that
    blocking retrieved via a different key than sk_ (e.g. a shared address
    number)."""
    return " ".join(phonetic_skeleton(t) for t in norm_name.split() if len(t) >= 3)


def compute_token_jaccard(tokens1: Set[str], tokens2: Set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def compute_number_features(numbers1: Set[str], numbers2: Set[str]) -> Dict[str, float]:
    """Features capturing numerical agreement (postal codes, street numbers)."""
    if not numbers1 and not numbers2:
        return {
            "num_exact_match": 1.0,
            "num_jaccard": 1.0,
            "num_overlap_count": 0.0,
            "num_conflict": 0.0,
        }
    if not numbers1 or not numbers2:
        return {
            "num_exact_match": 0.0,
            "num_jaccard": 0.0,
            "num_overlap_count": 0.0,
            "num_conflict": 0.0,
        }

    common = numbers1 & numbers2
    overlap_count = len(common)
    jaccard = overlap_count / len(numbers1 | numbers2)
    exact_match = 1.0 if numbers1 == numbers2 else 0.0
    # Conflict: both have numbers, but share none
    conflict = 1.0 if overlap_count == 0 else 0.0

    return {
        "num_exact_match": exact_match,
        "num_jaccard": jaccard,
        "num_overlap_count": float(overlap_count),
        "num_conflict": conflict,
    }


def extract_pair_features(
    s1_name_raw: str,
    s1_addr_raw: str,
    cand_name_raw: str,
    cand_addr_raw: str,
    cand_id: str,
    s1_norm_name: str = None,
    s1_norm_addr: str = None,
    cand_norm_name: str = None,
    cand_norm_addr: str = None,
) -> Dict[str, float]:
    """Compute dense feature vector for a single (S1, Candidate) pair.

    The four `*_norm_*` arguments let a caller that already normalized this
    text skip doing it again here - e.g. DiskBlockingEngine.retrieve_candidates()
    persists/returns each candidate's norm_name/norm_addr (computed once at
    index-build time), and an S1's own norm_name/norm_addr is the same for
    every one of its ~30 candidate pairs. Each defaults to None, in which
    case it's normalized from the corresponding `_raw` argument exactly as
    before, so existing callers are unaffected.
    """
    # Normalized strings
    s1_name = s1_norm_name if s1_norm_name is not None else normalize_business_name(s1_name_raw)
    cand_name = cand_norm_name if cand_norm_name is not None else normalize_business_name(cand_name_raw)
    s1_addr = s1_norm_addr if s1_norm_addr is not None else normalize_address(s1_addr_raw)
    cand_addr = cand_norm_addr if cand_norm_addr is not None else normalize_address(cand_addr_raw)

    s1_name_tokens = set(s1_name.split())
    cand_name_tokens = set(cand_name.split())
    s1_addr_tokens = set(s1_addr.split())
    cand_addr_tokens = set(cand_addr.split())

    # String similarity features via RapidFuzz
    if HAS_RAPIDFUZZ:
        name_ratio = fuzz.ratio(s1_name, cand_name) / 100.0
        name_partial_ratio = fuzz.partial_ratio(s1_name, cand_name) / 100.0
        name_token_sort = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        name_token_set = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        name_jw = distance.JaroWinkler.similarity(s1_name, cand_name)

        addr_ratio = fuzz.ratio(s1_addr, cand_addr) / 100.0
        addr_partial_ratio = fuzz.partial_ratio(s1_addr, cand_addr) / 100.0
        addr_token_sort = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
        addr_token_set = fuzz.token_set_ratio(s1_addr, cand_addr) / 100.0
    else:
        # Fallback when rapidfuzz is not yet loaded
        name_ratio = float(s1_name == cand_name)
        name_partial_ratio = float(s1_name in cand_name or cand_name in s1_name)
        name_token_sort = compute_token_jaccard(s1_name_tokens, cand_name_tokens)
        name_token_set = name_token_sort
        name_jw = name_ratio
        addr_ratio = float(s1_addr == cand_addr)
        addr_partial_ratio = float(s1_addr in cand_addr or cand_addr in s1_addr)
        addr_token_sort = compute_token_jaccard(s1_addr_tokens, cand_addr_tokens)
        addr_token_set = addr_token_sort

    # Token overlap
    name_jaccard = compute_token_jaccard(s1_name_tokens, cand_name_tokens)
    addr_jaccard = compute_token_jaccard(s1_addr_tokens, cand_addr_tokens)

    # Cross-script name similarity: without this, blocking can retrieve a
    # true Latin/Brahmic-script match (e.g. "Global Developers" / "ग्लोबल
    # डेवलपर्स") via its shared address, but every text-similarity feature
    # above sees ~0 name similarity and the screener rejects it anyway.
    if HAS_RAPIDFUZZ:
        name_skeleton_ratio = fuzz.token_set_ratio(_name_skeleton(s1_name), _name_skeleton(cand_name)) / 100.0
    else:
        name_skeleton_ratio = float(_name_skeleton(s1_name) == _name_skeleton(cand_name))

    # First token match (very strong signal for brand name)
    s1_first_token = s1_name.split()[0] if s1_name else ""
    cand_first_token = cand_name.split()[0] if cand_name else ""
    first_token_match = 1.0 if s1_first_token and s1_first_token == cand_first_token else 0.0

    # Number features
    s1_nums = extract_numbers(s1_addr_raw)
    cand_nums = extract_numbers(cand_addr_raw)
    num_feats = compute_number_features(s1_nums, cand_nums)

    # Length discrepancies
    name_len_diff = abs(len(s1_name) - len(cand_name)) / max(len(s1_name), len(cand_name), 1)
    addr_len_diff = abs(len(s1_addr) - len(cand_addr)) / max(len(s1_addr), len(cand_addr), 1)

    # Source prefix (S2 vs S3)
    is_source2 = 1.0 if cand_id.startswith("S2-") else 0.0

    features = {
        "name_ratio": name_ratio,
        "name_partial_ratio": name_partial_ratio,
        "name_token_sort": name_token_sort,
        "name_token_set": name_token_set,
        "name_jw": name_jw,
        "name_jaccard": name_jaccard,
        "name_skeleton_ratio": name_skeleton_ratio,
        "first_token_match": first_token_match,
        "name_len_diff": name_len_diff,
        "addr_ratio": addr_ratio,
        "addr_partial_ratio": addr_partial_ratio,
        "addr_token_sort": addr_token_sort,
        "addr_token_set": addr_token_set,
        "addr_jaccard": addr_jaccard,
        "addr_len_diff": addr_len_diff,
        "is_source2": is_source2,
        **num_feats,
    }
    return features


def add_group_relative_features(group_feats: List[Dict[str, float]]) -> None:
    """No-op - kept as a stable import target for train_screener.py /
    train_tf_specialist.py / pipeline_duckdb.py's call sites.

    A within-entity relative-rank feature set (rel_composite, rel_rank_frac,
    rel_gap_to_top, group_size_norm) was tried here and measured on the real
    mini-val pipeline: it improved non-singleton F_0.5 (0.8972 -> 0.8985)
    but hurt singleton accuracy badly (90.99% -> 84.71%), and even after
    re-sweeping the decode confidence floor (see decode.py) to trade some of
    that back, the best achievable macro F_0.5 (0.8946 @ floor=0.80) still
    fell short of the no-relative-features baseline (0.8980). Reverted - see
    documents/engineering_log.md for the full sweep numbers before trying
    this direction again with a different feature design.
    """
    return


FEATURE_NAMES = list(extract_pair_features("a", "b", "c", "d", "S2-1").keys())
