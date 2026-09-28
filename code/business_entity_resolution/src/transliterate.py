#!/usr/bin/env python3
"""
Learned Indic-script -> Latin word transliteration, built from this
dataset's own train_ground_truth matched pairs (not external data).

Why: about 23% of India S2/S3 business names/addresses are written in one
of 8 Indic scripts (Devanagari, Telugu, Tamil, Bengali, Gujarati, Gurmukhi,
Kannada, Malayalam), while every Source 1 record is Latin-script. Blocking
recall on these pairs measures at ~0.195 vs ~0.90 for India pairs that are
already Latin-script on both sides - normalize.py's phonetic_skeleton() is
a lossy consonant-only fold, not a real transliteration, and only weakly
bridges this gap.

Ground-truth matched pairs where the S1 name is Latin and the candidate
name is Indic-script overwhelmingly have equal token counts (measured
99.998% on this dataset), so a word-level alignment learns a near-complete
dictionary: build it from half of such pairs, and it covers 100% of the
distinct Indic words seen in the other half (dataset has ~1,498 distinct
words total). The same alignment is applied to addresses, where usually
one token (the state name) is in Indic script.

This dictionary is learned entirely from train_ground_truth.tsv/csv - the
same training data already used elsewhere in this pipeline - so it does
not violate the "no external data lookup" rule.
"""

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict

from .config import (
    TRAIN_GROUND_TRUTH_PATH,
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    SPLIT_DIR,
)
from .normalize import _is_brahmic_char, transliterate_brahmic_token

TRANSLIT_DICT_PATH = os.path.join(SPLIT_DIR, "translit_dict.json")


def _has_brahmic(text: str) -> bool:
    return any(_is_brahmic_char(ch) for ch in text)


def _read_entity_csv(path: str, id_col_hint: str = "entity_id"):
    """Yield (entity_id, business_name, business_address, country) from a
    csv_data-style S1/S2/S3 file (leading unnamed index column, quoted
    commas in address)."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader)
        id_idx = 1 if len(header) > 2 and header[1] == id_col_hint else 0
        name_idx, addr_idx, cntry_idx = id_idx + 1, id_idx + 2, id_idx + 3
        for row in reader:
            if len(row) <= cntry_idx:
                continue
            yield row[id_idx].strip(), row[name_idx].strip(), row[addr_idx].strip(), row[cntry_idx].strip()


def _align_and_count(latin_text: str, indic_text: str, counter: Dict[str, Counter]):
    """If latin_text and indic_text have the same token count, treat this as
    a word-aligned pair and record each (indic_word -> latin_word) vote."""
    latin_tokens = latin_text.split()
    indic_tokens = indic_text.split()
    if not latin_tokens or len(latin_tokens) != len(indic_tokens):
        return
    for lat, ind in zip(latin_tokens, indic_tokens):
        if _has_brahmic(ind) and not _has_brahmic(lat):
            counter.setdefault(ind, Counter())[lat.lower()] += 1


def build_transliteration_dict(
    ground_truth_path: str = TRAIN_GROUND_TRUTH_PATH,
    s1_path: str = TRAIN_SOURCE1_PATH,
    s2_path: str = TRAIN_SOURCE2_PATH,
    s3_path: str = TRAIN_SOURCE3_PATH,
    country: str = "India",
) -> Dict[str, str]:
    print(f"Loading {country} S1 records from {s1_path}...")
    s1_name = {}
    s1_addr = {}
    for eid, name, addr, cntry in _read_entity_csv(s1_path):
        if cntry == country:
            s1_name[eid] = name
            s1_addr[eid] = addr
    print(f"  {len(s1_name):,} {country} S1 records")

    print(f"Scanning ground truth ({ground_truth_path}) for {country} matched candidate ids...")
    wanted_cand_ids = set()
    pairs = []  # (s1_id, [cand_ids])
    is_csv = ground_truth_path.endswith(".csv")
    delim = "," if is_csv else "\t"
    with open(ground_truth_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delim)
        header = next(reader)
        s1_col = 1 if len(header) > 2 and header[1] == "source1_entity_id" else 0
        match_col = s1_col + 1
        for row in reader:
            if len(row) <= match_col:
                continue
            s1_id = row[s1_col].strip()
            if s1_id not in s1_name:
                continue
            match_val = row[match_col].strip()
            if not match_val:
                continue
            cand_ids = [c.strip() for c in match_val.split(",") if c.strip()]
            pairs.append((s1_id, cand_ids))
            wanted_cand_ids.update(cand_ids)
    print(f"  {len(pairs):,} {country} S1 entities with matches, {len(wanted_cand_ids):,} candidate ids to look up")

    print(f"Loading candidate name/address for {len(wanted_cand_ids):,} wanted ids...")
    cand_name = {}
    cand_addr = {}
    for path in (s2_path, s3_path):
        for eid, name, addr, _cntry in _read_entity_csv(path):
            if eid in wanted_cand_ids:
                cand_name[eid] = name
                cand_addr[eid] = addr
    print(f"  found {len(cand_name):,} of {len(wanted_cand_ids):,}")

    votes: Dict[str, Counter] = {}
    for s1_id, cand_ids in pairs:
        lat_name = s1_name[s1_id]
        lat_addr = s1_addr[s1_id]
        for cid in cand_ids:
            if cid in cand_name:
                _align_and_count(lat_name, cand_name[cid], votes)
            if cid in cand_addr:
                _align_and_count(lat_addr, cand_addr[cid], votes)

    translit_dict = {word: counter.most_common(1)[0][0] for word, counter in votes.items()}
    print(f"Learned {len(translit_dict):,} distinct Indic-script word -> Latin mappings")
    return translit_dict


def transliterate_text(text: str, translit_dict: Dict[str, str]) -> str:
    """Token-level lookup against the learned dictionary first (exact,
    field-tested against real matched pairs); any Indic-script word it
    hasn't seen falls back to transliterate_brahmic_token()'s rule-based
    phonetic transliteration (see normalize.py), which generalizes to
    words outside the dictionary's ~1,400-word vocabulary. A word that's
    neither in the dictionary nor Indic-script passes through unchanged.
    An empty translit_dict disables transliteration entirely (used by
    --no-translit for a clean before/after baseline), not just the
    dictionary lookup."""
    if not text or not translit_dict:
        return text
    out = []
    changed = False
    for tok in text.split():
        repl = translit_dict.get(tok)
        if repl is not None:
            out.append(repl)
            changed = True
        elif _has_brahmic(tok):
            out.append(transliterate_brahmic_token(tok))
            changed = True
        else:
            out.append(tok)
    return " ".join(out) if changed else text


def load_translit_dict(path: str = TRANSLIT_DICT_PATH) -> Dict[str, str]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_translit_dict(translit_dict: Dict[str, str], path: str = TRANSLIT_DICT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(translit_dict, f, ensure_ascii=False, indent=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default=TRANSLIT_DICT_PATH)
    args = parser.parse_args()
    d = build_transliteration_dict()
    save_translit_dict(d, args.out)
    print(f"Saved to {args.out}")
