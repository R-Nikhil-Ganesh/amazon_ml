#!/usr/bin/env python3
"""
Multilingual Text Normalization & Cleaning for Business Entity Resolution.

Handles:
- Unicode NFKC normalization (Devanagari / Hindi, French accents, ASCII).
- Case folding & punctuation cleaning.
- Business legal suffix normalization (US, India, France).
- Street / Address token normalization (English & French).
- Numerical token extraction (house numbers, postal / PIN codes).
"""

import re
import unicodedata
from typing import List, Tuple, Set


# Common legal entity suffixes across US, India, and France
LEGAL_SUFFIX_RE = re.compile(
    r"\b("
    # India / UK
    r"pvt\s*ltd|private\s*limited|pvt|private|ltd|limited|llp|limited\s*liability\s*partnership|"
    # US
    r"inc|incorporated|corp|corporation|llc|co|company|"
    # France (includes acronyms that survive dotted-initial collapsing, e.g. "s.a.s." -> "sas")
    r"sarl|sasu|sas|sa|eurl|eirl|sci|snc|scop|gie|cie"
    r")\b",
    re.IGNORECASE,
)

# Sequences of single-letter tokens (from dotted acronyms like "S.A.S." or
# "S.A." whose periods clean_basic already turned into spaces) get collapsed
# back into one token ("s a s" -> "sas") *before* legal-suffix stripping,
# otherwise the suffix regex above never matches them.
SINGLE_LETTER_RUN_RE = re.compile(r"\b(?:[a-z]\s+){1,5}[a-z]\b")

# Address term normalizations (English + French)
ADDRESS_ABBREVIATIONS = {
    # English
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bct\b": "court",
    r"\bste\b": "suite",
    r"\bapt\b": "apartment",
    r"\bhwy\b": "highway",
    r"\bpkwy\b": "parkway",
    r"\bcir\b": "circle",
    r"\bter\b": "terrace",
    r"\bsq\b": "square",
    # French
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\bimp\b": "impasse",
    r"\bpl\b": "place",
    r"\bche\b": "chemin",
    r"\brte\b": "route",
    r"\bfg\b": "faubourg",
}

# Unicode blocks treated as "Latin" for accent folding: Basic Latin, Latin-1
# Supplement, Latin Extended A/B/Additional. Accents on these are stripped
# (Cafe/Café, Àmicale/Amicale, Fédération/Federation) since French records
# mix accented and unaccented spellings of the same word. Combining marks on
# other scripts (e.g. Devanagari vowel signs) are left alone, since dropping
# those would change the word rather than just its diacritics.
_LATIN_RANGES = ((0x0000, 0x024F), (0x1E00, 0x1EFF))


def _is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _LATIN_RANGES)


def strip_accents(text: str) -> str:
    """Fold accented Latin characters to their unaccented form (NFKD-based)."""
    if not isinstance(text, str):
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    kept = []
    base_is_latin = False
    for ch in decomposed:
        if unicodedata.combining(ch):
            if base_is_latin:
                continue  # drop the accent mark
            kept.append(ch)  # keep combining marks on non-Latin scripts
        else:
            kept.append(ch)
            base_is_latin = _is_latin_char(ch)
    return unicodedata.normalize("NFKC", "".join(kept))


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (NFKC), fold Latin accents, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def clean_basic(text: str) -> str:
    """Lowercase, strip non-alphanumeric (except basic punctuation), and normalize whitespace."""
    text = normalize_unicode(text).lower()
    # Replace common separators with spaces. '#' is included so PO-box/unit
    # markers like "##16" or "#4005" normalize down to just the number
    # instead of leaving a literal '#' glued to the digits (which broke
    # \b\d+\b number extraction and produced a bogus "##16" address token).
    text = re.sub(r"[,/\\|_\-\(\)\[\]\{\}\.:;#\"\'`]", " ", text)
    return " ".join(text.split())


def normalize_business_name(name: str, strip_legal_suffix: bool = True) -> str:
    """Normalize business name, optionally stripping legal suffixes."""
    cleaned = clean_basic(name)
    # Re-join dotted acronyms ("s a s" -> "sas") before suffix stripping.
    cleaned = SINGLE_LETTER_RUN_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), cleaned)
    if strip_legal_suffix:
        cleaned = LEGAL_SUFFIX_RE.sub(" ", cleaned)
    return " ".join(cleaned.split())


def normalize_address(address: str) -> str:
    """Normalize business address, expanding standard street abbreviations."""
    cleaned = clean_basic(address)
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    return " ".join(cleaned.split())


def extract_numbers(text: str) -> Set[str]:
    """Extract all numerical tokens (e.g., house numbers, PIN codes, postal codes)."""
    if not isinstance(text, str):
        return set()
    return set(re.findall(r"\b\d+\b", text))


def get_token_ngrams(tokens: List[str], n: int = 2) -> List[str]:
    """Generate n-grams of words."""
    if len(tokens) < n:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# Devanagari-block consonant map, reused for every Brahmic script this
# dataset sees (Bengali, Gurmukhi, Gujarati, Odia, Tamil, Telugu, Kannada,
# Malayalam, Sinhala) by taking (codepoint - block_start) % 0x80: all of
# these blocks are laid out identically to Devanagari (a shared legacy of
# the ISCII standard they were all derived from), so one table covers them
# without a language-specific dependency. Vowel signs/marks are dropped
# entirely (skeletons are consonant-only); independent vowel letters (0x05-
# 0x14) are dropped too except word-initially, which is enough to line up
# e.g. "ग्लोबल" (global) with "global" without needing full transliteration.
_BRAHMIC_CONSONANTS = {
    0x15: "k", 0x16: "k", 0x17: "g", 0x18: "g", 0x19: "n",
    0x1A: "c", 0x1B: "c", 0x1C: "j", 0x1D: "j", 0x1E: "n",
    0x1F: "t", 0x20: "t", 0x21: "d", 0x22: "d", 0x23: "n",
    0x24: "t", 0x25: "t", 0x26: "d", 0x27: "d", 0x28: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "b", 0x2E: "m",
    0x30: "r", 0x32: "l", 0x33: "l", 0x35: "v",
    0x36: "s", 0x37: "s", 0x38: "s", 0x39: "h",
    0x58: "k", 0x59: "k", 0x5A: "g", 0x5B: "j", 0x5C: "d", 0x5D: "d", 0x5E: "f",
}
_BRAHMIC_INITIAL_VOWELS = {0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u",
                           0x0A: "u", 0x0F: "e", 0x10: "e", 0x13: "o", 0x14: "o"}
_BRAHMIC_BLOCK_STARTS = (0x0900, 0x0980, 0x0A00, 0x0A80, 0x0B00, 0x0B80, 0x0C00, 0x0C80, 0x0D00, 0x0D80)

_LATIN_DIGRAPHS = (
    ("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"),
    ("kh", "k"), ("gh", "g"), ("ch", "c"), ("ck", "k"),
    ("q", "k"), ("x", "ks"), ("w", "v"), ("z", "j"), ("y", ""),
)
_DOUBLE_LETTER_RE = re.compile(r"(.)\1+")


def _is_brahmic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(start <= cp < start + 0x80 for start in _BRAHMIC_BLOCK_STARTS)


def _skeleton_latin(token: str) -> str:
    w = token.lower()
    for a, b in _LATIN_DIGRAPHS:
        w = w.replace(a, b)
    w = re.sub(r"c(?=[eiy])", "s", w)
    w = w.replace("c", "k")
    first = w[:1] if w[:1] in "aeiou" else ""
    # Vowels are dropped, but standalone 'h' is kept as a consonant (unlike
    # the digraphs above, which already fold "th"/"bh"/"kh"/etc. away) so it
    # lines up with the Brahmic side, which maps its aspirate letter to "h".
    skeleton = first + re.sub(r"[aeiou]", "", w)
    return _DOUBLE_LETTER_RE.sub(r"\1", skeleton)


def _skeleton_brahmic(token: str) -> str:
    out = []
    for i, ch in enumerate(token):
        cp = ord(ch)
        block_start = next((s for s in _BRAHMIC_BLOCK_STARTS if s <= cp < s + 0x80), None)
        if block_start is None:
            if ch.isalnum():
                out.append(ch.lower())
            continue
        offset = cp - block_start
        if offset in _BRAHMIC_CONSONANTS:
            out.append(_BRAHMIC_CONSONANTS[offset])
        elif offset in _BRAHMIC_INITIAL_VOWELS and i == 0:
            out.append(_BRAHMIC_INITIAL_VOWELS[offset])
        # vowel signs / matras / virama / anusvara etc. are dropped
    return _DOUBLE_LETTER_RE.sub(r"\1", "".join(out))


def phonetic_skeleton(token: str) -> str:
    """
    Fold a single word to a script-agnostic consonant skeleton, so the same
    business name written in Latin script and in a Brahmic script (e.g.
    "Global Developers" / "ग्लोबल डेवलपर्स") folds to comparable strings
    ("glbl" / "glbl"-ish). Used only as an extra blocking anchor and as a
    matcher feature - it is deliberately lossy (consonant skeleton, not a
    real transliteration), so it should never be the *only* signal a match
    decision relies on.
    """
    if not token:
        return ""
    if any(_is_brahmic_char(ch) for ch in token):
        return _skeleton_brahmic(token)
    return _skeleton_latin(token)


# Full rule-based Brahmic -> Latin phonetic transliteration, separate from
# phonetic_skeleton() above (which already feeds the trained LightGBM
# screener's name_skeleton_ratio feature - kept untouched here so this
# doesn't shift that model's input distribution). Used in transliterate.py
# as a fallback for Indic words the learned train-ground-truth dictionary
# hasn't seen: the dictionary is exact-word-only, so any Indic word absent
# from the training pairs falls straight through today. This generalizes to
# *any* word in these scripts using only character-encoding facts (Unicode
# block layout), not any external corpus or model - same compliance
# footing as phonetic_skeleton() itself.
#
# Reuses _BRAHMIC_BLOCK_STARTS (all 10 scripts share Devanagari's relative
# layout, per the comment above _BRAHMIC_CONSONANTS). Unlike the skeleton
# map, consonants are NOT collapsed to base forms (kh/gh/ch/jh/th/dh/bh are
# kept distinct, matching how these loanwords are actually spelled in
# English, e.g. "Bhavan" not "Bavan"), and vowel signs are mapped to real
# vowel letters instead of being dropped.
_BRAHMIC_CONSONANTS_FULL = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng",
    0x1A: "ch", 0x1B: "chh", 0x1C: "j", 0x1D: "jh", 0x1E: "ny",
    0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh", 0x23: "n",
    0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n",
    0x2A: "p", 0x2B: "f", 0x2C: "b", 0x2D: "bh", 0x2E: "m",
    0x2F: "y", 0x30: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v",
    0x36: "sh", 0x37: "sh", 0x38: "s", 0x39: "h",
    # Nukta (loan-sound) precomposed forms
    0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "d", 0x5D: "dh", 0x5E: "f",
}
_BRAHMIC_VOWEL_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
    0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai",
    0x49: "o", 0x4A: "o", 0x4B: "o", 0x4C: "au",
}
_BRAHMIC_INDEP_VOWELS_FULL = {
    0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "li",
    0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai",
    0x11: "o", 0x12: "o", 0x13: "o", 0x14: "au",
}
_BRAHMIC_VIRAMA_OFFSET = 0x4D    # halant: suppresses the following inherent vowel
_BRAHMIC_NUKTA_OFFSET = 0x3C     # combining dot for loan sounds; consonant already handled above
_BRAHMIC_ANUSVARA_OFFSET = 0x02
_BRAHMIC_CANDRABINDU_OFFSET = 0x01
_BRAHMIC_VISARGA_OFFSET = 0x03


def _brahmic_block_offset(ch: str, block_start: int):
    cp = ord(ch)
    return cp - block_start if block_start <= cp < block_start + 0x80 else None


def transliterate_brahmic_token(token: str) -> str:
    """
    Full phonetic transliteration of one word from any of the 8 Indic
    scripts this dataset uses into Latin letters - e.g. "प्राइवेट" -> "praivet",
    "ग्लोबल" -> "global". Every consonant carries an inherent "a" vowel
    unless followed by a vowel sign (replaces it) or a virama/halant
    (suppresses it) - that's the core rule of these scripts' writing
    systems, not something inferred from data. The single exception: the
    inherent vowel on a word's *final* consonant (when nothing follows it)
    is dropped too - Hindi/Indic phonology drops this "schwa" in speech,
    and these business names are transliterating English loanwords, whose
    spelling follows the spoken form (e.g. "ग्लोबल" is "global", not "globala").
    Non-Brahmic characters (digits, ASCII, punctuation) pass through
    unchanged, so this is safe to run on a token with attached punctuation.
    """
    out = []
    n = len(token)
    i = 0
    while i < n:
        block_start = next((s for s in _BRAHMIC_BLOCK_STARTS if s <= ord(token[i]) < s + 0x80), None)
        if block_start is None:
            out.append(token[i].lower())
            i += 1
            continue

        offset = _brahmic_block_offset(token[i], block_start)

        if offset in _BRAHMIC_CONSONANTS_FULL:
            base = _BRAHMIC_CONSONANTS_FULL[offset]
            consumed = 1
            next_offset = _brahmic_block_offset(token[i + 1], block_start) if i + 1 < n else None
            if next_offset == _BRAHMIC_NUKTA_OFFSET:
                # Nukta modifies this consonant's sound (already folded into
                # `base` above); skip it and keep looking for a vowel sign.
                consumed += 1
                next_offset = _brahmic_block_offset(token[i + 2], block_start) if i + 2 < n else None

            if next_offset == _BRAHMIC_VIRAMA_OFFSET:
                out.append(base)
                i += consumed + 1
            elif next_offset in _BRAHMIC_VOWEL_SIGNS:
                out.append(base + _BRAHMIC_VOWEL_SIGNS[next_offset])
                i += consumed + 1
            else:
                is_word_final = (i + consumed >= n)
                out.append(base if is_word_final else base + "a")
                i += consumed
            continue

        if offset in _BRAHMIC_INDEP_VOWELS_FULL:
            out.append(_BRAHMIC_INDEP_VOWELS_FULL[offset])
            i += 1
            continue

        if offset in (_BRAHMIC_ANUSVARA_OFFSET, _BRAHMIC_CANDRABINDU_OFFSET):
            out.append("n")
            i += 1
            continue

        if offset == _BRAHMIC_VISARGA_OFFSET:
            out.append("h")
            i += 1
            continue

        # Vowel sign / virama / nukta with no preceding consonant (malformed
        # input or a rare script edge case) - drop, same as phonetic_skeleton().
        i += 1

    return "".join(out)


def extract_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text (useful for typos & fuzzy matching)."""
    text = "".join(text.split())
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}
