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
    # France
    r"sarl|sas|sasu|sa|eurl|sci|snc"
    r")\b",
    re.IGNORECASE,
)

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
    # French
    r"\br\b": "rue",
    r"\bbd\b": "boulevard",
    r"\bbvd\b": "boulevard",
    r"\bav\b": "avenue",
    r"\ball\b": "allee",
    r"\bimp\b": "impasse",
    r"\bpl\b": "place",
    r"\bche\b": "chemin",
}


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters (NFKC) and collapse extra whitespace."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def clean_basic(text: str) -> str:
    """Lowercase, strip non-alphanumeric (except basic punctuation), and normalize whitespace."""
    text = normalize_unicode(text).lower()
    # Replace common separators with spaces
    text = re.sub(r"[,/\\|_\-\(\)\[\]\{\}\.:;\"\'`]", " ", text)
    return " ".join(text.split())


def normalize_business_name(name: str, strip_legal_suffix: bool = True) -> str:
    """Normalize business name, optionally stripping legal suffixes."""
    cleaned = clean_basic(name)
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


def extract_character_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text (useful for typos & fuzzy matching)."""
    text = "".join(text.split())
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}
