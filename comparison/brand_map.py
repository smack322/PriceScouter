# comparison/brand_map.py
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from typing import Optional, Dict

# -----------------------------
# Core normalization utilities
# -----------------------------

_PUNCT_RE = re.compile(r"[^a-z0-9\s]+", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")
_DOT_RE = re.compile(r"\.(?!\d)")  # remove dots in acronyms, keep decimal numbers

# Common corporate/legal suffixes to strip from brand tails
_SUFFIXES = [
    "incorporated", "inc", "corp", "corporation", "co", "company",
    "llc", "ltd", "limited", "plc", "ag", "gmbh", "sa", "bv", "oy",
    "pty", "pte", "srl", "sro", "kk", "sas", "spa", "ab", "aps", "as",
    "usa", "us", "international", "intl", "group", "holdings", "tm", "®", "©"
]
_SUFFIX_TOKENS = set(_SUFFIXES)

# Heuristic stopwords for brand tokens (after normalization)
_BRAND_STOPWORDS = {"the", "and", "of"}

# Canonical aliases: left side = normalized key AFTER base_normalize()
# Right side = canonical, **also** in normalized form (spaces only, no punctuation)
# Keep these lowercase and punctuation-free to match canonicalization output.
BRAND_ALIASES: Dict[str, str] = {
    # Procter & Gamble
    "p g": "procter gamble",
    "pg": "procter gamble",
    "procter and gamble": "procter gamble",
    "procter & gamble": "procter gamble",  # tolerated (we normalize '&')

    # HP variants
    "hewlett packard": "hp",
    "hp inc": "hp",

    # Apple
    "apple inc": "apple",

    # Unilever
    "unilever plc": "unilever",

    # Nestlé
    "nestle s a": "nestle",
    "nestle sa": "nestle",

    # 3M
    "3 m": "3m",

    # P&G sub-brands mapped to the parent only if you intend to cluster at parent level.
    # If you want distinct clustering for sub-brands, remove/comment these.
    # "gillette": "procter gamble",
    # "tide": "procter gamble",

    # Misc retail/private label canonicalizations
    "amazon basics": "amazon basics",
    "walmart": "walmart",
    "target": "target",
}

# Optional vendor-specific corrections (normalized inbound -> canonical normalized)
# Use this to override known quirks per source, e.g. Keepa or eBay brand fields.
VENDOR_BRAND_OVERRIDES: Dict[str, Dict[str, str]] = {
    # Example:
    # "amazon": {"hp inc": "hp"},
    # "ebay":   {"hewlett-packard": "hp"},
}

# Pretty-case map for UI; keys must be **canonical normalized** strings.
CANONICAL_DISPLAY_CASE: Dict[str, str] = {
    "procter gamble": "Procter & Gamble",
    "hp": "HP",
    "apple": "Apple",
    "unilever": "Unilever",
    "nestle": "Nestlé",
    "3m": "3M",
    "amazon basics": "Amazon Basics",
    "walmart": "Walmart",
    "target": "Target",
}

# --------------------------------
# Public API
# --------------------------------

def display_brand(canonical_normalized: Optional[str]) -> Optional[str]:
    """
    Convert canonical normalized brand to a pretty display string for UI.
    Returns the input if no special case exists.
    """
    if canonical_normalized is None:
        return None
    return CANONICAL_DISPLAY_CASE.get(canonical_normalized, _title_case_safe(canonical_normalized))

def same_brand(a: Optional[str], b: Optional[str], vendor: Optional[str] = None) -> bool:
    """
    Brand equality after canonicalization.
    """
    return canonicalize_brand(a, vendor) == canonicalize_brand(b, vendor)

def canonicalize_brand(raw: Optional[str], vendor: Optional[str] = None) -> Optional[str]:
    """
    Main entry point: returns a canonical **normalized** brand or None.

    - Strips punctuation, legal suffixes
    - Collapses whitespace, lowercases
    - Expands & → 'and' during normalization
    - Applies vendor-specific and global alias maps
    - Returns None for empty/unknown brands
    """
    if raw is None:
        return None

    s = raw.strip()
    if not s:
        return None

    # Stage 1: base normalization (lowercase, dedot acronyms, '&' handling, strip punct, collapse ws)
    norm = _base_normalize_brand(s)

    if not norm:
        return None

    # Stage 2: strip trailing corporate/legal terms (tokens)
    norm = _strip_legal_suffixes(norm)

    if not norm:
        return None

    # Stage 3: drop trivial stopwords-only results
    toks = [t for t in norm.split() if t not in _BRAND_STOPWORDS]
    if not toks:
        return None
    norm = " ".join(toks)

    # Stage 4: vendor overrides (after normalization)
    if vendor:
        vmap = VENDOR_BRAND_OVERRIDES.get(vendor.lower())
        if vmap:
            if norm in vmap:
                norm = vmap[norm]

    # Stage 5: global alias map
    alias = BRAND_ALIASES.get(norm)
    if alias:
        norm = alias

    # Final sanity: single space separation
    norm = _WS_RE.sub(" ", norm).strip()

    return norm or None

# --------------------------------
# Internal helpers
# --------------------------------

def _ascii_fold(text: str) -> str:
    # Turn accents/diacritics into closest ASCII (é -> e, ñ -> n, ™ -> removed)
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

@lru_cache(maxsize=4096)
def _base_normalize_brand(s: str) -> str:
    s = s.lower()

    # Remove common legal marks BEFORE folding so they don't become 'tm'/'r' text
    # Include: ™, ®, ©, ℠, and parentheses variants some sources use
    s = (s.replace("™", " ")
           .replace("®", " ")
           .replace("©", " ")
           .replace("℠", " "))

    s = s.replace("&", " and ")
    s = _DOT_RE.sub("", s)           # 'S.A.' -> 'SA'
    s = _ascii_fold(s)               # 'nestlé' -> 'nestle' (no new 'tm' now)

    # Safety: strip trailing textual marks like '(tm)', 'tm', 'r', 'sm'
    s = re.sub(r"\b(?:tm|sm|r)\b", " ", s)     # standalone tokens
    s = re.sub(r"\((?:tm|sm|r)\)", " ", s)     # parenthesized forms

    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

def _strip_legal_suffixes(norm: str) -> str:
    """
    Remove common corporate/legal suffix tokens when they appear at the tail.
    Also removes repeated tails (e.g., 'hp inc co' -> 'hp').
    """
    toks = norm.split()
    # Work from the end backwards, stripping suffix tokens
    i = len(toks) - 1
    while i >= 0 and toks[i] in _SUFFIX_TOKENS:
        i -= 1
    if i < 0:
        return ""
    return " ".join(toks[: i + 1])

def _title_case_safe(norm: str) -> str:
    """
    A safe title-case that respects common all-caps brands like '3M'/'HP'.
    Uses simple heuristics; override via CANONICAL_DISPLAY_CASE for exceptions.
    """
    # Common heuristics: all letters short -> uppercase (hp -> HP), mixed tokens -> Title
    if norm.isalpha() and len(norm) <= 3:
        return norm.upper()
    # Keep numerics attached (e.g., '3m' -> '3M')
    parts = []
    for t in norm.split():
        if len(t) <= 3:
            parts.append(t.upper())
        else:
            parts.append(t.capitalize())
    return " ".join(parts)

# --------------------------------
# Mutation API (optional, for bootstrapping from data)
# --------------------------------

def add_alias(src_normalized: str, canonical_normalized: str) -> None:
    """
    Programmatically add/override an alias at runtime (e.g., during data cleanup scripts).
    Both args must already be normalized forms.
    """
    BRAND_ALIASES[src_normalized] = canonical_normalized

def set_display_case(canonical_normalized: str, display: str) -> None:
    """
    Programmatically set/override pretty-case for a canonical brand.
    """
    CANONICAL_DISPLAY_CASE[canonical_normalized] = display
