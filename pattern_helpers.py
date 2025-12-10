"""Shared pattern helper functions for S40 / 164950 / family tagging.

This module centralises the small helper functions used by multiple scripts
(precise_bet_engine, slot_recalc_engine, etc.) to avoid NameError issues and to
keep the family cache consistent across the chain.
"""
from typing import Dict, List, Set

from utils_2digit import to_2d_str

try:
    import pattern_packs

    _PACKS_AVAILABLE = True
except Exception:
    pattern_packs = None  # type: ignore
    _PACKS_AVAILABLE = False

S40: Set[str] = {
    "00",
    "06",
    "07",
    "09",
    "15",
    "16",
    "18",
    "19",
    "24",
    "25",
    "27",
    "28",
    "33",
    "34",
    "36",
    "37",
    "42",
    "43",
    "45",
    "46",
    "51",
    "52",
    "54",
    "55",
    "60",
    "61",
    "63",
    "64",
    "70",
    "72",
    "73",
    "79",
    "81",
    "82",
    "88",
    "89",
    "90",
    "91",
    "97",
    "98",
}

FAMILY_164950_DIGITS: Set[str] = {"0", "1", "4", "5", "6", "9"}

_FAMILY_CACHE: Dict[str, List[str]] = {}


def is_s40_number(num_str: str) -> bool:
    """Return True if the 2-digit string belongs to the fixed S40 pack."""
    try:
        if _PACKS_AVAILABLE and hasattr(pattern_packs, "is_s40"):
            return bool(pattern_packs.is_s40(num_str))
        return to_2d_str(num_str) in S40
    except Exception:
        return False


def is_164950_number(num_str: str) -> bool:
    """Return True if both digits of the number are in the 164950 family set."""
    try:
        if _PACKS_AVAILABLE and hasattr(pattern_packs, "is_164950_family"):
            return bool(pattern_packs.is_164950_family(num_str))
        digits = to_2d_str(num_str)
        return digits[0] in FAMILY_164950_DIGITS and digits[1] in FAMILY_164950_DIGITS
    except Exception:
        return False


def _build_family_cache() -> None:
    if _FAMILY_CACHE:
        return

    for i in range(100):
        num = to_2d_str(i)
        tags: List[str] = []

        if is_s40_number(num):
            tags.append("S40")
        if is_164950_number(num):
            tags.append("FAMILY_164950")

        try:
            if _PACKS_AVAILABLE:
                if hasattr(pattern_packs, "get_number_families"):
                    extra_tags = pattern_packs.get_number_families(num)
                elif hasattr(pattern_packs, "get_digit_pack_tags"):
                    extra_tags = pattern_packs.get_digit_pack_tags(num)
                else:
                    extra_tags = []
                for fam in extra_tags:
                    fam_str = str(fam).strip()
                    if fam_str and fam_str not in tags:
                        tags.append(fam_str)
        except Exception:
            pass

        _FAMILY_CACHE[num] = tags


def get_families_for_number(num_str: str) -> List[str]:
    """
    Return a list of family names that contain the given number.

    The mapping is precomputed once for efficiency and reuses the existing
    pattern_packs tagging helpers to stay aligned with the 27 family universe.
    """
    _build_family_cache()
    try:
        num = to_2d_str(num_str)
    except Exception:
        return []
    return list(_FAMILY_CACHE.get(num, []))


# Eagerly build the cache at import time to keep downstream modules fast.
_build_family_cache()
