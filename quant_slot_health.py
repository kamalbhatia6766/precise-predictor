"""Slot health loader for quant_reality_pnl signals.

This module centralizes reading the latest P&L summary JSON and exposes
safe accessors that never crash the calling engine. It is intentionally
lightweight and defensive so it can be used by CLI utilities and the bet
engine without altering any staking or selection logic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class SlotHealth:
    slot: str  # "FRBD", "GZBD", "GALI", "DSWR"
    roi_percent: float  # ROI for this slot, default 0.0 if missing
    wins: int
    losses: int
    hit_rate: float
    current_streak: int
    slump: bool  # True if slot is in slump (from JSON)
    roi_bucket: str  # "HIGH", "MID", "LOW", or "UNKNOWN"


_cached_slot_health: Optional[Dict[str, SlotHealth]] = None


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _compute_roi_bucket(roi_percent: float) -> str:
    if roi_percent >= 200:
        return "HIGH"
    if roi_percent >= 0:
        return "MID"
    if roi_percent < 0:
        return "LOW"
    return "UNKNOWN"


def _extract_slot_records(data: Dict) -> Dict[str, Dict]:
    """Normalize possible JSON shapes into a slot->dict map."""
    slot_map: Dict[str, Dict] = {}

    by_slot = data.get("by_slot")
    if isinstance(by_slot, list):
        for entry in by_slot:
            slot_key = str(entry.get("slot", "")).upper()
            if slot_key:
                slot_map[slot_key] = entry

    for key, value in data.items():
        key_upper = str(key).upper()
        if key_upper in {"FRBD", "GZBD", "GALI", "DSWR"} and isinstance(value, dict):
            slot_map.setdefault(key_upper, value)

    return slot_map


def load_slot_health(json_path: str = "logs/performance/quant_reality_pnl.json") -> Dict[str, SlotHealth]:
    """
    Read JSON summary file and return a dict:
        { "FRBD": SlotHealth(...),
          "GZBD": SlotHealth(...),
          "GALI": SlotHealth(...),
          "DSWR": SlotHealth(...) }
    If file missing or corrupted, return an empty dict.
    """

    path = Path(json_path)
    if not path.exists():
        print(f"[quant_slot_health] Warning: JSON file not found at {path}")
        return {}

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as exc:  # broad catch to avoid crashing the engine
        print(f"[quant_slot_health] Warning: Unable to parse {path}: {exc}")
        return {}

    slot_records = _extract_slot_records(data if isinstance(data, dict) else {})
    slot_health: Dict[str, SlotHealth] = {}

    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        record = slot_records.get(slot, {})
        roi_percent = _coerce_float(record.get("roi_percent", record.get("roi_pct", 0.0)))
        wins = _coerce_int(record.get("wins", record.get("main_hit", 0)))
        losses = _coerce_int(record.get("losses", 0))
        hit_rate = _coerce_float(record.get("hit_rate", 0.0))
        current_streak = _coerce_int(record.get("current_streak", record.get("current_losing_streak", 0)))
        slump = bool(record.get("slump", record.get("in_slump", False)))
        roi_bucket = _compute_roi_bucket(roi_percent)

        slot_health[slot] = SlotHealth(
            slot=slot,
            roi_percent=roi_percent,
            wins=wins,
            losses=losses,
            hit_rate=hit_rate,
            current_streak=current_streak,
            slump=slump,
            roi_bucket=roi_bucket,
        )

    return slot_health


def get_slot_health(slot: str, default_bucket: str = "UNKNOWN") -> SlotHealth:
    """
    Shortcut: load the JSON once (internally cache it in a module-level variable)
    and return a SlotHealth object for the requested slot.
    If the slot is missing, return a SlotHealth with neutral values:
        roi_percent=0.0, wins=0, losses=0, hit_rate=0.0,
        current_streak=0, slump=False, roi_bucket=default_bucket
    """

    global _cached_slot_health
    if _cached_slot_health is None:
        _cached_slot_health = load_slot_health()

    slot_key = str(slot).upper()
    if _cached_slot_health and slot_key in _cached_slot_health:
        return _cached_slot_health[slot_key]

    print(f"[quant_slot_health] Warning: Slot '{slot_key}' missing in slot health data; using defaults.")
    return SlotHealth(
        slot=slot_key,
        roi_percent=0.0,
        wins=0,
        losses=0,
        hit_rate=0.0,
        current_streak=0,
        slump=False,
        roi_bucket=default_bucket,
    )

