"""Helpers for consuming quant_reality_pnl.json slump signals."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import json

import quant_paths


@dataclass
class SlotHealth:
    slot: str
    roi_pct: float
    wins: int
    losses: int
    hit_rate: float
    last_hit_date: Optional[str]
    longest_losing_streak: int
    current_losing_streak: int
    others_avg_roi_pct: float
    roi_diff_vs_others_pct: float


def find_latest_pnl_json() -> Optional[Path]:
    """Locate the most recent quant_reality_pnl JSON file."""
    base_dir = quant_paths.get_base_dir()
    perf_dir = base_dir / "logs" / "performance"

    if not perf_dir.exists():
        return None

    pnl_files = list(perf_dir.glob("quant_reality_pnl*.json"))
    if not pnl_files:
        return None

    return max(pnl_files, key=lambda f: f.stat().st_mtime)


def load_latest_pnl_json() -> Dict[str, Any]:
    """Load and return the latest P&L JSON content with friendly errors."""
    latest_file = find_latest_pnl_json()
    if not latest_file:
        raise FileNotFoundError(
            f"No quant_reality_pnl*.json found under {quant_paths.get_base_dir() / 'logs' / 'performance'}"
        )

    try:
        with open(latest_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {latest_file}: {exc}")


def _compute_others_roi(slot: str, data: Dict[str, Any]) -> float:
    slot_key = slot.upper()
    rois = [row.get("roi_pct", 0.0) for row in data.get("by_slot", []) if row.get("slot") != slot_key]
    return sum(rois) / len(rois) if rois else 0.0


def get_slot_health(slot: str, data: Optional[Dict[str, Any]] = None) -> SlotHealth:
    """Return numeric P&L and slump diagnostics for a given slot."""

    slot_key = slot.upper()
    pnl_data = data if data is not None else load_latest_pnl_json()

    slot_row = next((row for row in pnl_data.get("by_slot", []) if row.get("slot") == slot_key), None)
    if not slot_row:
        raise ValueError(f"Slot '{slot_key}' not found in by_slot section of the latest P&L JSON.")

    slump_diag = pnl_data.get("slot_slump_diagnostics", {}).get(slot_key, {})

    roi_pct = float(slot_row.get("roi_pct", slump_diag.get("slot_roi_pct", 0.0)))
    wins = int(slot_row.get("wins", slot_row.get("main_hit", 0) or 0))
    losses = int(slot_row.get("losses", 0))

    hit_rate = slot_row.get("hit_rate")
    if hit_rate is None:
        total_outcomes = wins + losses
        hit_rate = (wins / total_outcomes) if total_outcomes else 0.0

    others_avg = float(slump_diag.get("others_avg_roi_pct", _compute_others_roi(slot_key, pnl_data)))
    roi_diff = float(slump_diag.get("roi_diff_vs_others_pct", roi_pct - others_avg))

    return SlotHealth(
        slot=slot_key,
        roi_pct=roi_pct,
        wins=wins,
        losses=losses,
        hit_rate=float(hit_rate),
        last_hit_date=slump_diag.get("last_hit_date"),
        longest_losing_streak=int(slump_diag.get("longest_losing_streak", 0) or 0),
        current_losing_streak=int(slump_diag.get("current_losing_streak", 0) or 0),
        others_avg_roi_pct=others_avg,
        roi_diff_vs_others_pct=roi_diff,
    )


def is_slot_in_slump(
    slot: str,
    roi_threshold: float = -30.0,
    losing_streak_min: int = 2,
    data: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Return True if the slot looks 'in slump' according to simple rules:
    - ROI below roi_threshold, and
    - current losing streak >= losing_streak_min.
    Uses SlotHealth derived from the JSON.
    """

    health = get_slot_health(slot, data=data)
    return health.roi_pct <= roi_threshold and health.current_losing_streak >= losing_streak_min


def _main():
    try:
        pnl_data = load_latest_pnl_json()
    except Exception as exc:
        print(f"❌ {exc}")
        return 1

    slots = []
    if pnl_data.get("slot_slump_diagnostics"):
        slots = list(pnl_data["slot_slump_diagnostics"].keys())
    elif pnl_data.get("by_slot"):
        slots = [row.get("slot") for row in pnl_data.get("by_slot", []) if row.get("slot")]

    if not slots:
        print("❌ No slot data available in the latest P&L JSON.")
        return 1

    print("=== SLOT HEALTH SNAPSHOT (from latest quant_reality_pnl.json) ===")
    for slot in slots:
        health = get_slot_health(slot, data=pnl_data)
        slump_flag = is_slot_in_slump(slot, data=pnl_data)
        print(
            f"{health.slot}: ROI={health.roi_pct:+.1f}%, wins={health.wins}, "
            f"losses={health.losses}, hit_rate={health.hit_rate:.2f}, "
            f"current_streak={health.current_losing_streak}, slump={str(slump_flag)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
