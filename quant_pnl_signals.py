"""Helpers for consuming quant_reality_pnl.json slump signals."""
from typing import Dict, Optional

from quant_slot_health import SlotHealth, get_slot_health, load_slot_health
import quant_paths


DEFAULT_JSON_PATH = "logs/performance/quant_reality_pnl.json"


def read_slot_health_snapshot(json_path: str = DEFAULT_JSON_PATH) -> Dict[str, SlotHealth]:
    """
    Thin wrapper around quant_slot_health.load_slot_health so that other tools can reuse it.
    Returns a mapping of slot name to SlotHealth.
    """

    return load_slot_health(json_path)


def is_slot_in_slump(
    slot: str,
    roi_threshold: float = -30.0,
    losing_streak_min: int = 2,
    data: Optional[Dict[str, SlotHealth]] = None,
) -> bool:
    """
    Return True if the slot looks 'in slump'.
    Prefers the slump flag from SlotHealth if present; otherwise, falls back to
    threshold-based evaluation using ROI and streak.
    """

    health_map = data if data is not None else None
    if health_map is None:
        health = get_slot_health(slot)
    else:
        health = health_map.get(slot.upper()) or get_slot_health(slot)

    if health.slump:
        return True

    return health.roi_percent <= roi_threshold and health.current_streak >= losing_streak_min


def _main():
    slot_health_map = read_slot_health_snapshot()
    if not slot_health_map:
        print(
            f"❌ No quant_reality_pnl data available under {quant_paths.get_base_dir() / DEFAULT_JSON_PATH}"
        )
        return 1

    slots = [slot for slot in ["FRBD", "GZBD", "GALI", "DSWR"] if slot in slot_health_map]
    if not slots:
        slots = list(slot_health_map.keys())

    print("=== SLOT HEALTH SNAPSHOT (from latest quant_reality_pnl.json) ===")
    for slot in slots:
        health = slot_health_map.get(slot) or get_slot_health(slot)
        slump_flag = is_slot_in_slump(slot, data=slot_health_map)
        print(
            f"{health.slot}: ROI={health.roi_percent:+.1f}%, wins={health.wins}, "
            f"losses={health.losses}, hit_rate={health.hit_rate:.2f}, "
            f"current_streak={health.current_streak}, slump={str(slump_flag)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
