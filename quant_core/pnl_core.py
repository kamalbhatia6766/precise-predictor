from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json

from quant_core.config_core import ROI_SLUMP_THRESHOLD, ROI_STRONG_THRESHOLD


PERFORMANCE_PATH = Path("logs") / "performance" / "quant_reality_pnl.json"


def load_pnl_snapshot(path: Path = PERFORMANCE_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def classify_slot_regime(pnl_snapshot: Optional[Dict[str, Any]], slot_name: str) -> str:
    if not pnl_snapshot:
        return "NORMAL"
    by_slot = pnl_snapshot.get("by_slot") or []
    for entry in by_slot:
        if str(entry.get("slot", "")).upper() == slot_name.upper():
            try:
                roi_val = float(entry.get("roi", 0.0))
            except Exception:
                roi_val = 0.0
            if roi_val >= ROI_STRONG_THRESHOLD:
                return "STRONG"
            if roi_val <= ROI_SLUMP_THRESHOLD:
                return "SLUMP"
            return "NORMAL"
    return "NORMAL"
