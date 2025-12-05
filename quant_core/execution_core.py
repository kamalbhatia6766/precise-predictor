from __future__ import annotations

from typing import Dict

import pandas as pd

from quant_core.config_core import STAKE_MULTIPLIERS


def build_weighted_script_weights(script_metrics_df: pd.DataFrame, window_days: int = 30) -> Dict[str, Dict[str, float]]:
    weights: Dict[str, Dict[str, float]] = {}
    if script_metrics_df is None or script_metrics_df.empty:
        return weights
    for slot, slot_df in script_metrics_df.groupby("slot", dropna=False):
        slot_weights: Dict[str, float] = {}
        for _, row in slot_df.iterrows():
            script_id = str(row.get("script_id", "")).upper()
            base = 1.0
            exact = float(row.get("hit_rate_exact", 0.0))
            score = float(row.get("score", 0.0))
            if row.get("exact_hits", 0) > 0 and score > 0:
                base += min(0.5, score)
            if row.get("exact_hits", 0) == 0 and row.get("near_hits", 0) > 5:
                base -= 0.3
            if base < 0:
                base = 0.0
            slot_weights[script_id] = base
        weights[slot] = slot_weights
    return weights


def apply_regime_to_stakes(slot_confidence: float, slot_regime: str) -> float:
    mult = STAKE_MULTIPLIERS.get(slot_regime.upper(), 1.0)
    return max(0.0, slot_confidence * mult)
