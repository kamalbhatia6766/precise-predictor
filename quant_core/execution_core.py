from __future__ import annotations

from typing import Dict

import pandas as pd

from quant_core.config_core import STAKE_MULTIPLIERS


def build_weighted_script_weights(
    metrics_df: pd.DataFrame,
    window_days: int = 30,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    weights: Dict[str, Dict[str, float]] = {}
    if metrics_df is None or metrics_df.empty:
        return {"window_days": window_days, "per_slot": weights}

    for slot_name, slot_df in metrics_df.groupby("slot", dropna=False):
        slot_weights: Dict[str, float] = {}
        ranked = slot_df.sort_values("score", ascending=False).copy()
        top_score = ranked["score"].max()
        median_score = ranked["score"].median()
        bottom_score = ranked["score"].min()

        for _, row in ranked.iterrows():
            script_id = str(row.get("script_id", "")).upper()
            score = float(row.get("score", 0.0))

            if pd.isna(score):
                score = 0.0

            if not pd.isna(top_score) and score >= float(top_score) - 0.005:
                w = 1.8
            elif not pd.isna(median_score) and score >= float(median_score):
                w = 1.3
            elif not pd.isna(bottom_score) and score > float(bottom_score) + 0.001:
                w = 0.9
            else:
                w = 0.6

            slot_weights[script_id] = w

        weights[str(slot_name)] = slot_weights

    return {"window_days": window_days, "per_slot": weights}


def apply_regime_to_stakes(slot_confidence: float, slot_regime: str) -> float:
    mult = STAKE_MULTIPLIERS.get(slot_regime.upper(), 1.0)
    return max(0.0, slot_confidence * mult)
