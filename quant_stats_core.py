from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

import quant_paths


def _load_ultimate_performance() -> pd.DataFrame:
    path = quant_paths.get_performance_logs_dir() / "ultimate_performance.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "result_date" in df.columns:
        df["date"] = pd.to_datetime(df["result_date"], errors="coerce")
    else:
        return pd.DataFrame()
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()
    df["slot"] = df.get("slot", "").astype(str).str.upper()
    return df


def compute_topn_roi(window_days: int = 30) -> Dict:
    df = _load_ultimate_performance()
    if df.empty:
        return {}

    latest_date = df["date"].max()
    cutoff = latest_date - timedelta(days=window_days - 1)
    window_df = df[df["date"] >= cutoff].copy()
    if window_df.empty:
        return {}

    window_start = window_df["date"].min().date()
    window_end = window_df["date"].max().date()
    available_days = window_df["date"].dt.date.nunique()

    topn_flags = {n: f"hit_top{n}" for n in range(1, 11)}

    def _roi_for_subset(subset: pd.DataFrame) -> Dict[int, float]:
        roi_map: Dict[int, float] = {}
        for n, flag_col in topn_flags.items():
            raw_flags = subset.get(flag_col, None)
            stake_rows = len(raw_flags) if raw_flags is not None else 0
            stake = stake_rows * n
            hits = 0
            if stake and hasattr(raw_flags, "fillna"):
                flags = raw_flags.fillna(False).astype(bool)
                hits = int(flags.sum())
            payout = hits * 90
            roi = ((payout - stake) / stake * 100.0) if stake else 0.0
            roi_map[n] = roi
        return roi_map

    roi_by_n = _roi_for_subset(window_df)
    best_N = max(roi_by_n, key=lambda k: roi_by_n[k]) if roi_by_n else None
    best_roi = roi_by_n.get(best_N) if best_N is not None else None

    per_slot: Dict[str, Dict[str, Dict[int, float]]] = {}
    for slot, slot_df in window_df.groupby("slot"):
        per_slot[slot] = {"roi_by_N": _roi_for_subset(slot_df)}

    return {
        "window_start": window_start,
        "window_end": window_end,
        "available_days": available_days,
        "overall": {"best_N": best_N, "best_roi": best_roi, "roi_by_N": roi_by_n},
        "per_slot": per_slot,
    }

