from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

import quant_paths
from script_hit_memory_utils import classify_relation, filter_hits_by_window, load_script_hit_memory


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


def compute_pack_hit_stats(window_days: int = 30, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    df = load_script_hit_memory(base_dir=base_dir)
    if df.empty:
        return {}

    df = df.copy()
    if "result_date" not in df.columns:
        raise ValueError("compute_pack_hit_stats: 'result_date' column missing from script-hit memory")

    df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce")
    df = df.dropna(subset=["result_date"])
    if not pd.api.types.is_datetime64_any_dtype(df["result_date"]):
        df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce")
        df = df.dropna(subset=["result_date"])

    if df.empty:
        return {}

    df["result_date"] = df["result_date"].dt.normalize()
    df["slot"] = df.get("slot", "").astype(str).str.upper()

    window_df, _ = filter_hits_by_window(df, window_days=window_days)
    if window_df.empty:
        return {}

    # ``filter_hits_by_window`` converts the date column to ``datetime.date`` to
    # perform window slicing. Convert it back to a normalised ``datetime64``
    # series so downstream ``.dt`` accessors (for unique-day counts, etc.) are
    # safe and consistent.
    window_df["result_date"] = pd.to_datetime(window_df["result_date"], errors="coerce").dt.normalize()
    window_df = window_df.dropna(subset=["result_date"])

    window_df["is_s40"] = pd.to_numeric(window_df.get("is_s40", 0), errors="coerce").fillna(0).astype(int)
    window_df["is_family_164950"] = pd.to_numeric(window_df.get("is_family_164950", 0), errors="coerce").fillna(0).astype(int)

    total_rows = len(window_df)
    days_total = window_df["result_date"].dt.date.nunique()
    s40_hits = int(window_df["is_s40"].sum())
    fam_hits = int(window_df["is_family_164950"].sum())
    s40_days = window_df.loc[window_df["is_s40"] > 0, "result_date"].dt.date.nunique()
    fam_days = window_df.loc[window_df["is_family_164950"] > 0, "result_date"].dt.date.nunique()

    summary: Dict[str, Any] = {
        "total_rows": total_rows,
        "days_total": int(days_total),
        "S40": {"hits": s40_hits, "hit_rate": s40_hits / total_rows if total_rows else 0.0},
        "FAMILY_164950": {"hits": fam_hits, "hit_rate": fam_hits / total_rows if total_rows else 0.0},
        "per_slot": {},
        "days_with_s40": int(s40_days),
        "days_with_fam": int(fam_days),
    }

    for slot, slot_df in window_df.groupby("slot"):
        if slot_df.empty:
            continue
        total_slot = len(slot_df)
        s40_slot_hits = int(slot_df["is_s40"].sum())
        fam_slot_hits = int(slot_df["is_family_164950"].sum())
        summary["per_slot"][slot] = {
            "total": total_slot,
            "s40_hits": s40_slot_hits,
            "fam_hits": fam_slot_hits,
            "s40_rate": s40_slot_hits / total_slot if total_slot else 0.0,
            "fam_rate": fam_slot_hits / total_slot if total_slot else 0.0,
            "days_total": int(slot_df["result_date"].dt.date.nunique()),
            "s40_days": int(slot_df.loc[slot_df["is_s40"] > 0, "result_date"].dt.date.nunique()),
            "fam_days": int(slot_df.loc[slot_df["is_family_164950"] > 0, "result_date"].dt.date.nunique()),
        }

    return summary


def compute_script_slot_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "total_predictions",
                "exact_hits",
                "near_hits",
                "hit_rate_exact",
                "near_miss_rate",
                "blind_miss_rate",
                "score",
            ]
        )

    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, Any] = {col: val for col, val in zip(group_cols, key_values)}

        total_predictions = len(group)
        relations = group.get("_relation") if "_relation" in group.columns else None
        if relations is None:
            relations = group.apply(lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1)

        exact_hits = int((relations == "EXACT").sum())
        near_hits = int(relations.isin({"MIRROR", "ADJACENT", "DIAGONAL_11", "REVERSE_CARRY", "SAME_DIGIT_COOL"}).sum())

        if total_predictions == 0:
            hit_rate_exact = near_miss_rate = blind_miss_rate = 0.0
            score = 0.0
            blind_misses = 0
        else:
            blind_misses = max(total_predictions - exact_hits - near_hits, 0)
            hit_rate_exact = exact_hits / total_predictions
            near_miss_rate = near_hits / total_predictions
            blind_miss_rate = max(blind_misses / total_predictions, 0)
            score = 120.0 * hit_rate_exact + 40.0 * near_miss_rate - 30.0 * blind_miss_rate

        rows.append(
            {
                **record,
                "total_predictions": int(total_predictions),
                "exact_hits": int(exact_hits),
                "near_hits": int(near_hits),
                "hit_rate_exact": hit_rate_exact,
                "near_miss_rate": near_miss_rate,
                "blind_miss_rate": blind_miss_rate,
                "score": score,
            }
        )

    return pd.DataFrame(rows)

