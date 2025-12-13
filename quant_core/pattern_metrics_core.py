from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

import pattern_packs
from quant_learning_core import slot_regime
from quant_stats_core import compute_pack_hit_stats
from script_hit_memory_utils import filter_hits_by_window, load_script_hit_memory

WINDOW_DAYS = 90
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def _load_windowed_memory(window_days: int, base_dir: Optional[Path] = None) -> pd.DataFrame:
    df = load_script_hit_memory(base_dir=base_dir)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["result_date"])
    if df.empty:
        return df

    df["slot"] = df.get("slot").astype(str).str.upper()
    df["real_number"] = df.get("real_number").astype(str)

    df, _ = filter_hits_by_window(df, window_days=window_days)
    if df.empty:
        return df

    df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce").dt.normalize()

    def _coerce_bool(series: pd.Series, default: bool = False) -> pd.Series:
        filled = series.copy()
        filled = filled.where(pd.notna(filled), default)
        lowered = filled.astype(str).str.lower()
        return lowered.isin({"true", "1", "yes", "y", "t"})

    df["is_exact_hit"] = _coerce_bool(df.get("is_exact_hit", False))
    df["is_s40"] = df["real_number"].apply(pattern_packs.is_s40)
    df["is_family_164950"] = df["real_number"].apply(pattern_packs.is_164950_family)
    return df


def _family_summary(df: pd.DataFrame, flag_col: str) -> Dict:
    total_rows = len(df)
    total_days = df["result_date"].dt.date.nunique() if not df.empty else 0
    hits_total = int(df[flag_col].sum()) if flag_col in df.columns else 0
    active_df = df[df[flag_col]] if flag_col in df.columns else pd.DataFrame()
    daily_cover_days = active_df["result_date"].dt.date.nunique() if not active_df.empty else 0
    daily_cover_pct = (daily_cover_days / total_days * 100.0) if total_days else 0.0

    per_slot: Dict[str, Dict[str, object]] = {}
    for slot, slot_df in df.groupby("slot"):
        slot_rows = len(slot_df)
        slot_hits = int(slot_df[flag_col].sum()) if flag_col in slot_df.columns else 0
        slot_hit_rate = slot_hits / slot_rows if slot_rows else 0.0
        slot_cover_days = slot_df[slot_df[flag_col]]["result_date"].dt.date.nunique() if flag_col in slot_df.columns else 0
        slot_total_days = slot_df["result_date"].dt.date.nunique() if slot_rows else 0
        per_slot[slot] = {
            "hits_total": slot_hits,
            "hit_rate": slot_hit_rate,
            "daily_cover_days": slot_cover_days,
            "daily_cover_total_days": slot_total_days,
            "daily_cover_pct": (slot_cover_days / slot_total_days * 100.0) if slot_total_days else 0.0,
            "regime": slot_regime(slot_hit_rate),
        }

    return {
        "hits_total": hits_total,
        "hit_rate": hits_total / total_rows if total_rows else 0.0,
        "daily_cover_days": daily_cover_days,
        "daily_cover_total_days": total_days,
        "daily_cover_pct": daily_cover_pct,
        "per_slot": per_slot,
    }


def compute_pattern_metrics(window_days: int = WINDOW_DAYS, base_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict]:
    df = _load_windowed_memory(window_days, base_dir=base_dir)
    pack_stats = compute_pack_hit_stats(window_days=window_days, base_dir=base_dir) or {}

    total_rows = int(pack_stats.get("total_rows", len(df)))
    result_days = int(pack_stats.get("days_total", df["result_date"].dt.date.nunique() if not df.empty else 0))
    exact_hits = (
        int(pd.to_numeric(df.get("is_exact_hit", False), errors="coerce").fillna(0).astype(int).sum())
        if not df.empty
        else 0
    )
    hit_rate_exact = exact_hits / total_rows if total_rows else 0.0

    summary = {
        "window_days": window_days,
        "total_rows": total_rows,
        "result_days": result_days,
        "exact_hits": exact_hits,
        "hit_rate_exact": hit_rate_exact,
    }

    def _slot_family_block(slot_stats: Dict[str, object], hit_field: str, rate_field: str, cover_field: str) -> Dict[str, object]:
        total_days_slot = int(slot_stats.get("days_total", 0) or 0)
        cover_days = int(slot_stats.get(cover_field, 0) or 0)
        hit_rate = float(slot_stats.get(rate_field, 0.0) or 0.0)
        return {
            "hits_total": int(slot_stats.get(hit_field, 0) or 0),
            "hit_rate": hit_rate,
            "daily_cover_days": cover_days,
            "daily_cover_total_days": total_days_slot,
            "daily_cover_pct": (cover_days / total_days_slot * 100.0) if total_days_slot else 0.0,
            "regime": slot_regime(hit_rate),
        }

    def _family_from_pack(
        family_key: str, hit_field: str, rate_field: str, cover_field: str, total_cover_key: str
    ) -> Dict[str, object]:
        fam_block = pack_stats.get(family_key, {}) if isinstance(pack_stats, dict) else {}
        hits = int(fam_block.get("hits", 0) or 0)
        hit_rate = float(fam_block.get("hit_rate", 0.0) or 0.0)
        cover_days = int(pack_stats.get(total_cover_key, 0) or 0)
        per_slot: Dict[str, Dict[str, object]] = {}
        for slot, slot_stats in (pack_stats.get("per_slot", {}) or {}).items():
            if not slot_stats:
                continue
            per_slot[slot] = _slot_family_block(slot_stats, hit_field, rate_field, cover_field)
        return {
            "hits_total": hits,
            "hit_rate": hit_rate,
            "daily_cover_days": cover_days,
            "daily_cover_total_days": result_days,
            "daily_cover_pct": (cover_days / result_days * 100.0) if result_days else 0.0,
            "per_slot": per_slot,
        }

    summary["s40"] = (
        _family_from_pack("S40", "s40_hits", "s40_rate", "s40_days", "days_with_s40")
        if pack_stats
        else _family_summary(df, "is_s40")
    )
    summary["family_164950"] = (
        _family_from_pack("FAMILY_164950", "fam_hits", "fam_rate", "fam_days", "days_with_fam")
        if pack_stats
        else _family_summary(df, "is_family_164950")
    )

    return df, summary
