from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import pattern_packs
from pattern_intelligence_engine import PatternIntelligenceEngine
from pattern_intelligence_enhanced import PatternIntelligenceEnhanced
from script_hit_memory_utils import load_script_hit_memory


DEFAULT_PATTERN_WINDOW = 90


def _ensure_hit_df(hit_df: Optional[pd.DataFrame], window_days: int) -> pd.DataFrame:
    if hit_df is not None and not hit_df.empty:
        df = hit_df.copy()
    else:
        df = load_script_hit_memory(base_dir=Path(__file__).resolve().parent.parent)
    if df.empty:
        return pd.DataFrame()
    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
    df = df.dropna(subset=["result_date"])
    latest = df["result_date"].max()
    if pd.isna(latest):
        return pd.DataFrame()
    cutoff = latest - pd.to_timedelta(window_days - 1, unit="D")
    df = df[df["result_date"] >= cutoff]
    df["real_number"] = pd.to_numeric(df.get("real_number"), errors="coerce")
    df["slot"] = df.get("slot").astype(str)
    df["is_exact_hit"] = df.get("is_exact_hit", False).astype(bool)
    df["is_near_miss"] = df.get("is_near_miss", False).astype(bool)
    return df


def run_basic_pattern_intel(hit_df: Optional[pd.DataFrame] = None, window_days: int = DEFAULT_PATTERN_WINDOW) -> Dict[str, Dict[str, float]]:
    engine = PatternIntelligenceEngine(window_days=window_days)
    df = _ensure_hit_df(hit_df, window_days)
    if df.empty:
        return {}
    return engine.analyse(df)


def run_enhanced_pattern_intel(hit_df: Optional[pd.DataFrame] = None, window_days: int = 120) -> Dict[str, Dict]:
    engine = PatternIntelligenceEnhanced(window_days=window_days)
    df = _ensure_hit_df(hit_df, window_days)
    if df.empty:
        return {}
    scripts = engine.summarise_scripts(df)
    slots = engine.summarise_slots(df)
    result = {
        "scripts": scripts,
        "slots": slots,
        "top_script": max(scripts.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0)) if scripts else None,
        "weak_script": min(scripts.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0)) if scripts else None,
        "top_slot": max(slots.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0)) if slots else None,
        "weak_slot": min(slots.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0)) if slots else None,
    }
    return result


def _pack_members(pack_id: str) -> List[str]:
    if pack_id == "S40":
        return [f"{n:02d}" for n in pattern_packs.S40_STRINGS]
    if pack_id == "PACK_164950":
        return [f"{a}{b}" for a in pattern_packs.FAMILY_164950 for b in pattern_packs.FAMILY_164950]
    if pack_id.startswith("PACK_"):
        try:
            start, end = pack_id.replace("PACK_", "").split("_")
            start_n = int(start)
            end_n = int(end)
            return [f"{n:02d}" for n in range(start_n, end_n + 1)]
        except Exception:
            return []
    return []


def build_pattern_config(
    hit_df: Optional[pd.DataFrame] = None,
    window_days: int = 120,
    out_path: str = "config/pattern_packs_auto.json",
) -> Dict:
    stats = run_basic_pattern_intel(hit_df=hit_df, window_days=window_days)
    packs = []
    for pack_id, pack_stats in stats.items():
        packs.append(
            {
                "id": pack_id,
                "label": pack_id,
                "members": _pack_members(pack_id),
                "hit_rate": pack_stats.get("hit_rate_exact"),
                "near_rate": pack_stats.get("near_miss_rate"),
                "best_slot": pack_stats.get("best_slot"),
                "weak_slot": pack_stats.get("weak_slot"),
                "status": "ON",
            }
        )
    config = {"version": "v001", "window_days": window_days, "packs": packs}
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
    return config
