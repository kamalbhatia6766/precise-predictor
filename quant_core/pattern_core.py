from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

import pattern_packs
from pattern_intelligence_enhanced import PatternIntelligenceEnhanced


DEFAULT_PATTERN_WINDOW = 90
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def _ensure_hit_df(hit_df: Optional[pd.DataFrame], window_days: int) -> pd.DataFrame:
    window_days = int(window_days)
    if hit_df is None:
        from . import hit_core

        hit_df = hit_core.rebuild_hit_memory(window_days=window_days)

    df = hit_df.copy()

    if df.empty:
        return pd.DataFrame()

    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

    if "result_date" in df.columns:
        df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
        df = df.dropna(subset=["result_date"])
        latest = df["result_date"].max()
        if pd.notna(latest):
            cutoff = latest - pd.to_timedelta(window_days - 1, unit="D")
            df = df[df["result_date"] >= cutoff]

    df["real_number"] = pd.to_numeric(df.get("real_number"), errors="coerce")
    df = df.dropna(subset=["real_number"])
    df["real_number"] = df["real_number"].astype(int)
    df["slot"] = df.get("slot").astype(str)

    if "is_exact_hit" in df.columns:
        df["is_exact_hit"] = df["is_exact_hit"].astype(bool)
    else:
        df["is_exact_hit"] = False

    near_col = None
    for candidate in ["is_near_hit", "is_neighbour_hit", "is_neighbor_hit", "is_near_miss"]:
        if candidate in df.columns:
            near_col = candidate
            break

    if near_col is not None:
        df["is_near_hit"] = df[near_col].astype(bool)
    else:
        df["is_near_hit"] = False

    return df


def run_basic_pattern_intel(hit_df: Optional[pd.DataFrame] = None, window_days: int = DEFAULT_PATTERN_WINDOW) -> Dict[str, Dict[str, float]]:
    window_days = int(window_days)
    df = _ensure_hit_df(hit_df, window_days)
    if df.empty:
        return {}

    total_rows = len(df)

    families: List[str] = [
        "S40",
        "PACK_164950",
        "PACK_00_19",
        "PACK_20_39",
        "PACK_40_59",
        "PACK_60_79",
        "PACK_80_99",
    ]

    def _belongs_to_family(number: int, family: str) -> bool:
        if family == "S40":
            return pattern_packs.is_s40(number)
        if family == "PACK_164950":
            return pattern_packs.is_164950_family(number)
        tags = pattern_packs.get_digit_pack_tags(number)
        return family in tags

    stats: Dict[str, Dict[str, float]] = {}
    epsilon = 1e-6

    for family in families:
        fam_df = df[df["real_number"].apply(lambda n: _belongs_to_family(n, family))]
        if fam_df.empty:
            stats[family] = {
                "observations": 0,
                "exact_hits": 0,
                "near_hits": 0,
                "hit_rate_exact": 0.0,
                "near_miss_rate": 0.0,
                "hit_rate_overall": 0.0,
                "near_miss_overall": 0.0,
                "best_slot": "n/a",
                "weak_slot": "n/a",
            }
            continue

        exact_hits = int(fam_df["is_exact_hit"].sum())
        near_hits = int(fam_df["is_near_hit"].sum())
        hit_rate_exact = exact_hits / len(fam_df) if len(fam_df) else 0.0
        near_rate = near_hits / len(fam_df) if len(fam_df) else 0.0
        hit_rate_overall = exact_hits / total_rows if total_rows else 0.0
        near_overall = near_hits / total_rows if total_rows else 0.0

        slot_scores: Dict[str, float] = {}
        for slot in SLOTS:
            slot_df = fam_df[fam_df["slot"] == slot]
            n = len(slot_df)
            if n == 0:
                slot_scores[slot] = 0.0
                continue
            slot_exact = int(slot_df["is_exact_hit"].sum())
            slot_near = int(slot_df["is_near_hit"].sum())
            hit_rate = slot_exact / n if n else 0.0
            near_slot_rate = slot_near / n if n else 0.0
            slot_scores[slot] = hit_rate + 0.2 * near_slot_rate

        if slot_scores:
            max_score = max(slot_scores.values())
            min_score = min(slot_scores.values())
            if len(fam_df) == 0 or abs(max_score - min_score) < epsilon:
                best_slot = weak_slot = "n/a"
            else:
                best_slot = max(slot_scores, key=slot_scores.get)
                weak_slot = min(slot_scores, key=slot_scores.get)
        else:
            best_slot = weak_slot = "n/a"

        stats[family] = {
            "observations": len(fam_df),
            "exact_hits": exact_hits,
            "near_hits": near_hits,
            "hit_rate_exact": hit_rate_exact,
            "near_miss_rate": near_rate,
            "hit_rate_overall": hit_rate_overall,
            "near_miss_overall": near_overall,
            "best_slot": best_slot,
            "weak_slot": weak_slot,
        }

    return stats


def _family_filter(number: int, family: str) -> bool:
    if family == "S40":
        return pattern_packs.is_s40(number)
    if family == "PACK_164950":
        return pattern_packs.is_164950_family(number)
    tags = pattern_packs.get_digit_pack_tags(number)
    return family in tags


def _normalise_hit_df(hit_df: Optional[pd.DataFrame], window_days: int) -> pd.DataFrame:
    df = _ensure_hit_df(hit_df, window_days)
    if df.empty:
        return df

    df = df.copy()
    if "result_date" in df.columns:
        df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df.get("DATE"), errors="coerce")
        df["result_date"] = df.get("result_date", df["DATE"])

    df = df.dropna(subset=["result_date"])
    df["result_date"] = df["result_date"].dt.normalize()
    return df


def build_pattern_summary(hit_df: Optional[pd.DataFrame], window_days: int = DEFAULT_PATTERN_WINDOW) -> Dict:
    """Build a standard pattern summary payload for JSON export."""

    df = _normalise_hit_df(hit_df, window_days)
    if df.empty:
        return {}

    total_rows = len(df)
    total_exact = int(df.get("is_exact_hit", False).sum())
    total_near = int(df.get("is_near_hit", False).sum())
    overall_days = df["result_date"].dt.date.nunique()

    families: List[str] = [
        "S40",
        "PACK_164950",
        "PACK_00_19",
        "PACK_20_39",
        "PACK_40_59",
        "PACK_60_79",
        "PACK_80_99",
    ]

    patterns: Dict[str, Dict] = {}
    for family in families:
        fam_df = df[df["real_number"].apply(lambda n: _family_filter(n, family))]
        rows = len(fam_df)
        exact_hits = int(fam_df.get("is_exact_hit", False).sum()) if rows else 0
        near_hits = int(fam_df.get("is_near_hit", False).sum()) if rows else 0

        fam_by_slot: Dict[str, Dict[str, float]] = {}
        best_slot = None
        weak_slot = None
        slot_rates: Dict[str, float] = {}
        for slot in SLOTS:
            slot_df = fam_df[fam_df["slot"] == slot]
            slot_rows = len(slot_df)
            slot_exact = int(slot_df.get("is_exact_hit", False).sum()) if slot_rows else 0
            slot_near = int(slot_df.get("is_near_hit", False).sum()) if slot_rows else 0
            hit_rate_exact = slot_exact / slot_rows if slot_rows else 0.0
            near_miss_rate = slot_near / slot_rows if slot_rows else 0.0
            slot_rates[slot] = hit_rate_exact

            days_with_results = slot_df["result_date"].dt.date.nunique() if not slot_df.empty else 0
            active_days = slot_df[(slot_df.get("is_exact_hit", False)) | (slot_df.get("is_near_hit", False))][
                "result_date"
            ].dt.date.nunique() if not slot_df.empty else 0

            fam_by_slot[slot] = {
                "rows": slot_rows,
                "exact_hits": slot_exact,
                "near_hits": slot_near,
                "hit_rate_exact": hit_rate_exact,
                "near_miss_rate": near_miss_rate,
                "days_with_results": days_with_results,
                "active_days": active_days,
            }

        if slot_rates:
            max_slot_rate = max(slot_rates.values())
            min_slot_rate = min(slot_rates.values())
            if rows == 0 or max_slot_rate == min_slot_rate:
                best_slot = weak_slot = None
            else:
                best_slot = max(slot_rates, key=slot_rates.get)
                weak_slot = min(slot_rates, key=slot_rates.get)

        hit_rate_exact = exact_hits / rows if rows else 0.0
        near_miss_rate = near_hits / rows if rows else 0.0
        hit_rate_overall = exact_hits / total_rows if total_rows else 0.0
        near_rate_overall = near_hits / total_rows if total_rows else 0.0

        if family == "PACK_40_59" and (rows == 0 or (exact_hits == 0 and near_hits == 0)):
            best_slot = None
            weak_slot = None

        patterns[family] = {
            "rows": rows,
            "exact_hits": exact_hits,
            "near_hits": near_hits,
            "hit_rate_exact": hit_rate_exact,
            "near_miss_rate": near_miss_rate,
            "hit_rate_overall": hit_rate_overall,
            "near_miss_overall": near_rate_overall,
            "best_slot": best_slot,
            "weak_slot": weak_slot,
            "by_slot": fam_by_slot,
        }

    def _cover_for_family(family: str) -> Dict[str, int]:
        fam_df = df[df["real_number"].apply(lambda n: _family_filter(n, family))]
        covered_days = fam_df["result_date"].dt.date.nunique() if not fam_df.empty else 0
        return {"total_days": overall_days, "covered_days": covered_days}

    daily_cover = {
        "S40": _cover_for_family("S40"),
        "PACK_164950": _cover_for_family("PACK_164950"),
    }

    latest_date = df["result_date"].max() if not df.empty else None
    earliest_date = latest_date - timedelta(days=window_days - 1) if latest_date is not None else None

    summary = {
        "window_days": window_days,
        "rows": total_rows,
        "exact_hits": total_exact,
        "near_hits": total_near,
        "patterns": patterns,
        "daily_cover": daily_cover,
    }

    if earliest_date is not None and latest_date is not None:
        summary["window"] = {
            "start": earliest_date.strftime("%Y-%m-%d"),
            "end": latest_date.strftime("%Y-%m-%d"),
        }

    return summary


def save_pattern_summary(summary: Dict, base_dir: Path, window_days: int) -> Optional[Path]:
    if not summary:
        return None
    output_dir = base_dir / "logs" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pattern_intel_summary_{window_days}d.json"
    path.write_text(json.dumps(summary, indent=2))
    export_pattern_strength_signals(summary, base_dir)
    return path


def export_pattern_strength_signals(summary: Dict, base_dir: Path) -> Optional[Path]:
    if not summary:
        return None

    patterns = summary.get("patterns", {}) or {}
    total_rows = summary.get("rows", 0) or 0
    if total_rows == 0:
        return None

    output_dir = base_dir / "logs" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pattern_strength_signals.json"

    signals: Dict[str, Dict[str, Dict[str, float]]] = {}
    families = list(patterns.keys())
    for family, fam_data in patterns.items():
        by_slot = fam_data.get("by_slot", {}) or {}
        for slot, slot_data in by_slot.items():
            slot_rows = slot_data.get("rows", 0) or 0
            exact_hits = slot_data.get("exact_hits", 0) or 0
            near_hits = slot_data.get("near_hits", 0) or 0
            slot_hit_overall = exact_hits / total_rows if total_rows else 0.0
            slot_near_overall = near_hits / total_rows if total_rows else 0.0
            score = max(0.0, min(1.0, slot_hit_overall + 0.3 * slot_near_overall))
            signals.setdefault(slot, {})[family] = {
                "hit_rate_exact": slot_hit_overall,
                "near_miss_rate": slot_near_overall,
                "strength_score": score,
                "rows": slot_rows,
            }

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "window_days": summary.get("window_days"),
        "families": families,
        "signals": signals,
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def run_enhanced_pattern_intel(hit_df: Optional[pd.DataFrame] = None, window_days: int = 120) -> Dict[str, Dict]:
    window_days = int(window_days)
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
        members: List[str] = []
        for n in pattern_packs.S40_STRINGS:
            try:
                members.append(f"{int(n):02d}")
            except Exception:
                try:
                    members.append(f"{n:0>2}")
                except Exception:
                    continue
        return members
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
    window_days = int(window_days)
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
