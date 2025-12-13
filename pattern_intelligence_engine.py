from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from quant_core import hit_core, pattern_core
from quant_learning_core import slot_regime
from quant_stats_core import compute_pack_hit_stats
from script_hit_memory_utils import classify_relation, filter_hits_by_window, load_script_hit_memory
import pattern_packs

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


def build_near_miss_boosts(
    window_days: int = 60, decay: float = 0.85, base_dir: Optional[Path] = None
) -> Dict[str, Dict[str, float]]:
    df = load_script_hit_memory(base_dir=base_dir)
    boosts: Dict[str, Dict[str, float]] = {}

    if df is None or df.empty:
        return boosts

    df = df.copy()
    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["result_date", "slot", "predicted_number"])
    if df.empty:
        return boosts

    latest_date = df["result_date"].max().date()
    cutoff = latest_date - timedelta(days=window_days - 1)
    df = df[df["result_date"].dt.date >= cutoff]
    df["slot"] = df.get("slot").astype(str).str.upper()
    df["predicted_number"] = df.get("predicted_number").astype(str).str.zfill(2)
    df["HIT_TYPE"] = df.get("HIT_TYPE", df.get("hit_type", "")).astype(str).str.upper()

    weight_map = {
        "NEIGHBOR": 1.0,
        "NEAR": 1.0,
        "MIRROR": 0.9,
        "CROSS_SLOT": 0.8,
        "CROSS_DAY": 0.8,
        "REVERSE": 0.7,
    }

    for _, row in df.iterrows():
        slot = row.get("slot")
        number = row.get("predicted_number")
        if not slot or not number:
            continue

        hit_type = str(row.get("HIT_TYPE", "")).upper()
        relation = classify_relation(row.get("predicted_number"), row.get("real_number"))
        weight = weight_map.get(hit_type, 0.0)
        if not weight and relation in {"ADJACENT", "MIRROR", "REVERSE_CARRY"}:
            weight = 0.8 if relation == "MIRROR" else 0.6
        if not weight and bool(row.get("is_near_miss")):
            weight = 0.5
        if weight <= 0:
            continue

        days_ago = (latest_date - row["result_date"].date()).days if pd.notna(row.get("result_date")) else 0
        contribution = weight * (decay ** max(days_ago, 0))
        slot_map = boosts.setdefault(str(slot).upper(), {})
        slot_map[number] = slot_map.get(number, 0.0) + contribution

    # Gentle decay for dormant numbers
    for slot, num_map in boosts.items():
        for num, score in list(num_map.items()):
            num_map[num] = max(score * decay, 0.0)

    output_root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    output_path = output_root / "data" / "near_miss_boosts.json"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({k: v for k, v in boosts.items()}, indent=2))
    except Exception:
        pass

    return boosts


def load_near_miss_boosts(base_dir: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    root = Path(base_dir) if base_dir else Path(__file__).resolve().parent
    path = root / "data" / "near_miss_boosts.json"
    try:
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return {str(slot).upper(): {str(n).zfill(2): float(v) for n, v in entries.items()} for slot, entries in data.items()}
    except Exception:
        pass
    return {}


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


class PatternIntelligenceEngine:
    """Compact pattern intelligence summary built on real result rows."""

    def __init__(self, window_days: int = WINDOW_DAYS) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.window_days = window_days

    def save_stats(self, df: pd.DataFrame, summary: Dict) -> Path:
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "pattern_intel_summary.json"
        payload = {"timestamp": datetime.now().isoformat(), **summary}
        path.write_text(json.dumps(payload, indent=2))
        try:
            summary = pattern_core.build_pattern_summary(df, window_days=self.window_days)
            pattern_core.save_pattern_summary(summary, base_dir=self.base_dir, window_days=self.window_days)
        except Exception:
            pass
        return path

    def print_summary(self, summary: Dict) -> None:
        print(
            f"[PatternIntel] Window: {self.window_days}d, rows: {summary.get('total_rows', 0)}, "
            f"result_days: {summary.get('result_days', 0)}, exact hits: {summary.get('exact_hits', 0)}"
        )
        s40 = summary.get("s40", {}) or {}
        fam = summary.get("family_164950", {}) or {}
        if s40:
            print(
                f"[PatternIntel] S40 hits={s40.get('hits_total', 0)}, "
                f"hit_rate={s40.get('hit_rate', 0.0):.4f}, daily_cover={s40.get('daily_cover_days', 0)}/{s40.get('daily_cover_total_days', 0)}"
            )
        if fam:
            print(
                f"[PatternIntel] 164950 hits={fam.get('hits_total', 0)}, "
                f"hit_rate={fam.get('hit_rate', 0.0):.4f}, daily_cover={fam.get('daily_cover_days', 0)}/{fam.get('daily_cover_total_days', 0)}"
            )

    def run(self) -> bool:
        hit_core.rebuild_hit_memory(window_days=self.window_days)
        df, summary = compute_pattern_metrics(window_days=self.window_days, base_dir=self.base_dir)
        if df.empty:
            print(
                f"[PatternIntel] Not enough hit data in the last {self.window_days} days (found 0 rows). Skipping pattern analysis."
            )
            return True
        self.save_stats(df, summary)
        try:
            boosts = build_near_miss_boosts(window_days=60, base_dir=self.base_dir)
            if boosts:
                sample_slot = next(iter(boosts.keys()))
                sample_count = len(boosts.get(sample_slot, {}))
                print(f"[PatternIntel] Near-miss boosts generated for {len(boosts)} slots (e.g., {sample_slot}: {sample_count} numbers)")
        except Exception as exc:
            print(f"[PatternIntel] Warning: could not refresh near_miss_boosts.json ({exc})")
        self.print_summary(summary)
        return True


def main() -> int:
    engine = PatternIntelligenceEngine()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
