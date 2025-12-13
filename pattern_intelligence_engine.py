from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from quant_core import hit_core, pattern_core
from quant_core.pattern_metrics_core import compute_pattern_metrics
from script_hit_memory_utils import classify_relation, load_script_hit_memory

WINDOW_DAYS = 90
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


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
