from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

import pandas as pd

from quant_core import hit_core, pattern_core
from script_hit_memory_utils import load_script_hit_memory

WINDOW_DAYS = 120
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


class PatternIntelligenceEnhanced:
    """Higher level summaries (scripts, slots) with quiet console output."""

    def __init__(self, window_days: int = WINDOW_DAYS) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.window_days = window_days

    def load_window(self) -> pd.DataFrame:
        df = load_script_hit_memory(base_dir=self.base_dir)
        if df.empty:
            return pd.DataFrame()
        df = df.copy()
        df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
        df = df.dropna(subset=["result_date"])
        latest = df["result_date"].max()
        if pd.isna(latest):
            return pd.DataFrame()
        cutoff = latest - timedelta(days=self.window_days - 1)
        df = df[df["result_date"] >= cutoff]
        df["is_exact_hit"] = df.get("is_exact_hit", False).astype(bool)
        df["is_near_miss"] = df.get("is_near_miss", False).astype(bool)
        df["slot"] = df.get("slot").astype(str)
        df["script_id"] = df.get("script_id").astype(str)
        return df

    def summarise_scripts(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        if df.empty:
            return summaries
        for script, group in df.groupby("script_id"):
            total = len(group)
            exact_hits = int(group["is_exact_hit"].sum())
            near_hits = int(group["is_near_miss"].sum())
            summaries[script] = {
                "rows": total,
                "exact_hits": exact_hits,
                "near_hits": near_hits,
                "hit_rate_exact": exact_hits / total if total else 0.0,
                "near_miss_rate": near_hits / total if total else 0.0,
            }
        return summaries

    def summarise_slots(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        if df.empty:
            return summaries
        for slot, group in df.groupby("slot"):
            total = len(group)
            exact_hits = int(group["is_exact_hit"].sum())
            near_hits = int(group["is_near_miss"].sum())
            summaries[slot] = {
                "rows": total,
                "exact_hits": exact_hits,
                "near_hits": near_hits,
                "hit_rate_exact": exact_hits / total if total else 0.0,
                "near_miss_rate": near_hits / total if total else 0.0,
            }
        return summaries

    def save(self, payload: Dict) -> Path:
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "pattern_intelligence_enhanced.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    def print_summary(self, df: pd.DataFrame, scripts: Dict[str, Dict[str, float]], slots: Dict[str, Dict[str, float]]) -> None:
        total_exact = int(df.get("is_exact_hit", False).sum()) if not df.empty else 0
        print(f"[PatternIntel+] Window: {self.window_days}d, rows: {len(df)}, exact hits: {total_exact}")
        if scripts:
            top_script = max(scripts.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0))
            weak_script = min(scripts.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0))
            print(
                f"[PatternIntel+] Top script {top_script[0]} hit_rate={top_script[1]['hit_rate_exact']:.3f}; "
                f"Weak script {weak_script[0]} hit_rate={weak_script[1]['hit_rate_exact']:.3f}"
            )
        if slots:
            top_slot = max(slots.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0))
            weak_slot = min(slots.items(), key=lambda kv: kv[1].get("hit_rate_exact", 0))
            print(
                f"[PatternIntel+] Strong slot {top_slot[0]} hit_rate={top_slot[1]['hit_rate_exact']:.3f}; "
                f"Weak slot {weak_slot[0]} hit_rate={weak_slot[1]['hit_rate_exact']:.3f}"
            )

    def run(self) -> bool:
        hit_core.rebuild_hit_memory(window_days=self.window_days)
        df = self.load_window()
        if df.empty:
            print(
                f"[PatternIntel+] Not enough hit data in the last {self.window_days} days (found 0 rows). Skipping enhanced analysis."
            )
            return True
        enhanced = pattern_core.run_enhanced_pattern_intel(hit_df=df, window_days=self.window_days)
        scripts = enhanced.get("scripts", {}) if isinstance(enhanced, dict) else {}
        slots = enhanced.get("slots", {}) if isinstance(enhanced, dict) else {}
        payload = {
            "timestamp": datetime.now().isoformat(),
            "window_days": self.window_days,
            "scripts": scripts,
            "slots": slots,
        }
        self.save(payload)
        self.print_summary(df, scripts, slots)
        return True


def main() -> int:
    engine = PatternIntelligenceEnhanced()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
