from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from quant_core import hit_core, pattern_core
from pattern_intelligence_engine import compute_pattern_metrics

WINDOW_DAYS = 120
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


class PatternIntelligenceEnhanced:
    """Higher level summaries (scripts, slots) with quiet console output."""

    def __init__(self, window_days: int = WINDOW_DAYS) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.window_days = window_days

    def summarise_scripts(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        if df.empty:
            return summaries
        for slot, group in df.groupby("slot"):
            total = len(group)
            summaries[slot] = {
                "rows": total,
                "exact_hits": total,
                "near_hits": 0,
                "hit_rate_exact": 1.0 if total else 0.0,
                "near_miss_rate": 0.0,
            }
        return summaries

    def summarise_slots(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        if df.empty:
            return summaries
        for slot, group in df.groupby("slot"):
            total = len(group)
            summaries[slot] = {
                "rows": total,
                "exact_hits": total,
                "near_hits": 0,
                "hit_rate_exact": 1.0 if total else 0.0,
                "near_miss_rate": 0.0,
            }
        return summaries

    def save(self, payload: Dict, df: pd.DataFrame) -> Path:
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"pattern_intel_summary_{self.window_days}d.json"
        path.write_text(json.dumps(payload, indent=2))
        try:
            summary = pattern_core.build_pattern_summary(df, window_days=self.window_days)
            pattern_core.save_pattern_summary(summary, base_dir=self.base_dir, window_days=self.window_days)
        except Exception:
            pass
        return path

    def print_summary(self, df: pd.DataFrame, scripts: Dict[str, Dict[str, float]], slots: Dict[str, Dict[str, float]]) -> None:
        total_exact = len(df.get("is_exact_hit", [])) if not df.empty else 0
        result_days = df["result_date"].dt.date.nunique() if not df.empty and "result_date" in df.columns else 0
        print(
            f"[PatternIntel+] Window: {self.window_days}d, rows: {len(df)}, result_days: {result_days}, exact hits: {total_exact}"
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
        df, base_summary = compute_pattern_metrics(window_days=self.window_days, base_dir=self.base_dir)
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
            "summary": base_summary,
            "scripts": scripts,
            "slots": slots,
        }
        self.save(payload, df)
        self.print_summary(df, scripts, slots)
        return True


def main() -> int:
    engine = PatternIntelligenceEnhanced()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
