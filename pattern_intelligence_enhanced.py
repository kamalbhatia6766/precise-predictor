from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from quant_core import hit_core, pattern_core
from pattern_intelligence_engine import compute_pattern_metrics
from quant_stats_core import compute_pack_hit_stats
import quant_paths

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
            exact_hits = int(pd.to_numeric(group.get("is_exact_hit", False), errors="coerce").fillna(0).astype(int).sum())
            summaries[slot] = {
                "rows": total,
                "exact_hits": exact_hits,
                "near_hits": 0,
                "hit_rate_exact": exact_hits / total if total else 0.0,
                "near_miss_rate": 0.0,
            }
        return summaries

    def summarise_slots(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        summaries: Dict[str, Dict[str, float]] = {}
        if df.empty:
            return summaries
        for slot, group in df.groupby("slot"):
            total = len(group)
            exact_hits = int(pd.to_numeric(group.get("is_exact_hit", False), errors="coerce").fillna(0).astype(int).sum())
            summaries[slot] = {
                "rows": total,
                "exact_hits": exact_hits,
                "near_hits": 0,
                "hit_rate_exact": exact_hits / total if total else 0.0,
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
        total_exact = int(pd.to_numeric(df.get("is_exact_hit", False), errors="coerce").fillna(0).astype(int).sum()) if not df.empty else 0
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
        summary = {
            "timestamp": datetime.now().isoformat(),
            "window_days": self.window_days,
            "summary": base_summary,
            "scripts": scripts,
            "slots": slots,
        }
        self._latest_summary = summary
        self.save(summary, df)
        self._export_regime_summary(base_summary)
        self.print_summary(df, scripts, slots)
        return True

    def _export_regime_summary(self, summary: Dict) -> None:
        try:
            base_dir = self.base_dir
            output_path = base_dir / "logs" / "performance" / "pattern_regimes_summary.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            pattern_window = int(summary.get("window_days", self.window_days))
            short_window = 30
            short_stats = compute_pack_hit_stats(window_days=short_window, base_dir=base_dir) or {}

            def _regime_score(label: str) -> int:
                mapping = {"BOOST": 2, "NORMAL": 1, "OFF": 0}
                return mapping.get(str(label).upper(), 0)

            def _family_block(key: str, label: str) -> Dict:
                fam = summary.get(key, {}) or {}
                per_slot_block = {}
                best_slot = None
                weak_slot = None
                best_score = (float("-inf"), float("-inf"))
                weak_score = (float("inf"), float("inf"))
                for slot in SLOTS:
                    slot_stats = fam.get("per_slot", {}).get(slot, {}) if isinstance(fam, dict) else {}
                    hits = int(slot_stats.get("hits_total", 0) or 0)
                    cover = int(slot_stats.get("daily_cover_days", 0) or 0)
                    rate = float(slot_stats.get("hit_rate", 0.0) or 0.0)
                    regime = slot_stats.get("regime", "NORMAL")
                    per_slot_block[slot] = {
                        "regime": regime,
                        "hits": hits,
                        "days_covered": cover,
                    }
                    score = (rate, _regime_score(regime))
                    if score > best_score:
                        best_score = score
                        best_slot = slot
                    if score < weak_score:
                        weak_score = score
                        weak_slot = slot

                return {
                    "per_slot": per_slot_block,
                    "overall": {
                        "daily_cover_percent": float(fam.get("daily_cover_pct", 0.0) or 0.0),
                        "total_hits": int(fam.get("hits_total", 0) or 0),
                        "best_slot": best_slot,
                        "weak_slot": weak_slot,
                    },
                }

            payload = {
                "window_days": pattern_window,
                "short_window_days": short_window,
                "families": {
                    "S40": _family_block("s40", "S40"),
                    "FAMILY_164950": _family_block("family_164950", "FAMILY_164950"),
                },
                "timestamp": datetime.now().isoformat(),
            }

            if short_stats:
                payload["short_window"] = {"available_days": short_stats.get("days_total")}

            output_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[pattern_intelligence_enhanced] Warning: unable to write pattern_regimes_summary.json: {exc}")


def main() -> int:
    engine = PatternIntelligenceEnhanced()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
