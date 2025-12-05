from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

import pattern_packs
from quant_core import hit_core, pattern_core
from script_hit_memory_utils import load_script_hit_memory

WINDOW_DAYS = 90
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def _belongs_to_family(number: int, family: str) -> bool:
    if family == "S40":
        return pattern_packs.is_s40(number)
    if family == "PACK_164950":
        return pattern_packs.is_164950_family(number)
    tags = pattern_packs.get_digit_pack_tags(number)
    return family in tags


class PatternIntelligenceEngine:
    """Compact pattern intelligence summary built on script hit memory."""

    def __init__(self, window_days: int = WINDOW_DAYS) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.window_days = window_days
        self.families: List[str] = [
            "S40",
            "PACK_164950",
            "PACK_00_19",
            "PACK_20_39",
            "PACK_40_59",
            "PACK_60_79",
            "PACK_80_99",
        ]

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
        df["real_number"] = pd.to_numeric(df.get("real_number"), errors="coerce")
        df = df.dropna(subset=["real_number"])
        df["real_number"] = df["real_number"].astype(int)
        df["slot"] = df.get("slot").astype(str)
        df["is_exact_hit"] = df.get("is_exact_hit", False).astype(bool)
        df["is_near_miss"] = df.get("is_near_miss", False).astype(bool)
        return df

    def analyse(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        total_rows = len(df)
        for family in self.families:
            stats[family] = {
                "observations": total_rows,
                "exact_hits": 0,
                "near_hits": 0,
                "hit_rate_exact": 0.0,
                "near_miss_rate": 0.0,
                "best_slot": None,
                "weak_slot": None,
            }

        if df.empty:
            return stats

        for family in self.families:
            fam_df = df[df["real_number"].apply(lambda n: _belongs_to_family(n, family))]
            if fam_df.empty:
                continue
            exact_hits = int(fam_df["is_exact_hit"].sum())
            near_hits = int(fam_df["is_near_miss"].sum())
            hit_rate_exact = exact_hits / len(fam_df) if len(fam_df) else 0.0
            near_rate = near_hits / len(fam_df) if len(fam_df) else 0.0

            by_slot = fam_df.groupby("slot")
            best_slot = None
            weak_slot = None
            if not by_slot.ngroups:
                best_slot = weak_slot = None
            else:
                slot_rates = by_slot["is_exact_hit"].mean()
                best_slot = slot_rates.idxmax()
                weak_slot = slot_rates.idxmin()

            stats[family].update(
                {
                    "observations": len(fam_df),
                    "exact_hits": exact_hits,
                    "near_hits": near_hits,
                    "hit_rate_exact": hit_rate_exact,
                    "near_miss_rate": near_rate,
                    "best_slot": best_slot,
                    "weak_slot": weak_slot,
                }
            )
        return stats

    def save_stats(self, stats: Dict[str, Dict[str, float]]) -> Path:
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "pattern_intelligence_summary.json"
        payload = {"timestamp": datetime.now().isoformat(), "window_days": self.window_days, "stats": stats}
        path.write_text(json.dumps(payload, indent=2))
        return path

    def print_summary(self, df: pd.DataFrame, stats: Dict[str, Dict[str, float]]) -> None:
        total_exact = int(df.get("is_exact_hit", False).sum()) if not df.empty else 0
        print(f"[PatternIntel] Window: {self.window_days}d, rows: {len(df)}, exact hits: {total_exact}")
        for family in self.families:
            fam = stats.get(family, {})
            if not fam:
                continue
            line = (
                f"[PatternIntel] {family}: hit_rate={fam.get('hit_rate_exact', 0):.3f}, "
                f"near={fam.get('near_miss_rate', 0):.3f}, "
                f"best={fam.get('best_slot') or 'n/a'}, weak={fam.get('weak_slot') or 'n/a'}"
            )
            print(line)

    def run(self) -> bool:
        hit_core.rebuild_hit_memory(window_days=self.window_days)
        df = self.load_window()
        if df.empty:
            print(f"[PatternIntel] Not enough hit data in the last {self.window_days} days (found 0 rows). Skipping pattern analysis.")
            return True
        stats = pattern_core.run_basic_pattern_intel(hit_df=df, window_days=self.window_days) or {}
        self.save_stats(stats)
        self.print_summary(df, stats)
        return True


def main() -> int:
    engine = PatternIntelligenceEngine()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
