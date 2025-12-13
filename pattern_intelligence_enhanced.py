from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from typing import Dict

import pandas as pd

from quant_core import hit_core, pattern_core
from quant_core.pattern_metrics_core import compute_pattern_metrics
from quant_stats_core import compute_pack_hit_stats
from script_hit_memory_utils import filter_by_window, filter_hits_by_window
import quant_paths
import pattern_packs

WINDOW_DAYS = 120
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


class PatternIntelligenceEnhanced:
    """Higher level summaries (scripts, slots) with quiet console output."""

    def __init__(self, window_days: int = WINDOW_DAYS) -> None:
        self.base_dir = Path(__file__).resolve().parent
        self.window_days = window_days
        self.logger = logging.getLogger(__name__)

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
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        date_col = None
        for candidate in ("result_date", "predict_date", "date"):
            if candidate in df.columns:
                date_col = candidate
                break
        filtered_df = df
        if date_col:
            try:
                filtered_df, window_start, window_end = filter_by_window(df, date_col=date_col, window_days=self.window_days)
                if self.logger.isEnabledFor(logging.INFO):
                    self.logger.info(
                        "[PatternIntel+] window=%sd start=%s end=%s rows=%s",
                        self.window_days,
                        getattr(window_start, "date", lambda: None)(),
                        getattr(window_end, "date", lambda: None)(),
                        len(filtered_df),
                    )
                print(
                    f"[PatternIntel+] Window slice: days={self.window_days}, start={getattr(window_start, 'date', lambda: None)()}, end={getattr(window_end, 'date', lambda: None)()}, rows={len(filtered_df)}"
                )
            except Exception:
                filtered_df = df
        if filtered_df.empty:
            print(
                f"[PatternIntel+] Not enough hit data in the last {self.window_days} days (found 0 rows). Skipping enhanced analysis."
            )
            return True
        enhanced = pattern_core.run_enhanced_pattern_intel(hit_df=filtered_df, window_days=self.window_days)
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
        self.save(summary, filtered_df)
        self._export_regime_summary(base_summary)
        self._export_family_regimes(filtered_df)
        self.print_summary(filtered_df, scripts, slots)
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

    def _export_family_regimes(self, df: pd.DataFrame) -> None:
        try:
            if df is None or df.empty:
                return

            work_df = df.copy()
            work_df["result_date"] = pd.to_datetime(work_df.get("result_date"), errors="coerce")
            work_df = work_df.dropna(subset=["result_date"])
            if work_df.empty:
                return

            work_df["slot"] = work_df.get("slot", "").astype(str).str.upper()

            def _families_for_number(n: object) -> List[str]:
                try:
                    fams = pattern_packs.get_number_families(n)
                    return [str(f).strip() for f in fams if str(f).strip()]
                except Exception:
                    return []

            def _is_hit(row: pd.Series) -> bool:
                hit_type = str(row.get("hit_type") or row.get("HIT_TYPE") or "").strip().upper()
                if hit_type:
                    return hit_type == "HIT"
                return bool(row.get("is_exact_hit"))

            work_df["_families"] = work_df.get("real_number").apply(_families_for_number)
            work_df["_is_hit"] = work_df.apply(_is_hit, axis=1)

            families: List[str] = []
            for fams in work_df["_families"]:
                if isinstance(fams, list):
                    families.extend([str(f).strip() for f in fams if str(f).strip()])
            families_set = set(families)
            families_set.update({"S40", "FAMILY_164950"})
            families = sorted(families_set)

            window_config = {"long_days": 90, "mid_days": 30, "short_days": 7}
            window_defs = {
                "90d": window_config["long_days"],
                "30d": window_config["mid_days"],
                "7d": window_config["short_days"],
            }

            slots_payload: Dict[str, Dict] = {}

            def _last_hit_date(slot_df: pd.DataFrame, family: str) -> Optional[str]:
                fam_hits = slot_df[slot_df["_families"].apply(lambda tags: family in tags) & slot_df["_is_hit"]]
                if fam_hits.empty:
                    return None
                date_val = fam_hits["result_date"].max()
                return date_val.date().isoformat() if pd.notna(date_val) else None

            for slot in SLOTS:
                slot_df_all = work_df[work_df["slot"] == slot].copy()
                slot_df_all["result_date"] = pd.to_datetime(slot_df_all["result_date"], errors="coerce")
                slot_payload: Dict[str, Dict] = {"families": {}}

                window_hit_rates: Dict[str, List[float]] = {label: [] for label in window_defs}

                for label, days in window_defs.items():
                    window_df, _ = filter_hits_by_window(slot_df_all, window_days=days)
                    if window_df.empty:
                        base_df = pd.DataFrame()
                    else:
                        base_df = window_df.copy()
                        base_df["result_date"] = pd.to_datetime(base_df["result_date"], errors="coerce")
                        base_df = base_df.dropna(subset=["result_date"])

                    rows_count = len(base_df)

                    for family in families:
                        fam_block = slot_payload["families"].setdefault(family, {})
                        fam_hits = 0
                        if not base_df.empty:
                            fam_mask = base_df["_families"].apply(lambda tags: family in tags)
                            fam_hits = int(base_df[fam_mask & base_df["_is_hit"]].shape[0])

                        hit_rate = fam_hits / max(rows_count, 1)
                        fam_block[f"rows_{label}"] = rows_count
                        fam_block[f"hits_{label}"] = fam_hits
                        fam_block[f"hit_rate_{label}"] = hit_rate
                        window_hit_rates[label].append(hit_rate)

                for label, rates in window_hit_rates.items():
                    avg_rate = sum(rates) / len(families) if families else 0.0
                    boost_cut = 1.3 * avg_rate
                    off_cut = 0.3 * avg_rate
                    for family in families:
                        fam_block = slot_payload["families"].setdefault(family, {})
                        rate_val = fam_block.get(f"hit_rate_{label}", 0.0)
                        if rate_val >= boost_cut:
                            regime = "BOOST"
                        elif rate_val <= off_cut:
                            regime = "OFF"
                        else:
                            regime = "NORMAL"
                        fam_block[f"regime_{label}"] = regime

                # Drift computation (R5/R6): compare 30d vs 90d behaviour
                for family in families:
                    fam_block = slot_payload["families"].setdefault(family, {})
                    hit_rate_90d = fam_block.get("hit_rate_90d", 0.0)
                    hit_rate_30d = fam_block.get("hit_rate_30d", 0.0)
                    drift_value = hit_rate_30d - hit_rate_90d
                    drift_label = "NORMAL"
                    # Thresholds tuned to be conservative to avoid over-smoothing
                    if drift_value >= max(0.0, abs(hit_rate_90d) * 0.25 + 0.002):
                        drift_label = "BOOST_DRIFT"
                    elif drift_value <= -(abs(hit_rate_90d) * 0.25 + 0.002):
                        drift_label = "COOL_OFF"

                    fam_block["hit_rate_90d"] = hit_rate_90d
                    fam_block["hit_rate_30d"] = hit_rate_30d
                    fam_block["drift_value"] = drift_value
                    fam_block["drift_label"] = drift_label

                for family in families:
                    fam_block = slot_payload["families"].setdefault(family, {})
                    fam_block["last_hit_date"] = _last_hit_date(slot_df_all, family)

                slots_payload[slot] = slot_payload

            output_path = self.base_dir / "logs" / "performance" / "pattern_regime_summary.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            payload = {
                "generated_at": datetime.now().isoformat(),
                "window_config": window_config,
                "slots": slots_payload,
            }

            output_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            print(f"[pattern_intelligence_enhanced] Warning: unable to write pattern_regime_summary.json: {exc}")


def main() -> int:
    engine = PatternIntelligenceEnhanced()
    success = engine.run()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
