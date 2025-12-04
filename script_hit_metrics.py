"""Script-level hit metrics and lightweight ROI proxies.

This module reads the unified ``logs/performance/script_hit_memory.csv``
and produces per-script, per-slot performance windows. It is designed to
operate in preview-only mode so existing behaviours remain unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

import quant_paths
from script_hit_memory_utils import get_script_hit_memory_path


def _load_memory_df() -> pd.DataFrame:
    path = get_script_hit_memory_path()
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        return df
    except Exception:
        return pd.DataFrame()


def compute_script_metrics(window_days: int = 30) -> pd.DataFrame:
    """Compute per-script metrics for the trailing window."""

    df = _load_memory_df()
    if df.empty:
        return pd.DataFrame()

    if "date" not in df.columns:
        return pd.DataFrame()

    latest_date = max([d for d in df["date"] if pd.notna(d)], default=None)
    if latest_date is None:
        return pd.DataFrame()

    cutoff = latest_date - timedelta(days=window_days - 1)
    df_window = df[df["date"] >= cutoff]

    required_cols = ["script_name", "slot", "hit_flag"]
    for col in required_cols:
        if col not in df_window.columns:
            return pd.DataFrame()

    groups = []
    for (script, slot), group in df_window.groupby(["script_name", "slot"]):
        n_rows = len(group)
        n_final_hits = int((group["hit_flag"] == "FINAL_HIT").sum())
        n_script_hits_only = int((group["hit_flag"] == "SCRIPT_HIT_BUT_NOT_FINAL").sum())
        n_blind_miss = int((group["hit_flag"] == "BLIND_MISS").sum())

        hit_rate_final = n_final_hits / n_rows if n_rows else 0.0
        hit_rate_any = (n_final_hits + n_script_hits_only) / n_rows if n_rows else 0.0
        blind_miss_rate = n_blind_miss / n_rows if n_rows else 0.0

        virtual_total_bet = n_rows * 10
        virtual_return = n_final_hits * 90
        roi_proxy_pct = (
            (virtual_return - virtual_total_bet) / max(virtual_total_bet, 1) * 100.0
        )

        groups.append(
            {
                "script_name": script,
                "slot": slot,
                "n_rows": n_rows,
                "n_final_hits": n_final_hits,
                "n_script_hits_only": n_script_hits_only,
                "n_blind_miss": n_blind_miss,
                "hit_rate_final": round(hit_rate_final, 4),
                "hit_rate_any": round(hit_rate_any, 4),
                "blind_miss_rate": round(blind_miss_rate, 4),
                "roi_proxy_pct": round(roi_proxy_pct, 2),
            }
        )

    return pd.DataFrame(groups)


def _save_metrics_outputs(metrics_df: pd.DataFrame, window_days: int) -> None:
    perf_dir = quant_paths.get_performance_logs_dir()
    perf_dir.mkdir(parents=True, exist_ok=True)

    csv_path = perf_dir / f"script_metrics_{window_days}d.csv"
    json_path = perf_dir / f"script_metrics_{window_days}d.json"

    metrics_df.to_csv(csv_path, index=False)

    payload = {
        "window_days": window_days,
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics_df.to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_script_metrics(window_days: int = 30) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return metrics as slot → script → metrics mapping."""

    df = compute_script_metrics(window_days=window_days)
    result: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    if df.empty:
        return result

    for _, row in df.iterrows():
        slot = str(row["slot"]).upper()
        script = str(row["script_name"]).upper()
        result[slot][script] = {
            "hit_rate_any": float(row.get("hit_rate_any", 0.0)),
            "blind_miss_rate": float(row.get("blind_miss_rate", 0.0)),
            "roi_proxy_pct": float(row.get("roi_proxy_pct", 0.0)),
            "n_rows": float(row.get("n_rows", 0)),
        }
    return result


def compute_slot_heroes_and_weak(
    metrics_df: pd.DataFrame, min_rows: int = 10
) -> Dict[str, Dict[str, List[str]]]:
    """Compute hero/weak script suggestions for brief printing."""

    summary: Dict[str, Dict[str, List[str]]] = {}
    if metrics_df.empty:
        return summary

    for slot, group in metrics_df.groupby("slot"):
        eligible = group[group["n_rows"] >= min_rows]
        if eligible.empty:
            summary[slot] = {"heroes": [], "weak": []}
            continue
        heroes = (
            eligible.sort_values("hit_rate_any", ascending=False)
            .head(2)["script_name"]
            .tolist()
        )
        weak = (
            eligible.sort_values("blind_miss_rate", ascending=False)
            .head(1)["script_name"]
            .tolist()
        )
        summary[slot] = {"heroes": heroes, "weak": weak}
    return summary


def _print_console_summary(metrics_df: pd.DataFrame, window_days: int) -> None:
    if metrics_df.empty:
        print("No script hit memory available for metrics.")
        return

    summary = compute_slot_heroes_and_weak(metrics_df)
    print(f"SCRIPT WEIGHT PREVIEW (last {window_days} days):")
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        slot_summary = summary.get(slot, {"heroes": [], "weak": []})
        hero_str = ",".join(slot_summary.get("heroes") or []) or "n/a"
        weak_str = ",".join(slot_summary.get("weak") or []) or "n/a"
        print(f"  {slot}: hero=[{hero_str}] weak=[{weak_str}] window={window_days}d")


def _run_cli(window: int, verbose: bool) -> None:
    metrics_df = compute_script_metrics(window_days=window)
    if metrics_df.empty:
        print("No metrics generated (missing or empty script_hit_memory.csv)")
        return

    _save_metrics_outputs(metrics_df, window)
    if verbose:
        print(metrics_df)
    _print_console_summary(metrics_df, window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute script hit metrics")
    parser.add_argument("--window", type=int, default=30, help="Window in days")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    _run_cli(args.window, args.verbose)

