from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from quant_core import hit_core, pattern_core, pnl_core
from quant_core.data_core import load_results_dataframe


def _as_date_string(value: object) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    try:
        dt = pd.to_datetime(value)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return str(value)


def _backup_file(path: Path) -> None:
    if not path.exists():
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
    backup.write_bytes(path.read_bytes())
    print(f"Backup created: {backup}")


def rebuild_from_date(from_date: str, rebuild_hit_memory: bool, rebuild_pnl: bool, rebuild_patterns: bool) -> None:
    print(f"QUANT TIME MACHINE – Rebuilding from {_as_date_string(from_date)}")
    base_dir = Path(__file__).resolve().parent
    target_date = pd.to_datetime(from_date).date()
    results_df = load_results_dataframe()
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    results_df = results_df[results_df["DATE"] >= target_date]

    if rebuild_hit_memory:
        path = base_dir / "logs" / "performance" / "script_hit_memory.xlsx"
        _backup_file(path)
        hit_df = hit_core.rebuild_hit_memory(window_days=90)
    else:
        hit_df = None

    if rebuild_pnl:
        path = base_dir / "logs" / "performance" / "quant_reality_pnl.json"
        _backup_file(path)
        snapshot = pnl_core.load_pnl_snapshot()
        print(f"P&L snapshot preserved entries: {len(snapshot.get('by_slot', [])) if snapshot else 0}")

    if rebuild_patterns:
        path = Path("config/pattern_packs_auto.json")
        _backup_file(path)
        try:
            pattern_window = 120
            hit_df = hit_core.rebuild_hit_memory(window_days=pattern_window)
            if hit_df is not None and not hit_df.empty:
                for col in ["DATE", "result_date", "date"]:
                    if col in hit_df.columns:
                        hit_df[col] = pd.to_datetime(hit_df[col], errors="coerce")
            pattern_core.build_pattern_config(hit_df=hit_df, window_days=pattern_window)
            print(
                "Pattern config rebuilt from Time Machine."
                f" window={pattern_window}d latest={_as_date_string(hit_df['DATE'].max()) if hit_df is not None and 'DATE' in hit_df.columns else 'n/a'}"
            )
        except Exception as e:
            print(f"[WARN] Pattern rebuild failed in Time Machine: {e}")

    print("Rebuild completed.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset/time-machine utilities for Precise Predictor")
    parser.add_argument("--from-date", dest="from_date", required=True, help="YYYY-MM-DD from which to rebuild")
    parser.add_argument("--rebuild-hit-memory", action="store_true")
    parser.add_argument("--rebuild-pnl", action="store_true")
    parser.add_argument("--rebuild-patterns", action="store_true")
    args = parser.parse_args()

    rebuild_flags = [args.rebuild_hit_memory, args.rebuild_pnl, args.rebuild_patterns]
    if not any(rebuild_flags):
        args.rebuild_hit_memory = args.rebuild_pnl = args.rebuild_patterns = True

    rebuild_from_date(
        args.from_date,
        rebuild_hit_memory=args.rebuild_hit_memory,
        rebuild_pnl=args.rebuild_pnl,
        rebuild_patterns=args.rebuild_patterns,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
