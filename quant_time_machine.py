from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from quant_core import hit_core, pattern_core, pnl_core
from quant_core.data_core import load_results_dataframe


def _as_date_string(value: object) -> str:
    import pandas as pd

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


def _write_golden_snapshot(snapshot: Optional[Dict], base_dir: Path) -> Optional[Path]:
    daily = snapshot.get("daily", []) if snapshot else []
    if not daily:
        return None

    df = pd.DataFrame(daily)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    if df.empty:
        return None

    golden_start = date(2025, 11, 9)
    golden_end = date(2025, 11, 13)
    mask = (df["date"] >= golden_start) & (df["date"] <= golden_end)
    golden_df = df[mask]
    if golden_df.empty:
        return None

    total_stake = float(golden_df.get("total_stake", 0).sum())
    total_pnl = float(golden_df.get("pnl", 0).sum())
    roi = total_pnl / total_stake * 100 if total_stake else None

    output_dir = base_dir / "logs" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "golden_days_snapshot.json"
    payload = {
        "window": {"start": golden_start.isoformat(), "end": golden_end.isoformat()},
        "total_stake": total_stake,
        "total_pnl": total_pnl,
        "roi_pct": roi,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"Golden days snapshot updated: {path}")
    return path


def rebuild_from_date(from_date: str, rebuild_hit_memory: bool, rebuild_pnl: bool, rebuild_patterns: bool) -> None:
    parsed_from = pd.to_datetime(from_date, errors="coerce")
    if pd.isna(parsed_from):
        parsed_from = None
    from_label = _as_date_string(parsed_from) if parsed_from is not None else _as_date_string(from_date)
    print(f"QUANT TIME MACHINE – Rebuilding from {from_label}")
    base_dir = Path(__file__).resolve().parent
    target_date = parsed_from.date() if parsed_from is not None else None
    results_df = load_results_dataframe()
    results_df["DATE"] = pd.to_datetime(results_df["DATE"], errors="coerce").dt.date
    if target_date:
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
        if parsed_from is not None:
            daily = snapshot.get("daily", []) if snapshot else []
            legacy_daily = []
            for item in daily:
                d = pd.to_datetime(item.get("date"), errors="coerce")
                if pd.isna(d):
                    continue
                if d.date() < parsed_from.date():
                    legacy_daily.append(item)
            if legacy_daily:
                legacy_path = base_dir / "logs" / "performance" / "quant_reality_pnl_legacy.json"
                legacy_payload = {
                    "preserved_before": _as_date_string(parsed_from),
                    "daily": legacy_daily,
                }
                legacy_path.write_text(json.dumps(legacy_payload, indent=2))
                print(f"Legacy P&L preserved to {legacy_path} ({len(legacy_daily)} days)")
        _write_golden_snapshot(snapshot, base_dir)

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
            try:
                if hit_df is not None and not hit_df.empty:
                    summary = pattern_core.build_pattern_summary(hit_df, window_days=pattern_window)
                    pattern_core.save_pattern_summary(summary, base_dir=base_dir, window_days=pattern_window)
            except Exception:
                pass
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
