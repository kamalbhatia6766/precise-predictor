from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

# PR57: controlled regeneration for predictions + reporting backup

import pandas as pd

import quant_paths
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


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def _iter_dates(start: date, end: Optional[date]) -> Iterable[date]:
    cursor = start
    while True:
        yield cursor
        if end is None or cursor >= end:
            break
        cursor += timedelta(days=1)


def _extract_date_from_name(path: Path) -> Optional[date]:
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d").date()
    except Exception:
        return None


def _backup_prediction_files(script_dir: Path, date_range: Sequence[date]) -> None:
    if not script_dir.exists():
        return
    date_set = set(date_range)
    targets: List[Path] = []
    for item in script_dir.iterdir():
        if not item.is_file():
            continue
        file_date = _extract_date_from_name(item)
        if file_date is None or (date_set and file_date not in date_set):
            continue
        targets.append(item)
    if not targets:
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = script_dir.parent / "backups" / f"{script_dir.name}_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    for item in targets:
        dest = backup_dir / item.name
        shutil.copy2(item, dest)
    print(f"Backed up {len(targets)} files for {script_dir.name} to {backup_dir}")


def regenerate_predictions(from_date: date, to_date: Optional[date], scripts: Sequence[str]) -> None:
    predictions_root = quant_paths.get_predictions_dir()
    if not predictions_root.exists():
        print("[TimeMachine] predictions directory missing; nothing to regenerate.")
        return

    if not scripts:
        print("[TimeMachine] No scripts provided for regeneration.")
        return

    date_range = list(_iter_dates(from_date, to_date))
    script_ids = [s.strip().upper() for s in scripts if s.strip()]
    if "ALL" in script_ids:
        script_ids = [p.name for p in predictions_root.iterdir() if p.is_dir()]
    available_dirs = {p.name.lower(): p for p in predictions_root.iterdir() if p.is_dir()}

    def _matching_dirs(script_id: str) -> List[Path]:
        lowered = script_id.lower()
        direct = available_dirs.get(lowered)
        if direct:
            return [direct]
        candidates = [p for name, p in available_dirs.items() if lowered in name]
        return candidates

    for script_id in script_ids:
        dirs = _matching_dirs(script_id)
        for script_dir in dirs:
            _backup_prediction_files(script_dir, date_range)

        scr_num = script_id.lower().replace("scr", "")
        script_path = Path(__file__).resolve().parent / f"deepseek_scr{scr_num}.py"
        if not script_path.exists():
            print(f"[TimeMachine] Script path not found for {script_id}: {script_path}")
            continue
        print(f"[TimeMachine] Regenerating {script_id} via {script_path.name} for {len(date_range)} day(s)...")
        result = subprocess.run([sys.executable, str(script_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"[TimeMachine] Regeneration failed for {script_id}: {result.returncode} | "
                f"{(result.stderr or '').splitlines()[-1:]}"
            )
        else:
            print(f"[TimeMachine] {script_id} regeneration completed.")


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
    parser.add_argument("--from-date", dest="from_date", required=True, help="YYYY-MM-DD from which to rebuild/regenerate")
    parser.add_argument("--to-date", dest="to_date", help="YYYY-MM-DD upper bound for regen")
    parser.add_argument("--rebuild-hit-memory", action="store_true")
    parser.add_argument("--rebuild-pnl", action="store_true")
    parser.add_argument("--rebuild-patterns", action="store_true")
    parser.add_argument("--regen", action="store_true", help="Regenerate predictions for selected scripts")
    parser.add_argument(
        "--scripts",
        type=str,
        default="",
        help="Comma-separated script ids for regeneration (e.g., SCR1,SCR2)",
    )
    args = parser.parse_args()

    parsed_from = _parse_date(args.from_date)
    parsed_to = _parse_date(args.to_date)
    if parsed_from is None:
        raise SystemExit("--from-date must be in YYYY-MM-DD format")

    if args.regen:
        script_list = args.scripts.split(",") if args.scripts else []
        regenerate_predictions(parsed_from, parsed_to, script_list)

    rebuild_flags = [args.rebuild_hit_memory, args.rebuild_pnl, args.rebuild_patterns]
    if args.regen and not any(rebuild_flags):
        # regen-only invocation
        return 0
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
