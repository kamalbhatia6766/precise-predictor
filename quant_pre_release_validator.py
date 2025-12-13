"""
Pre-release validator for Precise Predictor.
This script is read-only and performs sanity checks before live betting.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import quant_paths

    PROJECT_ROOT = quant_paths.get_project_root()
except Exception:
    PROJECT_ROOT = Path(__file__).resolve().parent


any_fail: bool = False
fail_count: int = 0


def report(test_id: str, ok: bool, msg: str, warn: bool = False) -> None:
    """Pretty print test results and update global failure counters."""
    global any_fail, fail_count
    if ok and not warn:
        print(f"✅ [{test_id}] {msg}")
    elif warn:
        print(f"⚠️ [{test_id}] {msg}")
    else:
        print(f"❌ [{test_id}] {msg}")
        any_fail = True
        fail_count += 1


def load_excel_safe(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists():
        return None, "missing"
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return df, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"error: {exc}"


def has_nan_or_inf(df: pd.DataFrame, numeric_only: bool = True) -> Dict[str, Dict[str, int]]:
    if numeric_only:
        cols = df.select_dtypes(include=["number"]).columns
    else:
        cols = df.columns
    issues: Dict[str, Dict[str, int]] = {}
    for col in cols:
        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        nan_count = numeric_series.isna().sum()
        inf_count = np.isinf(numeric_series).sum()
        if nan_count or inf_count:
            issues[str(col)] = {"nan": int(nan_count), "inf": int(inf_count)}
    return issues


def safe_float(x: Any) -> float:
    return float(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))


def normalize_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {str(col).upper(): str(col) for col in df.columns}


def find_column(columns_map: Dict[str, str], target: str) -> Optional[str]:
    return columns_map.get(target.upper())


def is_valid_2d(value: Any) -> bool:
    try:
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                return False
            if not value.isdigit():
                return False
            num = int(value)
        else:
            num = int(float(value))
        return 0 <= num <= 99
    except Exception:
        return False


def run_all_tests(skip_heavy: bool = False) -> bool:
    global any_fail, fail_count
    any_fail = False
    fail_count = 0

    # T1 – Core input file exists and loads
    t1_path = PROJECT_ROOT / "number prediction learn.xlsx"
    df_t1, status = load_excel_safe(t1_path)
    if status == "missing":
        report("T1", False, f"{t1_path.name} not found at {t1_path}")
    elif status.startswith("error"):
        report("T1", False, f"Failed to load {t1_path.name}: {status}")
    else:
        col_map = normalize_columns(df_t1)
        required_cols = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]
        missing = [col for col in required_cols if col not in col_map]
        if missing:
            report("T1", False, f"Missing columns in dataset: {', '.join(missing)}")
        else:
            rows = len(df_t1)
            if rows < 200:
                report("T1", True, f"{t1_path.name} loaded ({rows} rows); dataset small (<200).", warn=True)
            else:
                report("T1", True, f"{t1_path.name} loaded ({rows} rows, DATE+4 slots OK).")

    # T2 – script_hit_memory date sanity + overflow guard
    t2_path = PROJECT_ROOT / "logs" / "performance" / "script_hit_memory.xlsx"
    df_hit: Optional[pd.DataFrame] = None
    status = ""
    df_hit, status = load_excel_safe(t2_path)
    if status == "missing":
        report("T2", True, "script_hit_memory.xlsx not found; run prediction_hit_memory.py first.", warn=True)
    elif status.startswith("error"):
        report("T2", False, f"Failed to load script_hit_memory.xlsx: {status}")
    else:
        col_map = normalize_columns(df_hit)
        predict_col = find_column(col_map, "PREDICT_DATE")
        result_col = find_column(col_map, "RESULT_DATE")
        script_col = find_column(col_map, "SCRIPT_ID")
        slot_col = find_column(col_map, "SLOT")

        missing_cols = [name for name, col in {"predict_date": predict_col, "result_date": result_col, "script_id": script_col, "slot": slot_col}.items() if col is None]
        if missing_cols:
            report("T2", False, f"Missing required columns: {', '.join(missing_cols)}")
        else:
            date_cols = [predict_col, result_col]
            out_of_range = []
            nat_counts: List[str] = []
            for col in date_cols:
                with np.errstate(over="ignore", invalid="ignore"):
                    parsed = pd.to_datetime(df_hit[col], errors="coerce")
                invalid_mask = (parsed < pd.Timestamp("1990-01-01")) | (parsed > pd.Timestamp("2050-12-31"))
                if invalid_mask.any():
                    samples = parsed[invalid_mask].head(3).dt.strftime("%Y-%m-%d").tolist()
                    out_of_range.append(f"{col} ({invalid_mask.sum()} invalid: {samples})")
                nat_ratio = parsed.isna().mean()
                if nat_ratio > 0.05:
                    nat_counts.append(f"{col} NaT {nat_ratio:.1%}")
            if out_of_range:
                report("T2", False, f"Date values out of 1990-2050 range: {'; '.join(out_of_range)}")
            elif nat_counts:
                report("T2", True, f"High NaT ratio in date columns: {', '.join(nat_counts)}", warn=True)
            else:
                report("T2", True, "script_hit_memory dates within 1990–2050; NaT <5%.")

    # T3 – NaN / inf check on script_hit_memory.xlsx
    if df_hit is None or status == "missing":
        report("T3", True, "script_hit_memory.xlsx not available; skipping NaN/inf check.", warn=True)
    elif status.startswith("error"):
        report("T3", True, "script_hit_memory load failed; skipping NaN/inf check.", warn=True)
    else:
        issues = has_nan_or_inf(df_hit)
        if issues:
            summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issues.items()])
            report("T3", True, f"script_hit_memory has NaN/inf in columns: {summary}", warn=True)
        else:
            report("T3", True, "script_hit_memory numeric columns are finite.")

    # T4 – P&L master summary NaN / inf + slot sanity
    pnl_xlsx = PROJECT_ROOT / "logs" / "performance" / "quant_reality_pnl.xlsx"
    pnl_json = PROJECT_ROOT / "logs" / "performance" / "quant_reality_pnl.json"
    df_pnl: Optional[pd.DataFrame] = None
    pnl_status = ""
    df_pnl, pnl_status = load_excel_safe(pnl_xlsx)
    if pnl_status == "missing" and not pnl_json.exists():
        report("T4", True, "No P&L summary files found; run bet_pnl_tracker.py first.", warn=True)
    elif pnl_status.startswith("error"):
        report("T4", False, f"Failed to load quant_reality_pnl.xlsx: {pnl_status}")
    else:
        col_map = normalize_columns(df_pnl) if df_pnl is not None else {}
        target_cols = ["TOTAL_BET", "TOTAL_RETURN", "NET_PNL", "ROI_%"]
        missing = [col for col in target_cols if col not in col_map]
        if missing:
            report("T4", False, f"quant_reality_pnl.xlsx missing columns: {', '.join(missing)}")
        else:
            issues = has_nan_or_inf(df_pnl)
            hard_cols = [col_map[c] for c in ["TOTAL_BET", "TOTAL_RETURN", "ROI_%"] if c in col_map]
            hard_issue_rows: List[str] = []
            for col in hard_cols:
                series = df_pnl[col]
                nan_mask = series.isna()
                inf_mask = np.isinf(series.astype(float, errors="ignore"))
                if nan_mask.any() or inf_mask.any():
                    idx_list = series[nan_mask | inf_mask].index[:5].tolist()
                    hard_issue_rows.append(f"{col} rows {idx_list}")
            if hard_issue_rows:
                report("T4", False, f"NaN/inf detected in critical columns: {', '.join(hard_issue_rows)}")
            elif issues:
                summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issues.items()])
                report("T4", True, f"quant_reality_pnl.xlsx has minor NaN/inf issues: {summary}", warn=True)
            else:
                report("T4", True, "quant_reality_pnl.xlsx numeric stats finite and sane.")

            # Slot-level sanity
            slot_col = find_column(col_map, "SLOT")
            slot_list = ["FRBD", "GZBD", "GALI", "DSWR"]
            roi_col = find_column(col_map, "ROI_%")
            bet_col = find_column(col_map, "TOTAL_BET")
            return_col = find_column(col_map, "TOTAL_RETURN")
            for slot in slot_list:
                if slot_col is None or bet_col is None or return_col is None or roi_col is None:
                    break
                slot_rows = df_pnl[df_pnl[slot_col].astype(str).str.upper() == slot]
                if slot_rows.empty:
                    continue
                bet_negative = slot_rows[slot_rows[bet_col] < 0]
                return_negative = slot_rows[slot_rows[return_col] < 0]
                if not bet_negative.empty or not return_negative.empty:
                    report("T4", False, f"Slot {slot} has negative totals (bet or return).")
                    continue
                zero_bet = slot_rows[slot_rows[bet_col] == 0]
                if not zero_bet.empty and roi_col is not None:
                    non_zero_roi = zero_bet[zero_bet[roi_col] != 0]
                    if not non_zero_roi.empty:
                        report("T4", True, f"Slot {slot} has ROI with zero bet rows ({len(non_zero_roi)}).", warn=True)
                if roi_col is not None:
                    extreme_roi = slot_rows[slot_rows[roi_col].abs() > 2000]
                    if not extreme_roi.empty:
                        report("T4", True, f"Slot {slot} has {len(extreme_roi)} ROI outliers (>2000%).", warn=True)

    # T5 – Top-N ROI invariants
    perf_csv = PROJECT_ROOT / "logs" / "performance" / "ultimate_performance.csv"
    df_perf: Optional[pd.DataFrame] = None
    perf_status = ""
    if not perf_csv.exists():
        report("T5", True, "ultimate_performance.csv not found; run deepseek_scr9 / SCR11 backtest first.", warn=True)
    else:
        try:
            df_perf = pd.read_csv(perf_csv)
            perf_status = "ok"
        except Exception as exc:
            perf_status = f"error: {exc}"
            report("T5", False, f"Failed to load ultimate_performance.csv: {exc}")
        if perf_status == "ok":
            col_map = normalize_columns(df_perf)
            date_col = find_column(col_map, "DATE")
            slot_col = find_column(col_map, "SLOT")
            if date_col and slot_col:
                df_perf[date_col] = pd.to_datetime(df_perf[date_col], errors="coerce")
                recent = df_perf[df_perf[date_col] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
                missing_slots = [slot for slot in ["FRBD", "GZBD", "GALI", "DSWR"] if recent[slot_col].astype(str).str.upper().eq(slot).sum() == 0]
                if missing_slots:
                    report("T5", True, f"Recent data missing for slots: {', '.join(missing_slots)} (last 30d)", warn=True)
            numeric_cols = df_perf.select_dtypes(include=["number"]).columns
            numeric_issue = has_nan_or_inf(df_perf[numeric_cols]) if not df_perf.empty else {}
            if numeric_issue:
                summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in numeric_issue.items()])
                report("T5", False, f"ultimate_performance.csv has NaN/inf in numeric columns: {summary}")
            else:
                report("T5", True, "ultimate_performance.csv basic sanity passed (finite numeric columns).")

    # Optional topn ROI snapshot
    topn_files = list((PROJECT_ROOT / "logs" / "performance").glob("*topn*roi*.csv"))
    for topn_file in topn_files:
        try:
            df_topn = pd.read_csv(topn_file)
        except Exception as exc:  # pragma: no cover - defensive
            report("T5", False, f"Failed to load {topn_file.name}: {exc}")
            continue
        col_map = normalize_columns(df_topn)
        roi_cols = [find_column(col_map, name) for name in ["TOP1_ROI", "TOP5_ROI", "TOP10_ROI", "TOP15_ROI"]]
        roi_cols = [c for c in roi_cols if c]
        if roi_cols:
            issue = has_nan_or_inf(df_topn[roi_cols])
            if issue:
                summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issue.items()])
                report("T5", False, f"{topn_file.name} has NaN/inf in ROI columns: {summary}")
            identical_count = 0
            for _, row in df_topn.iterrows():
                values = [safe_float(row[c]) for c in roi_cols]
                rounded = [round(v, 1) for v in values]
                if len(set(rounded)) == 1 and rounded[0] != 0:
                    identical_count += 1
            if identical_count:
                report("T5", True, f"{topn_file.name}: {identical_count} rows with identical topN ROI values.", warn=True)

    # T6 – Today’s bet plan structural sanity
    bet_plan_dir = PROJECT_ROOT / "predictions" / "bet_engine"
    bet_plan_files = sorted(bet_plan_dir.glob("bet_plan_master_*.xlsx"))
    if not bet_plan_files:
        report("T6", True, "No bet_plan_master_*.xlsx found; skipping bet plan checks.", warn=True)
    else:
        latest_plan = bet_plan_files[-1]
        df_plan, status = load_excel_safe(latest_plan)
        if status != "ok" or df_plan is None:
            report("T6", False, f"Failed to load {latest_plan.name}: {status}")
        else:
            col_map = normalize_columns(df_plan)
            date_col = find_column(col_map, "DATE") or find_column(col_map, "BET_DATE")
            slot_col = find_column(col_map, "SLOT")
            number_col = find_column(col_map, "NUMBER")
            stake_col = find_column(col_map, "STAKE")
            missing = [name for name, col in {"date/bet_date": date_col, "slot": slot_col, "number": number_col, "stake": stake_col}.items() if col is None]
            if missing:
                report("T6", False, f"Bet plan missing columns: {', '.join(missing)}")
            else:
                stakes_negative = df_plan[df_plan[stake_col] < 0]
                nan_stake = df_plan[df_plan[stake_col].isna()]
                nan_slot = df_plan[df_plan[slot_col].isna()]
                nan_number = df_plan[df_plan[number_col].isna()]
                invalid_numbers = df_plan[~df_plan[number_col].apply(is_valid_2d)]

                if not stakes_negative.empty:
                    report("T6", False, f"Found {len(stakes_negative)} rows with negative stakes in {latest_plan.name}.")
                if not nan_stake.empty:
                    report("T6", False, f"Stake column has NaN in {len(nan_stake)} rows.")
                if not nan_slot.empty:
                    report("T6", True, f"Slot column has NaN in {len(nan_slot)} rows.", warn=True)
                if not nan_number.empty:
                    report("T6", True, f"Number column has NaN in {len(nan_number)} rows.", warn=True)
                if not invalid_numbers.empty:
                    samples = invalid_numbers[number_col].head(5).tolist()
                    report("T6", False, f"Invalid numbers outside 0-99 found: {samples}")

                if stakes_negative.empty and nan_stake.empty and nan_slot.empty and nan_number.empty and invalid_numbers.empty:
                    report("T6", True, f"Latest bet plan {latest_plan.name}: stakes non-negative, numbers 00–99 valid.")

    # T7 – compute_pack_hit_stats smoke test
    if skip_heavy:
        report("T7", True, "Skipped by --skip-heavy flag.", warn=True)
    else:
        try:
            from quant_stats_core import compute_pack_hit_stats

            _stats = compute_pack_hit_stats(window_days=30, base_dir=PROJECT_ROOT)
            if _stats is None:
                report("T7", True, "compute_pack_hit_stats(30d) returned None (no crash).", warn=True)
            else:
                report("T7", True, "compute_pack_hit_stats(30d) completed without exceptions.")
        except Exception as exc:  # pragma: no cover - defensive
            report("T7", False, f"compute_pack_hit_stats(30d) raised exception: {exc}")

    # T8 – quant_daily_brief smoke test
    if skip_heavy:
        report("T8", True, "Skipped by --skip-heavy flag.", warn=True)
    else:
        try:
            import quant_daily_brief  # noqa: F401

            report("T8", True, "quant_daily_brief imported successfully (smoke test).")
        except ImportError as exc:
            report("T8", False, f"quant_daily_brief not importable: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            report("T8", False, f"quant_daily_brief smoke test failed: {exc}")

    return any_fail


def print_header() -> None:
    print("=" * 70)
    print("PRE-RELEASE VALIDATOR – Precise Predictor")
    print(f"Root: {PROJECT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-release validator for Precise Predictor.")
    parser.add_argument("--skip-heavy", action="store_true", help="Skip heavy tests like T7/T8.")
    args = parser.parse_args()

    print_header()
    any_fail = run_all_tests(skip_heavy=args.skip_heavy)

    print("\n" + "=" * 70)
    if fail_count:
        print(f"OVERALL STATUS: FAIL ({fail_count} failing tests)")
    elif any_fail:
        print("OVERALL STATUS: FAIL")
    else:
        print("OVERALL STATUS: PASS")
    print("=" * 70)

    if any_fail:
        sys.exit(1)
    else:
        sys.exit(0)
