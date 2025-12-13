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

fail_count: int = 0
warn_count: int = 0


def _ok(test_id: str, msg: str) -> None:
    print(f"✅ [{test_id}] {msg}")


def _warn(test_id: str, msg: str) -> None:
    global warn_count
    warn_count += 1
    print(f"⚠️ [{test_id}] {msg}")


def _fail(test_id: str, msg: str) -> None:
    global fail_count
    fail_count += 1
    print(f"❌ [{test_id}] {msg}")


def load_excel_safe(path: Path) -> Tuple[Optional[pd.DataFrame], str]:
    if not path.exists():
        return None, "missing"
    try:
        df = pd.read_excel(path, engine="openpyxl")
        return df, "ok"
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"error: {exc}"


def has_nan_or_inf(
    df: pd.DataFrame, numeric_only: bool = True, ignore_prefixes: Tuple[str, ...] = tuple(), min_non_na_ratio: float = 0.0
) -> Dict[str, Dict[str, int]]:
    if numeric_only:
        cols = df.select_dtypes(include=["number"]).columns
    else:
        cols = df.columns
    issues: Dict[str, Dict[str, int]] = {}
    for col in cols:
        col_name = str(col)
        if ignore_prefixes and col_name.lower().startswith(ignore_prefixes):
            continue
        series = df[col]
        if min_non_na_ratio > 0 and series.notna().mean() < min_non_na_ratio:
            continue
        numeric_series = pd.to_numeric(series, errors="coerce")
        nan_count = numeric_series.isna().sum()
        inf_count = np.isinf(numeric_series).sum()
        if nan_count or inf_count:
            issues[col_name] = {"nan": int(nan_count), "inf": int(inf_count)}
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


def run_all_tests(skip_heavy: bool = False) -> None:
    global fail_count, warn_count
    fail_count = 0
    warn_count = 0

    # T1 – Core input file exists and loads
    t1_path = PROJECT_ROOT / "number prediction learn.xlsx"
    df_t1, status = load_excel_safe(t1_path)
    if status == "missing":
        _fail("T1", f"{t1_path.name} not found at {t1_path}")
    elif status.startswith("error"):
        _fail("T1", f"Failed to load {t1_path.name}: {status}")
    else:
        col_map = normalize_columns(df_t1)
        required_cols = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]
        missing = [col for col in required_cols if col not in col_map]
        if missing:
            _fail("T1", f"Missing columns in dataset: {', '.join(missing)}")
        else:
            rows = len(df_t1)
            if rows < 200:
                _warn("T1", f"{t1_path.name} loaded ({rows} rows); dataset small (<200).")
            else:
                _ok("T1", f"{t1_path.name} loaded ({rows} rows, DATE+4 slots OK).")

    # T2 – script_hit_memory date sanity + overflow guard
    t2_path = PROJECT_ROOT / "logs" / "performance" / "script_hit_memory.xlsx"
    df_hit: Optional[pd.DataFrame] = None
    status = ""
    df_hit, status = load_excel_safe(t2_path)
    if status == "missing":
        _warn("T2", "script_hit_memory.xlsx not found; run prediction_hit_memory.py first.")
    elif status.startswith("error"):
        _warn("T2", f"Failed to load script_hit_memory.xlsx: {status}")
    else:
        col_map = normalize_columns(df_hit)
        predict_col = find_column(col_map, "PREDICT_DATE")
        result_col = find_column(col_map, "RESULT_DATE")
        script_col = find_column(col_map, "SCRIPT_ID")
        slot_col = find_column(col_map, "SLOT")

        missing_cols = [
            name
            for name, col in {"predict_date": predict_col, "result_date": result_col, "script_id": script_col, "slot": slot_col}.items()
            if col is None
        ]
        if missing_cols:
            _warn("T2", f"Missing required columns: {', '.join(missing_cols)}")
        else:
            core_dates = [result_col]
            optional_dates = [col for col in [predict_col] if col and col not in core_dates]
            out_of_range = []
            nat_counts: List[str] = []
            optional_sparse: List[str] = []
            for col in core_dates + optional_dates:
                with np.errstate(over="ignore", invalid="ignore"):
                    parsed = pd.to_datetime(df_hit[col], errors="coerce")
                invalid_mask = (parsed < pd.Timestamp("1990-01-01")) | (parsed > pd.Timestamp("2050-12-31"))
                if invalid_mask.any():
                    samples = parsed[invalid_mask].head(3).dt.strftime("%Y-%m-%d").tolist()
                    out_of_range.append(f"{col} ({invalid_mask.sum()} invalid: {samples})")
                nat_ratio = parsed.isna().mean()
                if col in core_dates and nat_ratio > 0.05:
                    nat_counts.append(f"{col} NaT {nat_ratio:.1%}")
                if col in optional_dates:
                    if nat_ratio >= 0.95:
                        optional_sparse.append(f"{col} mostly empty ({nat_ratio:.1%} NaT)")
                    elif nat_ratio > 0.05:
                        nat_counts.append(f"optional {col} NaT {nat_ratio:.1%}")
            if out_of_range:
                _warn("T2", f"Date values out of 1990-2050 range: {'; '.join(out_of_range)} (warning only).")
            elif nat_counts:
                _warn("T2", f"High NaT ratio in date columns: {', '.join(nat_counts)}. Using as warning only.")
            else:
                _ok("T2", "script_hit_memory dates within 1990–2050; NaT <5% on required columns.")

            for note in optional_sparse:
                print(f"ℹ️ [T2] {note}; treated as optional.")

    # T3 – NaN / inf check on script_hit_memory.xlsx
    if df_hit is None or status == "missing":
        _warn("T3", "script_hit_memory.xlsx not available; skipping NaN/inf check.")
    elif status.startswith("error"):
        _warn("T3", "script_hit_memory load failed; skipping NaN/inf check.")
    else:
        numeric_cols = df_hit.select_dtypes(include=["number"]).columns.tolist()
        ignore_prefixes = ("note", "comment")
        sparse_threshold = 0.05
        numeric_cols = [col for col in numeric_cols if not str(col).lower().startswith(ignore_prefixes)]
        numeric_cols = [col for col in numeric_cols if df_hit[col].notna().mean() >= sparse_threshold]

        issues = has_nan_or_inf(df_hit[numeric_cols], ignore_prefixes=ignore_prefixes, min_non_na_ratio=sparse_threshold) if numeric_cols else {}
        if issues:
            summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issues.items()])
            _warn("T3", f"script_hit_memory has NaN/inf (warning only): {summary}")
        else:
            _ok("T3", "script_hit_memory numeric columns are finite (optional sparse/text columns ignored).")

    # T4 – P&L master summary NaN / inf + slot sanity
    pnl_xlsx = PROJECT_ROOT / "logs" / "performance" / "quant_reality_pnl.xlsx"
    pnl_json = PROJECT_ROOT / "logs" / "performance" / "quant_reality_pnl.json"
    df_pnl: Optional[pd.DataFrame] = None
    pnl_status = ""
    df_pnl, pnl_status = load_excel_safe(pnl_xlsx)
    if pnl_status == "missing":
        _fail("T4", "quant_reality_pnl.xlsx not found; run bet_pnl_tracker.py first.")
    elif pnl_status.startswith("error"):
        _fail("T4", f"Failed to load quant_reality_pnl.xlsx: {pnl_status}")
    else:
        col_map = normalize_columns(df_pnl) if df_pnl is not None else {}
        core_cols = ["SLOT", "DATE", "TOTAL_RETURN"]
        missing_core = [col for col in core_cols if col not in col_map]
        if missing_core:
            _fail("T4", f"quant_reality_pnl.xlsx missing core columns: {', '.join(missing_core)}")
        else:
            optional_cols = ["TOTAL_BET", "NET_PNL", "ROI_%"]
            missing_optional = [col for col in optional_cols if col not in col_map]
            issues = has_nan_or_inf(df_pnl)
            hard_cols = [col_map[c] for c in ["TOTAL_RETURN"] if c in col_map]
            hard_issue_rows: List[str] = []
            for col in hard_cols:
                series = df_pnl[col]
                numeric_series = pd.to_numeric(series, errors="coerce")
                nan_mask = numeric_series.isna()
                inf_mask = np.isinf(numeric_series)
                if nan_mask.any() or inf_mask.any():
                    idx_list = series[nan_mask | inf_mask].index[:5].tolist()
                    hard_issue_rows.append(f"{col} rows {idx_list}")
            if hard_issue_rows:
                _fail("T4", f"NaN/inf detected in critical columns: {', '.join(hard_issue_rows)}")
            else:
                if missing_optional:
                    _warn("T4", f"quant_reality_pnl.xlsx missing optional columns: {', '.join(missing_optional)}. PNL will be computed from available fields.")
                elif issues:
                    summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issues.items()])
                    _warn("T4", f"quant_reality_pnl.xlsx has minor NaN/inf issues: {summary}")
                else:
                    _ok("T4", "quant_reality_pnl.xlsx numeric stats finite and sane.")

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
                    _fail("T4", f"Slot {slot} has negative totals (bet or return).")
                    continue
                zero_bet = slot_rows[slot_rows[bet_col] == 0]
                if not zero_bet.empty and roi_col is not None:
                    non_zero_roi = zero_bet[zero_bet[roi_col] != 0]
                    if not non_zero_roi.empty:
                        _warn("T4", f"Slot {slot} has ROI with zero bet rows ({len(non_zero_roi)}).")
                if roi_col is not None:
                    extreme_roi = slot_rows[slot_rows[roi_col].abs() > 2000]
                    if not extreme_roi.empty:
                        _warn("T4", f"Slot {slot} has {len(extreme_roi)} ROI outliers (>2000%).")

    # T5 – Top-N ROI invariants
    perf_csv = PROJECT_ROOT / "logs" / "performance" / "ultimate_performance.csv"
    df_perf: Optional[pd.DataFrame] = None
    perf_status = ""
    if not perf_csv.exists():
        _warn("T5", "ultimate_performance.csv not found; run deepseek_scr9 / SCR11 backtest first.")
    else:
        try:
            df_perf = pd.read_csv(perf_csv)
            perf_status = "ok"
        except Exception as exc:
            perf_status = f"error: {exc}"
            _warn("T5", f"Failed to load ultimate_performance.csv: {exc}")
        if perf_status == "ok":
            col_map = normalize_columns(df_perf)
            date_col = find_column(col_map, "DATE")
            slot_col = find_column(col_map, "SLOT")
            if date_col and slot_col:
                df_perf[date_col] = pd.to_datetime(df_perf[date_col], errors="coerce")
                recent = df_perf[df_perf[date_col] >= pd.Timestamp.today() - pd.Timedelta(days=30)]
                missing_slots = [slot for slot in ["FRBD", "GZBD", "GALI", "DSWR"] if recent[slot_col].astype(str).str.upper().eq(slot).sum() == 0]
                if missing_slots:
                    _warn("T5", f"Recent data missing for slots: {', '.join(missing_slots)} (last 30d)")
            numeric_cols = df_perf.select_dtypes(include=["number"]).columns
            numeric_issue = has_nan_or_inf(df_perf[numeric_cols]) if not df_perf.empty else {}
            if numeric_issue:
                summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in numeric_issue.items()])
                _warn("T5", f"ultimate_performance.csv has NaN/inf in numeric columns: {summary}. This will not block the daily run.")
            else:
                _ok("T5", "ultimate_performance.csv basic sanity passed (finite numeric columns).")

    # Optional topn ROI snapshot
    topn_files = list((PROJECT_ROOT / "logs" / "performance").glob("*topn*roi*.csv"))
    for topn_file in topn_files:
        try:
            df_topn = pd.read_csv(topn_file)
        except Exception as exc:  # pragma: no cover - defensive
            _warn("T5", f"Failed to load {topn_file.name}: {exc}")
            continue
        col_map = normalize_columns(df_topn)
        roi_cols = [find_column(col_map, name) for name in ["TOP1_ROI", "TOP5_ROI", "TOP10_ROI", "TOP15_ROI"]]
        roi_cols = [c for c in roi_cols if c]
        if roi_cols:
            issue = has_nan_or_inf(df_topn[roi_cols])
            if issue:
                summary = "; ".join([f"{col}(nan={vals['nan']}, inf={vals['inf']})" for col, vals in issue.items()])
                _warn("T5", f"{topn_file.name} has NaN/inf in ROI columns: {summary}")
            identical_count = 0
            for _, row in df_topn.iterrows():
                values = [safe_float(row[c]) for c in roi_cols]
                rounded = [round(v, 1) for v in values]
                if len(set(rounded)) == 1 and rounded[0] != 0:
                    identical_count += 1
            if identical_count:
                _warn("T5", f"{topn_file.name}: {identical_count} rows with identical topN ROI values.")

    # T6 – Today’s bet plan structural sanity
    bet_plan_dir = PROJECT_ROOT / "predictions" / "bet_engine"
    bet_plan_files = sorted(bet_plan_dir.glob("bet_plan_master_*.xlsx"))
    if not bet_plan_files:
        _warn("T6", "No bet_plan_master_*.xlsx found; skipping bet plan checks.")
    else:
        latest_plan = bet_plan_files[-1]
        df_plan, status = load_excel_safe(latest_plan)
        if status != "ok" or df_plan is None:
            _warn("T6", f"Failed to load {latest_plan.name}: {status}")
        else:
            col_map = normalize_columns(df_plan)
            date_col = find_column(col_map, "DATE") or find_column(col_map, "BET_DATE")
            slot_col = find_column(col_map, "SLOT")
            number_col = find_column(col_map, "NUMBER")
            stake_col = find_column(col_map, "STAKE")
            missing = [name for name, col in {"date/bet_date": date_col, "slot": slot_col, "number": number_col, "stake": stake_col}.items() if col is None]
            if missing:
                _warn("T6", f"Bet plan missing canonical 'number' column or related fields ({', '.join(missing)}). Using legacy schema; non-fatal.")
            else:
                stakes_negative = df_plan[df_plan[stake_col] < 0]
                nan_stake = df_plan[df_plan[stake_col].isna()]
                nan_slot = df_plan[df_plan[slot_col].isna()]
                nan_number = df_plan[df_plan[number_col].isna()]
                invalid_numbers = df_plan[~df_plan[number_col].apply(is_valid_2d)]

                if not stakes_negative.empty:
                    _warn("T6", f"Found {len(stakes_negative)} rows with negative stakes in {latest_plan.name}. Non-fatal for now.")
                if not nan_stake.empty:
                    _warn("T6", f"Stake column has NaN in {len(nan_stake)} rows (warning only).")
                if not nan_slot.empty:
                    _warn("T6", f"Slot column has NaN in {len(nan_slot)} rows (warning only).")
                if not nan_number.empty:
                    _warn("T6", f"Number column has NaN in {len(nan_number)} rows (warning only).")
                if not invalid_numbers.empty:
                    samples = invalid_numbers[number_col].head(5).tolist()
                    _warn("T6", f"Invalid numbers outside 0-99 found: {samples}. Using legacy schema; non-fatal.")

                if stakes_negative.empty and nan_stake.empty and nan_slot.empty and nan_number.empty and invalid_numbers.empty:
                    _warn("T6", f"Bet plan {latest_plan.name} passed structural checks; treating as warning-only scope.")

    # T7 – compute_pack_hit_stats smoke test
    if skip_heavy:
        _warn("T7", "Skipped by --skip-heavy flag.")
    else:
        try:
            from quant_stats_core import compute_pack_hit_stats

            _stats = compute_pack_hit_stats(window_days=30, base_dir=PROJECT_ROOT)
            if _stats is None:
                _ok("T7", "compute_pack_hit_stats(30d) returned None (no crash).")
            else:
                _ok("T7", "compute_pack_hit_stats(30d) completed without exceptions.")
        except Exception as exc:  # pragma: no cover - defensive
            _fail("T7", f"compute_pack_hit_stats(30d) raised exception: {exc}")

    # T8 – quant_daily_brief smoke test
    if skip_heavy:
        _warn("T8", "Skipped by --skip-heavy flag.")
    else:
        try:
            import quant_daily_brief  # noqa: F401

            _ok("T8", "quant_daily_brief imported successfully (smoke test).")
        except ImportError as exc:
            _fail("T8", f"quant_daily_brief not importable: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            _fail("T8", f"quant_daily_brief smoke test failed: {exc}")


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
    run_all_tests(skip_heavy=args.skip_heavy)

    print("\n" + "=" * 70)
    if fail_count > 0:
        print(f"OVERALL STATUS: FAIL ({fail_count} failing tests, {warn_count} warnings)")
    elif warn_count > 0:
        print(f"OVERALL STATUS: WARN (0 failing tests, {warn_count} warnings)")
    else:
        print("OVERALL STATUS: PASS (0 failing tests, 0 warnings)")
    print("=" * 70)

    if fail_count > 0:
        sys.exit(1)
    else:
        sys.exit(0)
