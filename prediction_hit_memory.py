from __future__ import annotations

"""Build and maintain script-level hit memory from prediction files."""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

import quant_data_core
from script_hit_memory_utils import (
    SCRIPT_HIT_MEMORY_HEADERS,
    append_script_hit_row,
    get_script_hit_memory_path,
    load_script_hit_memory,
    rebuild_script_hit_memory,
)

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]

SCRIPT_PATTERNS: Dict[str, List[str]] = {
    "scr1": ["scr1_precise_predictions_*.xlsx", "scr1_predictions_*.xlsx", "scr1_detailed_predictions_*.xlsx"],
    "scr2": ["scr2_predictions_*.xlsx"],
    "scr3": ["scr3_predictions_*.xlsx", "scr3_predictions_latest.xlsx"],
    "scr4": ["scr4_predictions_*.xlsx"],
    "scr5": ["scr5_predictions_*.xlsx"],
    "scr6": ["ultimate_predictions_long_*.xlsx", "ultimate_predictions_*.xlsx"],
    "scr7": ["advanced_predictions_*.xlsx"],
    "scr8": ["scr10_predictions_*.xlsx"],
    "scr9": ["ultimate_predictions_*.xlsx"],
}


# --------------------------------------------------------------------------------------
# Real results helpers
# --------------------------------------------------------------------------------------

def _parse_number(value: Any) -> Optional[int]:
    if pd.isna(value):
        return None
    try:
        num = int(float(str(value).strip()))
        return num % 100
    except Exception:
        return None


def load_real_results_long(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load results into long form with columns: date, slot, real_num."""

    df = quant_data_core.load_results_dataframe()
    if df is None or df.empty:
        raise ValueError("Real results file is empty or failed to load")

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        date_val = pd.to_datetime(row.get("DATE"), errors="coerce")
        if pd.isna(date_val):
            continue
        for slot in SLOTS:
            val = row.get(slot)
            num = _parse_number(val)
            if num is None:
                continue
            rows.append({"date": date_val.date(), "slot": slot, "real_num": num})

    real_df = pd.DataFrame(rows)
    if real_df.empty:
        raise ValueError("No valid real results found after parsing")

    min_date = real_df["date"].min()
    max_date = real_df["date"].max()
    print(
        f"📅 Real results loaded: {len(real_df)} rows from {min_date.isoformat()} to {max_date.isoformat()}"
    )
    return real_df


def get_completed_dates(real_df: pd.DataFrame) -> List[date]:
    counts = real_df.groupby("date")["slot"].nunique()
    return sorted([d for d, c in counts.items() if c == len(SLOTS)])


# --------------------------------------------------------------------------------------
# Prediction file helpers
# --------------------------------------------------------------------------------------

def normalize_script_name(path: Path) -> Optional[str]:
    lower = path.name.lower()
    for idx in range(1, 10):
        token = f"scr{idx}"
        if token in lower:
            return token
    for part in path.parts:
        lower_part = part.lower()
        for idx in range(1, 10):
            if f"scr{idx}" in lower_part:
                return f"scr{idx}"
    return None


def find_prediction_files(base_dir: Path) -> Dict[str, Path]:
    predictions_root = base_dir / "predictions"
    files: Dict[str, Path] = {}

    for script, patterns in SCRIPT_PATTERNS.items():
        script_dir = predictions_root / f"deepseek_{script}"
        if not script_dir.exists():
            print(f"⚠️  Directory not found for {script.upper()}: {script_dir}")
            continue

        matches: List[Path] = []
        for pattern in patterns:
            matches.extend(list(script_dir.glob(pattern)))
        if not matches:
            print(f"⚠️  No prediction files found for {script.upper()} in {script_dir}")
            continue
        latest = max(matches, key=lambda p: p.stat().st_mtime)
        files[script] = latest
    return files


def _explode_numbers(value: Any) -> List[int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        text = str(value)
        splitter = "," if "," in text else "|"
        values = [v.strip() for v in text.split(splitter) if v.strip()]
    numbers: List[int] = []
    for item in values:
        parsed = _parse_number(item)
        if parsed is not None:
            numbers.append(parsed)
    return numbers


def _ensure_target_and_slot_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "slot" not in cols:
        slot_candidate = next((c for c in df.columns if "slot" in str(c).lower()), None)
        if slot_candidate:
            df = df.rename(columns={slot_candidate: "slot"})
    if "target_date" not in cols:
        date_candidate = cols.get("date") or cols.get("result_date") or cols.get("game_date")
        if date_candidate:
            df = df.rename(columns={date_candidate: "target_date"})
    if "predict_date" not in cols:
        bet_candidate = cols.get("bet_date") or cols.get("prediction_date")
        if bet_candidate:
            df = df.rename(columns={bet_candidate: "predict_date"})
    return df


def load_prediction_dataframe(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    df = raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Wide format: slot columns
    slot_columns = [c for c in df.columns if any(slot.lower() in c for slot in SLOTS)]
    if slot_columns and "slot" not in df.columns:
        date_col = next((c for c in ["target_date", "date", "result_date", "game_date"] if c in df.columns), df.columns[0])
        long_rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            target_val = row.get(date_col)
            for col in slot_columns:
                slot_name = next((slot for slot in SLOTS if slot in col.upper()), col).upper()
                preds = _explode_numbers(row.get(col))
                for pred in preds:
                    long_rows.append(
                        {
                            "target_date": target_val,
                            "slot": slot_name,
                            "predicted": pred,
                            "predict_date": row.get("predict_date") or row.get("bet_date"),
                            "pack_family": row.get("pack_family"),
                        }
                    )
        return pd.DataFrame(long_rows)

    # Long format with numbers list
    if "numbers" in df.columns:
        long_rows = []
        slot_col = "slot"
        if "slot" not in df.columns:
            slot_col = next((c for c in df.columns if "slot" in c), "slot")
        date_col = next((c for c in ["target_date", "date", "result_date", "game_date"] if c in df.columns), "date")
        for _, row in df.iterrows():
            preds = _explode_numbers(row.get("numbers"))
            for pred in preds:
                long_rows.append(
                    {
                        "target_date": row.get(date_col),
                        "slot": row.get(slot_col),
                        "predicted": pred,
                        "predict_date": row.get("predict_date") or row.get("bet_date"),
                        "pack_family": row.get("pack_family"),
                    }
                )
        return pd.DataFrame(long_rows)

    df = _ensure_target_and_slot_columns(df)
    if "predicted" not in df.columns:
        pred_candidate = next((c for c in ["pred", "prediction", "number", "num"] if c in df.columns), None)
        if pred_candidate:
            df = df.rename(columns={pred_candidate: "predicted"})
    return df


def load_predictions_map(base_dir: Path) -> Dict[str, pd.DataFrame]:
    files = find_prediction_files(base_dir)
    predictions: Dict[str, pd.DataFrame] = {}
    for script, path in files.items():
        try:
            df = load_prediction_dataframe(path)
            row_count = len(df) if isinstance(df, pd.DataFrame) else 0
            print(f"📁 {script.upper()}: {path.name} ({row_count} rows)")
            predictions[script] = df
        except Exception as exc:
            print(f"❌ Error loading predictions for {script.upper()} from {path}: {exc}")
    return predictions


# --------------------------------------------------------------------------------------
# Row builder
# --------------------------------------------------------------------------------------

def build_script_hit_rows_for_dates(
    real_df: pd.DataFrame,
    predictions_map: Dict[str, pd.DataFrame],
    dates: List[date],
) -> List[Dict[str, Any]]:
    real_lookup = {
        (row["date"], row["slot"]): int(row["real_num"])
        for _, row in real_df.iterrows()
        if pd.notna(row.get("real_num"))
    }

    def make_row(date_val, slot, script_name, predicted, result, predict_date=None, pack_family=None, is_near_miss=False):
        row = {key: None for key in SCRIPT_HIT_MEMORY_HEADERS}
        row["date"] = date_val
        row["result_date"] = date_val
        row["slot"] = slot
        row["script_name"] = script_name
        row["predicted"] = int(predicted) if pd.notna(predicted) else None
        row["result"] = int(result) if result is not None else None
        row["hit_flag"] = int(predicted == result) if (predicted is not None and result is not None) else 0
        row["hit_type"] = "HIT" if row["hit_flag"] == 1 else "MISS"
        row["predict_date"] = predict_date or date_val
        row["is_near_miss"] = int(bool(is_near_miss))
        row["pack_family"] = pack_family
        return row

    rows: List[Dict[str, Any]] = []

    for script_name, df in predictions_map.items():
        if df is None or df.empty:
            continue
        df_columns = {c.lower(): c for c in df.columns}
        target_col = df_columns.get("target_date") or df_columns.get("date") or df_columns.get("result_date")
        slot_col = df_columns.get("slot")
        pred_col = df_columns.get("predicted") or df_columns.get("number") or df_columns.get("pred")
        predict_date_col = df_columns.get("predict_date") or df_columns.get("bet_date") or target_col
        pack_family_col = df_columns.get("pack_family")

        for _, row in df.iterrows():
            if target_col is None or slot_col is None or pred_col is None:
                continue
            target_val = row[target_col]
            if pd.isna(target_val):
                continue
            target_date = pd.to_datetime(target_val).date()
            if target_date not in dates:
                continue

            slot_val = str(row[slot_col]).strip().upper()
            key = (target_date, slot_val)
            if key not in real_lookup:
                continue
            result_val = real_lookup[key]
            predicted_val = row[pred_col]
            if pd.isna(predicted_val):
                continue

            predict_date_val = None
            if predict_date_col is not None:
                predict_raw = row[predict_date_col]
                if pd.notna(predict_raw):
                    predict_date_val = pd.to_datetime(predict_raw).date()

            pack_family_val = row[pack_family_col] if pack_family_col is not None else None

            rows.append(
                make_row(
                    date_val=target_date,
                    slot=slot_val,
                    script_name=script_name,
                    predicted=int(predicted_val),
                    result=int(result_val),
                    predict_date=predict_date_val,
                    pack_family=pack_family_val,
                    is_near_miss=False,
                )
            )
    return rows


# --------------------------------------------------------------------------------------
# CLI operations
# --------------------------------------------------------------------------------------

def _select_window_dates(real_df: pd.DataFrame, window_days: int) -> List[date]:
    completed_dates = get_completed_dates(real_df)
    if not completed_dates:
        return []
    max_date = completed_dates[-1]
    if window_days:
        start_date = max_date - timedelta(days=window_days - 1)
        return [d for d in completed_dates if start_date <= d <= max_date]
    return completed_dates


def _rebuild_script_hit_memory(window_days: int) -> None:
    base_dir = Path(__file__).resolve().parent
    real_df = load_real_results_long(base_dir)
    dates = _select_window_dates(real_df, window_days)
    if not dates:
        print("No complete result dates available for rebuild.")
        return

    predictions_map = load_predictions_map(base_dir)
    rows = build_script_hit_rows_for_dates(real_df, predictions_map, dates)
    memory_path = rebuild_script_hit_memory(rows, base_dir=base_dir)
    print(
        f"Built {len(rows)} script-hit rows for {len(dates)} dates and {len(predictions_map)} scripts."
    )
    print(f"Script hit memory rebuilt at {memory_path}")


def _update_latest_script_hit_memory() -> None:
    base_dir = Path(__file__).resolve().parent
    real_df = load_real_results_long(base_dir)
    completed_dates = get_completed_dates(real_df)
    if not completed_dates:
        print("No completed dates found in real results.")
        return
    latest_completed = completed_dates[-1]

    memory_df = load_script_hit_memory(base_dir=base_dir)
    last_result_date: Optional[date] = None
    if not memory_df.empty:
        if "result_date" in memory_df.columns:
            last_result_date = pd.to_datetime(memory_df["result_date"], errors="coerce").dt.date.max()
        elif "date" in memory_df.columns:
            last_result_date = pd.to_datetime(memory_df["date"], errors="coerce").dt.date.max()

    if last_result_date is not None and latest_completed <= last_result_date:
        print(f"Hit memory already up to date for {last_result_date}")
        return

    dates_to_update = [d for d in completed_dates if last_result_date is None or d > last_result_date]
    predictions_map = load_predictions_map(base_dir)
    rows = build_script_hit_rows_for_dates(real_df, predictions_map, dates_to_update)
    for row in rows:
        append_script_hit_row(row, base_dir=base_dir)

    memory_path = get_script_hit_memory_path(base_dir)
    print(f"Appended {len(rows)} rows for dates {dates_to_update} to {memory_path}")


# --------------------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Prediction hit memory toolkit")
    parser.add_argument("--mode", choices=["legacy", "rebuild", "update-latest"], default="legacy")
    parser.add_argument("--window", type=int, default=30, help="Window in days for rebuild mode")
    args = parser.parse_args()

    try:
        if args.mode == "rebuild":
            _rebuild_script_hit_memory(args.window)
            return 0
        if args.mode == "update-latest":
            _update_latest_script_hit_memory()
            return 0

        # Legacy mode now mirrors update-latest for convenience
        _update_latest_script_hit_memory()
        return 0
    except Exception as exc:
        print(f"❌ Error: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
