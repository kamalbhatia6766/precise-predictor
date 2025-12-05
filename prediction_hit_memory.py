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
    update_latest_script_hit_memory,
)

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
SLOT_ID_TO_NAME = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}
SLOT_NAME_TO_ID = {v: k for k, v in SLOT_ID_TO_NAME.items()}

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

def _parse_number(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() == "XX":
        return None
    try:
        num = int(float(text)) % 100
        return f"{num:02d}"
    except Exception:
        return None


def load_real_results_long(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load results into long form with columns: DATE, SLOT, REAL_NUM."""

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
            rows.append({"DATE": date_val.date(), "SLOT": slot, "REAL_NUM": num})

    real_df = pd.DataFrame(rows)
    if real_df.empty:
        raise ValueError("No valid real results found after parsing")

    min_date = real_df["DATE"].min()
    max_date = real_df["DATE"].max()
    print(
        f"📅 Real results loaded: {len(real_df)} rows from {min_date.isoformat()} to {max_date.isoformat()}"
    )
    return real_df


def get_completed_dates(real_df: pd.DataFrame) -> List[date]:
    counts = real_df.groupby("DATE")["SLOT"].nunique()
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


def find_prediction_files(base_dir: Path) -> Dict[str, List[Path]]:
    predictions_root = base_dir / "predictions"
    files: Dict[str, List[Path]] = {}

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
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
        files[script] = matches
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


def _normalise_slot_value(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.upper() in SLOTS:
        return text.upper()
    try:
        slot_id = int(float(text))
        return SLOT_ID_TO_NAME.get(slot_id)
    except Exception:
        return None


def _format_two_digit(num: Any) -> Optional[str]:
    parsed = _parse_number(num)
    return parsed


def load_prediction_dataframe(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path)
    if isinstance(raw, dict):
        raw = next(iter(raw.values()))
    df = pd.DataFrame(raw).copy()
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
                for idx, pred in enumerate(preds, start=1):
                    long_rows.append(
                        {
                            "target_date": target_val,
                            "slot": slot_name,
                            "predicted": pred,
                            "rank": idx,
                            "score": None,
                            "predict_date": row.get("predict_date") or row.get("bet_date"),
                            "pack_family": row.get("pack_family"),
                        }
                    )
        df = pd.DataFrame(long_rows)
    elif "numbers" in df.columns:
        # Long format with numbers list
        long_rows = []
        slot_col = "slot" if "slot" in df.columns else next((c for c in df.columns if "slot" in c), "slot")
        date_col = next((c for c in ["target_date", "date", "result_date", "game_date"] if c in df.columns), "date")
        for _, row in df.iterrows():
            preds = _explode_numbers(row.get("numbers"))
            for idx, pred in enumerate(preds, start=1):
                long_rows.append(
                    {
                        "target_date": row.get(date_col),
                        "slot": row.get(slot_col),
                        "predicted": pred,
                        "rank": idx,
                        "score": None,
                        "predict_date": row.get("predict_date") or row.get("bet_date"),
                        "pack_family": row.get("pack_family"),
                    }
                )
        df = pd.DataFrame(long_rows)
    else:
        df = _ensure_target_and_slot_columns(df)
        if "predicted" not in df.columns:
            pred_candidate = next((c for c in ["pred", "prediction", "number", "num"] if c in df.columns), None)
            if pred_candidate:
                df = df.rename(columns={pred_candidate: "predicted"})

    # Standardise types
    df["target_date"] = pd.to_datetime(df.get("target_date"), errors="coerce")
    df = df.dropna(subset=["target_date"])
    df["target_date"] = df["target_date"].dt.date
    if "slot" in df.columns:
        df["slot"] = df["slot"].apply(_normalise_slot_value)
    df = df.dropna(subset=["slot", "predicted"])

    df["predicted"] = df["predicted"].apply(_format_two_digit)
    df = df.dropna(subset=["predicted"])
    if "rank" in df.columns:
        df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
    if "predict_date" in df.columns:
        df["predict_date"] = pd.to_datetime(df["predict_date"], errors="coerce").dt.date
    return df


def load_predictions_map(base_dir: Path) -> Dict[str, pd.DataFrame]:
    files = find_prediction_files(base_dir)
    predictions: Dict[str, pd.DataFrame] = {}
    for script, paths in files.items():
        frames: List[pd.DataFrame] = []
        for path in paths:
            try:
                df = load_prediction_dataframe(path)
                if df is None or df.empty:
                    continue
                df["source_file"] = path.name
                frames.append(df)
                print(f"📁 {script.upper()}: {path.name} ({len(df)} rows)")
            except Exception as exc:
                print(f"❌ Error loading predictions for {script.upper()} from {path}: {exc}")
        if frames:
            predictions[script] = pd.concat(frames, ignore_index=True)
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
        (row["DATE"], row["SLOT"]): row["REAL_NUM"]
        for _, row in real_df.iterrows()
        if pd.notna(row.get("REAL_NUM"))
    }

    def classify_hit(predicted: Optional[str], actual: Optional[str]) -> str:
        if not predicted or not actual:
            return "MISS"
        if predicted == actual:
            return "EXACT"
        if predicted[::-1] == actual:
            return "MIRROR"
        try:
            pred_num = int(predicted)
            actual_num = int(actual)
        except Exception:
            return "MISS"
        if actual_num in {(pred_num + 1) % 100, (pred_num - 1) % 100}:
            return "NEIGHBOR"
        return "MISS"

    def pick_top_row(group: pd.DataFrame) -> pd.Series:
        if "rank" in group.columns and group["rank"].notna().any():
            sorted_group = group.sort_values("rank")
            return sorted_group.iloc[0]
        for score_col in ["score", "probability", "confidence", "weight"]:
            if score_col in group.columns and group[score_col].notna().any():
                sorted_group = group.sort_values(score_col, ascending=False)
                return sorted_group.iloc[0]
        return group.iloc[0]

    rows: List[Dict[str, Any]] = []

    for script_name, df in predictions_map.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        if "target_date" not in df.columns or "slot" not in df.columns:
            continue

        for current_date in dates:
            date_df = df[df["target_date"] == current_date]
            if date_df.empty:
                continue
            for slot in SLOTS:
                slot_df = date_df[date_df["slot"] == slot]
                if slot_df.empty:
                    continue
                key = (current_date, slot)
                if key not in real_lookup:
                    continue
                top_row = pick_top_row(slot_df)
                predicted_val = _format_two_digit(top_row.get("predicted"))
                if predicted_val is None:
                    continue
                actual_val = _format_two_digit(real_lookup[key])
                hit_type = classify_hit(predicted_val, actual_val)
                rank_val = top_row.get("rank")
                if pd.isna(rank_val):
                    rank_val = 1
                row: Dict[str, Any] = {key: None for key in SCRIPT_HIT_MEMORY_HEADERS}
                script_id = script_name.upper()
                row["DATE"] = current_date
                row["result_date"] = current_date
                row["SLOT"] = slot
                row["SCRIPT_ID"] = script_id
                row["script_name"] = script_id
                row["PREDICTED"] = predicted_val
                row["ACTUAL"] = actual_val
                row["result"] = actual_val
                row["HIT_TYPE"] = hit_type
                row["HIT_FLAG"] = int(hit_type == "EXACT")
                row["RANK"] = int(rank_val) if not pd.isna(rank_val) else None
                row["PREDICT_DATE"] = top_row.get("predict_date") or top_row.get("predict_day")
                row["SOURCE_FILE"] = top_row.get("source_file")
                row["is_near_miss"] = int(hit_type in {"MIRROR", "NEIGHBOR"})
                row["pack_family"] = top_row.get("pack_family")
                rows.append(row)
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
        f"Built {len(rows)} script-hit rows for {len(dates)} dates and 9 scripts."
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
        if "DATE" in memory_df.columns:
            last_result_date = pd.to_datetime(memory_df["DATE"], errors="coerce").dt.date.max()
        elif "result_date" in memory_df.columns:
            last_result_date = pd.to_datetime(memory_df["result_date"], errors="coerce").dt.date.max()
        elif "date" in memory_df.columns:
            last_result_date = pd.to_datetime(memory_df["date"], errors="coerce").dt.date.max()

    if last_result_date is not None and latest_completed <= last_result_date:
        print(f"Hit memory already up to date for {last_result_date}")
        return

    dates_to_update = [d for d in completed_dates if last_result_date is None or d > last_result_date]
    predictions_map = load_predictions_map(base_dir)
    rows = build_script_hit_rows_for_dates(real_df, predictions_map, dates_to_update)
    memory_path = update_latest_script_hit_memory(rows, base_dir=base_dir)
    if rows:
        print(f"Appended {len(rows)} rows for dates {dates_to_update} to {memory_path}")
    else:
        print("No new script hit rows to append (no predictions matched new dates).")


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
