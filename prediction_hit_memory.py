from __future__ import annotations

"""Build and maintain script-level hit memory from prediction files."""

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

import quant_data_core
from quant_core import hit_core
import pattern_packs
from script_hit_memory_utils import (
    SCRIPT_HIT_MEMORY_HEADERS,
    get_script_hit_memory_path,
    get_script_hit_memory_xlsx_path,
    load_script_hit_memory,
    overwrite_script_hit_memory,
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

PREDICTION_ROOT_CANDIDATES: Tuple[str, ...] = (
    "predictions",
    "predictions_archive",
    "predictions_archives",
    "old_predictions",
)


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
    files: Dict[str, List[Path]] = {}

    for root_name in PREDICTION_ROOT_CANDIDATES:
        root_dir = base_dir / root_name
        if not root_dir.exists():
            continue
        for script, patterns in SCRIPT_PATTERNS.items():
            script_dir = root_dir / f"deepseek_{script}"
            if not script_dir.exists():
                continue
            matches: List[Path] = []
            for pattern in patterns:
                matches.extend(script_dir.glob(pattern))
            if matches:
                files.setdefault(script, []).extend(matches)

    for script, paths in files.items():
        unique_paths = {p.resolve() for p in paths}
        sorted_paths = sorted(unique_paths, key=lambda p: p.stat().st_mtime, reverse=True)
        files[script] = sorted_paths
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
                df["script_id"] = script.upper()
                frames.append(df)
            except Exception as exc:
                print(f"❌ Error loading predictions for {script.upper()} from {path}: {exc}")
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            combined = combined.sort_values(["target_date", "slot", "rank"], na_position="last")
            predictions[script] = combined
            print(
                f"📁 {script.upper()}: {len(paths)} files, {len(combined)} rows (all sources)"
            )
    if not predictions:
        print("⚠️  No prediction files found in any configured root.")
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
        (row["DATE"], row["SLOT"]): _format_two_digit(row["REAL_NUM"])
        for _, row in real_df.iterrows()
        if pd.notna(row.get("REAL_NUM"))
    }
    s40_set = set(pattern_packs.S40_STRINGS)

    def _is_neighbor(pred: int, actual: int) -> bool:
        return actual in {(pred + 1) % 100, (pred - 1) % 100}

    def classify_hit(
        predicted: Optional[str],
        actual: Optional[str],
        other_slots: Iterable[str],
        slot: str,
        current_date: date,
    ) -> str:
        if not predicted or not actual:
            return "MISS"

        try:
            pred_num = int(predicted)
            actual_num = int(actual)
        except Exception:
            return "MISS"

        if predicted == actual:
            return "DIRECT"
        if predicted[::-1] == actual:
            return "MIRROR"
        if _is_neighbor(pred_num, actual_num):
            return "NEIGHBOR"

        other_hits = {num for num in other_slots if num and num != actual}
        if predicted in other_hits:
            return "CROSS_SLOT"

        prev_actual = real_lookup.get((current_date - timedelta(days=1), slot))
        next_actual = real_lookup.get((current_date + timedelta(days=1), slot))
        if predicted in {prev_actual, next_actual}:
            return "CROSS_DAY"
        return "MISS"

    rows: List[Dict[str, Any]] = []

    for script_name, df in predictions_map.items():
        if df is None or df.empty:
            continue

        df = df.copy()
        if "target_date" not in df.columns or "slot" not in df.columns:
            continue

        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce").dt.date
        df["slot"] = df["slot"].apply(_normalise_slot_value)
        df = df.dropna(subset=["target_date", "slot"])
        df["predicted"] = df["predicted"].apply(_format_two_digit)
        df = df.dropna(subset=["predicted"])
        df["rank"] = pd.to_numeric(df.get("rank"), errors="coerce")

        for current_date in dates:
            date_df = df[df["target_date"] == current_date]
            if date_df.empty:
                continue
            actuals_for_date = {slot: real_lookup.get((current_date, slot)) for slot in SLOTS}
            for slot in SLOTS:
                slot_actual = actuals_for_date.get(slot)
                if not slot_actual:
                    continue
                slot_df = date_df[date_df["slot"] == slot].copy()
                if slot_df.empty:
                    continue

                fallback_rank = pd.Series(range(1, len(slot_df) + 1), index=slot_df.index)
                slot_df["rank"] = slot_df["rank"].fillna(fallback_rank)
                slot_df = slot_df.sort_values("rank")

                other_slots = [num for s, num in actuals_for_date.items() if s != slot]
                for _, pred_row in slot_df.iterrows():
                    predicted_val = _format_two_digit(pred_row.get("predicted"))
                    if predicted_val is None:
                        continue
                    hit_type = classify_hit(predicted_val, slot_actual, other_slots, slot, current_date)
                    rank_val = pred_row.get("rank")
                    pack_family = None
                    if predicted_val in s40_set:
                        pack_family = "S40"
                    is_exact = int(hit_type == "DIRECT")
                    is_near = int(hit_type in {"MIRROR", "NEIGHBOR", "CROSS_SLOT", "CROSS_DAY"})
                    row: Dict[str, Any] = {key: None for key in SCRIPT_HIT_MEMORY_HEADERS}
                    script_id = script_name.upper()
                    row.update(
                        {
                            "DATE": current_date,
                            "result_date": current_date,
                            "SLOT": slot,
                            "real_slot": slot,
                            "SCRIPT_ID": script_id,
                            "script_name": script_id,
                            "PREDICTED": predicted_val,
                            "predicted_number": predicted_val,
                            "ACTUAL": slot_actual,
                            "real_number": slot_actual,
                            "result": slot_actual,
                            "HIT_TYPE": hit_type,
                            "hit_type": hit_type,
                            "HIT_FLAG": int(hit_type != "MISS"),
                            "is_neighbor": int(hit_type == "NEIGHBOR"),
                            "is_mirror": int(hit_type == "MIRROR"),
                            "is_s40": int(predicted_val in s40_set),
                            "is_family_164950": 0,
                            "is_exact_hit": is_exact,
                            "is_near_hit": is_near,
                            "RANK": int(rank_val) if not pd.isna(rank_val) else None,
                            "rank_in_script": int(rank_val) if not pd.isna(rank_val) else None,
                            "PREDICT_DATE": pred_row.get("predict_date") or pred_row.get("predict_day"),
                            "SOURCE_FILE": pred_row.get("source_file"),
                            "is_near_miss": is_near,
                            "pack_family": pack_family or pred_row.get("pack_family"),
                        }
                    )
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
    df = pd.DataFrame(rows)
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    memory_path = overwrite_script_hit_memory(df, base_dir=base_dir)
    print(f"Built {len(rows)} script-hit rows for {len(dates)} dates and 9 scripts.")
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
        last_result_date = pd.to_datetime(memory_df["result_date"], errors="coerce").dt.date.max()
        if pd.isna(last_result_date):
            last_result_date = None

    if last_result_date is not None and latest_completed <= last_result_date:
        print(f"Hit memory already up to date for {last_result_date}")
        return

    dates_to_update = [d for d in completed_dates if last_result_date is None or d > last_result_date]
    predictions_map = load_predictions_map(base_dir)
    rows = build_script_hit_rows_for_dates(real_df, predictions_map, dates_to_update)

    if rows:
        new_df = pd.DataFrame(rows)
        combined = pd.concat([memory_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["result_date", "slot", "script_id", "predicted_number", "source_file"],
            keep="last",
        )
        combined = combined.copy()
        if "date" in combined.columns:
            combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        memory_path = overwrite_script_hit_memory(combined, base_dir=base_dir)
        if dates_to_update:
            first_date = min(dates_to_update)
            last_date = max(dates_to_update)
            print(
                f"Appended {len(rows)} rows for dates "
                f"{first_date} → {last_date} to {memory_path}"
            )
        else:
            print(f"Appended {len(rows)} rows to {memory_path}")
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
            hit_core.rebuild_hit_memory(window_days=args.window)
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
