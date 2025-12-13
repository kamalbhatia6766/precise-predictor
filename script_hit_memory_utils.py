from datetime import date, datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from openpyxl.utils.exceptions import InvalidFileException
import quant_paths
import pattern_packs
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=__name__,
)


SCRIPT_HIT_MEMORY_HEADERS: List[str] = [
    "date",
    "result_date",
    "real_slot",
    "slot",
    "script_id",
    "script_name",
    "predicted",
    "predicted_number",
    "actual",
    "real_number",
    "result",
    "hit_flag",
    "is_exact_hit",
    "is_near_hit",
    "hit_type",
    "is_neighbor",
    "is_mirror",
    "is_s40",
    "is_family_164950",
    "rank",
    "rank_in_script",
    "predict_date",
    "source_file",
    "is_near_miss",
    "pack_family",
    "note",
]


def _resolve_base_dir(base_dir: Optional[Path] = None) -> Path:
    return Path(base_dir) if base_dir else quant_paths.get_project_root()


def _normalise_slot(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().upper()
    if not text or text == "NAN":
        return None
    mapping = {"1": "FRBD", "2": "GZBD", "3": "GALI", "4": "DSWR"}
    return mapping.get(text, text)


def _clean_script(value: Optional[str]) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).replace(" ", "").strip().upper()
    return text if text else None


def get_script_hit_memory_path(base_dir: Optional[Path] = None) -> Path:
    """
    Return the absolute path to script_hit_memory.csv inside the project's logs/performance folder.
    Use quant_paths.get_project_root() / "logs" / "performance" / "script_hit_memory.csv".
    Ensure parent folders exist.
    """

    project_root = _resolve_base_dir(base_dir)
    logs_dir = project_root / "logs" / "performance"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "script_hit_memory.csv"


def get_script_hit_memory_xlsx_path(base_dir: Optional[Path] = None) -> Path:
    """Return the path to the canonical script_hit_memory.xlsx file."""

    project_root = _resolve_base_dir(base_dir)
    logs_dir = project_root / "logs" / "performance"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "script_hit_memory.xlsx"


def _parse_excel_date_series(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.NaT

    s = pd.to_datetime(series, errors="coerce")
    if s.notna().sum() > 0:
        return s

    # Fallback for true Excel-numeric date
    try:
        return pd.to_datetime(series, errors="coerce", unit="D", origin="1899-12-30")
    except Exception:
        return pd.to_datetime(series, errors="coerce")


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise a raw hit-memory frame into SCRIPT_HIT_MEMORY_HEADERS.

    This function is deliberately defensive:
    - Never relies on the truth value of a Series (to avoid "truth value is ambiguous").
    - Accepts a variety of upstream column name variants.
    - Always returns a frame with exactly SCRIPT_HIT_MEMORY_HEADERS (in that order).
    """
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    df = df.copy()

    # Map lower-case column name -> original name
    col_map = {c.lower(): c for c in df.columns if isinstance(c, str)}

    def _take_series(*names: str):
        """Return first matching column as Series, or None if none exist."""
        for name in names:
            key = name.lower()
            if key in col_map:
                return df[col_map[key]]
        return None

    out = pd.DataFrame(index=df.index)

    # --- date / result_date / predict_date ---------------------------------
    s_date = _take_series("date")
    if s_date is not None:
        out["date"] = _parse_excel_date_series(s_date)
    else:
        out["date"] = pd.NaT

    s_res_date = _take_series("result_date")
    if s_res_date is not None:
        out["result_date"] = _parse_excel_date_series(s_res_date)
    else:
        out["result_date"] = out["date"]

    s_pred_date = _take_series("predict_date", "bet_date")
    if s_pred_date is not None:
        out["predict_date"] = _parse_excel_date_series(s_pred_date)
    else:
        out["predict_date"] = out["date"]

    # --- slot ---------------------------------------------------------------
    s_slot = _take_series("slot", "slot_name", "clock", "slot_id", "real_slot")
    if s_slot is not None:
        out["slot"] = s_slot.astype(str).map(_normalise_slot)
    else:
        out["slot"] = "NONE"
    out["real_slot"] = out["slot"]

    # --- script_id / script_name -------------------------------------------
    s_script_id = _take_series("script_id", "script", "script_name",
                               "script_file", "scriptid", "script_id")
    if s_script_id is not None:
        out["script_id"] = s_script_id.map(_clean_script)
    else:
        out["script_id"] = "NONE"

    s_script_name = _take_series("script_name")
    if s_script_name is not None:
        out["script_name"] = s_script_name.astype(str)
    else:
        out["script_name"] = out["script_id"]

    # --- predicted / actual -------------------------------------------------
    s_pred = _take_series("predicted_number", "predicted", "prediction", "number",
                          "num", "top1", "top_1")
    if s_pred is not None:
        out["predicted"] = s_pred.astype(str).str.zfill(2)
    else:
        out["predicted"] = ""
    out["predicted_number"] = out["predicted"]

    s_actual = _take_series("real_number", "actual", "result", "outcome")
    if s_actual is not None:
        out["actual"] = s_actual.astype(str).str.zfill(2)
    else:
        out["actual"] = ""
    out["real_number"] = out["actual"]

    # --- result / hit flags -------------------------------------------------
    s_hit_flag = _take_series("hit_flag", "hit")
    if s_hit_flag is not None:
        out["hit_flag"] = s_hit_flag.astype(str)
    else:
        out["hit_flag"] = ""

    s_hit_type = _take_series("HIT_TYPE", "hit_type")
    if s_hit_type is not None:
        out["hit_type"] = s_hit_type.astype(str).str.upper()
    else:
        out["hit_type"] = ""

    # For convenience, mirror overall "result" as a simple text flag
    out["result"] = out["hit_flag"]

    # Neighbor / mirror flags
    s_is_neighbor = _take_series("is_neighbor", "neighbor_hit")
    if s_is_neighbor is not None:
        out["is_neighbor"] = s_is_neighbor.fillna(False).astype(bool)
    else:
        out["is_neighbor"] = False

    s_is_mirror = _take_series("is_mirror", "mirror_hit")
    if s_is_mirror is not None:
        out["is_mirror"] = s_is_mirror.fillna(False).astype(bool)
    else:
        out["is_mirror"] = False

    # S40 / 164950 family flags
    s_is_s40 = _take_series("is_s40")
    if s_is_s40 is not None:
        out["is_s40"] = s_is_s40.fillna(False).astype(bool)
    else:
        out["is_s40"] = False

    s_is_family = _take_series("is_family_164950", "is_164950")
    if s_is_family is not None:
        out["is_family_164950"] = s_is_family.fillna(False).astype(bool)
    else:
        out["is_family_164950"] = False

    # Rank (may be float / string)
    s_rank = _take_series("rank_in_script", "rank", "position", "pos")
    if s_rank is not None:
        out["rank"] = pd.to_numeric(s_rank, errors="coerce")
    else:
        out["rank"] = pd.NA
    out["rank_in_script"] = out["rank"]

    # Source file
    s_src = _take_series("source_file", "file", "file_name", "filename")
    if s_src is not None:
        out["source_file"] = s_src.astype(str)
    else:
        out["source_file"] = ""

    # Near-miss flag
    s_near = _take_series("is_near_miss", "near_miss")
    if s_near is not None:
        out["is_near_miss"] = s_near.fillna(False).astype(bool)
    else:
        out["is_near_miss"] = out["is_neighbor"] | out["is_mirror"]

    # Pack family + free-form note
    s_pack = _take_series("pack_family")
    if s_pack is not None:
        out["pack_family"] = s_pack.astype(str)
    else:
        out["pack_family"] = ""

    s_note = _take_series("note")
    if s_note is not None:
        out["note"] = s_note.astype(str)
    else:
        out["note"] = ""

    # Finally, enforce column order
    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in out.columns:
            # If any new header was added later and not handled above
            out[col] = pd.NA

    date_cols = ["predict_date", "result_date", "date"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out = out.dropna(subset=["predict_date", "result_date"], how="any")
    for col in ("predict_date", "result_date"):
        if col in out.columns:
            out[col] = out[col].dt.date

    return out[SCRIPT_HIT_MEMORY_HEADERS]


def ensure_script_hit_memory_exists(base_dir: Optional[Path] = None) -> Path:
    """
    Ensure that script_hit_memory.csv exists with the correct headers.
    Return the Path to the CSV file.
    """

    csv_path = get_script_hit_memory_path(base_dir=base_dir)

    xlsx_path = get_script_hit_memory_xlsx_path(base_dir=base_dir)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        empty_df = pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
        empty_df.to_csv(csv_path, index=False)
        empty_df.to_excel(xlsx_path, index=False)
        return csv_path

    df = pd.read_csv(csv_path, dtype=str)
    df = _align_columns(df)
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    return csv_path


def load_script_hit_memory(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load script hit memory, preferring the Excel source of truth."""

    base = _resolve_base_dir(base_dir)
    xlsx_path = get_script_hit_memory_xlsx_path(base_dir=base)
    csv_path = get_script_hit_memory_path(base_dir=base)

    df: Optional[pd.DataFrame] = None
    excel_error: Optional[Exception] = None

    if xlsx_path.exists():
        try:
            df = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")
        except (BadZipFile, InvalidFileException, ValueError) as exc:
            excel_error = exc

    if df is None and csv_path.exists():
        df = pd.read_csv(csv_path, dtype=str)

    if df is None:
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    df = _align_columns(df)

    if excel_error is not None:
        try:
            df.to_excel(xlsx_path, index=False)
        except Exception:
            pass

    def _to_bool(series: pd.Series, default: bool = False) -> pd.Series:
        filled = series.copy()
        filled = filled.infer_objects(copy=False)
        filled = filled.where(pd.notna(filled), default)
        filled = filled.astype(str)
        lowered = filled.str.lower()
        return lowered.isin({"true", "1", "yes", "y"})

    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
    df["slot"] = df.get("slot").astype(str)
    df["script_id"] = df.get("script_id").astype(str)
    df["HIT_TYPE"] = df.get("hit_type").astype(str).str.upper()
    df["is_exact_hit"] = df["HIT_TYPE"].eq("DIRECT") | df["HIT_TYPE"].eq("EXACT")
    df["is_near_hit"] = _to_bool(df.get("is_near_hit", False))
    df["is_mirror_hit"] = df["HIT_TYPE"].eq("MIRROR")
    df["is_neighbor_hit"] = df["HIT_TYPE"].eq("NEIGHBOR")
    df["is_near_miss"] = _to_bool(df.get("is_near_miss", False)) | df["HIT_TYPE"].isin(
        ["MIRROR", "NEIGHBOR", "CROSS_SLOT", "CROSS_DAY"]
    )
    return df


def overwrite_script_hit_memory(df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    """
    Overwrite script_hit_memory.csv with df, realigned to SCRIPT_HIT_MEMORY_HEADERS.
    """

    csv_path = get_script_hit_memory_path(base_dir=base_dir)
    xlsx_path = get_script_hit_memory_xlsx_path(base_dir=base_dir)

    if df is None or df.empty:
        aligned_df = pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
    else:
        aligned_df = _align_columns(df)

    aligned_df.to_csv(csv_path, index=False)
    aligned_df.to_excel(xlsx_path, index=False)
    return xlsx_path


def append_script_hit_row(row: Dict[str, object], base_dir: Optional[Path] = None) -> None:
    """
    Append one logical row to script_hit_memory.csv.
    Keys are normalised to SCRIPT_HIT_MEMORY_HEADERS.
    Missing keys are filled with None.
    """

    ensure_script_hit_memory_exists(base_dir=base_dir)

    normalised = {k.lower(): v for k, v in row.items()} if row else {}
    new_df = pd.DataFrame([normalised])
    new_df = _align_columns(new_df)

    current_df = load_script_hit_memory(base_dir=base_dir)
    combined_df = pd.concat([current_df, new_df], ignore_index=True)
    overwrite_script_hit_memory(combined_df, base_dir=base_dir)


def _choose_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("predict_date", "result_date", "date"):
        if col in df.columns and not df[col].isna().all():
            return col
    return None


def filter_hits_by_window(df: pd.DataFrame, window_days: int) -> Tuple[pd.DataFrame, int]:
    """Return a filtered frame limited to the last ``window_days``.

    The function is intentionally lightweight and reusable by other modules. It
    normalises the recognised date column (``result_date`` preferred, falling
    back to ``date``), applies a simple cutoff based on the latest available
    date, and returns both the filtered DataFrame and the number of distinct
    days present in that filtered slice.
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else []), 0

    date_col = _choose_date_column(df)
    if not date_col:
        return pd.DataFrame(columns=df.columns), 0

    work_df = df.copy()
    work_df[date_col] = pd.to_datetime(work_df[date_col], errors="coerce").dt.normalize()
    work_df = work_df.dropna(subset=[date_col])
    if work_df.empty:
        return work_df, 0

    window_end = work_df[date_col].max().normalize()
    window_start = window_end - timedelta(days=window_days)
    filtered = work_df[(work_df[date_col] >= window_start) & (work_df[date_col] <= window_end)]

    used_days = filtered[date_col].dt.date.nunique()
    logger = logging.getLogger(__name__)
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "filter_hits_by_window: days=%s, start=%s, end=%s, rows=%s, column=%s",
            window_days,
            window_start.date(),
            window_end.date(),
            len(filtered),
            date_col,
        )

    return filtered, int(used_days)


def classify_relation(pred: str, actual: str) -> str:
    pred = str(pred or "").zfill(2)
    actual = str(actual or "").zfill(2)
    if not pred.strip() or not actual.strip():
        return "NONE"
    if pred == actual:
        return "EXACT"
    if pattern_packs.is_mirror(pred, actual):
        return "MIRROR"
    if pattern_packs.is_adjacent(pred, actual):
        return "ADJACENT"
    if pattern_packs.is_diagonal_11(pred, actual):
        return "DIAGONAL_11"
    if pattern_packs.are_in_same_reverse_carry_cluster(pred, actual):
        return "REVERSE_CARRY"
    if pattern_packs.are_in_same_same_digit_cluster(pred, actual):
        return "SAME_DIGIT_COOL"
    return "NONE"


def _neutral_weight_map() -> Dict[Tuple[str, str], float]:
    scripts = [f"SCR{i}" for i in range(1, 10)]
    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    return {(script, slot): 1.0 for script in scripts for slot in slots}


def load_script_weights(window_days: int = 30, base_dir: Optional[Path] = None) -> Dict[Tuple[str, str], float]:
    """Lightweight slot-aware script weights based on recent hit memory.

    The output is a dict keyed by (script_name, slot) with conservative weights
    clipped to [0.4, 1.8]. If there is insufficient data, a neutral map of 1.0
    weights is returned.
    """

    df = load_script_hit_memory(base_dir=base_dir)
    if df is None or df.empty:
        return _neutral_weight_map()

    df = df.copy()
    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_name"] = df.get("script_name").astype(str).str.upper()

    date_col = _choose_date_column(df)
    if date_col is None:
        return _neutral_weight_map()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    latest_date = df[date_col].max()
    if pd.isna(latest_date):
        return _neutral_weight_map()
    cutoff = latest_date - timedelta(days=window_days - 1)
    window_df = df[df[date_col] >= cutoff]
    if window_df.empty:
        return _neutral_weight_map()

    weights: Dict[Tuple[str, str], float] = {}
    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    for slot in slots:
        slot_df = window_df[window_df["slot"] == slot]
        if slot_df.empty:
            for script in window_df["script_name"].unique():
                if not script:
                    continue
                weights[(str(script), slot)] = 1.0
            continue
        for script, group in slot_df.groupby("script_name"):
            if not script:
                continue
            script = str(script)
            total = len(group)
            hit_types = group.get("hit_type", "").astype(str).str.upper()
            exact_hits = hit_types.isin({"EXACT", "DIRECT"}).sum()
            ext_hits = hit_types.isin({"NEIGHBOR", "MIRROR", "S40", "FAMILY_164950", "CROSS_SLOT", "CROSS_DAY"}).sum()
            hit_rate_ext = (exact_hits + ext_hits) / total if total else 0.0
            base_weight = 0.8 + 1.2 * hit_rate_ext
            weight = max(0.4, min(1.8, base_weight))
            if total < 8:
                weight = 1.0 + (weight - 1.0) * 0.5
            weights[(script, slot)] = weight

    neutral = _neutral_weight_map()
    neutral.update(weights)
    return neutral


def rebuild_script_hit_memory(rows: List[Dict[str, object]], base_dir: Optional[Path] = None) -> Path:
    """Rebuild the entire script hit memory CSV from provided rows."""

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
    aligned_df = _align_columns(df)
    return overwrite_script_hit_memory(aligned_df, base_dir=base_dir)


def update_latest_script_hit_memory(rows: List[Dict[str, object]], base_dir: Optional[Path] = None) -> Path:
    """
    Append new rows to script_hit_memory.csv while avoiding duplicates on (date, slot, script_id).
    """

    ensure_script_hit_memory_exists(base_dir=base_dir)
    existing = load_script_hit_memory(base_dir=base_dir)
    new_df = _align_columns(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    if not new_df.empty:
        combined = pd.concat([existing, new_df], ignore_index=True)
        dedup_subset = [col for col in ["date", "slot"] if col in combined.columns]
        if "script_id" in combined.columns and combined["script_id"].notna().any():
            dedup_subset.append("script_id")
        elif "script_name" in combined.columns:
            dedup_subset.append("script_name")
        if dedup_subset:
            combined = combined.drop_duplicates(subset=dedup_subset, keep="last")
    else:
        combined = existing

    return overwrite_script_hit_memory(combined, base_dir=base_dir)
