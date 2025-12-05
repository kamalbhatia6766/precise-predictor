from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import quant_paths


SCRIPT_HIT_MEMORY_HEADERS: List[str] = [
    "date",
    "result_date",
    "slot",
    "script_id",
    "script_name",
    "predicted",
    "actual",
    "result",
    "hit_flag",
    "hit_type",
    "is_neighbor",
    "is_mirror",
    "is_s40",
    "is_family_164950",
    "rank",
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


def _align_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    df = df.copy()
    # normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "date": "date",
        "result_date": "result_date",
        "slot": "slot",
        "script_id": "script_id",
        "script_name": "script_name",
        "scriptid": "script_id",
        "script": "script_name",
        "predicted": "predicted",
        "predict": "predicted",
        "actual": "actual",
        "result": "result",
        "hit_flag": "hit_flag",
        "hit": "hit_flag",
        "hit_type": "hit_type",
        "rank": "rank",
        "predict_date": "predict_date",
        "predict_day": "predict_date",
        "source_file": "source_file",
        "is_near_miss": "is_near_miss",
        "near_miss": "is_near_miss",
        "pack_family": "pack_family",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ensure all expected columns exist
    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in df.columns:
            df[col] = None

    # collapse duplicate "result" / "actual" columns if they exist
    for col in ("result", "actual"):
        if col in df.columns:
            mask = df.columns == col
            if mask.sum() > 1:
                # merge duplicate columns by row-wise backfill and keep the first
                merged = df.loc[:, mask]
                series = merged.bfill(axis=1).iloc[:, 0]
                # drop all duplicates
                df = df.loc[:, ~mask]
                # re-attach a single canonical column
                df[col] = series

    # fill result from actual where result is missing
    if "result" in df.columns and "actual" in df.columns:
        result_series = df["result"]
        actual_series = df["actual"]
        df["result"] = result_series.where(result_series.notna(), actual_series)

    # enforce final column order
    df = df[SCRIPT_HIT_MEMORY_HEADERS]

    def _clean_script(value):
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).replace(" ", "").strip().upper()
        return text if text else None

    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_name"] = df.get("script_name").apply(_clean_script)
    df["script_id"] = df.get("script_id").apply(_clean_script)
    if "hit_type" in df.columns:
        mask = df.columns == "hit_type"
        if mask.sum() > 1:
            merged = df.loc[:, mask]
            series = merged.bfill(axis=1).iloc[:, 0]
            df = df.loc[:, ~mask]
            df["hit_type"] = series
        df["hit_type"] = df["hit_type"].astype(str).str.strip().str.lower()
    else:
        df["hit_type"] = "exact"
    for flag_col in [
        "hit_flag",
        "is_near_miss",
        "is_neighbor",
        "is_mirror",
        "is_s40",
        "is_family_164950",
    ]:
        if flag_col in df.columns:
            df[flag_col] = pd.to_numeric(df.get(flag_col), errors="coerce").fillna(0).astype(int)
    return df


def ensure_script_hit_memory_exists(base_dir: Optional[Path] = None) -> Path:
    """
    Ensure that script_hit_memory.csv exists with the correct headers.
    Return the Path to the CSV file.
    """

    csv_path = get_script_hit_memory_path(base_dir=base_dir)

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS).to_csv(csv_path, index=False)
        return csv_path

    df = pd.read_csv(csv_path, dtype=str)
    df = _align_columns(df)
    df.to_csv(csv_path, index=False)
    return csv_path


def load_script_hit_memory(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Load script_hit_memory.csv as a DataFrame with columns exactly SCRIPT_HIT_MEMORY_HEADERS.
    """

    ensure_script_hit_memory_exists(base_dir=base_dir)
    df = pd.read_csv(get_script_hit_memory_path(base_dir=base_dir), dtype=str)
    df = _align_columns(df)
    return df


def overwrite_script_hit_memory(df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    """
    Overwrite script_hit_memory.csv with df, realigned to SCRIPT_HIT_MEMORY_HEADERS.
    """

    csv_path = get_script_hit_memory_path(base_dir=base_dir)

    if df is None or df.empty:
        pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS).to_csv(csv_path, index=False)
        return csv_path

    aligned_df = _align_columns(df)
    aligned_df.to_csv(csv_path, index=False)
    return csv_path


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
    for col in ("result_date", "date"):
        if col in df.columns and not df[col].isna().all():
            return col
    return None


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
            exact_hits = (hit_types == "EXACT").sum()
            ext_hits = hit_types.isin({"NEIGHBOR", "MIRROR", "S40", "FAMILY_164950"}).sum()
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
