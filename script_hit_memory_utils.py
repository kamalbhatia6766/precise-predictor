from pathlib import Path
from typing import Dict, List, Optional

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
    "rank",
    "predict_date",
    "source_file",
    "is_near_miss",
    "pack_family",
]


def _resolve_base_dir(base_dir: Optional[Path] = None) -> Path:
    return Path(base_dir) if base_dir else quant_paths.get_project_root()


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

    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in df.columns:
            df[col] = None

    if "result" in df.columns and "actual" in df.columns:
        df["result"] = df["result"].where(df["result"].notna(), df["actual"])

    df = df[SCRIPT_HIT_MEMORY_HEADERS]
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
