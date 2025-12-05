from pathlib import Path
from typing import List, Dict

import pandas as pd
import quant_paths


SCRIPT_HIT_MEMORY_HEADERS: List[str] = [
    "date",
    "result_date",
    "slot",
    "script_name",
    "predicted",
    "result",
    "hit_flag",
    "hit_type",
    "predict_date",
    "is_near_miss",
    "pack_family",
]


def get_script_hit_memory_path() -> Path:
    """
    Return the absolute path to script_hit_memory.csv inside the project's logs folder.
    Use quant_paths.get_project_root() / "logs" / "script_hit_memory.csv".
    Ensure that the parent "logs" folder exists.
    """

    project_root = quant_paths.get_project_root()
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / "script_hit_memory.csv"


def ensure_script_hit_memory_exists() -> Path:
    """
    Ensure that script_hit_memory.csv exists with the correct header row.
    Return the Path to the CSV file.
    """

    csv_path = get_script_hit_memory_path()

    if not csv_path.exists():
        pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS).to_csv(csv_path, index=False)
        return csv_path

    df = pd.read_csv(csv_path, dtype=str) if csv_path.stat().st_size > 0 else pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in df.columns:
            df[col] = None

    df = df[SCRIPT_HIT_MEMORY_HEADERS]
    df.to_csv(csv_path, index=False)
    return csv_path


def load_script_hit_memory() -> pd.DataFrame:
    """
    Load script_hit_memory.csv as a DataFrame.
    Always returns a DataFrame with columns exactly SCRIPT_HIT_MEMORY_HEADERS.
    """

    ensure_script_hit_memory_exists()
    df = pd.read_csv(get_script_hit_memory_path(), dtype=str)

    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in df.columns:
            df[col] = None

    return df[SCRIPT_HIT_MEMORY_HEADERS]


def overwrite_script_hit_memory(df: pd.DataFrame) -> None:
    """
    Overwrite script_hit_memory.csv with df, realigned to SCRIPT_HIT_MEMORY_HEADERS.
    """

    csv_path = get_script_hit_memory_path()

    if df is None or df.empty:
        pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS).to_csv(csv_path, index=False)
        return

    aligned_df = df.copy()
    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in aligned_df.columns:
            aligned_df[col] = None

    aligned_df = aligned_df[SCRIPT_HIT_MEMORY_HEADERS]
    aligned_df.to_csv(csv_path, index=False)


def append_script_hit_row(row: Dict[str, object]) -> None:
    """
    Append one logical row to script_hit_memory.csv.
    Keys in row MUST be a subset of SCRIPT_HIT_MEMORY_HEADERS.
    Missing keys are filled with None.
    """

    ensure_script_hit_memory_exists()

    new_data = {col: row.get(col) if col in row else None for col in SCRIPT_HIT_MEMORY_HEADERS}
    new_df = pd.DataFrame([new_data], columns=SCRIPT_HIT_MEMORY_HEADERS)

    current_df = load_script_hit_memory()
    combined_df = pd.concat([current_df, new_df], ignore_index=True)
    overwrite_script_hit_memory(combined_df)
