"""Utilities for maintaining script-level hit memory logs.

This module standardises append and load helpers for
``logs/performance/script_hit_memory.csv`` while keeping existing
behaviour backward compatible. All helpers are additive and safe to call
from batch or rebuild workflows.
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

import quant_paths
from utils_2digit import to_2d_str


SCRIPT_HIT_MEMORY_HEADERS = [
    "date",
    "slot",
    "script_name",
    "real_number",
    "top_predictions",
    "is_in_final_shortlist",
    "hit_flag",
    "hit_type",
    "created_at",
]


def get_script_hit_memory_path() -> Path:
    """Return the canonical CSV path for script hit memory."""

    performance_dir = quant_paths.get_performance_logs_dir()
    performance_dir.mkdir(parents=True, exist_ok=True)
    return performance_dir / "script_hit_memory.csv"


def ensure_script_hit_memory(headers: Iterable[str] = SCRIPT_HIT_MEMORY_HEADERS) -> Path:
    """Ensure the CSV exists with headers.

    This function is idempotent; if the file already exists, it is left
    untouched. Otherwise a new file is created with the provided headers.
    """

    memory_path = get_script_hit_memory_path()
    if not memory_path.exists():
        with memory_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(headers))
    return memory_path


def append_script_hit_row(
    date: str,
    slot: str,
    script_name: str,
    real_number: str,
    top_predictions: Iterable[int] | Iterable[str],
    is_in_final_shortlist: bool,
    hit_flag: str,
    hit_type: str,
    created_at: Optional[str] = None,
) -> None:
    """Append a single script hit record to the unified CSV.

    The caller is responsible for providing validated slot and script
    names. ``top_predictions`` will be normalised to a pipe-separated
    string of zero-padded numbers.
    """

    memory_path = ensure_script_hit_memory()
    created_ts = created_at or datetime.now().isoformat()

    top_pred_str = "|".join([to_2d_str(p) for p in top_predictions])

    row = {
        "date": date,
        "slot": slot,
        "script_name": script_name,
        "real_number": to_2d_str(real_number),
        "top_predictions": top_pred_str,
        "is_in_final_shortlist": bool(is_in_final_shortlist),
        "hit_flag": hit_flag,
        "hit_type": hit_type,
        "created_at": created_ts,
    }

    df = pd.DataFrame([row])
    if memory_path.exists():
        try:
            existing = pd.read_csv(memory_path)
        except Exception:
            existing = pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = _dedupe_memory(combined)
        combined.to_csv(memory_path, index=False)
    else:
        df.to_csv(memory_path, index=False)


def _dedupe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on the primary key fields."""

    key_cols = ["date", "slot", "script_name"]
    if not set(key_cols).issubset(df.columns):
        return df
    return (
        df.drop_duplicates(subset=key_cols, keep="last")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )


def rebuild_script_hit_memory(rows: List[Dict]) -> Path:
    """Overwrite the CSV with a provided collection of rows."""

    memory_path = ensure_script_hit_memory()
    df = pd.DataFrame(rows)
    if not df.empty:
        df = _dedupe_memory(df)
    df.to_csv(memory_path, index=False)
    return memory_path


def load_script_hit_memory() -> pd.DataFrame:
    """Load the existing script hit memory as a DataFrame.

    Returns an empty DataFrame if the file does not exist.
    """

    memory_path = get_script_hit_memory_path()
    if not memory_path.exists():
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
    try:
        df = pd.read_csv(memory_path)
    except Exception:
        df = pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
    return df

