"""Utilities for maintaining script-level hit memory logs.

This module standardises append and load helpers for
``logs/performance/script_hit_memory.csv`` while keeping existing
behaviour backward compatible. All helpers are additive and safe to call
from batch or rebuild workflows.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

import quant_paths


# Required columns for the new hit-memory layer. Legacy columns are preserved
# for backward compatibility.
_BASE_HEADERS = [
    "date",  # result date (game date)
    "slot",  # FRBD / GZBD / GALI / DSWR
    "script_name",  # scr1..scr9
    "predicted",  # predicted number
    "result",  # actual result
    "hit_flag",  # 1/0
    "hit_type",  # HIT / MISS
    "predict_date",  # when the prediction was made
    "result_date",  # same as date but explicit
    "is_near_miss",  # 1/0
    "pack_family",  # optional
]

_LEGACY_HEADERS = [
    "real_number",
    "top_predictions",
    "is_in_final_shortlist",
    "created_at",
]

SCRIPT_HIT_MEMORY_HEADERS = _BASE_HEADERS + [
    col for col in _LEGACY_HEADERS if col not in _BASE_HEADERS
]


def get_script_hit_memory_path(base_dir: Optional[Path] = None) -> Path:
    """Return the canonical CSV path for script hit memory.

    The path is ``logs/performance/script_hit_memory.csv`` under the provided
    ``base_dir`` (or the project root via ``quant_paths``). Parent directories
    are created as needed.
    """

    if base_dir is None:
        performance_dir = quant_paths.get_performance_logs_dir()
    else:
        performance_dir = Path(base_dir) / "logs" / "performance"
    performance_dir.mkdir(parents=True, exist_ok=True)
    return performance_dir / "script_hit_memory.csv"


def _ensure_script_hit_memory(
    headers: Iterable[str] = SCRIPT_HIT_MEMORY_HEADERS,
    base_dir: Optional[Path] = None,
) -> Path:
    """Ensure the CSV exists with headers.

    This function is idempotent; if the file already exists, it is left
    untouched. Otherwise a new file is created with the provided headers.
    """

    memory_path = get_script_hit_memory_path(base_dir)
    if not memory_path.exists():
        with memory_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(list(headers))
    return memory_path


def append_script_hit_row(row: Dict[str, Any], base_dir: Optional[Path] = None) -> None:
    """Append a single script hit record to the unified CSV.

    Missing fields are filled with ``None``; extra keys are ignored.
    """

    memory_path = _ensure_script_hit_memory(base_dir=base_dir)
    prepared = {key: row.get(key) if isinstance(row, dict) else None for key in SCRIPT_HIT_MEMORY_HEADERS}
    df = pd.DataFrame([prepared])

    if memory_path.exists():
        try:
            existing = pd.read_csv(memory_path)
        except Exception as exc:
            print(f"⚠️  Error reading existing script_hit_memory.csv: {exc}")
            existing = pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
        combined = pd.concat([existing, df], ignore_index=True)
        combined.to_csv(memory_path, index=False)
    else:
        df.to_csv(memory_path, index=False)


def rebuild_script_hit_memory(
    rows: List[Dict[str, Any]], base_dir: Optional[Path] = None
) -> Path:
    """Overwrite the CSV with a provided collection of rows."""

    memory_path = _ensure_script_hit_memory(base_dir=base_dir)
    df = pd.DataFrame(rows, columns=SCRIPT_HIT_MEMORY_HEADERS)
    df.to_csv(memory_path, index=False)
    return memory_path


def _normalise_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return the first matching column name from a list of candidates."""

    normalised = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in normalised:
            return normalised[key]
    return None


def load_script_hit_memory(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and normalise the script hit memory CSV.

    If the file does not exist, an empty DataFrame with the canonical headers
    is returned. Any parsing error is surfaced to help diagnostics.
    """

    memory_path = get_script_hit_memory_path(base_dir)
    if not memory_path.exists():
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    try:
        raw_df = pd.read_csv(memory_path)
    except Exception as exc:
        print(f"⚠️  Error reading script_hit_memory.csv: {exc}")
        raise

    if raw_df.empty:
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    raw_df = raw_df.dropna(how="all")
    if raw_df.empty:
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)

    # Ensure required columns exist even if missing in file
    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in raw_df.columns:
            raw_df[col] = None

    # Normalise date columns
    for col in ["date", "predict_date", "result_date"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce").dt.date

    raw_df = raw_df.dropna(how="all")
    return raw_df.reset_index(drop=True)

