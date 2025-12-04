"""Utilities for maintaining script-level hit memory logs.

This module standardises append and load helpers for
``logs/performance/script_hit_memory.csv`` while keeping existing
behaviour backward compatible. All helpers are additive and safe to call
from batch or rebuild workflows.
"""Utilities for maintaining script-level hit memory logs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

import quant_paths


# Required columns for the new hit-memory layer. Legacy columns are preserved
# for backward compatibility. Canonical columns are upper-case to match the
# contract shared with downstream dashboards.
_CORE_HEADERS = [
    "DATE",  # result date (game date)
    "SLOT",  # FRBD / GZBD / GALI / DSWR
    "SCRIPT_ID",  # SCR1..SCR9
    "HIT_TYPE",  # EXACT / MIRROR / NEIGHBOR / MISS
    "PREDICTED",  # predicted number (two-digit string)
    "ACTUAL",  # actual result (two-digit string)
    "RANK",  # integer rank (1 = top)
    "SOURCE_FILE",  # filename used for the prediction
    "PREDICT_DATE",  # when the prediction was made
]

_LEGACY_HEADERS = [
    "script_name",  # keep legacy alias
    "result",  # legacy name for ACTUAL
    "result_date",
    "hit_flag",
    "is_near_miss",
    "pack_family",
    "real_number",
    "top_predictions",
    "is_in_final_shortlist",
    "created_at",
]

SCRIPT_HIT_MEMORY_HEADERS: List[str] = []
for col in _CORE_HEADERS + _LEGACY_HEADERS:
    if col not in SCRIPT_HIT_MEMORY_HEADERS:
        SCRIPT_HIT_MEMORY_HEADERS.append(col)


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

    # Normalize columns to expected names (case-insensitive)
    rename_map = {}
    for col in list(raw_df.columns):
        key = str(col).strip()
        upper_key = key.upper()
        if upper_key in SCRIPT_HIT_MEMORY_HEADERS and key != upper_key:
            rename_map[col] = upper_key
        elif key.lower() in {"date", "result_date"}:
            rename_map[col] = "DATE"
        elif key.lower() == "slot":
            rename_map[col] = "SLOT"
        elif key.lower() in {"script", "script_name", "script_id"}:
            rename_map[col] = "SCRIPT_ID"
        elif key.lower() in {"predicted", "prediction", "number"}:
            rename_map[col] = "PREDICTED"
        elif key.lower() in {"actual", "result"}:
            rename_map[col] = "ACTUAL"
    if rename_map:
        raw_df = raw_df.rename(columns=rename_map)

    for col in SCRIPT_HIT_MEMORY_HEADERS:
        if col not in raw_df.columns:
            raw_df[col] = None

    for col in ["DATE", "PREDICT_DATE", "RESULT_DATE"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce").dt.date

    # Fill compatibility aliases
    raw_df["script_name"] = raw_df.get("script_name") if "script_name" in raw_df.columns else raw_df.get("SCRIPT_ID")
    raw_df["result"] = raw_df.get("result") if "result" in raw_df.columns else raw_df.get("ACTUAL")

    raw_df = raw_df.dropna(how="all")
    return raw_df.reset_index(drop=True)

