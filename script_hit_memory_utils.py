"""Utilities for maintaining script-level hit memory logs.

This module standardises append and load helpers for
``logs/performance/script_hit_memory.csv`` while keeping existing
behaviour backward compatible. All helpers are additive and safe to call
from batch or rebuild workflows.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
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
    df = pd.DataFrame(rows, columns=SCRIPT_HIT_MEMORY_HEADERS)
    if not df.empty:
        df = _dedupe_memory(df)
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


def load_script_hit_memory(window_days: int | None = None) -> pd.DataFrame:
    """Load and normalise the script hit memory CSV.

    The on-disk format is treated as canonical and left untouched. The returned
    DataFrame exposes a consistent schema for downstream consumers. If
    ``window_days`` is provided, the data is filtered to that trailing window
    based on the latest available ``date``.
    """

    memory_path = get_script_hit_memory_path()
    if not memory_path.exists():
        return pd.DataFrame()

    try:
        raw_df = pd.read_csv(memory_path)
    except Exception as exc:
        print(f"⚠️  Error reading script_hit_memory.csv: {exc}")
        return pd.DataFrame()

    if raw_df.empty:
        return pd.DataFrame()

    # Flexible column resolution
    date_col = _normalise_column(raw_df, ["date", "target_date"])
    slot_col = _normalise_column(raw_df, ["slot", "real_slot"])
    script_col = _normalise_column(raw_df, ["script_id", "script", "script_name"])
    number_col = _normalise_column(raw_df, ["number", "real_number"])
    hit_type_col = _normalise_column(raw_df, ["hit_type", "hit_flag", "hit"])
    is_final_col = _normalise_column(raw_df, ["is_final", "is_in_final_shortlist", "final", "final_shortlist"])
    layer_col = _normalise_column(raw_df, ["layer", "layer_name"])
    rank_col = _normalise_column(raw_df, ["top_rank", "rank"])
    source_col = _normalise_column(raw_df, ["source_file", "source", "file"])

    if date_col is None or slot_col is None or script_col is None or number_col is None:
        return pd.DataFrame()

    df = raw_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
    df = df[pd.notna(df[date_col])]
    df[slot_col] = df[slot_col].astype(str).str.upper()
    df[script_col] = df[script_col].astype(str).str.upper()
    df[number_col] = df[number_col].apply(lambda x: to_2d_str(x) if pd.notna(x) else None)
    if hit_type_col:
        df[hit_type_col] = df[hit_type_col].astype(str).str.upper()

    canonical = pd.DataFrame(
        {
            "date": df[date_col],
            "slot": df[slot_col],
            "script_id": df[script_col],
            "number": df[number_col],
            "hit_type": df[hit_type_col] if hit_type_col else None,
            "is_final": df[is_final_col].astype(bool) if is_final_col else False,
            "layer": df[layer_col] if layer_col else None,
            "top_rank": pd.to_numeric(df[rank_col], errors="coerce") if rank_col else pd.NA,
            "source_file": df[source_col] if source_col else None,
        }
    )

    if window_days:
        valid_dates = canonical["date"][pd.notna(canonical["date"])]
        if not valid_dates.empty:
            max_date = valid_dates.max()
            cutoff = max_date - timedelta(days=window_days - 1)
            canonical = canonical[canonical["date"] >= cutoff]

    canonical = canonical.reset_index(drop=True)
    return canonical

