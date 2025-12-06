from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import quant_data_core


DEFAULT_COLUMNS = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]


def load_results_dataframe() -> pd.DataFrame:
    """Wrapper for quant_data_core.load_results_dataframe with basic cleaning."""

    df = quant_data_core.load_results_dataframe()
    if df is None or df.empty:
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    df = df.copy()
    # Standardise column order and names
    if "DATE" not in df.columns:
        date_col = next((c for c in df.columns if str(c).strip().upper() == "DATE"), None)
        if date_col:
            df = df.rename(columns={date_col: "DATE"})
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        if slot not in df.columns:
            candidate = next((c for c in df.columns if str(c).strip().upper() == slot), None)
            if candidate:
                df = df.rename(columns={candidate: slot})
    missing = [c for c in DEFAULT_COLUMNS if c not in df.columns]
    for col in missing:
        df[col] = pd.NA
    df = df[DEFAULT_COLUMNS]
    # Closed days are represented as "XX" in the existing pipelines
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        df[slot] = df[slot].fillna("XX").astype(str).str.upper()
    return df


def get_date_range(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df is None or df.empty or "DATE" not in df.columns:
        return None, None
    dates = pd.to_datetime(df["DATE"], errors="coerce")
    return dates.min(), dates.max()
