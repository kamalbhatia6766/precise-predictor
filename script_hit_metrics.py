from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import pandas as pd

from script_hit_memory_utils import load_script_hit_memory


def _load_memory() -> pd.DataFrame:
    """
    Load script_hit_memory.csv and return a DataFrame.
    Ensure 'date' is parsed as datetime.date for filtering.
    """

    df = load_script_hit_memory()
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce").dt.date
    if "hit_flag" in df.columns:
        df["hit_flag"] = pd.to_numeric(df.get("hit_flag"), errors="coerce")
    if "is_near_miss" in df.columns:
        df["is_near_miss"] = pd.to_numeric(df.get("is_near_miss"), errors="coerce")

    return df


def get_metrics_table(window_days: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute per-script metrics for the last `window_days` days.

    Returns:
        metrics_df: DataFrame with columns
            ["SCRIPT_ID", "DAYS", "EVENTS", "EXACT", "MIRROR", "NEIGHBOR",
             "EXTENDED", "EXACT_PCT", "EXTENDED_PCT", "SIGNAL"]
        summary: dict with high-level information about the window.
    """

    df = _load_memory()
    summary: Dict = {
        "has_data": False,
        "window_days": window_days,
        "from_date": None,
        "to_date": None,
        "total_events": 0,
    }

    if df.empty or "date" not in df.columns or df["date"].isna().all():
        return pd.DataFrame(), summary

    latest_date = df["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame(), summary

    cutoff_date = latest_date - timedelta(days=window_days - 1)
    df_window = df[df["date"] >= cutoff_date]
    if df_window.empty:
        return pd.DataFrame(), summary

    df_window = df_window.copy()
    # derive a canonical SCRIPT_ID from script_id / script_name
    if "script_id" in df_window.columns and df_window["script_id"].notna().any():
        df_window["SCRIPT_ID"] = df_window["script_id"].astype(str).str.strip()
    elif "script_name" in df_window.columns:
        df_window["SCRIPT_ID"] = df_window["script_name"].astype(str).str.strip()
    else:
        df_window["SCRIPT_ID"] = None

    # if script_id had gaps but script_name is present, backfill from script_name
    if "script_name" in df_window.columns:
        name_series = df_window["script_name"].astype(str).str.strip()
        df_window["SCRIPT_ID"] = df_window["SCRIPT_ID"].where(
            df_window["SCRIPT_ID"].notna() & (df_window["SCRIPT_ID"] != ""),
            name_series,
        )

    # drop rows that still do not have any identifier
    df_window = df_window.dropna(subset=["SCRIPT_ID"])
    df_window = df_window[df_window["SCRIPT_ID"].astype(str).str.strip() != ""]
    if df_window.empty:
        return pd.DataFrame(), summary

    df_window["hit_type"] = df_window.get("hit_type", "").fillna("").str.upper()
    df_window["hit_flag"] = pd.to_numeric(df_window.get("hit_flag", 0), errors="coerce").fillna(0).astype(int)
    df_window["is_near_miss"] = pd.to_numeric(df_window.get("is_near_miss", 0), errors="coerce").fillna(0).astype(int)

    grouped = df_window.groupby("SCRIPT_ID")
    records: List[Dict] = []
    for script_id, group in grouped:
        total_events = len(group)
        exact_hits = int((group["hit_type"] == "EXACT").sum())
        mirror_hits = int((group["hit_type"] == "MIRROR").sum())
        neighbor_hits = int((group["hit_type"] == "NEIGHBOR").sum())
        extended_hits = exact_hits + mirror_hits + neighbor_hits
        exact_pct = (exact_hits / total_events * 100.0) if total_events else 0.0
        extended_pct = (extended_hits / total_events * 100.0) if total_events else 0.0
        days = group["date"].nunique()
        records.append(
            {
                "SCRIPT_ID": script_id,
                "DAYS": days,
                "EVENTS": total_events,
                "EXACT": exact_hits,
                "MIRROR": mirror_hits,
                "NEIGHBOR": neighbor_hits,
                "EXTENDED": extended_hits,
                "EXACT_PCT": exact_pct,
                "EXTENDED_PCT": extended_pct,
            }
        )

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df, summary

    metrics_df = metrics_df.sort_values("EXTENDED_PCT", ascending=False).reset_index(drop=True)
    num_rows = len(metrics_df)
    if num_rows:
        top_cut = max(1, num_rows // 3)
        mid_cut = max(1, (2 * num_rows) // 3)
        metrics_df["SIGNAL"] = "LOW"
        metrics_df.loc[: top_cut - 1, "SIGNAL"] = "HIGH"
        metrics_df.loc[top_cut: mid_cut - 1, "SIGNAL"] = "MEDIUM"
    else:
        metrics_df["SIGNAL"] = "LOW"

    summary.update(
        {
            "has_data": True,
            "from_date": df_window["date"].min(),
            "to_date": df_window["date"].max(),
            "total_events": int(metrics_df["EVENTS"].sum()),
        }
    )

    return metrics_df, summary


def compute_script_metrics(window_days: int = 30):
    """
    Backwards-compatible wrapper for older scripts (e.g. deepseek_scr9.py).

    Returns the same (metrics_df, summary) tuple as get_metrics_table.
    """
    return get_metrics_table(window_days=window_days)


def compute_slot_heroes_and_weak(window_days: int = 30):
    """
    Backwards-compatible helper for older scripts (e.g. deepseek_scr9.py).

    Returns:
        slot_heroes: dict[slot -> list of "hero" SCRIPT_IDs]
        slot_weak:   dict[slot -> list of "weak" SCRIPT_IDs]
        metrics_df:  DataFrame from get_metrics_table()
        summary:     summary dict from get_metrics_table()
    """
    metrics_df, summary = get_metrics_table(window_days=window_days)

    # default slot structure
    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    empty_map = {slot: [] for slot in slots}

    # if there is no data in this window, return empty structures
    if (not summary.get("has_data")) or metrics_df is None or metrics_df.empty:
        return empty_map, empty_map, metrics_df, summary

    # hero scripts = those with SIGNAL == "HIGH"
    heroes = (
        metrics_df.loc[metrics_df["SIGNAL"] == "HIGH", "SCRIPT_ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # weak scripts = those with SIGNAL == "LOW"
    weak = (
        metrics_df.loc[metrics_df["SIGNAL"] == "LOW", "SCRIPT_ID"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # use the same hero/weak lists for all slots for now.
    # (If in future we track per-slot contributions separately,
    #  this function can be extended without breaking the signature.)
    slot_heroes = {slot: list(heroes) for slot in slots}
    slot_weak = {slot: list(weak) for slot in slots}

    return slot_heroes, slot_weak, metrics_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script-wise hit metrics for Precise Predictor.")
    parser.add_argument("--window", type=int, default=30, help="Number of days to look back from the latest date.")
    args = parser.parse_args()

    df_metrics, summary = get_metrics_table(window_days=args.window)

    if not summary.get("has_data"):
        print(f"No script hit memory data available for last {args.window} days.")
    else:
        print(f"=== SCRIPT HIT METRICS (last {args.window} days) ===")
        print(df_metrics.to_string(index=False))
