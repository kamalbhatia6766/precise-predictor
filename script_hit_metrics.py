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
    df_window["SCRIPT_ID"] = df_window.get("script_id") or df_window.get("script_name")
    df_window["SCRIPT_ID"] = df_window["SCRIPT_ID"].fillna(df_window.get("script_name"))
    df_window = df_window.dropna(subset=["SCRIPT_ID"])
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
