from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import List

import pandas as pd

from script_hit_memory_utils import load_script_hit_memory


def _load_memory() -> pd.DataFrame:
    """
    Load script_hit_memory.csv and return a DataFrame.
    Ensure 'date' is parsed as datetime.date for filtering.
    """

    df = load_script_hit_memory()
    if df.empty:
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), errors="coerce").dt.date

    return df


def compute_script_metrics(window_days: int) -> pd.DataFrame:
    """
    Compute per-script hit metrics for the last `window_days` worth of results.
    Returns a DataFrame with one row per script.
    """

    df = _load_memory()
    if df.empty or "date" not in df.columns or df["date"].isna().all():
        return pd.DataFrame()

    latest_date = df["date"].max()
    if pd.isna(latest_date):
        return pd.DataFrame()

    cutoff_date = latest_date - timedelta(days=window_days - 1)
    df_window = df[df["date"] >= cutoff_date]

    if df_window.empty or "script_name" not in df_window.columns:
        return pd.DataFrame()

    df_window = df_window.dropna(subset=["script_name"])
    if df_window.empty:
        return pd.DataFrame()

    df_window["hit_flag"] = pd.to_numeric(df_window.get("hit_flag", 0), errors="coerce").fillna(0).astype(int)
    df_window["is_near_miss"] = pd.to_numeric(df_window.get("is_near_miss", 0), errors="coerce").fillna(0).astype(int)

    grouped = df_window.groupby("script_name")

    records: List[dict] = []
    for script_name, group in grouped:
        n_predictions = len(group)
        n_exact_hits = int((group["hit_flag"] == 1).sum())
        n_near_misses = int((group["is_near_miss"] == 1).sum())
        n_extended_hits = int(((group["hit_flag"] == 1) | (group["is_near_miss"] == 1)).sum())

        exact_hit_rate = n_exact_hits / n_predictions if n_predictions else 0.0
        extended_hit_rate = n_extended_hits / n_predictions if n_predictions else 0.0

        records.append(
            {
                "script_name": script_name,
                "n_predictions": n_predictions,
                "n_exact_hits": n_exact_hits,
                "n_near_misses": n_near_misses,
                "n_extended_hits": n_extended_hits,
                "exact_hit_rate": exact_hit_rate,
                "extended_hit_rate": extended_hit_rate,
            }
        )

    result_df = pd.DataFrame(records)
    if result_df.empty:
        return result_df

    result_df = result_df.sort_values(by=["extended_hit_rate", "n_predictions"], ascending=[False, False]).reset_index(drop=True)
    return result_df


def get_metrics_table(window_days: int = 30) -> pd.DataFrame:
    """
    Public wrapper used by quant_daily_brief.py.
    Simply calls compute_script_metrics(window_days).
    """

    return compute_script_metrics(window_days)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script-wise hit metrics for Precise Predictor.")
    parser.add_argument("--window", type=int, default=30, help="Number of days to look back from the latest date.")
    args = parser.parse_args()

    df_metrics = get_metrics_table(window_days=args.window)

    if df_metrics.empty:
        print(f"No script hit memory data available for last {args.window} days.")
    else:
        print(f"=== SCRIPT HIT METRICS (last {args.window} days) ===")
        print(df_metrics.to_string(index=False))
