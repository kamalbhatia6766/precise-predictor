from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

import script_hit_metrics
from prediction_hit_memory import (
    _rebuild_script_hit_memory,
    _select_window_dates,
    build_script_hit_rows_for_dates,
    get_completed_dates,
    load_predictions_map,
    load_real_results_long,
)
from script_hit_memory_utils import overwrite_script_hit_memory


DEFAULT_WINDOW_DAYS = 90


def rebuild_hit_memory(window_days: int = DEFAULT_WINDOW_DAYS, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Rebuild script hit memory and return the resulting dataframe."""

    base_dir = base_dir or Path(__file__).resolve().parent.parent
    real_df = load_real_results_long(base_dir)
    dates = _select_window_dates(real_df, window_days)
    if not dates:
        print("No complete result dates available for rebuild.")
        return pd.DataFrame()

    predictions_map = load_predictions_map(base_dir)
    rows = build_script_hit_rows_for_dates(real_df, predictions_map, dates)
    hit_df = pd.DataFrame(rows)
    memory_path = overwrite_script_hit_memory(hit_df, base_dir=base_dir)
    print(f"Built {len(rows)} script-hit rows for {len(dates)} dates and 9 scripts.")
    print(f"Script hit memory rebuilt at {memory_path}")
    return hit_df


def compute_script_metrics(hit_df: Optional[pd.DataFrame] = None, window_days: int = 30, base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Compute metrics using the existing script_hit_metrics helper and persist to CSV."""

    base_dir = base_dir or Path(__file__).resolve().parent.parent
    if hit_df is None or hit_df.empty:
        metrics_df, _ = script_hit_metrics.get_metrics_table(window_days=window_days, base_dir=base_dir, mode="per_slot")
    else:
        temp_path = overwrite_script_hit_memory(hit_df, base_dir=base_dir)
        metrics_df, _ = script_hit_metrics.get_metrics_table(window_days=window_days, base_dir=base_dir, mode="per_slot")
    output_dir = base_dir / "logs" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"script_hit_metrics_window{window_days}.csv"
    metrics_df.to_csv(out_path, index=False)
    return metrics_df
