"""Script-level hit metrics based on script_hit_memory.csv."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd

import quant_paths
from script_hit_memory_utils import load_script_hit_memory

DEFAULT_WINDOW_DAYS = 30


def _load_memory_df(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and normalise the raw script hit memory CSV."""

    df = load_script_hit_memory(base_dir=base_dir)
    if df is None or df.empty:
        print("script_hit_memory.csv is present but empty.")
        return pd.DataFrame()

    rename_map = {}
    for col in df.columns:
        key = str(col).strip()
        if key.lower() in {"date", "result_date"}:
            rename_map[col] = "DATE"
        elif key.lower() in {"script_id", "script_name", "script"}:
            rename_map[col] = "SCRIPT_ID"
        elif key.lower() in {"hit_type"}:
            rename_map[col] = "HIT_TYPE"
        elif key.lower() in {"predicted", "prediction"}:
            rename_map[col] = "PREDICTED"
        elif key.lower() in {"actual", "result"}:
            rename_map[col] = "ACTUAL"
    if rename_map:
        df = df.rename(columns=rename_map)

    required_cols = {"DATE", "SCRIPT_ID", "HIT_TYPE", "PREDICTED", "ACTUAL"}
    if missing := [c for c in required_cols if c not in df.columns]:
        print(f"script_hit_memory.csv missing required columns: {missing}")
        return pd.DataFrame()

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    df = df.dropna(subset=["DATE"])
    if df.empty:
        print("script_hit_memory.csv has no valid dates after parsing.")
        return pd.DataFrame()

    df["SCRIPT_ID"] = df["SCRIPT_ID"].astype(str).str.upper()
    df["HIT_TYPE"] = df["HIT_TYPE"].astype(str).str.upper()
    df["ACTUAL"] = df["ACTUAL"].astype(str).str.zfill(2)
    df["PREDICTED"] = df["PREDICTED"].astype(str).str.zfill(2)
    return df


def _normalise_hit_flag(values: Iterable) -> pd.Series:
    """Return a boolean Series indicating whether a prediction was a hit."""

    def _to_bool(value: object) -> bool:
        if isinstance(value, bool):
            return value
        try:
            num = float(value)
            if not pd.isna(num):
                return num != 0
        except Exception:
            pass

        text = str(value).strip().upper()
        if text in {"HIT", "EXACT", "TRUE", "YES", "Y", "T", "1"}:
            return True
        if text in {"MISS", "FALSE", "NO", "N", "F", "0"}:
            return False
        return False

    return pd.Series([_to_bool(v) for v in values])


def compute_script_metrics(
    df: pd.DataFrame | None = None, window_days: int = DEFAULT_WINDOW_DAYS
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Compute windowed script metrics from script_hit_memory.csv."""

    base_df = df if df is not None else _load_memory_df()
    if base_df is None or base_df.empty:
        print("No script hit memory available for metrics.")
        return pd.DataFrame(), {}

    working_df = base_df.copy()
    latest_date = working_df["DATE"].max()
    if window_days and window_days > 0:
        cutoff = latest_date - timedelta(days=window_days - 1)
        working_df = working_df[(working_df["DATE"] >= cutoff) & (working_df["DATE"] <= latest_date)]

    if working_df.empty:
        print(f"No script hit memory rows in the last {window_days} days.")
        return pd.DataFrame(), {}

    records = []
    for script_id, group in working_df.groupby("SCRIPT_ID"):
        total_events = len(group)
        days_covered = group["DATE"].nunique()
        hit_types = group["HIT_TYPE"].astype(str).str.upper()
        exact_hits = int((hit_types == "EXACT").sum())
        mirror_hits = int((hit_types == "MIRROR").sum())
        neighbor_hits = int((hit_types == "NEIGHBOR").sum())
        extended_hits = exact_hits + mirror_hits + neighbor_hits
        exact_rate = exact_hits / total_events if total_events else 0.0
        extended_rate = extended_hits / total_events if total_events else 0.0

        records.append(
            {
                "SCRIPT_ID": str(script_id).upper(),
                "days_covered": int(days_covered),
                "total_events": int(total_events),
                "exact_hits": int(exact_hits),
                "mirror_hits": int(mirror_hits),
                "neighbor_hits": int(neighbor_hits),
                "extended_hits": int(extended_hits),
                "exact_hit_rate": float(exact_rate),
                "extended_hit_rate": float(extended_rate),
            }
        )

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df, {}

    high_cut = metrics_df["extended_hit_rate"].quantile(0.66)
    low_cut = metrics_df["extended_hit_rate"].quantile(0.33)
    signals = []
    for rate in metrics_df["extended_hit_rate"]:
        if rate >= high_cut:
            signals.append("HIGH")
        elif rate >= low_cut:
            signals.append("MEDIUM")
        else:
            signals.append("LOW")
    metrics_df["signal"] = signals

    metrics_df = metrics_df.sort_values(
        ["extended_hit_rate", "exact_hit_rate", "total_events"], ascending=[False, False, False]
    ).reset_index(drop=True)

    summary = {
        "window_days": window_days,
        "latest_date": latest_date,
        "total_rows": int(working_df.shape[0]),
        "total_scripts": metrics_df.shape[0],
        "overall_exact_hits": int(metrics_df["exact_hits"].sum()),
        "overall_extended_hits": int(metrics_df["extended_hits"].sum()),
        "overall_events": int(metrics_df["total_events"].sum()),
    }
    return metrics_df, summary


def load_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> Dict:
    """Public API to obtain script metrics safely as a mapping."""

    try:
        df, summary = compute_script_metrics(window_days=window_days)
        if df.empty:
            return {}

        metrics_by_script: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            script_id = str(row.get("SCRIPT_ID", "")).upper()
            metrics_by_script[script_id] = {
                "total_events": int(row.get("total_events", 0) or 0),
                "days_covered": int(row.get("days_covered", 0) or 0),
                "exact_hits": int(row.get("exact_hits", 0) or 0),
                "mirror_hits": int(row.get("mirror_hits", 0) or 0),
                "neighbor_hits": int(row.get("neighbor_hits", 0) or 0),
                "extended_hits": int(row.get("extended_hits", 0) or 0),
                "exact_hit_rate": float(row.get("exact_hit_rate", 0.0) or 0.0),
                "extended_hit_rate": float(row.get("extended_hit_rate", 0.0) or 0.0),
                "signal": row.get("signal"),
            }

        sorted_scripts = df["SCRIPT_ID"].tolist()
        best_script = sorted_scripts[0] if sorted_scripts else None
        worst_script = sorted_scripts[-1] if sorted_scripts else None

        return {
            "window_days": window_days,
            "script_count": len(sorted_scripts),
            "total_rows": int(df["total_events"].sum()),
            "metrics_by_script": metrics_by_script,
            "sorted_scripts": sorted_scripts,
            "best_script": best_script,
            "worst_script": worst_script,
            "summary": summary,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"⚠️  Failed to load script metrics: {exc}")
        return {}


def compute_slot_heroes_and_weak(
    metrics_df: pd.DataFrame, min_rows: int = 10
) -> Dict[str, Dict[str, list]]:
    """Provide a lightweight hero/weak summary for previews."""

    summary: Dict[str, Dict[str, list]] = {}
    if metrics_df is None or metrics_df.empty:
        return summary

    eligible = metrics_df[metrics_df["total_events"] >= min_rows]
    heroes = eligible.sort_values("extended_hit_rate", ascending=False).head(3)[
        "SCRIPT_ID"
    ].tolist()
    weak = eligible.sort_values("extended_hit_rate", ascending=True).head(3)[
        "SCRIPT_ID"
    ].tolist()
    summary["ALL"] = {"heroes": heroes, "weak": weak}
    return summary


def get_metrics_table(window_days: int = DEFAULT_WINDOW_DAYS) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Convenience accessor for windowed metrics (returns DataFrame and summary)."""

    memory_df = _load_memory_df()
    if memory_df is None or memory_df.empty:
        return pd.DataFrame(), {}
    return compute_script_metrics(memory_df, window_days=window_days)


def _run_cli(window: int) -> None:
    df = _load_memory_df()
    if df.empty:
        print("No script hit memory available for metrics.")
        return

    metrics_df, summary = compute_script_metrics(df, window_days=window)
    if metrics_df.empty:
        print("No metrics generated (script_hit_memory empty or no rows in this window).")
        return

    logs_dir = quant_paths.get_performance_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / f"script_hit_metrics_window{window}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"SCRIPT METRICS (last {window} days)")
    header = "SCRIPT  DAYS  EVENTS  EXACT  EXT(H+M+N)  EXACT_%  EXT_%  SIGNAL"
    print(header)
    for _, row in metrics_df.iterrows():
        print(
            f"{row['SCRIPT_ID']:<6} "
            f"{row['days_covered']:>4}  "
            f"{row['total_events']:>6}  "
            f"{row['exact_hits']:>5}  "
            f"{row['extended_hits']:>11}  "
            f"{row['exact_hit_rate'] * 100:>6.1f}  "
            f"{row['extended_hit_rate'] * 100:>5.1f}  "
            f"{row['signal']}"
        )
    print(
        f"GLOBAL: events={summary.get('overall_events', 0)} | exact_hits={summary.get('overall_exact_hits', 0)} | "
        f"extended_hits={summary.get('overall_extended_hits', 0)}"
    )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute script hit metrics based on script_hit_memory.csv"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Trailing window in days to include for metrics.",
    )
    args = parser.parse_args()

    _run_cli(args.window)
