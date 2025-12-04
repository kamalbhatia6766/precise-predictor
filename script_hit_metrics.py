"""Script-level hit metrics based on script_hit_memory.csv."""

from __future__ import annotations

import argparse
from typing import Dict

import pandas as pd

from script_hit_memory_utils import get_script_hit_memory_path

DEFAULT_WINDOW_DAYS = 30


def _load_memory_df() -> pd.DataFrame:
    """Load and normalise the raw script hit memory CSV."""

    path = get_script_hit_memory_path()
    if not path.exists():
        print("No script_hit_memory.csv found; nothing to analyse yet.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error reading script_hit_memory.csv: {exc}")
        return pd.DataFrame()

    if df.empty:
        print("script_hit_memory.csv is present but empty.")
        return df

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Date column normalisation
    if "date" not in df.columns:
        if "result_date" in df.columns:
            df = df.rename(columns={"result_date": "date"})
        elif "prediction_date" in df.columns:
            df = df.rename(columns={"prediction_date": "date"})
    if "date" not in df.columns:
        print("script_hit_memory.csv has no usable date column.")
        return pd.DataFrame()
    df["date"] = df["date"].astype(str).str.strip()

    # Script column normalisation
    if "script_name" not in df.columns:
        if "script" in df.columns:
            df = df.rename(columns={"script": "script_name"})
    if "script_name" not in df.columns:
        print("script_hit_memory.csv has no usable script_name column.")
        return pd.DataFrame()

    # Slot column normalisation (optional)
    if "slot" not in df.columns and "real_slot" in df.columns:
        df = df.rename(columns={"real_slot": "slot"})

    # Hit type normalisation
    if "hit_type" not in df.columns:
        for alt in ["hit", "hit_tag", "hit_type".upper()]:
            if alt.lower() in df.columns:
                df = df.rename(columns={alt.lower(): "hit_type"})
                break
    if "hit_type" in df.columns:
        df["hit_type"] = df["hit_type"].astype(str).str.strip().str.upper()

    print(
        f"Loaded script_hit_memory.csv → rows={len(df)}, cols={list(df.columns)}"
    )
    return df


def compute_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """Compute windowed script metrics from script_hit_memory.csv."""

    df = _load_memory_df()
    if df.empty:
        print("No script hit memory available for metrics.")
        return pd.DataFrame()

    unique_dates_in_order = list(dict.fromkeys(df["date"]))
    if window_days <= 0 or len(unique_dates_in_order) <= window_days:
        recent_dates = unique_dates_in_order
    else:
        recent_dates = unique_dates_in_order[-window_days:]

    df_window = df[df["date"].isin(recent_dates)]
    if df_window.empty:
        print(
            f"No script hit memory rows found within the last {window_days} unique dates."
        )
        return pd.DataFrame()

    records = []
    for script_name, group in df_window.groupby("script_name"):
        total_rows = len(group)
        if "hit_type" in group.columns:
            hit_types = group["hit_type"].astype(str).str.strip().str.upper()
        else:
            hit_types = pd.Series(["" for _ in range(total_rows)])

        counts_by_type = hit_types.value_counts().to_dict()
        misses = int(counts_by_type.get("MISS", 0))
        hits_any = int(total_rows - misses)
        hit_rate_any = hits_any / total_rows if total_rows else 0.0

        records.append(
            {
                "script_id": str(script_name).upper(),
                "total_rows": int(total_rows),
                "hits_any": hits_any,
                "misses": misses,
                "hit_rate_any": float(hit_rate_any),
                "counts_by_hit_type": counts_by_type,
            }
        )

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(
        ["hit_rate_any", "total_rows"], ascending=[False, False]
    ).reset_index(drop=True)
    metrics_df.attrs["date_min"] = recent_dates[0] if recent_dates else None
    metrics_df.attrs["date_max"] = recent_dates[-1] if recent_dates else None
    metrics_df.attrs["window_days"] = window_days
    metrics_df.attrs["total_rows_window"] = len(df_window)
    metrics_df.attrs["script_count"] = len(metrics_df)
    return metrics_df


def load_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> Dict:
    """Public API to obtain script metrics safely."""

    try:
        metrics_df = compute_script_metrics(window_days=window_days)
        if metrics_df.empty:
            return {}

        date_min = metrics_df.attrs.get("date_min")
        date_max = metrics_df.attrs.get("date_max")
        metrics_by_script: Dict[str, Dict] = {}
        for _, row in metrics_df.iterrows():
            script_id = str(row.get("script_id", "")).upper()
            metrics_by_script[script_id] = {
                "total_rows": int(row.get("total_rows", 0) or 0),
                "counts_by_hit_type": row.get("counts_by_hit_type", {}) or {},
                "hits_any": int(row.get("hits_any", 0) or 0),
                "misses": int(row.get("misses", 0) or 0),
                "hit_rate_any": float(row.get("hit_rate_any", 0.0) or 0.0),
            }

        sorted_scripts = metrics_df["script_id"].tolist()
        best_script = sorted_scripts[0] if sorted_scripts else None
        worst_script = sorted_scripts[-1] if sorted_scripts else None

        return {
            "window_days": window_days,
            "date_min": date_min,
            "date_max": date_max,
            "total_rows": int(metrics_df.attrs.get("total_rows_window", 0) or 0),
            "script_count": int(metrics_df.attrs.get("script_count", 0) or 0),
            "metrics_by_script": metrics_by_script,
            "sorted_scripts": sorted_scripts,
            "best_script": best_script,
            "worst_script": worst_script,
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

    eligible = metrics_df[metrics_df["total_rows"] >= min_rows]
    heroes = eligible.sort_values("hit_rate_any", ascending=False).head(3)[
        "script_id"
    ].tolist()
    weak = eligible.sort_values("hit_rate_any", ascending=True).head(3)[
        "script_id"
    ].tolist()
    summary["ALL"] = {"heroes": heroes, "weak": weak}
    return summary


def _run_cli(window: int) -> None:
    metrics = load_script_metrics(window_days=window)
    if not metrics or metrics.get("script_count", 0) == 0:
        print("No metrics generated (script_hit_memory empty or no rows in this window).")
        return

    print("=" * 70)
    print(f"📊 SCRIPT HIT METRICS - last {metrics['window_days']} unique dates")
    print("=" * 70)
    print(f"Date range  : {metrics['date_min']} → {metrics['date_max']}")
    print(f"Scripts     : {metrics['script_count']}")
    print(f"Rows in win : {metrics['total_rows']}")
    print("")

    for script_name in metrics["sorted_scripts"]:
        m = metrics["metrics_by_script"][script_name]
        hr = m["hit_rate_any"] * 100.0 if m["total_rows"] > 0 else 0.0
        print(
            f"  {script_name}: hit_rate_any={hr:.1f}%  "
            f"(rows={m['total_rows']}, hits={m['hits_any']}, misses={m['misses']})"
        )

    print("=" * 70)
    if metrics["best_script"]:
        print(f"🏆 BEST SCRIPT : {metrics['best_script']}")
    if metrics["worst_script"]:
        print(f"📉 WORST SCRIPT: {metrics['worst_script']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute script hit metrics based on script_hit_memory.csv"
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of unique dates to include.",
    )
    args = parser.parse_args()

    _run_cli(args.window)
