"""Script-level hit metrics based on script_hit_memory.csv."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Dict, Iterable

import pandas as pd

import quant_paths
from script_hit_memory_utils import get_script_hit_memory_path

DEFAULT_WINDOW_DAYS = 30


def _load_memory_df() -> pd.DataFrame:
    """Load and normalise the raw script hit memory CSV."""

    path = get_script_hit_memory_path()
    if not path.exists():
        print("script_hit_memory.csv not found; no metrics to compute.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"Error reading script_hit_memory.csv: {exc}")
        return pd.DataFrame()

    df = df.dropna(how="all")
    if df.empty:
        print("script_hit_memory.csv is present but empty.")
        return pd.DataFrame()

    df.columns = [str(c).strip().lower() for c in df.columns]

    # Column normalisation
    if "script" in df.columns and "script_name" not in df.columns:
        df = df.rename(columns={"script": "script_name"})

    if "date" not in df.columns:
        print("script_hit_memory.csv missing 'date' column; cannot compute metrics.")
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    if df.empty:
        print("script_hit_memory.csv has no valid dates after parsing.")
        return pd.DataFrame()

    if "hit_flag" in df.columns:
        df["is_hit"] = _normalise_hit_flag(df["hit_flag"])
    else:
        df["is_hit"] = False

    if "hit_type" in df.columns:
        df["hit_type"] = df["hit_type"].astype(str).str.strip().str.upper()

    if "script_name" not in df.columns:
        print("script_hit_memory.csv missing 'script_name' column; cannot compute metrics.")
        return pd.DataFrame()

    df["script_name"] = df["script_name"].astype(str).str.upper()
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
) -> pd.DataFrame:
    """Compute windowed script metrics from script_hit_memory.csv."""

    base_df = df if df is not None else _load_memory_df()
    if base_df is None or base_df.empty:
        print("No script hit memory available for metrics.")
        return pd.DataFrame()

    working_df = base_df.copy()
    today = date.today()
    if window_days and window_days > 0:
        cutoff = today - timedelta(days=window_days)
        working_df = working_df[working_df["date"] >= cutoff]

    if working_df.empty:
        print(f"No script hit memory rows in the last {window_days} days.")
        return pd.DataFrame()

    records = []
    exact_tokens = {"EXACT", "HIT"}

    for script_name, group in working_df.groupby("script_name"):
        total_preds = len(group)
        hit_types = group["hit_type"].astype(str).str.upper() if "hit_type" in group.columns else pd.Series(["" for _ in range(total_preds)])

        hit_series = group["is_hit"] if "is_hit" in group.columns else pd.Series([False] * total_preds)
        hits = int(pd.Series(hit_series).astype(bool).sum())
        hit_rate = hits / total_preds if total_preds else 0.0

        exact_hits = int(hit_types.isin(exact_tokens).sum())
        exact_hit_rate = exact_hits / total_preds if total_preds else 0.0

        near_miss = int(
            (~hit_types.isin(exact_tokens | {"MISS", "NAN", "NONE", ""}))
            .sum()
        )

        records.append(
            {
                "script_name": str(script_name).upper(),
                "total_preds": int(total_preds),
                "hits": hits,
                "hit_rate": float(hit_rate),
                "exact_hits": exact_hits,
                "exact_hit_rate": float(exact_hit_rate),
                "near_miss": near_miss,
            }
        )

    metrics_df = pd.DataFrame(records)
    metrics_df = metrics_df.sort_values(
        ["hit_rate", "total_preds"], ascending=[False, False]
    ).reset_index(drop=True)
    return metrics_df


def load_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> Dict:
    """Public API to obtain script metrics safely as a mapping."""

    try:
        df = compute_script_metrics(window_days=window_days)
        if df.empty:
            return {}

        metrics_by_script: Dict[str, Dict] = {}
        for _, row in df.iterrows():
            script_id = str(row.get("script_name", "")).upper()
            metrics_by_script[script_id] = {
                "total_preds": int(row.get("total_preds", 0) or 0),
                "hits": int(row.get("hits", 0) or 0),
                "hit_rate": float(row.get("hit_rate", 0.0) or 0.0),
                "exact_hits": int(row.get("exact_hits", 0) or 0),
                "exact_hit_rate": float(row.get("exact_hit_rate", 0.0) or 0.0),
                "near_miss": int(row.get("near_miss", 0) or 0),
            }

        sorted_scripts = df["script_name"].tolist()
        best_script = sorted_scripts[0] if sorted_scripts else None
        worst_script = sorted_scripts[-1] if sorted_scripts else None

        return {
            "window_days": window_days,
            "script_count": len(sorted_scripts),
            "total_rows": int(df["total_preds"].sum()),
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

    eligible = metrics_df[metrics_df["total_preds"] >= min_rows]
    heroes = eligible.sort_values("hit_rate", ascending=False).head(3)[
        "script_name"
    ].tolist()
    weak = eligible.sort_values("hit_rate", ascending=True).head(3)[
        "script_name"
    ].tolist()
    summary["ALL"] = {"heroes": heroes, "weak": weak}
    return summary


def _run_cli(window: int) -> None:
    df = _load_memory_df()
    if df.empty:
        print("No script hit memory available for metrics.")
        return

    metrics_df = compute_script_metrics(df, window_days=window)
    if metrics_df.empty:
        print("No metrics generated (script_hit_memory empty or no rows in this window).")
        return

    logs_dir = quant_paths.get_performance_logs_dir()
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / f"script_hit_metrics_window{window}.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"SCRIPT METRICS (last {window} days)")
    print("script  total  hits  hit_rate  exact_hit_rate  near_miss")
    for _, row in metrics_df.iterrows():
        print(
            f"{row['script_name']:<6} {row['total_preds']:>5}  {row['hits']:>4}  "
            f"{row['hit_rate']:.3f}    {row['exact_hit_rate']:.3f}         {row['near_miss']}"
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
