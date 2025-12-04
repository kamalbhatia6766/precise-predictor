"""Script-level hit metrics and lightweight ROI proxies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Dict, List

import pandas as pd

import quant_paths
from script_hit_memory_utils import (
    SCRIPT_HIT_MEMORY_HEADERS,
    get_script_hit_memory_path,
    load_script_hit_memory,
)

DEFAULT_WINDOW_DAYS = 30


def _safe_rate(num: float, denom: float) -> float:
    return num / denom if denom else 0.0


def _score_row(hit_rate_final: float, coverage_rate: float, blind_miss_rate: float) -> float:
    score = 1.0
    score += 0.6 * hit_rate_final
    score += 0.3 * coverage_rate
    score -= 0.4 * blind_miss_rate
    return max(0.0, min(2.0, score))


def _load_memory_df(window_days: int) -> pd.DataFrame:
    path = get_script_hit_memory_path()
    try:
        if not path.exists():
            return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)
        return load_script_hit_memory(window_days=window_days)
    except Exception as exc:
        print(f"⚠️  Error reading script_hit_memory.csv: {exc}")
        return pd.DataFrame(columns=SCRIPT_HIT_MEMORY_HEADERS)


def compute_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> pd.DataFrame:
    """Compute per-script metrics for the trailing window."""

    df = _load_memory_df(window_days)
    if df.empty:
        print("No metrics generated (script_hit_memory empty for this window)")
        return pd.DataFrame()

    if "script_id" not in df.columns or "hit_type" not in df.columns:
        return pd.DataFrame()

    records: List[Dict] = []
    for script_id, group in df.groupby("script_id"):
        total_rows = len(group)
        hit_types = group["hit_type"].fillna("").astype(str).str.upper()
        final_hits = int((hit_types == "FINAL_HIT").sum())
        script_hits_not_final = int((hit_types == "SCRIPT_HIT_BUT_NOT_FINAL").sum())
        blind_misses = int((hit_types == "BLIND_MISS").sum())

        coverage = final_hits + script_hits_not_final
        hit_rate_final = _safe_rate(final_hits, total_rows)
        coverage_rate = _safe_rate(coverage, total_rows)
        blind_miss_rate = _safe_rate(blind_misses, total_rows)
        score = _score_row(hit_rate_final, coverage_rate, blind_miss_rate)

        approx_roi_pct = None
        if total_rows:
            stake = total_rows
            returns = final_hits * 90
            approx_roi_pct = (returns - stake) * 100 / stake

        records.append(
            {
                "script_id": str(script_id).upper(),
                "window_days": window_days,
                "total_rows": total_rows,
                "final_hits": final_hits,
                "script_hits_not_final": script_hits_not_final,
                "blind_misses": blind_misses,
                "hit_rate_final": round(hit_rate_final, 4),
                "coverage_rate": round(coverage_rate, 4),
                "blind_miss_rate": round(blind_miss_rate, 4),
                "score": round(score, 4),
                "approx_roi_pct": round(approx_roi_pct, 2) if approx_roi_pct is not None else None,
                "slot": "ALL",
            }
        )

    metrics_df = pd.DataFrame(records)
    return metrics_df.sort_values(["score", "total_rows"], ascending=[False, False]).reset_index(drop=True)


def _save_metrics_outputs(metrics_df: pd.DataFrame, window_days: int) -> None:
    perf_dir = quant_paths.get_performance_logs_dir()
    perf_dir.mkdir(parents=True, exist_ok=True)

    csv_path = perf_dir / "script_hit_metrics.csv"
    json_path = perf_dir / "script_hit_metrics.json"

    metrics_df.to_csv(csv_path, index=False)

    payload = {
        "window_days": window_days,
        "generated_at": datetime.now().isoformat(),
        "metrics": metrics_df.to_dict(orient="records"),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_script_metrics(window_days: int = DEFAULT_WINDOW_DAYS) -> Dict[str, Dict[str, float]]:
    """Return metrics keyed by script_id."""

    metrics_df = compute_script_metrics(window_days=window_days)
    result: Dict[str, Dict[str, float]] = {}
    if metrics_df.empty:
        return result

    for _, row in metrics_df.iterrows():
        script_id = str(row.get("script_id", "")).upper()
        if not script_id:
            continue
        result[script_id] = {
            "hit_rate_final": float(row.get("hit_rate_final", 0.0)),
            "coverage_rate": float(row.get("coverage_rate", 0.0)),
            "blind_miss_rate": float(row.get("blind_miss_rate", 0.0)),
            "score": float(row.get("score", 0.0)),
            "total_rows": float(row.get("total_rows", 0)),
        }
    return result


def compute_slot_heroes_and_weak(metrics_df: pd.DataFrame, min_rows: int = 10) -> Dict[str, Dict[str, List[str]]]:
    """Provide a lightweight hero/weak summary for previews."""

    summary: Dict[str, Dict[str, List[str]]] = {}
    if metrics_df is None or metrics_df.empty:
        return summary

    eligible = metrics_df[metrics_df["total_rows"] >= min_rows]
    heroes = eligible.sort_values("score", ascending=False).head(3)["script_id"].tolist()
    weak = eligible.sort_values("blind_miss_rate", ascending=False).head(3)["script_id"].tolist()
    summary["ALL"] = {"heroes": heroes, "weak": weak}
    return summary


def _print_console_summary(metrics_df: pd.DataFrame, window_days: int) -> None:
    if metrics_df.empty:
        print("No metrics generated (script_hit_memory empty for this window)")
        return

    cols = [
        "script_id",
        "total_rows",
        "final_hits",
        "script_hits_not_final",
        "coverage_rate",
        "blind_misses",
        "hit_rate_final",
        "blind_miss_rate",
        "score",
    ]
    metrics_df = metrics_df[cols]

    print("SCRIPT  ROWS  FINAL  COV  BLIND  HIT%   COV%   BLIND%  SCORE")
    for _, row in metrics_df.iterrows():
        script = row["script_id"]
        rows = int(row["total_rows"])
        final_hits = int(row["final_hits"])
        script_hits_not_final = int(row["script_hits_not_final"])
        coverage = float(row["coverage_rate"])
        blind = int(row["blind_misses"])
        hit_rate = float(row["hit_rate_final"])
        blind_rate = float(row["blind_miss_rate"])
        score = float(row["score"])
        print(
            f"{script:<6} {rows:>4}  {final_hits:>5}  {(final_hits + script_hits_not_final):>3}  {blind:>5}  "
            f"{hit_rate*100:>5.1f}%  {coverage*100:>5.1f}%  {blind_rate*100:>6.1f}%  {score:>5.2f}"
        )


def _run_cli(window: int, verbose: bool) -> None:
    metrics_df = compute_script_metrics(window_days=window)
    if metrics_df.empty:
        return

    _save_metrics_outputs(metrics_df, window)
    if verbose:
        print(metrics_df)
    _print_console_summary(metrics_df, window)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute script hit metrics")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_DAYS, help="Window in days")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    _run_cli(args.window, args.verbose)
