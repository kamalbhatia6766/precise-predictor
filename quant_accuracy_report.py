from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Tuple

import pandas as pd

import argparse
from datetime import timedelta
from typing import Tuple

import pandas as pd

from quant_excel_loader import load_results_excel
from script_hit_memory_utils import (
    classify_relation,
    get_script_hit_memory_xlsx_path,
    load_script_hit_memory,
)


def _format_percent(value: float) -> float:
    return round(value * 100, 1)


def _filter_window(df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    if df.empty:
        return df
    if "result_date" not in df.columns:
        return df

    result_dates = pd.to_datetime(df["result_date"], errors="coerce")
    mask = (result_dates >= window_start) & (result_dates <= window_end)
    return df[mask].copy()


def _coerce_bool(series: pd.Series, default: bool = False) -> pd.Series:
    filled = series.copy()
    filled = filled.where(pd.notna(filled), default)
    lowered = filled.astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def _build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "script_id",
                "slot",
                "total_predictions",
                "exact_hits",
                "near_hits",
                "hit_rate_exact",
                "near_miss_rate",
            ]
        )

    grouped = df.groupby(["script_id", "slot"], dropna=False)
    rows = []
    for (script, slot), group in grouped:
        total = len(group)
        relations = group.get("_relation") if "_relation" in group.columns else None
        if relations is None:
            relations = group.apply(
                lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1
            )

        if "is_exact_hit" in group.columns:
            exact_series = _coerce_bool(group.get("is_exact_hit"))
            exact = int(exact_series.sum())
        else:
            exact = int((relations == "EXACT").sum())

        if "is_near_miss" in group.columns:
            near_series = _coerce_bool(group.get("is_near_miss"))
            near = int(near_series.sum())
        else:
            near = int(
                relations.isin({"MIRROR", "ADJACENT", "DIAGONAL_11", "REVERSE_CARRY", "SAME_DIGIT_COOL"}).sum()
            )
        hit_rate = _format_percent(exact / total) if total else 0.0
        near_rate = _format_percent(near / total) if total else 0.0
        rows.append(
            {
                "script_id": str(script),
                "slot": str(slot),
                "total_predictions": total,
                "exact_hits": exact,
                "near_hits": near,
                "hit_rate_exact": hit_rate,
                "near_miss_rate": near_rate,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics = metrics.sort_values(["script_id", "slot"]).reset_index(drop=True)
    return metrics


def _hero_weak(df: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None]:
    if df.empty:
        return None, None

    agg = (
        df.groupby("script_id")
        .agg(
            total_predictions=("result_date", "size"),
            exact_hits=("is_exact_hit", "sum"),
            near_hits=("is_near_miss", "sum"),
        )
        .reset_index()
    )
    if agg.empty:
        return None, None

    agg["hit_rate_exact"] = agg.apply(
        lambda row: _format_percent(row["exact_hits"] / row["total_predictions"]) if row["total_predictions"] else 0.0,
        axis=1,
    )

    agg = agg.sort_values("script_id")
    heroes = agg[agg["total_predictions"] > 0]
    if heroes.empty:
        return None, None

    hero_row = heroes.sort_values(["hit_rate_exact", "total_predictions"], ascending=[False, False]).iloc[0]
    weak_row = heroes.sort_values(["hit_rate_exact", "total_predictions"], ascending=[True, False]).iloc[0]
    return hero_row, weak_row


def _print_table(title: str, df: pd.DataFrame) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if df.empty:
        print("(no data)")
        return
    display_cols = [
        "script_id",
        "slot",
        "total_predictions",
        "exact_hits",
        "near_hits",
        "hit_rate_exact",
        "near_miss_rate",
    ]
    print(df[display_cols].to_string(index=False))


def _print_hero_weak(label: str, df: pd.DataFrame) -> None:
    hero, weak = _hero_weak(df)
    print(f"\n{label} HERO/WEAK")
    print("-" * (len(label) + 9))
    if hero is None or weak is None:
        print("(no qualifying scripts)")
        return
    print(
        f"Hero: {hero['script_id']} | Exact={hero['hit_rate_exact']:.1f}% "
        f"over {int(hero['total_predictions'])} preds"
    )
    print(
        f"Weak: {weak['script_id']} | Exact={weak['hit_rate_exact']:.1f}% "
        f"over {int(weak['total_predictions'])} preds"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate script accuracy report")
    parser.add_argument(
        "--window_days",
        type=int,
        default=30,
        help="Recent window in days for hero/weak and slot metrics",
    )
    args = parser.parse_args()

    results_df = load_results_excel()
    if not results_df.empty:
        results_df["DATE"] = pd.to_datetime(results_df.get("date"), errors="coerce")
        min_date = results_df["DATE"].min()
        max_date = results_df["DATE"].max()
    else:
        min_date = max_date = None

    hit_df = load_script_hit_memory()
    hit_df["result_date"] = pd.to_datetime(hit_df.get("result_date"), errors="coerce")
    hit_df = hit_df.dropna(subset=["result_date"])
    hit_df["result_date"] = hit_df["result_date"].dt.date
    hit_df["slot"] = hit_df.get("slot", "").astype(str).str.upper()
    hit_df["script_id"] = hit_df.get("script_id", "").astype(str).str.upper()
    hit_df["is_exact_hit"] = _coerce_bool(hit_df.get("is_exact_hit", False))
    hit_df["is_near_miss"] = _coerce_bool(hit_df.get("is_near_miss", False))

    latest_date = hit_df["result_date"].max() if not hit_df.empty else None
    if latest_date is not None:
        window_start = latest_date - timedelta(days=args.window_days - 1)
        window_end = latest_date
        df_window = hit_df[
            (hit_df["result_date"] >= window_start) & (hit_df["result_date"] <= window_end)
        ].copy()
    else:
        window_start = window_end = None
        df_window = hit_df.copy()

    used_days = df_window["result_date"].nunique() if not df_window.empty else 0

    overall_metrics = _build_metrics(hit_df)

    print(
        f"=== QUANT ACCURACY REPORT (window_days={args.window_days}, used_days={used_days}) ==="
    )
    print(f"Hit memory: {get_script_hit_memory_xlsx_path()} → {len(hit_df)} rows")
    if min_date is not None and max_date is not None:
        print(f"Real results range: {min_date.date()} → {max_date.date()}")
    if latest_date is not None and window_start is not None and window_end is not None:
        print(
            f"Window: {window_start} to {window_end} (window_days={args.window_days}, used_days={used_days})"
        )

    recent_metrics = _build_metrics(df_window)

    print("\n=== QUANT ACCURACY REPORT (all-history) ===")
    _print_table("All-history accuracy (per script & slot)", overall_metrics)
    _print_hero_weak("All-history", hit_df)

    window_title = f"Last {args.window_days}d accuracy (per script & slot)"
    print(f"\n=== QUANT ACCURACY REPORT (window) ===")
    _print_table(window_title, recent_metrics)
    _print_hero_weak(f"Last {args.window_days}d", df_window)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
