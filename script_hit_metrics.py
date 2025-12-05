from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from script_hit_memory_utils import load_script_hit_memory

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def _normalise_slot(slot_value: object) -> Optional[str]:
    if slot_value is None:
        return None
    mapping = {"1": "FRBD", "2": "GZBD", "3": "GALI", "4": "DSWR"}
    slot_str = str(slot_value).strip().upper()
    return mapping.get(slot_str, slot_str if slot_str else None)


def _prepare_memory_df(base_dir: Optional[Path] = None) -> pd.DataFrame:
    df = load_script_hit_memory(base_dir=base_dir)
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    for col in ("date", "result_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_id"] = df.get("script_id").astype(str).str.strip().str.upper()
    df["script_name"] = (
        df.get("script_name").fillna(df.get("script_id")) if "script_name" in df.columns else df.get("script_id")
    )
    df["script_name"] = df["script_name"].astype(str).str.strip().str.upper()
    df["hit_type"] = df.get("hit_type", "none").astype(str).str.strip().str.lower()
    df["is_near_miss"] = pd.to_numeric(df.get("is_near_miss", 0), errors="coerce").fillna(0).astype(int)
    return df.dropna(subset=["slot", "script_name"])


def _choose_date_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("result_date", "date"):
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None


def _window_memory(df: pd.DataFrame, window_days: int, fallback: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return pd.DataFrame(), {}

    date_col = _choose_date_column(df)
    if date_col is None:
        return pd.DataFrame(), {}

    latest_date = df[date_col].max()
    earliest_date = df[date_col].min()
    if pd.isna(latest_date) or pd.isna(earliest_date):
        return pd.DataFrame(), {}

    candidate_windows: List[int] = [window_days]
    if fallback:
        candidate_windows.append(min(window_days * 2, 90))
        full_history = (latest_date - earliest_date).days + 1
        candidate_windows.append(full_history)

    candidate_windows = list(dict.fromkeys(candidate_windows))

    selected_df = pd.DataFrame()
    effective_window = window_days
    for win in candidate_windows:
        cutoff = latest_date - timedelta(days=win - 1)
        window_df = df[df[date_col] >= cutoff]
        if not window_df.empty:
            selected_df = window_df
            effective_window = win
            break

    summary = {
        "requested_window_days": window_days,
        "effective_window_days": effective_window,
        "latest_date": latest_date,
        "earliest_date": earliest_date,
        "total_rows": len(selected_df),
    }
    return selected_df, summary


def _aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, Any] = {}
        for col, value in zip(group_cols, key_values):
            record[col] = value
        total_predictions = len(group)
        hit_series = group.get("hit_type", "none")
        exact_hits = int((hit_series == "exact").sum())
        neighbor_hits = int((hit_series == "neighbor").sum())
        mirror_hits = int((hit_series == "mirror").sum())
        s40_hits = int((hit_series == "s40").sum())
        family_hits = int((hit_series == "family_164950").sum())
        near_miss_hits = int(pd.to_numeric(group.get("is_near_miss", 0), errors="coerce").fillna(0).sum())

        hit_rate_exact = exact_hits / total_predictions if total_predictions else 0.0
        hit_rate_extended = (exact_hits + neighbor_hits + mirror_hits) / total_predictions if total_predictions else 0.0
        near_miss_rate = near_miss_hits / total_predictions if total_predictions else 0.0
        pack_rate = (s40_hits + family_hits) / total_predictions if total_predictions else 0.0
        blind_miss_rate = max(0.0, 1.0 - hit_rate_extended - near_miss_rate)
        mirror_neighbor_rate = (neighbor_hits + mirror_hits) / total_predictions if total_predictions else 0.0
        score_raw = 100.0 * hit_rate_exact + 40.0 * mirror_neighbor_rate + 10.0 * pack_rate
        score = min(100.0, round(score_raw, 1))

        record.update(
            {
                "total_predictions": int(total_predictions),
                "exact_hits": int(exact_hits),
                "neighbor_hits": int(neighbor_hits),
                "mirror_hits": int(mirror_hits),
                "s40_hits": int(s40_hits),
                "family_164950_hits": int(family_hits),
                "near_miss_hits": int(near_miss_hits),
                "hit_rate_exact": hit_rate_exact,
                "hit_rate_extended": hit_rate_extended,
                "near_miss_rate": near_miss_rate,
                "blind_miss_rate": blind_miss_rate,
                "score": score,
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def compute_script_metrics(
    mode: str = "overall",
    window_days: int = 30,
    base_dir: Optional[Path] = None,
    fallback: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = _prepare_memory_df(base_dir=base_dir)
    window_df, summary = _window_memory(df, window_days=window_days, fallback=fallback)
    if window_df.empty:
        return pd.DataFrame(), summary

    if mode == "per_slot":
        metrics = _aggregate_metrics(window_df, ["slot", "script_name"])
        metrics = metrics.rename(columns={"script_name": "script_id"})
        metrics = metrics[[
            "slot",
            "script_id",
            "total_predictions",
            "exact_hits",
            "hit_rate_exact",
            "hit_rate_extended",
            "near_miss_rate",
            "blind_miss_rate",
            "s40_hits",
            "family_164950_hits",
            "score",
            "neighbor_hits",
            "mirror_hits",
            "near_miss_hits",
        ]]
        return metrics.sort_values(["slot", "score"], ascending=[True, False]).reset_index(drop=True), summary

    metrics = _aggregate_metrics(window_df, ["script_name"])
    metrics = metrics.rename(columns={"script_name": "script_id"})
    metrics = metrics[[
        "script_id",
        "total_predictions",
        "exact_hits",
        "hit_rate_exact",
        "hit_rate_extended",
        "near_miss_rate",
        "blind_miss_rate",
        "s40_hits",
        "family_164950_hits",
        "score",
        "neighbor_hits",
        "mirror_hits",
        "near_miss_hits",
    ]]
    return metrics.sort_values("score", ascending=False).reset_index(drop=True), summary


def hero_weak_table(metrics_per_slot: pd.DataFrame, min_samples: int = 15) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics_per_slot.empty:
        return pd.DataFrame(columns=[
            "slot",
            "hero_script",
            "hero_score",
            "hero_hit_rate_exact",
            "weak_script",
            "weak_score",
            "weak_hit_rate_exact",
        ])

    for slot in SLOTS:
        slot_df = metrics_per_slot[metrics_per_slot["slot"] == slot]
        eligible = slot_df[(slot_df["total_predictions"] >= min_samples) & (slot_df["hit_rate_exact"] > 0)]
        if eligible.empty:
            rows.append({
                "slot": slot,
                "hero_script": None,
                "hero_score": None,
                "hero_hit_rate_exact": None,
                "weak_script": None,
                "weak_score": None,
                "weak_hit_rate_exact": None,
            })
            continue
        hero_row = eligible.sort_values("score", ascending=False).iloc[0]
        weak_row = eligible.sort_values("score", ascending=True).iloc[0]
        rows.append(
            {
                "slot": slot,
                "hero_script": hero_row["script_id"],
                "hero_score": hero_row["score"],
                "hero_hit_rate_exact": hero_row["hit_rate_exact"],
                "weak_script": weak_row["script_id"],
                "weak_score": weak_row["score"],
                "weak_hit_rate_exact": weak_row["hit_rate_exact"],
            }
        )
    return pd.DataFrame(rows)


def build_script_weights_by_slot(metrics_per_slot_df: pd.DataFrame, min_samples: int = 15) -> Dict[str, Dict[str, float]]:
    weights: Dict[str, Dict[str, float]] = {}
    if metrics_per_slot_df is None or metrics_per_slot_df.empty:
        return weights

    for slot in SLOTS:
        slot_df = metrics_per_slot_df[(metrics_per_slot_df["slot"] == slot) & (metrics_per_slot_df["total_predictions"] >= min_samples)]
        if slot_df.empty:
            continue
        base_scores = {row["script_id"]: max(float(row["score"]), 0.0) for _, row in slot_df.iterrows()}
        if all(score <= 0 for score in base_scores.values()):
            base_scores = {script: 1.0 for script in base_scores}
        total_base = sum(base_scores.values()) or 1.0
        slot_weights = {script: score / total_base for script, score in base_scores.items()}
        weights[slot] = slot_weights
    return weights


# Backwards compatibility wrappers -------------------------------------------------------

def build_script_league(df: pd.DataFrame, min_predictions: int = 10, min_hits_for_hero: int = 1):
    metrics = df if df is not None else pd.DataFrame()
    if metrics.empty:
        return {"heroes": [], "weak": [], "window_rows": 0}
    overall = metrics.groupby("script_id", as_index=False).agg(
        total_predictions=("total_predictions", "sum"),
        exact_hits=("exact_hits", "sum"),
    )
    heroes = overall[(overall["total_predictions"] >= min_predictions) & (overall["exact_hits"] >= min_hits_for_hero)]
    heroes["hit_rate"] = heroes["exact_hits"] / heroes["total_predictions"].clip(lower=1)
    heroes = heroes.sort_values(["hit_rate", "total_predictions"], ascending=[False, False])
    weak = overall[(overall["total_predictions"] >= min_predictions) & (overall["exact_hits"] == 0)]
    heroes_list = [
        {
            "script": row["script_id"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": int(row["exact_hits"]),
            "hit_rate": float(row["hit_rate"]),
        }
        for _, row in heroes.iterrows()
    ]
    weak_list = [
        {
            "script": row["script_id"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": int(row["exact_hits"]),
            "hit_rate": 0.0,
        }
        for _, row in weak.iterrows()
    ]
    return {"heroes": heroes_list, "weak": weak_list, "window_rows": len(metrics)}


# CLI utilities -------------------------------------------------------------------------

def _print_metrics(metrics_df: pd.DataFrame, summary: Dict[str, Any], mode: str) -> None:
    header = (
        "Script hit metrics – "
        f"requested {summary.get('requested_window_days')}d, used {summary.get('effective_window_days')}d "
        f"(rows={summary.get('total_rows')})"
    )
    print(header)
    if metrics_df.empty:
        print("No script hit metrics rows to display.")
        return
    display_cols = [
        "script_id",
        "slot",
        "total_predictions",
        "exact_hits",
        "hit_rate_exact",
        "hit_rate_extended",
        "near_miss_rate",
        "blind_miss_rate",
        "score",
    ]
    existing = [c for c in display_cols if c in metrics_df.columns]
    print(metrics_df[existing].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Script-wise hit metrics for Precise Predictor.")
    parser.add_argument("--window", type=int, default=30, help="Number of days to look back from the latest date.")
    parser.add_argument("--mode", choices=["overall", "per_slot"], default="per_slot")
    args = parser.parse_args()

    metrics_df, summary = compute_script_metrics(mode=args.mode, window_days=args.window, fallback=True)
    _print_metrics(metrics_df, summary, args.mode)
    if args.mode == "per_slot":
        heroes_df = hero_weak_table(metrics_df)
        if not heroes_df.empty:
            print("\nHero/Weak per slot:")
            print(heroes_df.to_string(index=False))


if __name__ == "__main__":
    main()
