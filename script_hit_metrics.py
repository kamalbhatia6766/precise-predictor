from __future__ import annotations

import argparse
from datetime import timedelta
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


def _prepare_memory_df(base_dir: Optional[Path] = None) -> Tuple[pd.DataFrame, Optional[str]]:
    df = load_script_hit_memory(base_dir=base_dir)
    if df.empty:
        return pd.DataFrame(), None

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for col in ("date", "result_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_id"] = df.get("script_id").astype(str).str.strip().str.upper()
    df["hit_type"] = df.get("hit_type", "none").astype(str).str.strip().str.lower()
    date_col = "result_date" if "result_date" in df.columns else "date" if "date" in df.columns else None
    if date_col:
        df = df.dropna(subset=[date_col])
    df = df.dropna(subset=["slot", "script_id"])
    return df, date_col


def _window_memory(df: pd.DataFrame, date_col: Optional[str], window_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty or not date_col:
        return pd.DataFrame(), {}

    latest_date = df[date_col].max()
    earliest_date = df[date_col].min()
    if pd.isna(latest_date) or pd.isna(earliest_date):
        return pd.DataFrame(), {}

    cutoff = latest_date - timedelta(days=window_days - 1)
    window_df = df[df[date_col] >= cutoff]
    summary = {
        "requested_window_days": window_days,
        "effective_window_days": window_days,
        "latest_date": latest_date,
        "earliest_date": earliest_date,
        "total_rows": len(window_df),
    }
    return window_df, summary


def _aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for keys, group in df.groupby(group_cols):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, Any] = {col: value for col, value in zip(group_cols, key_values)}

        total_predictions = len(group)
        hit_series = group.get("hit_type", "none")
        exact_hits = int((hit_series == "exact").sum())
        neighbor_hits = int((hit_series == "neighbor").sum())
        mirror_hits = int((hit_series == "mirror").sum())
        near_hits = neighbor_hits + mirror_hits

        hit_rate_exact = exact_hits / total_predictions if total_predictions else 0.0
        near_miss_rate = near_hits / total_predictions if total_predictions else 0.0
        hit_rate_extended = (exact_hits + near_hits) / total_predictions if total_predictions else 0.0
        blind_miss_rate = max(0.0, 1.0 - hit_rate_extended)

        hit_rate = hit_rate_exact + 0.5 * near_miss_rate
        if hit_rate > 1.0:
            hit_rate = 1.0
        score = hit_rate_exact + 0.5 * near_miss_rate - 0.2 * blind_miss_rate

        record.update(
            {
                "total_predictions": int(total_predictions),
                "exact_hits": int(exact_hits),
                "mirror_hits": int(mirror_hits),
                "neighbor_hits": int(neighbor_hits),
                "near_hits": int(near_hits),
                "hit_rate_exact": hit_rate_exact,
                "near_miss_rate": near_miss_rate,
                "hit_rate_extended": hit_rate_extended,
                "blind_miss_rate": blind_miss_rate,
                "hit_rate": hit_rate,
                "score": score,
            }
        )
        records.append(record)

    return pd.DataFrame(records)


def get_metrics_table(
    window_days: int = 30,
    base_dir: Optional[Path] = None,
    mode: str = "per_slot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, date_col = _prepare_memory_df(base_dir=base_dir)
    window_df, summary = _window_memory(df, date_col, window_days)
    if window_df.empty:
        return pd.DataFrame(), summary

    group_cols = ["script_id"] if mode == "overall" else ["script_id", "slot"]
    metrics = _aggregate_metrics(window_df, group_cols)
    order_cols = ["script_id", "slot"] if "slot" in metrics.columns else ["script_id"]
    metrics = metrics.sort_values(order_cols + ["score"], ascending=[True] * len(order_cols) + [False]).reset_index(drop=True)
    return metrics, summary


def compute_script_metrics(
    mode: str = "overall",
    window_days: int = 30,
    base_dir: Optional[Path] = None,
    fallback: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    return get_metrics_table(window_days=window_days, base_dir=base_dir, mode=mode)


def hero_weak_table(metrics_df: pd.DataFrame, min_predictions: int = 10) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(
            columns=[
                "slot",
                "hero_script",
                "hero_score",
                "hero_hit_rate_exact",
                "weak_script",
                "weak_score",
                "weak_hit_rate_exact",
            ]
        )

    for slot in SLOTS:
        slot_df = metrics_df[metrics_df.get("slot") == slot]
        eligible = slot_df[slot_df["total_predictions"] >= min_predictions]
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
                "hero_script": hero_row.get("script_id"),
                "hero_score": hero_row.get("score"),
                "hero_hit_rate_exact": hero_row.get("hit_rate_exact"),
                "weak_script": weak_row.get("script_id"),
                "weak_score": weak_row.get("score"),
                "weak_hit_rate_exact": weak_row.get("hit_rate_exact"),
            }
        )
    return pd.DataFrame(rows)


def build_script_weights_by_slot(metrics_per_slot_df: pd.DataFrame, min_samples: int = 15) -> Dict[str, Dict[str, float]]:
    weights: Dict[str, Dict[str, float]] = {}
    if metrics_per_slot_df is None or metrics_per_slot_df.empty:
        return weights

    for slot in SLOTS:
        slot_df = metrics_per_slot_df[
            (metrics_per_slot_df.get("slot") == slot) & (metrics_per_slot_df["total_predictions"] >= min_samples)
        ]
        if slot_df.empty:
            continue
        base_scores = {row["script_id"]: max(float(row["score"]), 0.0) for _, row in slot_df.iterrows()}
        if all(score <= 0 for score in base_scores.values()):
            base_scores = {script: 1.0 for script in base_scores}
        total_base = sum(base_scores.values()) or 1.0
        slot_weights = {script: score / total_base for script, score in base_scores.items()}
        weights[slot] = slot_weights
    return weights


def compute_pack_hit_stats(window_days: int = 30, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    df, date_col = _prepare_memory_df(base_dir=base_dir)
    window_df, _ = _window_memory(df, date_col, window_days)
    if window_df.empty:
        return {}

    window_df["is_s40"] = pd.to_numeric(window_df.get("is_s40", 0), errors="coerce").fillna(0).astype(int)
    window_df["is_family_164950"] = pd.to_numeric(window_df.get("is_family_164950", 0), errors="coerce").fillna(0).astype(int)

    total_rows = len(window_df)
    s40_hits = int(window_df["is_s40"].sum())
    fam_hits = int(window_df["is_family_164950"].sum())
    summary = {
        "total_rows": total_rows,
        "S40": {"hits": s40_hits, "hit_rate": s40_hits / total_rows if total_rows else 0.0},
        "FAMILY_164950": {"hits": fam_hits, "hit_rate": fam_hits / total_rows if total_rows else 0.0},
        "per_slot": {},
    }

    for slot in SLOTS:
        slot_df = window_df[window_df["slot"] == slot]
        if slot_df.empty:
            continue
        total_slot = len(slot_df)
        s40_slot_hits = int(slot_df["is_s40"].sum())
        fam_slot_hits = int(slot_df["is_family_164950"].sum())
        summary["per_slot"][slot] = {
            "total": total_slot,
            "s40_hits": s40_slot_hits,
            "fam_hits": fam_slot_hits,
            "s40_rate": s40_slot_hits / total_slot if total_slot else 0.0,
            "fam_rate": fam_slot_hits / total_slot if total_slot else 0.0,
        }
    return summary


def build_script_league(df: pd.DataFrame, min_predictions: int = 10, min_hits_for_hero: int = 1):
    metrics = df if df is not None else pd.DataFrame()
    if metrics.empty:
        return {"heroes": [], "weak": [], "window_rows": 0}

    agg = metrics.groupby("script_id", as_index=False).agg(
        total_predictions=("total_predictions", "sum"),
        exact_hits=("exact_hits", "sum"),
        near_hits=("near_hits", "sum"),
        avg_hit_rate=("hit_rate", "mean"),
        avg_score=("score", "mean"),
    )
    agg["combined_hits"] = agg["exact_hits"] + 0.5 * agg["near_hits"]

    heroes_df = agg[(agg["total_predictions"] >= min_predictions) & (agg["exact_hits"] >= min_hits_for_hero)]
    heroes_df = heroes_df.sort_values(["avg_score", "combined_hits"], ascending=[False, False])
    weak_df = agg[agg["total_predictions"] >= min_predictions].sort_values(["avg_score", "combined_hits"], ascending=[True, False])

    heroes_list = [
        {
            "script": row["script_id"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": float(row["combined_hits"]),
            "hit_rate": float(row["avg_hit_rate"]),
        }
        for _, row in heroes_df.iterrows()
    ]
    weak_list = [
        {
            "script": row["script_id"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": float(row["combined_hits"]),
            "hit_rate": float(row["avg_hit_rate"]),
        }
        for _, row in weak_df.iterrows()
    ]
    return {"heroes": heroes_list, "weak": weak_list, "window_rows": len(metrics)}


def format_script_league(league_df: pd.DataFrame, max_rows: int = 20) -> str:
    if not league_df:
        return "No league data available."

    heroes = league_df.get("heroes", [])[:max_rows]
    weak = league_df.get("weak", [])[:max_rows]
    lines: List[str] = []
    if heroes:
        lines.append("Heroes:")
        for row in heroes:
            lines.append(
                f"  {row['script']}: hits={row['total_hits']:.1f} in {row['total_predictions']} (hit_rate={row['hit_rate']:.2f})"
            )
    if weak:
        lines.append("Weak:")
        for row in weak:
            lines.append(
                f"  {row['script']}: hits={row['total_hits']:.1f} in {row['total_predictions']} (hit_rate={row['hit_rate']:.2f})"
            )
    return "\n".join(lines) if lines else "No league data available."


def load_script_hit_metrics(window_days: int = 30, base_dir: Optional[Path] = None) -> pd.DataFrame:
    metrics_df, _ = get_metrics_table(window_days=window_days, base_dir=base_dir, mode="per_slot")
    return metrics_df


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
        "mirror_hits",
        "neighbor_hits",
        "hit_rate_exact",
        "near_miss_rate",
        "hit_rate",
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
