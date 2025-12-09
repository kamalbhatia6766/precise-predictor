from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import quant_paths
from quant_core import hit_core
from quant_stats_core import compute_pack_hit_stats as compute_pack_hit_stats_core, compute_script_slot_stats
from script_hit_memory_utils import (
    classify_relation,
    filter_hits_by_window,
    load_script_hit_memory,
)

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
    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce")
    df = df.dropna(subset=["result_date"])
    df["result_date"] = df["result_date"].dt.date
    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_id"] = df.get("script_id").astype(str).str.strip().str.upper()
    df["HIT_TYPE"] = df.get("HIT_TYPE", "MISS").astype(str).str.upper()
    df["is_exact_hit"] = df.get("is_exact_hit", False).astype(bool)
    df["is_near_miss"] = df.get("is_near_miss", False).astype(bool)
    df["is_near_hit"] = df.get("is_near_hit", False).astype(bool)
    df["_relation"] = df.apply(lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1)
    df = df.dropna(subset=["slot", "script_id"])
    return df, "result_date"


def _window_memory(df: pd.DataFrame, date_col: Optional[str], window_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty or not date_col:
        return pd.DataFrame(), {}

    filtered, used_days = filter_hits_by_window(df, window_days=window_days)
    if filtered.empty:
        return pd.DataFrame(), {}

    latest_date = filtered[date_col].max()
    earliest_date = filtered[date_col].min()

    summary = {
        "requested_window_days": window_days,
        "effective_window_days": int(used_days),
        "latest_date": latest_date,
        "earliest_date": earliest_date,
        "total_rows": len(filtered),
    }
    return filtered, summary


def _aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return compute_script_slot_stats(df, group_cols)


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
    if not metrics.empty and "score" in metrics.columns:
        metrics["score"] = (metrics["score"] + 0.20) * 100.0

    if mode == "per_slot":
        total_predictions = len(window_df)
        relations = window_df.get("_relation") if "_relation" in window_df.columns else None
        if relations is None:
            relations = window_df.apply(
                lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1
            )
        exact_hits = int((relations == "EXACT").sum())
        near_hits = int(
            relations.isin({"MIRROR", "ADJACENT", "DIAGONAL_11", "REVERSE_CARRY", "SAME_DIGIT_COOL"}).sum()
        )
        hit_rate_exact = exact_hits / total_predictions if total_predictions else 0.0
        near_miss_rate = near_hits / total_predictions if total_predictions else 0.0
        blind_misses = max(total_predictions - exact_hits - near_hits, 0)
        blind_miss_rate = blind_misses / total_predictions if total_predictions else 0.0
        score = (120.0 * hit_rate_exact + 40.0 * near_miss_rate - 30.0 * blind_miss_rate)
        score = (score + 0.20) * 100.0 if total_predictions else 0.0
        global_row = pd.DataFrame(
            [
                {
                    "script_id": "ALL",
                    "slot": "ALL",
                    "total_predictions": int(total_predictions),
                    "exact_hits": int(exact_hits),
                    "near_hits": int(near_hits),
                    "hit_rate_exact": hit_rate_exact,
                    "near_miss_rate": near_miss_rate,
                    "blind_miss_rate": blind_miss_rate,
                    "score": score,
                }
            ]
        )
        metrics = pd.concat([global_row, metrics], ignore_index=True)
    order_cols = ["script_id", "slot"] if "slot" in metrics.columns else ["script_id"]
    metrics = metrics.sort_values(order_cols + ["score"], ascending=[True] * len(order_cols) + [False]).reset_index(drop=True)
    export_nearhit_topn_analysis(metrics, summary, base_dir=base_dir)
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

    metrics_df = metrics_df[metrics_df.get("script_id") != "ALL"]
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
        hero_pool = eligible[eligible["exact_hits"] > 0]
        if hero_pool.empty:
            hero_pool = eligible[eligible["near_hits"] > 0]
        hero_row = hero_pool.sort_values("score", ascending=False).iloc[0] if not hero_pool.empty else None
        weak_row = eligible.sort_values("score", ascending=True).iloc[0]
        rows.append(
            {
                "slot": slot,
                "hero_script": hero_row.get("script_id") if hero_row is not None else None,
                "hero_score": hero_row.get("score") if hero_row is not None else None,
                "hero_hit_rate_exact": hero_row.get("hit_rate_exact") if hero_row is not None else None,
                "weak_script": weak_row.get("script_id"),
                "weak_score": weak_row.get("score"),
                "weak_hit_rate_exact": weak_row.get("hit_rate_exact"),
            }
        )
    return pd.DataFrame(rows)


def build_script_weights_by_slot(
    metrics_per_slot_df: pd.DataFrame,
    min_samples: int = 30,
    min_score: float = 0.01,
) -> Dict[str, Dict[str, float]]:
    """Convert per-slot metrics into normalized script weights.

    Scripts with insufficient samples or score are ignored. If nothing survives,
    the slot falls back to equal weights across available scripts.
    """

    weights: Dict[str, Dict[str, float]] = {}
    if metrics_per_slot_df is None or metrics_per_slot_df.empty:
        return weights

    for slot in SLOTS:
        slot_df = metrics_per_slot_df[
            (metrics_per_slot_df.get("slot") == slot)
            & (metrics_per_slot_df["total_predictions"] >= min_samples)
        ].copy()
        if slot_df.empty:
            continue

        # Ensure numeric score
        if "score" in slot_df.columns:
            slot_df["score"] = pd.to_numeric(slot_df["score"], errors="coerce").fillna(0.0)

        eligible_scripts = [sid for sid in slot_df["script_id"].astype(str).str.upper().tolist()]
        filtered_scores = {
            row["script_id"]: float(row.get("score", 0.0))
            for _, row in slot_df.iterrows()
            if float(row.get("score", 0.0)) >= min_score
        }

        if not filtered_scores:
            if eligible_scripts:
                equal_weight = 1.0 / len(eligible_scripts)
                weights[slot] = {script: equal_weight for script in eligible_scripts}
            continue

        adjusted_scores = {script: max(score, min_score) for script, score in filtered_scores.items()}
        total_base = sum(adjusted_scores.values()) or 1.0
        slot_weights = {script: score / total_base for script, score in adjusted_scores.items()}
        weights[slot] = slot_weights
    return weights


def compute_pack_hit_stats(window_days: int = 30, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    return compute_pack_hit_stats_core(window_days=window_days, base_dir=base_dir)


def _recommend_topn(hit_rate: float, near_rate: float) -> int:
    if near_rate >= max(0.08, hit_rate * 3):
        return 10
    if near_rate >= max(0.05, hit_rate * 2):
        return 5
    if near_rate > hit_rate:
        return 4
    return 3


def export_nearhit_topn_analysis(
    metrics_df: pd.DataFrame, summary: Dict[str, Any], base_dir: Optional[Path] = None
) -> Optional[Path]:
    if metrics_df is None or metrics_df.empty:
        return None

    project_root = Path(base_dir) if base_dir else quant_paths.get_project_root()
    output_dir = project_root / "logs" / "performance"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "nearhit_topn_analysis.csv"

    window_days = summary.get("effective_window_days") or summary.get("requested_window_days")

    rows: List[Dict[str, Any]] = []
    for _, row in metrics_df.iterrows():
        hit_rate = float(row.get("hit_rate_exact", 0.0) or 0.0)
        near_rate = float(row.get("near_miss_rate", 0.0) or 0.0)
        rows.append(
            {
                "script_id": row.get("script_id"),
                "slot": row.get("slot"),
                "total_predictions": int(row.get("total_predictions", 0) or 0),
                "exact_hits": int(row.get("exact_hits", 0) or 0),
                "near_hits": int(row.get("near_hits", 0) or 0),
                "hit_rate_exact": hit_rate,
                "near_miss_rate": near_rate,
                "recommended_topN": _recommend_topn(hit_rate, near_rate),
                "window_days": window_days,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_script_league(df: pd.DataFrame, min_predictions: int = 10, min_hits_for_hero: int = 1):
    metrics = df if df is not None else pd.DataFrame()
    league: Dict[str, Any] = {"by_slot": {}, "overall": {}, "window_rows": len(metrics)}

    if metrics.empty:
        league["overall"] = {"hero": None, "weak": None}
        return league

    metrics = metrics.copy()
    metrics["script_id"] = metrics.get("script_id").astype(str).str.strip().str.upper()
    if "slot" in metrics.columns:
        metrics["slot"] = metrics.get("slot").astype(str).str.strip().str.upper()

    for slot, sub in metrics.groupby("slot", dropna=False):
        eligible = sub[sub["total_predictions"] >= min_predictions]
        if eligible.empty:
            league["by_slot"][slot] = {"hero": None, "weak": None}
            continue

        hero_pool = eligible[eligible["exact_hits"] >= min_hits_for_hero]
        hero_source = hero_pool if not hero_pool.empty else eligible
        hero_row = hero_source.loc[hero_source["score"].idxmax()]
        weak_row = eligible.loc[eligible["score"].idxmin()]

        league["by_slot"][slot] = {
            "hero": {
                "script_id": str(hero_row.get("script_id")),
                "score": float(hero_row.get("score", 0.0)),
                "hit_rate_exact": float(hero_row.get("hit_rate_exact", 0.0)),
            },
            "weak": {
                "script_id": str(weak_row.get("script_id")),
                "score": float(weak_row.get("score", 0.0)),
                "hit_rate_exact": float(weak_row.get("hit_rate_exact", 0.0)),
            },
        }

    overall_source = metrics[metrics["total_predictions"] >= min_predictions]
    if overall_source.empty:
        league["overall"] = {"hero": None, "weak": None}
        return league

    overall_pool = overall_source[overall_source["exact_hits"] >= min_hits_for_hero]
    hero_source = overall_pool if not overall_pool.empty else overall_source
    hero_row = hero_source.loc[hero_source["score"].idxmax()]
    weak_row = overall_source.loc[overall_source["score"].idxmin()]

    league["overall"] = {
        "hero": {
            "script_id": str(hero_row.get("script_id")),
            "score": float(hero_row.get("score", 0.0)),
            "hit_rate_exact": float(hero_row.get("hit_rate_exact", 0.0)),
        },
        "weak": {
            "script_id": str(weak_row.get("script_id")),
            "score": float(weak_row.get("score", 0.0)),
            "hit_rate_exact": float(weak_row.get("hit_rate_exact", 0.0)),
        },
    }

    return league


def format_script_league(league_df: pd.DataFrame, max_rows: int = 20) -> str:
    if not league_df or not isinstance(league_df, dict):
        return "No league data available."

    lines: List[str] = []
    by_slot = league_df.get("by_slot", {}) or {}
    for slot in SLOTS:
        entry = by_slot.get(slot)
        if not entry:
            continue
        hero_id = entry.get("hero", {}).get("script_id") if isinstance(entry, dict) else None
        weak_id = entry.get("weak", {}).get("script_id") if isinstance(entry, dict) else None
        hero_label = hero_id or "n/a"
        weak_label = weak_id or "n/a"
        lines.append(f"{slot}: hero {hero_label} | weak {weak_label}")

    overall = league_df.get("overall") or {}
    hero_overall = overall.get("hero", {}) if isinstance(overall, dict) else {}
    weak_overall = overall.get("weak", {}) if isinstance(overall, dict) else {}
    hero_label = hero_overall.get("script_id") or "n/a"
    weak_label = weak_overall.get("script_id") or "n/a"
    lines.append(f"Overall hero {hero_label} | weak {weak_label}")

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
        "near_hits",
        "hit_rate_exact",
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

    metrics_df, summary = get_metrics_table(window_days=args.window, mode=args.mode)
    _print_metrics(metrics_df, summary, args.mode)
    if args.mode == "per_slot":
        heroes_df = hero_weak_table(metrics_df)
        if not heroes_df.empty:
            print("\nHero/Weak per slot:")
            print(heroes_df.to_string(index=False))


if __name__ == "__main__":
    main()
