from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pandas as pd

import quant_paths
from quant_core import hit_core
from quant_stats_core import compute_pack_hit_stats as compute_pack_hit_stats_core, compute_script_slot_stats
from script_hit_memory_utils import (
    classify_relation,
    filter_hits_by_window,
    load_script_hit_memory,
    normalize_date_column,
)

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def safe(value: object) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


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

    df, date_col = normalize_date_column(df)
    if df.empty:
        return pd.DataFrame(), None

    date_col = date_col or "result_date"

    df = df.copy()
    df[date_col] = pd.to_datetime(df.get(date_col), errors="coerce").dt.date
    df["slot"] = df.get("slot").apply(_normalise_slot)
    df["script_id"] = df.get("script_id").astype(str).str.strip().str.upper()
    df["HIT_TYPE"] = df.get("HIT_TYPE", "MISS").astype(str).str.upper()
    df["is_exact_hit"] = df.get("is_exact_hit", False).astype(bool)
    df["is_near_miss"] = df.get("is_near_miss", False).astype(bool)
    df["is_near_hit"] = df.get("is_near_hit", False).astype(bool)
    df["_relation"] = df.apply(lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1)
    df = df.dropna(subset=["slot", "script_id"])
    return df, date_col


def _window_memory(df: pd.DataFrame, date_col: Optional[str], window_days: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty_summary = {
        "requested_window_days": window_days,
        "effective_window_days": 0,
        "available_days": 0,
        "available_days_total": 0,
        "window_start": None,
        "window_end": None,
        "latest_date": None,
        "earliest_date": None,
        "total_rows": 0,
    }

    if df.empty or not date_col:
        return pd.DataFrame(), empty_summary

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col])
    if df.empty:
        return pd.DataFrame(), empty_summary

    window_df, effective_days = filter_hits_by_window(df, window_days)
    if window_df.empty:
        return pd.DataFrame(), empty_summary

    window_df[date_col] = pd.to_datetime(window_df[date_col], errors="coerce").dt.date
    latest_date = window_df[date_col].max()
    earliest_date = df[date_col].min().date() if hasattr(df[date_col].min(), "date") else df[date_col].min()
    total_available_days = int(df[date_col].dt.date.nunique())
    window_start = latest_date - timedelta(days=window_days - 1)

    filtered = window_df.copy()
    if filtered.empty:
        return pd.DataFrame(), empty_summary

    filtered_dates = pd.to_datetime(filtered[date_col], errors="coerce").dt.date
    filtered[date_col] = filtered_dates
    available_days = int(filtered_dates.dropna().nunique())

    summary = {
        "requested_window_days": int(window_days),
        "effective_window_days": int(effective_days),
        "available_days": int(effective_days),
        "available_days_total": total_available_days,
        "window_end": latest_date,
        "window_start": filtered[date_col].min(),
        "latest_date": latest_date,
        "earliest_date": earliest_date,
        "window_earliest_date": filtered[date_col].min(),
        "total_rows": len(filtered),
    }
    return filtered, summary


def _aggregate_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    return compute_script_slot_stats(df, group_cols)


def _score_block(metrics: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return a compact score block with a suffix for merging."""

    if metrics is None or metrics.empty:
        return pd.DataFrame()

    block = metrics.copy()
    suffix = f"_{label}" if label else ""
    rename_map = {
        "score": f"score{suffix}",
        "total_predictions": f"total_predictions{suffix}",
        "exact_hits": f"exact_hits{suffix}",
        "near_hits": f"near_hits{suffix}",
        "hit_rate_exact": f"hit_rate_exact{suffix}",
        "near_miss_rate": f"near_miss_rate{suffix}",
        "blind_miss_rate": f"blind_miss_rate{suffix}",
    }
    existing = {k: v for k, v in rename_map.items() if k in block.columns}
    return block.rename(columns=existing)


def get_metrics_table(
    window_days: int = 30,
    base_dir: Optional[Path] = None,
    mode: str = "per_slot",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df, date_col = _prepare_memory_df(base_dir=base_dir)
    if df.empty:
        return pd.DataFrame(), {"requested_window_days": window_days, "total_rows": 0}

    group_cols = ["script_id"] if mode == "overall" else ["script_id", "slot"]

    def _compute_window(label: str, days: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
        if days is None:
            window_df = df.copy()
            summary = {
                "requested_window_days": "FULL",
                "effective_window_days": int(window_df[date_col].nunique()) if date_col else 0,
                "available_days": int(window_df[date_col].nunique()) if date_col else 0,
                "available_days_total": int(window_df[date_col].nunique()) if date_col else 0,
                "window_start": window_df[date_col].min() if date_col and not window_df.empty else None,
                "window_end": window_df[date_col].max() if date_col and not window_df.empty else None,
                "latest_date": window_df[date_col].max() if date_col and not window_df.empty else None,
                "earliest_date": window_df[date_col].min() if date_col and not window_df.empty else None,
                "window_earliest_date": window_df[date_col].min() if date_col and not window_df.empty else None,
                "total_rows": len(window_df),
            }
        else:
            window_df, summary = _window_memory(df, date_col, days)
        if window_df.empty:
            return pd.DataFrame(), summary, window_df
        metrics = _aggregate_metrics(window_df, group_cols)
        if not metrics.empty and "score" in metrics.columns:
            metrics["score"] = (metrics["score"] + 0.20) * 100.0
        return metrics, summary, window_df

    # Base (requested) window for backward compatibility
    base_metrics, summary, base_window_df = _compute_window(f"{window_days}d", window_days)
    if base_metrics.empty:
        return pd.DataFrame(), summary

    window_defs = [("30d", 30), ("60d", 60), ("90d", 90), ("full", None)]
    window_tables: Dict[str, pd.DataFrame] = {}
    for label, days in window_defs:
        metrics, _, _ = _compute_window(label, days if days is None or days > 0 else window_days)
        if metrics is None or metrics.empty:
            continue
        window_tables[label] = _score_block(metrics, label)

    metrics = base_metrics.copy()
    merge_on = ["script_id"] if mode == "overall" else ["script_id", "slot"]
    for label, table in window_tables.items():
        metrics = metrics.merge(table[merge_on + [c for c in table.columns if c not in merge_on]], on=merge_on, how="outer")

    if mode == "per_slot":
        total_predictions = len(base_window_df)
        relations = base_window_df.get("_relation") if "_relation" in base_window_df.columns else base_window_df.apply(
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

    for label, _ in window_defs:
        score_col = f"score_{label}"
        if score_col in metrics.columns:
            continue
        source_col = f"score_{label}" if f"score_{label}" in metrics.columns else None
        if source_col:
            metrics.rename(columns={source_col: score_col}, inplace=True)

    # blended score using available window scores (defaults to 0)
    metrics["score_30d"] = np.nan_to_num(metrics.get("score_30d", 0.0), nan=0.0)
    metrics["score_60d"] = np.nan_to_num(metrics.get("score_60d", metrics.get("score_60", 0.0)), nan=0.0)
    metrics["score_90d"] = np.nan_to_num(metrics.get("score_90d", metrics.get("score_90", 0.0)), nan=0.0)
    metrics["score_full"] = np.nan_to_num(metrics.get("score_full", 0.0), nan=0.0)
    metrics["blended_score"] = (
        metrics["score_30d"].fillna(0.0) * 0.4
        + metrics["score_60d"].fillna(0.0) * 0.3
        + metrics["score_90d"].fillna(0.0) * 0.2
        + metrics["score_full"].fillna(0.0) * 0.1
    )
    metrics["blended_score"] = metrics["blended_score"].apply(safe)

    order_cols = ["script_id", "slot"] if "slot" in metrics.columns else ["script_id"]
    metrics = metrics.sort_values(order_cols + ["blended_score"], ascending=[True] * len(order_cols) + [False]).reset_index(drop=True)
    export_nearhit_topn_analysis(metrics, summary, base_dir=base_dir)
    _export_metrics_bundle(metrics, summary, window_days=window_days, base_dir=base_dir)
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
        score_col = "blended_score" if "blended_score" in eligible.columns else "score"
        hero_pool = eligible[eligible["exact_hits"] > 0]
        if hero_pool.empty:
            hero_pool = eligible[eligible["near_hits"] > 0]
        hero_row = hero_pool.sort_values(score_col, ascending=False).iloc[0] if not hero_pool.empty else None
        weak_row = eligible.sort_values(score_col, ascending=True).iloc[0]
        rows.append(
            {
                "slot": slot,
                "hero_script": hero_row.get("script_id") if hero_row is not None else None,
                "hero_score": hero_row.get(score_col) if hero_row is not None else None,
                "hero_hit_rate_exact": hero_row.get("hit_rate_exact") if hero_row is not None else None,
                "weak_script": weak_row.get("script_id"),
                "weak_score": weak_row.get(score_col),
                "weak_hit_rate_exact": weak_row.get("hit_rate_exact"),
            }
        )
    return pd.DataFrame(rows)


def _export_hero_weak_json(
    heroes_df: pd.DataFrame, metrics_df: pd.DataFrame, summary: Dict[str, Any], base_dir: Optional[Path] = None
) -> None:
    try:
        if heroes_df is None or heroes_df.empty or metrics_df is None or metrics_df.empty:
            return

        project_root = Path(base_dir) if base_dir else quant_paths.get_base_dir()
        output_path = project_root / "logs" / "performance" / "script_hero_weak.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        per_slot: Dict[str, Dict[str, Any]] = {}
        for _, row in heroes_df.iterrows():
            slot = row.get("slot")
            if not slot:
                continue
            per_slot[str(slot)] = {
                "hero_script": row.get("hero_script"),
                "hero_score": row.get("hero_score"),
                "hero_hit_rate_exact": row.get("hero_hit_rate_exact"),
                "weak_script": row.get("weak_script"),
                "weak_score": row.get("weak_score"),
                "weak_hit_rate_exact": row.get("weak_hit_rate_exact"),
            }

        eligible = metrics_df[metrics_df.get("script_id") != "ALL"]
        pred_counts = pd.to_numeric(eligible.get("total_predictions"), errors="coerce").fillna(0)
        eligible = eligible[pred_counts >= 10]

        score_col = "blended_score" if "blended_score" in eligible.columns else "score"
        hero_row = eligible.loc[eligible[score_col].idxmax()] if not eligible.empty else None
        weak_row = eligible.loc[eligible[score_col].idxmin()] if not eligible.empty else None

        overall = {
            "hero_script": hero_row.get("script_id") if hero_row is not None else None,
            "hero_score": hero_row.get(score_col) if hero_row is not None else None,
            "hero_hit_rate_exact": hero_row.get("hit_rate_exact") if hero_row is not None else None,
            "weak_script": weak_row.get("script_id") if weak_row is not None else None,
            "weak_score": weak_row.get(score_col) if weak_row is not None else None,
            "weak_hit_rate_exact": weak_row.get("hit_rate_exact") if weak_row is not None else None,
        }

        hero_weak = {
            "per_slot": per_slot,
            "overall": overall,
            "meta": {
                "window_days": summary.get("effective_window_days") or summary.get("requested_window_days"),
                "requested_window_days": summary.get("requested_window_days"),
                "effective_window_days": summary.get("effective_window_days"),
                "available_days": summary.get("available_days"),
                "window_start": summary.get("window_start") or summary.get("earliest_date"),
                "window_end": summary.get("window_end") or summary.get("latest_date"),
                "rows": summary.get("total_rows", len(metrics_df)),
            },
        }

        def _json_safe(val: Any) -> Any:
            if isinstance(val, (pd.Timestamp,)):
                return val.date().isoformat()
            if hasattr(val, "isoformat"):
                try:
                    return val.isoformat()
                except Exception:
                    pass
            if isinstance(val, (pd.Series, pd.DataFrame)):
                return val.to_dict()
            if isinstance(val, (pd.Int64Dtype,)):
                return int(val)
            if isinstance(val, (pd.Float64Dtype,)):
                return float(val)
            try:
                import numpy as np

                if isinstance(val, (np.integer,)):
                    return int(val)
                if isinstance(val, (np.floating,)):
                    return float(val)
            except Exception:
                pass
            return val

        safe_payload = json.loads(json.dumps(hero_weak, default=_json_safe))
        output_path.write_text(json.dumps(safe_payload, indent=2))
    except Exception as exc:
        print(f"[script_hit_metrics] Warning: unable to write script_hero_weak.json: {exc}")


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


def _export_metrics_bundle(
    metrics_df: pd.DataFrame, summary: Dict[str, Any], window_days: int, base_dir: Optional[Path] = None
) -> None:
    if metrics_df is None or metrics_df.empty:
        return

    project_root = Path(base_dir) if base_dir else quant_paths.get_project_root()
    output_dir = project_root / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _aggregate_slot(slot: str) -> Dict[str, Any]:
        subset = metrics_df[(metrics_df.get("slot") == slot) & (metrics_df.get("script_id") != "ALL")]
        total_predictions = int(subset.get("total_predictions", pd.Series(dtype=int)).sum()) if not subset.empty else 0
        exact_hits = int(subset.get("exact_hits", pd.Series(dtype=int)).sum()) if not subset.empty else 0
        near_hits = int(subset.get("near_hits", pd.Series(dtype=int)).sum()) if not subset.empty else 0
        hit_rate_exact = exact_hits / total_predictions if total_predictions else 0.0
        near_miss_rate = near_hits / total_predictions if total_predictions else 0.0
        blind_misses = max(total_predictions - exact_hits - near_hits, 0)
        blind_miss_rate = blind_misses / total_predictions if total_predictions else 0.0
        score = (120.0 * hit_rate_exact + 40.0 * near_miss_rate - 30.0 * blind_miss_rate)
        score = (score + 0.20) * 100.0 if total_predictions else 0.0
        return {
            "slot": slot,
            "total_predictions": total_predictions,
            "exact_hits": exact_hits,
            "near_hits": near_hits,
            "hit_rate_exact": hit_rate_exact,
            "near_miss_rate": near_miss_rate,
            "blind_miss_rate": blind_miss_rate,
            "score": score,
        }

    slot_rows = [_aggregate_slot(slot) for slot in SLOTS]
    total_predictions = sum(row["total_predictions"] for row in slot_rows)
    exact_hits = sum(row["exact_hits"] for row in slot_rows)
    near_hits = sum(row["near_hits"] for row in slot_rows)
    hit_rate_exact = exact_hits / total_predictions if total_predictions else 0.0
    near_miss_rate = near_hits / total_predictions if total_predictions else 0.0
    blind_misses = max(total_predictions - exact_hits - near_hits, 0)
    blind_miss_rate = blind_misses / total_predictions if total_predictions else 0.0
    score = (120.0 * hit_rate_exact + 40.0 * near_miss_rate - 30.0 * blind_miss_rate)
    score = (score + 0.20) * 100.0 if total_predictions else 0.0

    summary_row = {
        "slot": "ALL",
        "total_predictions": total_predictions,
        "exact_hits": exact_hits,
        "near_hits": near_hits,
        "hit_rate_exact": hit_rate_exact,
        "near_miss_rate": near_miss_rate,
        "blind_miss_rate": blind_miss_rate,
        "score": score,
    }

    payload = {row["slot"]: {k: v for k, v in row.items() if k != "slot"} for row in slot_rows + [summary_row]}
    json_path = output_dir / f"script_metrics_{window_days}d.json"
    csv_path = output_dir / f"script_metrics_{window_days}d.csv"

    try:
        json_path.write_text(json.dumps(payload, indent=2))
    except Exception as exc:
        print(f"[script_hit_metrics] Warning: unable to write {json_path}: {exc}")

    try:
        pd.DataFrame(slot_rows + [summary_row]).to_csv(csv_path, index=False)
    except Exception as exc:
        print(f"[script_hit_metrics] Warning: unable to write {csv_path}: {exc}")


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
    rows_display = summary.get("total_rows") if isinstance(summary, dict) else None
    rows_display = rows_display if rows_display is not None else len(metrics_df)
    effective_days = summary.get("effective_window_days") or summary.get("available_days")
    header = (
        "Script hit metrics – "
        f"requested {summary.get('requested_window_days')}d, used {effective_days}d "
        f"(rows={rows_display})"
    )
    print(header)
    if summary.get("window_start") and summary.get("window_end"):
        print(
            f"Window range: {summary.get('window_start')} → {summary.get('window_end')} | "
            f"available days: {summary.get('available_days')} / total distinct: {summary.get('available_days_total')}"
        )
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
        "score_30d",
        "score_60d",
        "score_90d",
        "score_full",
        "blended_score",
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
        _export_hero_weak_json(heroes_df, metrics_df, summary)


if __name__ == "__main__":
    main()
