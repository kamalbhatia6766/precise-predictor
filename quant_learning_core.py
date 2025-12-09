"""Shared learning/pattern utilities.

This module stays lightweight and focuses on consuming existing outputs
without changing any R0/R1/R2 behaviour.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

import quant_paths
from quant_core import pnl_core
from utils_2digit import to_2d_str
from script_hit_memory_utils import load_script_hit_memory


PERFORMANCE_DIR = quant_paths.get_performance_logs_dir()
PATTERN_LOW_THRESHOLD = 0.001
PATTERN_HIGH_THRESHOLD = 0.010


def load_pattern_summary_json(window_days: int = 90) -> Optional[Dict]:
    """Load the standard pattern summary JSON if present."""

    paths: Iterable[Path] = [
        PERFORMANCE_DIR / f"pattern_intel_summary_{window_days}d.json",
        PERFORMANCE_DIR / "pattern_intelligence_summary.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            return json.loads(path.read_text())
        except Exception:
            continue
    return None


def slot_regime(hit_rate_exact: Optional[float]) -> str:
    if hit_rate_exact is None:
        return "NORMAL"
    if hit_rate_exact >= PATTERN_HIGH_THRESHOLD:
        return "BOOST"
    if hit_rate_exact <= PATTERN_LOW_THRESHOLD:
        return "OFF"
    return "NORMAL"


def _normalise_number(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        return to_2d_str(int(value))
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit():
        return to_2d_str(int(text))
    return None


def _parse_predictions(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        numbers: List[str] = []
        for item in value:
            numbers.extend(_parse_predictions(item))
        return numbers
    text = str(value)
    parts = [p.strip() for p in text.replace(";", ",").replace("|", ",").split(",") if p.strip()]
    results: List[str] = []
    for part in parts:
        norm = _normalise_number(part)
        if norm is not None:
            results.append(norm)
    return results


def _extract_topn_records(snapshot: Dict) -> List[Dict]:
    if not snapshot:
        return []

    candidates = snapshot.get("records")
    if not candidates:
        for key in ("bets", "bet_history", "entries"):
            if snapshot.get(key):
                candidates = snapshot.get(key)
                break

    records: List[Dict] = []
    for item in candidates or []:
        date_raw = item.get("bet_date") or item.get("date") or item.get("result_date")
        try:
            date_val = datetime.strptime(str(date_raw)[:10], "%Y-%m-%d").date()
        except Exception:
            continue

        slot = str(item.get("slot", "")).upper()
        if not slot:
            continue

        preds = (
            item.get("bet_numbers")
            or item.get("numbers")
            or item.get("predictions")
            or item.get("top_predictions")
        )
        pred_list = _parse_predictions(preds)
        if not pred_list:
            continue

        result_val = _normalise_number(
            item.get("result") or item.get("result_number") or item.get("actual") or item.get("real_number")
        )
        if result_val is None:
            continue

        records.append({"date": date_val, "slot": slot, "predictions": pred_list, "result": result_val})

    return records


def _build_records_from_hit_memory(window_days: int = 90) -> List[Dict]:
    try:
        df = load_script_hit_memory()
    except Exception:
        return []

    if df is None or df.empty:
        return []

    df = df.copy()
    df["result_date"] = pd.to_datetime(df.get("result_date"), errors="coerce").dt.date
    df = df.dropna(subset=["result_date"])
    df["slot"] = df.get("slot").astype(str).str.upper()

    latest = df["result_date"].max()
    if pd.isna(latest):
        return []
    cutoff = latest - timedelta(days=window_days - 1)
    df = df[df["result_date"] >= cutoff]

    records: List[Dict] = []
    for (dt_val, slot), group in df.groupby(["result_date", "slot"]):
        ordered = group.sort_values(["rank", "rank_in_script"], na_position="last")
        predictions: List[str] = []
        for _, row in ordered.iterrows():
            pred = _normalise_number(row.get("predicted_number") or row.get("predicted"))
            if pred and pred not in predictions:
                predictions.append(pred)
            if len(predictions) >= 10:
                break

        result_candidates = [_normalise_number(val) for val in ordered.get("real_number", []) if _normalise_number(val)]
        if not predictions or not result_candidates:
            continue

        records.append({"date": dt_val, "slot": slot, "predictions": predictions, "result": result_candidates[0]})

    return records


def compute_topn_roi_from_snapshot(
    snapshot: Optional[Dict] = None, window_days: int = 30, max_n: int = 10
) -> Optional[Dict]:
    data = snapshot if snapshot is not None else pnl_core.load_pnl_snapshot()
    records = _extract_topn_records(data)
    if not records:
        records = _build_records_from_hit_memory(window_days=window_days)
    if not records:
        return None

    unique_dates = sorted({r["date"] for r in records})
    selected_dates = unique_dates[-window_days:] if window_days else unique_dates
    window_records = [r for r in records if r["date"] in set(selected_dates)]
    if not window_records:
        return None

    unique_days = {r["date"] for r in window_records}
    start_date = min(unique_days)
    latest_date = max(unique_days)

    def _roi_for_n(recs: List[Dict], n: int):
        stake = profit = 0.0
        slot_map: Dict[str, Dict[str, float]] = {}
        for rec in recs:
            top_preds = rec["predictions"][:n]
            if not top_preds:
                continue
            slot = rec.get("slot") or "NA"
            stake += len(top_preds)
            hit = rec["result"] in top_preds
            profit_step = 90 - len(top_preds) if hit else -len(top_preds)
            profit += profit_step

            slot_entry = slot_map.setdefault(slot, {"stake": 0.0, "profit": 0.0})
            slot_entry["stake"] += len(top_preds)
            slot_entry["profit"] += profit_step

        overall = {"roi": (profit / stake) if stake else 0.0, "stake": stake, "profit": profit}
        per_slot = {
            slot: {**vals, "roi": (vals["profit"] / vals["stake"] if vals.get("stake") else 0.0)}
            for slot, vals in slot_map.items()
        }
        return overall, per_slot

    results: Dict[int, Dict[str, float]] = {}
    per_slot: Dict[str, Dict[int, Dict[str, float]]] = {}

    for n in range(1, max_n + 1):
        overall, slot_data = _roi_for_n(window_records, n)
        results[n] = overall
        for slot, stats in slot_data.items():
            per_slot.setdefault(slot, {})[n] = stats

    return {
        "start_date": start_date,
        "end_date": latest_date,
        "unique_days": len(unique_days),
        "bets": len(window_records),
        "results": results,
        "per_slot": per_slot,
    }
