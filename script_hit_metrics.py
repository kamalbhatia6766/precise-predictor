from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from script_hit_memory_utils import load_script_hit_memory


METRIC_COLUMNS: List[str] = [
    "script_name",
    "slot",
    "total_predictions",
    "total_hits",
    "hit_rate",
    "primary_hits",
    "neighbor_hits",
    "mirror_hits",
    "s40_hits",
    "family_164950_hits",
    "last_hit_date",
    "window_days",
]


def _format_date(value: object) -> Optional[str]:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _prepare_memory_df(base_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load and normalise script hit memory for downstream metrics."""

    df = load_script_hit_memory(base_dir=base_dir)
    if df.empty:
        return pd.DataFrame(columns=df.columns)

    df = df.copy()
    for col in ("date", "result_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce").dt.date
    if "hit_flag" in df.columns:
        df["hit_flag"] = pd.to_numeric(df.get("hit_flag"), errors="coerce").fillna(0).astype(int)
    if "is_near_miss" in df.columns:
        df["is_near_miss"] = pd.to_numeric(df.get("is_near_miss"), errors="coerce").fillna(0).astype(int)
    if "slot" in df.columns:
        df["slot"] = df["slot"].apply(_normalise_slot)
    return df


def _normalise_slot(slot_value: object) -> str:
    mapping = {"1": "FRBD", "2": "GZBD", "3": "GALI", "4": "DSWR"}
    slot_str = str(slot_value).strip()
    return mapping.get(slot_str, slot_str)


def _choose_date_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("result_date", "date"):
        if candidate in df.columns and not df[candidate].isna().all():
            return candidate
    return None


def _build_script_identifier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    script_series = None
    if "script_name" in df.columns and df["script_name"].notna().any():
        script_series = df["script_name"]
    elif "script_id" in df.columns:
        script_series = df["script_id"]

    if script_series is None:
        df["script_name"] = None
    else:
        df["script_name"] = script_series.astype(str).str.strip()
    df = df.dropna(subset=["script_name"])
    df = df[df["script_name"].astype(str).str.strip() != ""]
    return df


def compute_script_metrics(
    df: pd.DataFrame, window_days: int = 30, date_col: Optional[str] = None
) -> pd.DataFrame:
    """Compute per-script metrics for a pre-filtered DataFrame."""

    if df is None or df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    working_df = _build_script_identifier(df)
    if working_df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    date_col = date_col or _choose_date_column(working_df)
    if date_col is None:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    if "slot" not in working_df.columns:
        working_df["slot"] = None

    working_df["hit_type"] = working_df.get("hit_type", "").fillna("").astype(str).str.upper()
    working_df["pack_family"] = working_df.get("pack_family", "").fillna("").astype(str).str.strip()
    working_df["hit_flag"] = pd.to_numeric(working_df.get("hit_flag", 0), errors="coerce").fillna(0).astype(int)
    working_df["is_neighbor"] = pd.to_numeric(working_df.get("is_neighbor", 0), errors="coerce").fillna(0).astype(int)
    working_df["is_mirror"] = pd.to_numeric(working_df.get("is_mirror", 0), errors="coerce").fillna(0).astype(int)
    working_df["is_s40"] = pd.to_numeric(working_df.get("is_s40", 0), errors="coerce").fillna(0).astype(int)
    working_df["is_family_164950"] = pd.to_numeric(
        working_df.get("is_family_164950", 0), errors="coerce"
    ).fillna(0).astype(int)

    working_df["_is_hit"] = working_df["hit_type"] != "MISS"

    records: List[Dict[str, object]] = []
    for (script_name, slot), group in working_df.groupby(["script_name", "slot"], dropna=False):
        total_predictions = len(group)
        total_hits = int((group["hit_type"] != "MISS").sum())
        primary_hits = int((group["hit_type"] == "EXACT").sum())
        neighbor_hits = int((group["hit_type"] == "NEIGHBOR").sum())
        mirror_hits = int((group["hit_type"] == "MIRROR").sum())
        s40_hits = int(((group["hit_type"] == "S40") | (group["is_s40"] == 1)).sum())
        family_hits = int(
            ((group["hit_type"] == "FAMILY_164950") | (group["is_family_164950"] == 1)).sum()
        )
        hit_rate = (total_hits / total_predictions) if total_predictions else 0.0
        last_hit_raw = group.loc[group["_is_hit"], date_col].max() if total_hits else None
        last_hit_date = _format_date(last_hit_raw)

        records.append(
            {
                "script_name": script_name,
                "slot": slot,
                "total_predictions": total_predictions,
                "total_hits": total_hits,
                "hit_rate": hit_rate,
                "primary_hits": primary_hits,
                "neighbor_hits": neighbor_hits,
                "mirror_hits": mirror_hits,
                "s40_hits": s40_hits,
                "family_164950_hits": family_hits,
                "last_hit_date": last_hit_date,
                "window_days": window_days,
            }
        )

    metrics_df = pd.DataFrame(records)
    if metrics_df.empty:
        return metrics_df

    metrics_df = metrics_df.sort_values("hit_rate", ascending=False).reset_index(drop=True)
    return metrics_df


def compute_slot_heroes_and_weak(metrics_df: pd.DataFrame) -> Dict[str, Dict[str, List[str]]]:
    """Return hero/weak scripts per slot based on hit_rate ordering."""

    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    default = {slot: {"heroes": [], "weak": []} for slot in slots}

    if metrics_df is None or metrics_df.empty or "slot" not in metrics_df.columns:
        return default

    for slot in slots:
        slot_df = metrics_df[metrics_df["slot"] == slot]
        if slot_df.empty or "hit_rate" not in slot_df.columns:
            continue
        slot_df = slot_df.sort_values("hit_rate", ascending=False)
        count = len(slot_df)
        band = max(1, count // 3)
        heroes = slot_df.head(band)["script_name"].astype(str).str.strip().tolist()
        weak = slot_df.tail(band)["script_name"].astype(str).str.strip().tolist()
        default[slot] = {"heroes": heroes, "weak": weak}

    return default


def build_script_league(
    df: pd.DataFrame, min_predictions: int = 10, min_hits_for_hero: int = 1
) -> Dict[str, object]:
    """Aggregate metrics into a simple hero/weak league."""

    window_rows = len(df) if df is not None else 0
    if df is None or df.empty:
        return {"heroes": [], "weak": [], "window_rows": window_rows}

    working_df = df.copy()
    working_df = working_df.dropna(subset=["script_name"]) if "script_name" in working_df.columns else pd.DataFrame()
    if working_df.empty:
        return {"heroes": [], "weak": [], "window_rows": window_rows}

    working_df["script_name"] = working_df["script_name"].astype(str).str.strip()
    working_df = working_df[working_df["script_name"] != ""]
    if working_df.empty:
        return {"heroes": [], "weak": [], "window_rows": window_rows}

    agg_df = (
        working_df.groupby("script_name", as_index=False)
        .agg(
            total_predictions=("total_predictions", "sum"),
            total_hits=("total_hits", "sum"),
            last_hit_date=("last_hit_date", "max"),
        )
    )

    if agg_df.empty:
        return {"heroes": [], "weak": [], "window_rows": window_rows}

    agg_df["hit_rate"] = agg_df.apply(
        lambda row: (row["total_hits"] / row["total_predictions"]) if row["total_predictions"] else 0.0,
        axis=1,
    )

    filtered = agg_df[agg_df["total_predictions"] >= min_predictions]
    if filtered.empty:
        return {"heroes": [], "weak": [], "window_rows": window_rows}

    heroes_df = filtered[filtered["total_hits"] >= min_hits_for_hero]
    heroes_df = heroes_df.sort_values(["hit_rate", "total_predictions"], ascending=[False, False])

    weak_df = filtered[filtered["total_hits"] == 0]
    weak_df = weak_df.sort_values(["total_predictions"], ascending=[False])

    heroes = [
        {
            "script": row["script_name"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": int(row["total_hits"]),
            "hit_rate": float(row["hit_rate"]),
        }
        for _, row in heroes_df.iterrows()
    ]

    weak = [
        {
            "script": row["script_name"],
            "total_predictions": int(row["total_predictions"]),
            "total_hits": int(row["total_hits"]),
            "hit_rate": float(row["hit_rate"]),
        }
        for _, row in weak_df.iterrows()
    ]

    return {"heroes": heroes, "weak": weak, "window_rows": window_rows}


def format_script_league(league: Dict[str, object]) -> str:
    """Return a compact, printable summary of the league."""

    if not league or league.get("window_rows", 0) == 0:
        return "Script league: no data in this window."

    heroes = league.get("heroes") or []
    weak = league.get("weak") or []

    if not heroes and not weak:
        return "Script league: warming up (no strong signals yet in this window)."

    lines = [f"Script league (window rows: {league.get('window_rows')})"]

    if heroes:
        lines.append("  Heroes (by hit-rate):")
        for entry in heroes:
            lines.append(
                "    - "
                + f"{entry['script']}: hits={entry['total_hits']} / preds={entry['total_predictions']} "
                + f"(hit_rate={entry['hit_rate']:.2f})"
            )

    if weak:
        lines.append("  Under watch (0-hit, high volume):")
        for entry in weak:
            lines.append(
                "    - "
                + f"{entry['script']}: hits={entry['total_hits']} / preds={entry['total_predictions']}"
            )

    return "\n".join(lines)


def load_script_metrics(
    window_days: int = 30, fallback: bool = True, project_root: Optional[Path] = None
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    """
    Load script hit memory, apply windowing with fallback, and compute metrics.

    Returns (metrics_df, summary_dict) or (None, None) if no data is available.
    """

    memory_df = _prepare_memory_df(base_dir=project_root)
    if memory_df.empty:
        return None, None

    date_col = _choose_date_column(memory_df)
    if date_col is None:
        return None, None

    earliest_date = memory_df[date_col].min()
    latest_date = memory_df[date_col].max()
    if pd.isna(earliest_date) or pd.isna(latest_date):
        return None, None

    candidate_windows: List[int] = [window_days]
    if fallback:
        double_window = min(2 * window_days, 90)
        full_history = (latest_date - earliest_date).days + 1
        for candidate in (double_window, full_history):
            if candidate not in candidate_windows:
                candidate_windows.append(candidate)

    effective_window = None
    window_df: Optional[pd.DataFrame] = None

    for candidate in candidate_windows:
        cutoff = latest_date - timedelta(days=candidate - 1)
        df_slice = memory_df[memory_df[date_col] >= cutoff]
        if not df_slice.empty:
            effective_window = candidate
            window_df = df_slice.copy()
            break

    if effective_window is None or window_df is None or window_df.empty:
        return None, None

    metrics_df = compute_script_metrics(window_df, window_days=effective_window, date_col=date_col)
    summary = {
        "requested_window_days": window_days,
        "effective_window_days": effective_window,
        "earliest_date": _format_date(earliest_date),
        "latest_date": _format_date(latest_date),
        "total_rows": len(window_df),
        "total_scripts": int(metrics_df["script_name"].nunique()) if metrics_df is not None else 0,
    }

    return metrics_df, summary


def build_script_weight_map(
    window_days: int = 30, project_root: Optional[Path] = None
) -> Dict[str, Dict[str, Any]]:
    """Build a conservative script weight map from hit metrics.

    Returns a dict keyed by script name (e.g., "SCR1") with fields:
    weight, hit_rate, extended_hit_rate, blind_miss_rate, total_predictions,
    total_hits, last_hit_date.
    """

    def _neutral_weights() -> Dict[str, Dict[str, Any]]:
        return {
            f"SCR{i}": {
                "weight": 1.0,
                "hit_rate": 0.0,
                "extended_hit_rate": 0.0,
                "blind_miss_rate": 1.0,
                "total_predictions": 0,
                "total_hits": 0,
                "last_hit_date": None,
            }
            for i in range(1, 10)
        }

    metrics_df, summary = load_script_metrics(
        window_days=window_days, fallback=True, project_root=project_root
    )

    if metrics_df is None or summary is None or metrics_df.empty:
        return _neutral_weights()

    agg_df = (
        metrics_df.copy()
        .fillna({
            "total_predictions": 0,
            "total_hits": 0,
            "primary_hits": 0,
            "neighbor_hits": 0,
            "mirror_hits": 0,
        })
        .groupby("script_name", as_index=False)
        .agg(
            total_predictions=("total_predictions", "sum"),
            total_hits=("total_hits", "sum"),
            primary_hits=("primary_hits", "sum"),
            last_hit_date=("last_hit_date", "max"),
        )
    )

    reference_date = None
    latest_date = summary.get("latest_date") if isinstance(summary, dict) else None
    if latest_date:
        try:
            reference_date = pd.to_datetime(latest_date).date()
        except Exception:
            reference_date = None

    raw_scores: List[float] = []
    weight_entries: Dict[str, Dict[str, Any]] = {}
    for _, row in agg_df.iterrows():
        script_name = str(row.get("script_name", "")).strip()
        if not script_name:
            continue
        script_name = script_name.upper()

        total_predictions = int(row.get("total_predictions", 0) or 0)
        total_hits = int(row.get("total_hits", 0) or 0)
        primary_hits = int(row.get("primary_hits", 0) or 0)

        exact_hit_rate = primary_hits / max(1, total_predictions)
        extended_hit_rate = total_hits / max(1, total_predictions)
        blind_miss_rate = 1.0 - extended_hit_rate

        recency_penalty = 0.0
        last_hit_raw = row.get("last_hit_date")
        if pd.notna(last_hit_raw):
            try:
                last_hit = pd.to_datetime(str(last_hit_raw)).date()
                anchor = reference_date or datetime.utcnow().date()
                days_since = max(0, (anchor - last_hit).days)
                recency_penalty = min(0.1, (days_since / max(1, window_days)) * 0.05)
            except Exception:
                recency_penalty = 0.0

        base_score = extended_hit_rate
        bonus_exact = 0.3 * exact_hit_rate
        penalty_blind = 0.2 * blind_miss_rate
        raw_score = base_score + bonus_exact - penalty_blind - recency_penalty
        raw_score = float(raw_score)
        if raw_score < 0:
            raw_score = 0.0

        weight_entries[script_name] = {
            "hit_rate": exact_hit_rate,
            "extended_hit_rate": extended_hit_rate,
            "blind_miss_rate": blind_miss_rate,
            "total_predictions": total_predictions,
            "total_hits": total_hits,
            "last_hit_date": row.get("last_hit_date"),
            "raw_score": raw_score,
        }
        raw_scores.append(raw_score)

    if not raw_scores or all(score == raw_scores[0] for score in raw_scores):
        for entry in weight_entries.values():
            entry["weight"] = 1.0
        for name, neutral_entry in _neutral_weights().items():
            if name not in weight_entries:
                weight_entries[name] = neutral_entry
        return weight_entries

    adjusted_scores = [score if score > 0 else 0.001 for score in raw_scores]
    mean_score = sum(adjusted_scores) / len(adjusted_scores)
    if mean_score == 0:
        for entry in weight_entries.values():
            entry["weight"] = 1.0
        for name, neutral_entry in _neutral_weights().items():
            if name not in weight_entries:
                weight_entries[name] = neutral_entry
        return weight_entries

    for script, entry in weight_entries.items():
        weight = entry["raw_score"] / mean_score if mean_score else 1.0
        weight = max(0.5, min(1.5, weight))
        entry["weight"] = weight

    for name, neutral_entry in _neutral_weights().items():
        if name not in weight_entries:
            weight_entries[name] = neutral_entry

    return weight_entries


def get_metrics_table(
    window_days: int = 30,
    fallback: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, object]]]:
    """
    Backwards-compatible shim for older callers.

    Returns (metrics_df, summary_dict), or (None, None) if no data is available
    even after applying the fallback logic.
    """
    metrics_df, summary = load_script_metrics(
        window_days=window_days, fallback=fallback
    )

    # Mirror the CLI behaviour: if no data, print a clear message
    if metrics_df is None or summary is None:
        print("No script hit memory data available (even after fallback up to ALL history).")
        return None, None

    return metrics_df, summary


def _format_header(summary: Dict[str, object]) -> str:
    return (
        "Script hit metrics – requested "
        f"{summary.get('requested_window_days')}d, used {summary.get('effective_window_days')}d "
        f"(rows={summary.get('total_rows')})"
    )


def _print_metrics(metrics_df: pd.DataFrame, summary: Dict[str, object]) -> None:
    print(_format_header(summary))
    if metrics_df.empty:
        print("No script hit metrics rows to display.")
    else:
        display_cols = [
            "script_name",
            "slot",
            "total_predictions",
            "total_hits",
            "hit_rate",
            "primary_hits",
            "neighbor_hits",
            "mirror_hits",
            "s40_hits",
            "family_164950_hits",
            "last_hit_date",
        ]
        existing_cols = [c for c in display_cols if c in metrics_df.columns]
        print(metrics_df[existing_cols].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script-wise hit metrics for Precise Predictor.")
    parser.add_argument("--window", type=int, default=30, help="Number of days to look back from the latest date.")
    args = parser.parse_args()

    metrics_df, summary = load_script_metrics(window_days=args.window, fallback=True)

    if metrics_df is None or summary is None:
        print("No script hit memory data available (even after fallback up to ALL history).")
    else:
        _print_metrics(metrics_df, summary)
        league = build_script_league(metrics_df)
        print(format_script_league(league))
