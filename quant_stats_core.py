from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

import quant_paths
from script_hit_memory_utils import classify_relation, filter_hits_by_window, load_script_hit_memory
import pattern_packs


logger = logging.getLogger(__name__)


NEAR_HIT_TYPES = {
    "NEAR",
    "NEIGHBOR",
    "MIRROR",
    "S40",
    "FAMILY_164950",
    "CROSS_SLOT",
    "CROSS_DAY",
}

UNIT_STAKE = 10
FULL_PAYOUT_PER_UNIT = 90
NEAR_HIT_WEIGHT = 0.3


def _load_ultimate_performance() -> pd.DataFrame:
    path = quant_paths.get_performance_logs_dir() / "ultimate_performance.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    elif "result_date" in df.columns:
        df["date"] = pd.to_datetime(df["result_date"], errors="coerce")
    else:
        return pd.DataFrame()
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.normalize()
    df["slot"] = df.get("slot", "").astype(str).str.upper()
    return df


def safe(value: object) -> float:
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _pick_best_n(roi_map: Dict[int, float]) -> Optional[int]:
    if not roi_map:
        return None
    best_roi = max(roi_map.values())
    tied = [n for n, roi in roi_map.items() if roi == best_roi]
    return min(tied) if tied else None


def _series_from_any(x: Any) -> pd.Series:
    """Normalise scalars, arrays or frames to a pandas Series for safe math ops."""
    if x is None:
        return pd.Series(dtype="float64")
    if isinstance(x, pd.Series):
        return x
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            return pd.Series(dtype="float64")
        return x.iloc[:, 0]
    if np.isscalar(x):
        return pd.Series([x], dtype="float64")
    try:
        return pd.Series(x)
    except Exception:
        return pd.Series(dtype="float64")


def _parse_numbers_list(value: Any) -> List[str]:
    """
    Best-effort parser that converts the stored numbers list (stringified list, CSV,
    or already-materialised list) into a normalised list of 2-digit strings.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).zfill(2) for x in value]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        # JSON-style list
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = json.loads(txt)
                if isinstance(parsed, list):
                    return [str(x).zfill(2) for x in parsed]
            except Exception:
                pass
        # Fallback: comma-separated string
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        return [str(p).zfill(2) for p in parts]
    return []


def _collect_top_numbers(window_df: pd.DataFrame, max_n: int) -> Dict[str, Dict[int, List[str]]]:
    """
    Build frequency-ranked top-number lists per slot for each N bucket. The intent
    is to expose which numbers were most responsible for the Top-N ROI bands within
    the evaluation window, without re-scanning rows repeatedly.
    """
    number_counts: Dict[str, Dict[str, int]] = {}
    if "top_numbers" in window_df.columns:
        source_col = "top_numbers"
    elif "numbers" in window_df.columns:
        source_col = "numbers"
    else:
        source_col = None

    if source_col:
        for slot, slot_df in window_df.groupby("slot"):
            slot_counts: Dict[str, int] = {}
            for raw in slot_df[source_col]:
                for num in _parse_numbers_list(raw):
                    slot_counts[num] = slot_counts.get(num, 0) + 1
            if slot_counts:
                number_counts[slot] = slot_counts

    result: Dict[str, Dict[int, List[str]]] = {}
    for slot, counts in number_counts.items():
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        numbers_only = [num for num, _ in ranked]
        slot_map: Dict[int, List[str]] = {}
        for n in range(1, max_n + 1):
            slot_map[n] = numbers_only[:n]
        result[slot] = slot_map
    return result


def compute_topn_roi(window_days: int = 30, max_n: int = 10) -> Dict:
    df = _load_ultimate_performance()
    if df.empty:
        return {}

    latest_date = df["date"].max()
    cutoff = latest_date - timedelta(days=window_days - 1)
    window_df = df[df["date"] >= cutoff].copy()
    window_df = window_df.copy()
    # Safe handling for per_number_stake:
    # - If column exists, coerce to numeric and fill NaN with UNIT_STAKE
    # - If column missing, use constant UNIT_STAKE for all rows
    if "per_number_stake" in window_df.columns:
        window_df["per_number_stake"] = (
            pd.to_numeric(window_df["per_number_stake"], errors="coerce")
            .fillna(UNIT_STAKE)
        )
    else:
        window_df["per_number_stake"] = UNIT_STAKE
    if "andar_stake" in window_df.columns:
        window_df["andar_stake"] = pd.to_numeric(window_df["andar_stake"], errors="coerce").fillna(0)
    else:
        window_df["andar_stake"] = 0
    if "bahar_stake" in window_df.columns:
        window_df["bahar_stake"] = pd.to_numeric(window_df["bahar_stake"], errors="coerce").fillna(0)
    else:
        window_df["bahar_stake"] = 0
    if window_df.empty:
        return {}

    window_start = window_df["date"].min().date()
    window_end = window_df["date"].max().date()
    available_days = window_df["date"].dt.date.nunique()

    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Top-N ROI window: requested=%s start=%s end=%s rows=%s",
            window_days,
            window_start,
            window_end,
            len(window_df),
        )

    topn_flags = {n: f"hit_top{n}" for n in range(1, max_n + 1)}

    def _roi_for_subset(subset: pd.DataFrame, label: str) -> Dict[str, Dict[int, float]]:
        roi_map: Dict[int, float] = {}
        days_map: Dict[int, float] = {}
        hits_map: Dict[int, float] = {}
        near_hits_map: Dict[int, float] = {}

        per_number_stake = pd.to_numeric(_series_from_any(subset.get("per_number_stake")), errors="coerce").fillna(
            UNIT_STAKE
        )
        andar_stake = pd.to_numeric(_series_from_any(subset.get("andar_stake")), errors="coerce").fillna(0)
        bahar_stake = pd.to_numeric(_series_from_any(subset.get("bahar_stake")), errors="coerce").fillna(0)

        hit_type_series = subset.get("hit_type") if isinstance(subset, pd.DataFrame) else None
        near_mask = None
        if hit_type_series is not None:
            try:
                near_mask = hit_type_series.astype(str).str.upper().isin(NEAR_HIT_TYPES)
            except Exception:
                near_mask = None
        for n, flag_col in topn_flags.items():
            raw_flags = subset.get(flag_col, None)
            if raw_flags is None:
                continue

            valid_mask = raw_flags.notna() if hasattr(raw_flags, "notna") else None
            stake_components = (per_number_stake * n) + andar_stake + bahar_stake
            if valid_mask is not None:
                stake_components = stake_components.where(valid_mask, 0)
            total_stake = float(stake_components.sum())

            hits = 0
            near_hits = 0
            total_return_exact = 0.0
            near_return = 0.0
            if total_stake and hasattr(raw_flags, "fillna"):
                flags = raw_flags.fillna(False).astype(bool)
                hits = int(flags.sum())
                payout_per_row = FULL_PAYOUT_PER_UNIT * per_number_stake
                total_return_exact = float((flags.astype(int) * payout_per_row).sum())
                if near_mask is not None:
                    try:
                        candidate_mask = near_mask
                        if hasattr(raw_flags, "notna"):
                            candidate_mask = candidate_mask & raw_flags.notna()
                        candidate_mask = candidate_mask.fillna(False)
                        near_hits = int(candidate_mask.sum())
                        near_return = float((candidate_mask.astype(int) * payout_per_row * NEAR_HIT_WEIGHT).sum())
                    except Exception:
                        near_hits = 0
                        near_return = 0.0

            total_return = total_return_exact + near_return
            roi = ((total_return - total_stake) / total_stake * 100.0) if total_stake else 0.0
            roi_map[n] = safe(roi)
            days_map[n] = safe(valid_mask.sum() if valid_mask is not None else len(raw_flags))
            hits_map[n] = safe(hits)
            near_hits_map[n] = safe(near_hits)
            if logger.isEnabledFor(logging.INFO):
                logger.info(
                    "Top-N ROI band %s N=%s stake=%.2f return=%.2f hits=%s near_hits=%s",
                    label,
                    n,
                    total_stake,
                    total_return,
                    hits,
                    near_hits,
                )
        return {"roi": roi_map, "days": days_map, "hits": hits_map, "near_hits": near_hits_map}

    overall_maps = _roi_for_subset(window_df, label="overall")
    roi_by_n = overall_maps.get("roi", {})
    best_N = _pick_best_n(roi_by_n)
    best_roi = roi_by_n.get(best_N) if best_N is not None else None

    numbers_by_slot = _collect_top_numbers(window_df, max_n=max_n)

    per_slot: Dict[str, Dict[str, Dict[int, float]]] = {}
    for slot, slot_df in window_df.groupby("slot"):
        slot_maps = _roi_for_subset(slot_df, label=f"slot {slot}")
        slot_roi = slot_maps.get("roi", {})
        slot_best_n = _pick_best_n(slot_roi)
        slot_best_roi = slot_roi.get(slot_best_n) if slot_best_n is not None else None
        per_slot[slot] = {
            "roi_by_N": slot_roi,
            "days_by_N": slot_maps.get("days", {}),
            "hits_by_N": slot_maps.get("hits", {}),
            "near_hits_by_N": slot_maps.get("near_hits", {}),
        }
        if slot_best_n is not None:
            per_slot[slot]["best_N"] = slot_best_n
            per_slot[slot]["best_roi"] = slot_best_roi
        if slot in numbers_by_slot:
            per_slot[slot]["numbers_by_N"] = numbers_by_slot.get(slot, {})

    return {
        "window_days_requested": window_days,
        "window_days_used": int(available_days),
        "window_start": window_start,
        "window_end": window_end,
        "available_days": available_days,
        "overall": {
            "best_N": best_N,
            "best_roi": best_roi,
            "roi_by_N": roi_by_n,
            "days_by_N": overall_maps.get("days", {}),
            "hits_by_N": overall_maps.get("hits", {}),
            "near_hits_by_N": overall_maps.get("near_hits", {}),
        },
        "per_slot": per_slot,
        "numbers_by_slot": numbers_by_slot,
    }


def load_topn_roi_stats(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    base_path = Path(base_dir) if base_dir else quant_paths.get_base_dir()
    path = base_path / "logs" / "performance" / "topn_roi_scan.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def get_quant_stats(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    project_root = Path(base_dir) if base_dir else quant_paths.get_base_dir()

    def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    pnl_path = project_root / "logs" / "performance" / "quant_reality_pnl.json"
    slot_health_path = project_root / "data" / "slot_health.json"
    scripts_path = project_root / "logs" / "performance" / "script_hero_weak.json"
    patterns_path = project_root / "logs" / "performance" / "pattern_regime_summary.json"
    topn_path = project_root / "logs" / "performance" / "topn_roi_scan.json"

    return {
        "pnl": _safe_load(pnl_path),
        "slot_health": _safe_load(slot_health_path),
        "scripts": _safe_load(scripts_path),
        "patterns": _safe_load(patterns_path),
        "topn": _safe_load(topn_path) or {},
    }


def compute_pack_hit_stats(window_days: int = 30, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    df = load_script_hit_memory(base_dir=base_dir)
    if df.empty:
        return {}

    df = df.copy()
    if "result_date" not in df.columns:
        raise ValueError("compute_pack_hit_stats: 'result_date' column missing from script-hit memory")

    df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce")
    df = df.dropna(subset=["result_date"])
    if not pd.api.types.is_datetime64_any_dtype(df["result_date"]):
        df["result_date"] = pd.to_datetime(df["result_date"], errors="coerce")
        df = df.dropna(subset=["result_date"])

    if df.empty:
        return {}

    df["result_date"] = df["result_date"].dt.normalize()
    df["slot"] = df.get("slot", "").astype(str).str.upper()

    window_df, _ = filter_hits_by_window(df, window_days=window_days)
    if window_df.empty:
        return {}

    # ``filter_hits_by_window`` converts the date column to ``datetime.date`` to
    # perform window slicing. Convert it back to a normalised ``datetime64``
    # series so downstream ``.dt`` accessors (for unique-day counts, etc.) are
    # safe and consistent.
    window_df["result_date"] = pd.to_datetime(window_df["result_date"], errors="coerce").dt.normalize()
    window_df = window_df.dropna(subset=["result_date"])

    real_numbers = window_df.get("real_number")
    if real_numbers is not None:
        num_strings = real_numbers.astype(str).str.zfill(2)
        window_df["is_s40"] = num_strings.apply(pattern_packs.is_s40).astype(int)
        window_df["is_family_164950"] = num_strings.apply(pattern_packs.is_164950_family).astype(int)
    else:
        window_df["is_s40"] = pd.to_numeric(window_df.get("is_s40", 0), errors="coerce").fillna(0).astype(int)
        window_df["is_family_164950"] = pd.to_numeric(window_df.get("is_family_164950", 0), errors="coerce").fillna(0).astype(int)

    total_rows = len(window_df)
    days_total = window_df["result_date"].dt.date.nunique()
    s40_hits = int(window_df["is_s40"].sum())
    fam_hits = int(window_df["is_family_164950"].sum())
    s40_days = window_df.loc[window_df["is_s40"] > 0, "result_date"].dt.date.nunique()
    fam_days = window_df.loc[window_df["is_family_164950"] > 0, "result_date"].dt.date.nunique()

    summary: Dict[str, Any] = {
        "total_rows": total_rows,
        "days_total": int(days_total),
        "S40": {"hits": s40_hits, "hit_rate": s40_hits / total_rows if total_rows else 0.0},
        "FAMILY_164950": {"hits": fam_hits, "hit_rate": fam_hits / total_rows if total_rows else 0.0},
        "per_slot": {},
        "days_with_s40": int(s40_days),
        "days_with_fam": int(fam_days),
    }

    for slot, slot_df in window_df.groupby("slot"):
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
            "days_total": int(slot_df["result_date"].dt.date.nunique()),
            "s40_days": int(slot_df.loc[slot_df["is_s40"] > 0, "result_date"].dt.date.nunique()),
            "fam_days": int(slot_df.loc[slot_df["is_family_164950"] > 0, "result_date"].dt.date.nunique()),
        }

    return summary


def compute_script_slot_stats(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                *group_cols,
                "total_predictions",
                "exact_hits",
                "near_hits",
                "hit_rate_exact",
                "near_miss_rate",
                "blind_miss_rate",
                "score",
            ]
        )

    rows: List[Dict[str, Any]] = []
    grouped = df.groupby(group_cols, dropna=False)

    for keys, group in grouped:
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: Dict[str, Any] = {col: val for col, val in zip(group_cols, key_values)}

        total_predictions = len(group)
        relations = group.get("_relation") if "_relation" in group.columns else None
        if relations is None:
            relations = group.apply(lambda r: classify_relation(r.get("predicted_number"), r.get("real_number")), axis=1)

        exact_hits = int((relations == "EXACT").sum())
        near_hits = int(relations.isin({"MIRROR", "ADJACENT", "DIAGONAL_11", "REVERSE_CARRY", "SAME_DIGIT_COOL"}).sum())

        if total_predictions == 0:
            hit_rate_exact = near_miss_rate = blind_miss_rate = 0.0
            score = 0.0
            blind_misses = 0
        else:
            blind_misses = max(total_predictions - exact_hits - near_hits, 0)
            hit_rate_exact = exact_hits / total_predictions
            near_miss_rate = near_hits / total_predictions
            blind_miss_rate = max(blind_misses / total_predictions, 0)
            score = 120.0 * hit_rate_exact + 40.0 * near_miss_rate - 30.0 * blind_miss_rate

        rows.append(
            {
                **record,
                "total_predictions": int(total_predictions),
                "exact_hits": int(exact_hits),
                "near_hits": int(near_hits),
                "hit_rate_exact": hit_rate_exact,
                "near_miss_rate": near_miss_rate,
                "blind_miss_rate": blind_miss_rate,
                "score": score,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI dashboard orchestrator
# ---------------------------------------------------------------------------


def _run_cmd(args: List[str]) -> None:
    result = subprocess.run([sys.executable] + args, check=False)
    if result.returncode != 0:
        print(f"[WARN] {' '.join(args)} exited with code {result.returncode}")


def run_dashboard(window_days: int = 30) -> None:
    print("================================")
    print("QUANT STATS CORE DASHBOARD")
    print(f"Window days: {window_days}")
    print("================================\n")

    print("[1] SCRIPT-HIT METRICS")
    _run_cmd(["script_hit_metrics.py", "--window", str(window_days)])

    print("\n[2] QUANT ACCURACY REPORT")
    _run_cmd(["quant_accuracy_report.py", "--window_days", str(window_days)])

    print("\n[3] TOP-N ROI SCANNER")
    _run_cmd(["topn_roi_scanner.py"])

    print("\n[4] PATTERN INTELLIGENCE (ENHANCED)")
    _run_cmd(["pattern_intelligence_enhanced.py"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run combined quant statistics dashboard.")
    parser.add_argument("--window_days", type=int, default=30, help="Window (in days) to pass to windowed scripts.")
    args = parser.parse_args()

    run_dashboard(window_days=args.window_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

