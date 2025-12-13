"""Progress gate tracker for Precise Predictor runs."""
from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import quant_data_core
import quant_paths

GATE_TOTAL = 5

BASE_DIR = quant_paths.get_project_root()
SCR9_DIR = BASE_DIR / "predictions" / "deepseek_scr9"
LATEST_SCR9_DIR = BASE_DIR / "latest predictions" / "deepseek_scr9"
BET_ENGINE_DIR = BASE_DIR / "predictions" / "bet_engine"
LATEST_BET_DIR = BASE_DIR / "latest predictions" / "bet_engine"
PERFORMANCE_DIR = BASE_DIR / "logs" / "performance"
PROGRESS_LOG = PERFORMANCE_DIR / "progress_status.json"


def _parse_anchor(run_anchor: Optional[str]) -> Optional[datetime]:
    if not run_anchor:
        return None
    try:
        parsed = datetime.fromisoformat(run_anchor.strip())
        return parsed
    except Exception:
        return None


def get_run_anchor(default: Optional[str] = None) -> Optional[str]:
    """Return SCR9 run anchor string from env or provided default."""

    return os.environ.get("SCR9_RUN_STARTED_AT") or default


def _latest_file(patterns: List[Tuple[Path, str]]) -> Optional[Path]:
    candidates: List[Path] = []
    for base_dir, pattern in patterns:
        if not base_dir.exists():
            continue
        candidates.extend(base_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _gate_scr9_predictions(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    latest = _latest_file(
        [
            (LATEST_SCR9_DIR, "ultimate_predictions_*.xlsx"),
            (SCR9_DIR, "ultimate_predictions_*.xlsx"),
        ]
    )
    if not latest:
        return False, "No SCR9 ultimate_predictions file found"

    if anchor_dt and datetime.fromtimestamp(latest.stat().st_mtime) < anchor_dt:
        return False, f"Latest SCR9 predictions stale ({latest.name})"
    return True, latest.name


def _gate_latest_predictions(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    latest = _latest_file(
        [
            (LATEST_SCR9_DIR, "ultimate_predictions_*.xlsx"),
            (SCR9_DIR, "ultimate_predictions_*.xlsx"),
        ]
    )
    if not latest:
        return False, "No latest predictions available"

    if anchor_dt and datetime.fromtimestamp(latest.stat().st_mtime) < anchor_dt:
        return False, f"Selected predictions predate run anchor ({latest.name})"
    return True, latest.name


def _gate_bet_plan(target_date: Optional[date]) -> Tuple[bool, str]:
    patterns: List[Tuple[Path, str]] = [
        (LATEST_BET_DIR, "bet_plan_master_*.xlsx"),
        (BET_ENGINE_DIR, "bet_plan_master_*.xlsx"),
    ]
    latest = _latest_file(patterns)
    if not latest:
        return False, "No bet_plan_master file found"

    if target_date:
        date_str = target_date.strftime("%Y%m%d")
        if date_str not in latest.name:
            return False, f"Bet plan date mismatch ({latest.name} for {date_str})"
    return True, latest.name


def _load_latest_result_date() -> Optional[date]:
    try:
        results_df = quant_data_core.load_results_dataframe()
        if results_df.empty:
            return None
        latest_result = quant_data_core.get_latest_result_date(results_df)
        return latest_result
    except Exception:
        return None


def _gate_quant_reality(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    pnl_file = PERFORMANCE_DIR / "quant_reality_pnl.json"
    latest_result_date = _load_latest_result_date()

    if not pnl_file.exists():
        return False, "quant_reality_pnl.json missing"

    mtime = datetime.fromtimestamp(pnl_file.stat().st_mtime)
    if anchor_dt and mtime < anchor_dt:
        return False, "quant_reality_pnl.json not refreshed for this run"

    if latest_result_date:
        if mtime.date() >= latest_result_date:
            return True, f"Updated through {latest_result_date}"
        return False, f"quant_reality_pnl.json stale vs results ({latest_result_date})"

    return True, "No results dataset; PNL unchanged"


def _gate_topn_roi() -> Tuple[bool, str]:
    profile_path = PERFORMANCE_DIR / "topn_roi_profile.json"
    if not profile_path.exists():
        return False, "topn_roi_profile.json missing"

    try:
        with profile_path.open("r") as f:
            data = json.load(f)
    except Exception as exc:
        return False, f"Cannot read topn_roi_profile.json: {exc}"

    main_rows = int(data.get("main_rows_detected") or data.get("main_rows", 0) or 0)
    if main_rows > 0:
        return True, f"MAIN rows: {main_rows}"
    return False, "TopN ROI scanner found no MAIN rows"


def compute_progress_status(
    target_date: Optional[date] = None, run_anchor: Optional[str] = None
) -> Dict[str, object]:
    anchor_dt = _parse_anchor(run_anchor)

    gates: List[Tuple[bool, str]] = []
    gates.append(_gate_scr9_predictions(anchor_dt))
    gates.append(_gate_latest_predictions(anchor_dt))
    gates.append(_gate_bet_plan(target_date))
    gates.append(_gate_quant_reality(anchor_dt))
    gates.append(_gate_topn_roi())

    passed = sum(1 for ok, _ in gates if ok)
    notes = [msg for _, msg in gates]
    percent = int(round((passed / GATE_TOTAL) * 100)) if GATE_TOTAL else 0

    status = {
        "timestamp": datetime.now().isoformat(),
        "run_anchor": run_anchor,
        "run_target_date": target_date.isoformat() if target_date else None,
        "gates_passed": passed,
        "gates_total": GATE_TOTAL,
        "progress_percent": percent,
        "notes": notes,
    }
    return status


def _append_status(status: Dict[str, object]) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if PROGRESS_LOG.exists():
        try:
            with PROGRESS_LOG.open("r") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    existing = loaded
        except Exception:
            existing = []

    existing.append(status)
    with PROGRESS_LOG.open("w") as f:
        json.dump(existing, f, indent=2)


def capture_progress_status(
    target_date: Optional[date] = None, run_anchor: Optional[str] = None
) -> Dict[str, object]:
    status = compute_progress_status(target_date=target_date, run_anchor=run_anchor)
    _append_status(status)

    print(
        f"[PROGRESS] gates={status['gates_passed']}/{status['gates_total']} => "
        f"{status['progress_percent']}% (stable)"
    )
    return status
