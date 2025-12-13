"""Progress gate tracker for Precise Predictor runs."""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import quant_data_core
import quant_paths

BASE_DIR = quant_paths.get_project_root()
SCR9_DIR = BASE_DIR / "predictions" / "deepseek_scr9"
LATEST_SCR9_DIR = BASE_DIR / "latest predictions" / "deepseek_scr9"
BET_ENGINE_DIR = BASE_DIR / "predictions" / "bet_engine"
LATEST_BET_DIR = BASE_DIR / "latest predictions" / "bet_engine"
PERFORMANCE_DIR = BASE_DIR / "logs" / "performance"
PROGRESS_LOG = PERFORMANCE_DIR / "progress_status.json"
IST = ZoneInfo("Asia/Kolkata")

GATES = [
    "results_loaded",
    "hit_memory_built",
    "pattern_intel_done",
    "topn_policy_saved",
    "scr9_predictions_saved",
    "bet_plan_saved",
    "pnl_tracker_saved",
    "daily_brief_printed",
    "snapshot_saved",
]


def to_ist_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=IST)
    return dt.astimezone(IST)


def _parse_anchor(run_anchor: Optional[str]) -> Optional[datetime]:
    if not run_anchor:
        return None
    try:
        parsed = datetime.fromisoformat(run_anchor.strip())
        return to_ist_aware(parsed)
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

    latest_time = to_ist_aware(
        datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    )
    if anchor_dt and latest_time < anchor_dt:
        return False, f"Latest SCR9 predictions stale ({latest.name})"
    return True, latest_time.isoformat()


def _gate_latest_predictions(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    latest = _latest_file(
        [
            (LATEST_SCR9_DIR, "ultimate_predictions_*.xlsx"),
            (SCR9_DIR, "ultimate_predictions_*.xlsx"),
        ]
    )
    if not latest:
        return False, "No latest predictions available"

    latest_time = to_ist_aware(
        datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    )
    if anchor_dt and latest_time < anchor_dt:
        return False, f"Selected predictions predate run anchor ({latest.name})"
    return True, latest_time.isoformat()


def _gate_bet_plan(target_date: Optional[date]) -> Tuple[bool, str]:
    patterns: List[Tuple[Path, str]] = [
        (LATEST_BET_DIR, "bet_plan_master_*.xlsx"),
        (BET_ENGINE_DIR, "bet_plan_master_*.xlsx"),
    ]
    latest = _latest_file(patterns)
    if not latest:
        return False, "No bet_plan_master file found"

    bet_time = to_ist_aware(
        datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    )

    if target_date:
        date_str = target_date.strftime("%Y%m%d")
        if date_str not in latest.name:
            return False, f"Bet plan date mismatch ({latest.name} for {date_str})"
    return True, bet_time.isoformat()


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

    mtime = to_ist_aware(datetime.fromtimestamp(pnl_file.stat().st_mtime, tz=timezone.utc))
    if anchor_dt and mtime < anchor_dt:
        return False, "quant_reality_pnl.json not refreshed for this run"

    if latest_result_date:
        if mtime.date() >= latest_result_date:
            return True, f"Updated through {latest_result_date}"
        return False, f"quant_reality_pnl.json stale vs results ({latest_result_date})"

    return True, mtime.isoformat()


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
    ts_value = data.get("timestamp") or data.get("updated_at")
    ts = None
    try:
        if ts_value:
            ts = to_ist_aware(datetime.fromisoformat(str(ts_value)))
    except Exception:
        ts = None
    if main_rows > 0:
        return True, (ts.isoformat() if ts else f"MAIN rows: {main_rows}")
    return False, "TopN ROI scanner found no MAIN rows"


def _gate_hit_memory(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    hit_path = PERFORMANCE_DIR / "script_hit_memory.xlsx"
    if not hit_path.exists():
        return False, "script_hit_memory.xlsx missing"
    mtime = to_ist_aware(datetime.fromtimestamp(hit_path.stat().st_mtime, tz=timezone.utc))
    if anchor_dt and mtime < anchor_dt:
        return False, "script_hit_memory.xlsx stale"
    return True, mtime.isoformat()


def _gate_pattern_intel(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    intel_path = BASE_DIR / "data" / "pattern_intelligence.json"
    if not intel_path.exists():
        return False, "pattern_intelligence.json missing"
    mtime = to_ist_aware(datetime.fromtimestamp(intel_path.stat().st_mtime, tz=timezone.utc))
    if anchor_dt and mtime < anchor_dt:
        return False, "pattern_intelligence.json stale"
    return True, mtime.isoformat()


def _gate_topn_policy(anchor_dt: Optional[datetime]) -> Tuple[bool, str]:
    policy_path = BASE_DIR / "data" / "topn_policy.json"
    if not policy_path.exists():
        return False, "topn_policy.json missing"
    mtime = to_ist_aware(datetime.fromtimestamp(policy_path.stat().st_mtime, tz=timezone.utc))
    if anchor_dt and mtime < anchor_dt:
        return False, "topn_policy.json stale"
    return True, mtime.isoformat()


def compute_progress_status(
    target_date: Optional[date] = None,
    run_anchor: Optional[str] = None,
    gate_overrides: Optional[Dict[str, Tuple[bool, Optional[datetime]]]] = None,
) -> Dict[str, object]:
    anchor_dt = _parse_anchor(run_anchor)
    now_ts = to_ist_aware(datetime.now())

    gates: Dict[str, Dict[str, object]] = {}
    resolved_overrides = gate_overrides or {}

    def _assign(name: str, result: Tuple[bool, str]):
        done, raw_ts = result
        ts_val: Optional[str] = None
        note = raw_ts
        try:
            if isinstance(raw_ts, str) and "T" in raw_ts:
                ts_val = raw_ts
        except Exception:
            ts_val = None
        gates[name] = {"done": done, "timestamp": ts_val, "note": note}

    try:
        results_df = quant_data_core.load_results_dataframe()
        results_ok = not results_df.empty
    except Exception:
        results_ok = False
    _assign("results_loaded", (results_ok, "results parsed" if results_ok else "No results"))
    _assign("hit_memory_built", _gate_hit_memory(anchor_dt))
    _assign("pattern_intel_done", _gate_pattern_intel(anchor_dt))
    policy_ok, policy_note = _gate_topn_policy(anchor_dt)
    roi_ok, roi_note = _gate_topn_roi()
    combined_note = roi_note if policy_ok else policy_note
    _assign("topn_policy_saved", (policy_ok and roi_ok, combined_note))
    _assign("scr9_predictions_saved", _gate_scr9_predictions(anchor_dt))
    _assign("bet_plan_saved", _gate_bet_plan(target_date))
    _assign("pnl_tracker_saved", _gate_quant_reality(anchor_dt))

    for name in ["daily_brief_printed", "snapshot_saved"]:
        if name in resolved_overrides:
            done, ts = resolved_overrides[name]
            gates[name] = {
                "done": bool(done),
                "timestamp": to_ist_aware(ts).isoformat() if isinstance(ts, datetime) else None,
                "note": "override",
            }
        else:
            gates.setdefault(name, {"done": False, "timestamp": None, "note": "pending"})

    for name in GATES:
        gates.setdefault(name, {"done": False, "timestamp": None, "note": "pending"})

    passed = sum(1 for name in GATES if gates.get(name, {}).get("done"))
    total = len(GATES)
    percent = int(round((passed / total) * 100)) if total else 0

    status = {
        "timestamp": now_ts.isoformat(),
        "run_anchor": run_anchor,
        "run_anchor_ist": anchor_dt.isoformat() if anchor_dt else None,
        "run_target_date": target_date.isoformat() if target_date else None,
        "gates": gates,
        "progress_percent": percent,
        "gates_passed": passed,
        "gates_total": total,
    }
    return status


def _append_status(status: Dict[str, object]) -> None:
    PROGRESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_LOG.open("w") as f:
        json.dump(status, f, indent=2)


def capture_progress_status(
    target_date: Optional[date] = None,
    run_anchor: Optional[str] = None,
    gate_overrides: Optional[Dict[str, Tuple[bool, Optional[datetime]]]] = None,
) -> Dict[str, object]:
    status: Dict[str, object]
    try:
        status = compute_progress_status(
            target_date=target_date, run_anchor=run_anchor, gate_overrides=gate_overrides
        )
    except Exception as exc:  # pragma: no cover - best effort logging
        status = {
            "timestamp": to_ist_aware(datetime.now()).isoformat(),
            "run_anchor": run_anchor,
            "run_target_date": target_date.isoformat() if target_date else None,
            "error": str(exc),
        }
    try:
        _append_status(status)
    except Exception:
        status.setdefault("error", "Unable to persist progress_status.json")

    gates_passed = status.get("gates_passed") or 0
    gates_total = status.get("gates_total") or len(GATES)
    percent = status.get("progress_percent") or 0
    print(f"[PROGRESS] gates={gates_passed}/{gates_total} => {percent}% (stable)")
    return status
