"""Quant Daily Brief orchestrator.

This script decides between intraday vs next-day workflows, triggers the
appropriate engines quietly, and prints a concise human-readable daily brief.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import quant_data_core
import quant_paths
import quant_learning_core
from quant_core import pattern_core
from quant_core.config_core import PACK_164950_FAMILY, S40 as S40_SET
from quant_stats_core import get_quant_stats
from script_hit_metrics import (
    compute_pack_hit_stats,
    get_metrics_table,
    build_script_league,
    format_script_league,
    hero_weak_table,
)
from script_hit_memory_utils import load_script_hit_memory
from quant_stats_core import compute_topn_roi
from utils_2digit import is_valid_2d_number, to_2d_str

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
quant_stats = get_quant_stats()
PERFORMANCE_DIR = quant_paths.get_performance_logs_dir()
BET_ENGINE_DIR = quant_paths.get_bet_engine_dir()
SCRIPT_METRICS_WINDOW_DAYS = 30
SLOT_HEALTH_PATH = Path("data") / "slot_health.json"
TOPN_POLICY_PATH = Path("data") / "topn_policy.json"
TOPN_WINDOW_DAYS = 30


@dataclass
class PlanSlot:
    slot: str
    main_numbers: List[str]
    andar: Optional[str]
    andar_stake: float
    bahar: Optional[str]
    bahar_stake: float
    slot_stake: float


@dataclass
class PlanSummary:
    slots: List[PlanSlot]
    total_stake: float
    open_slots: List[str]


@dataclass
class PnLSnapshot:
    overall_pnl: Optional[float]
    overall_roi: Optional[float]
    last7_pnl: Optional[float]
    last7_roi: Optional[float]
    last30_pnl: Optional[float]
    last30_roi: Optional[float]
    best_slot: Optional[Tuple[str, float]]
    worst_slot: Optional[Tuple[str, float]]
    golden_roi: Optional[float] = None
    golden_pnl: Optional[float] = None


@dataclass
class ExecutionReadiness:
    mode: Optional[str]
    multiplier: Optional[float]
    base_stake: Optional[float]
    recommended_stake: Optional[float]
    environment_score: Optional[float]


@dataclass
class PatternSummary:
    total_hits: Optional[int]
    s40: Optional[Dict[str, float]]
    fam_164950: Optional[Dict[str, float]]
    extra_family: Optional[Dict[str, object]]
    notes: List[str]
    per_slot: Dict[str, Dict[str, Dict[str, float]]]


def _choose_date_column(df: pd.DataFrame) -> Optional[str]:
    for col in ("result_date", "date"):
        if col in df.columns and not df[col].isna().all():
            return col
    return None


@dataclass
class StrategySummary:
    recommended: Optional[str]
    top_family: Optional[str]
    confidence: Optional[str]
    risk_mode: Optional[str]


@dataclass
class MoneyManagerSummary:
    risk_mode: Optional[str]
    daily_cap: Optional[float]
    single_cap: Optional[float]


@dataclass
class ConfidenceSummary:
    scores: Dict[str, float]
    labels: Dict[str, str]
    reasons: Dict[str, str] = field(default_factory=dict)


def refresh_script_hit_memory_and_metrics(
    window_days: int = SCRIPT_METRICS_WINDOW_DAYS,
) -> None:
    """
    Run prediction_hit_memory.py --mode update-latest and
    script_hit_metrics.py --window <window_days> before printing the brief.
    Failures should be logged as warnings but must not abort the brief.
    """

    try:
        subprocess.run(
            [sys.executable, "prediction_hit_memory.py", "--mode", "update-latest"],
            check=False,
        )
    except Exception as exc:
        print(f"⚠️ Could not refresh script hit memory: {exc}")

    try:
        subprocess.run(
            [sys.executable, "script_hit_metrics.py", "--window", str(window_days)],
            check=False,
        )
    except Exception as exc:
        print(f"⚠️ Could not recompute script hit metrics: {exc}")


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def load_results_df() -> pd.DataFrame:
    df = quant_data_core.load_results_dataframe()
    if not df.empty and "DATE" in df.columns:
        with np.errstate(over="ignore", invalid="ignore"):
            safe_date = pd.to_datetime(df["DATE"], errors="coerce")
        df["DATE_ONLY"] = safe_date.dt.date
    return df


def find_latest_bet_date() -> date:
    latest = quant_paths.find_latest_bet_plan_master()
    if latest:
        parsed = quant_paths.parse_date_from_filename(latest.name)
        if parsed:
            return parsed
    return datetime.now().date()


def decide_mode(bet_date: date, explicit_mode: str, results_df: pd.DataFrame) -> Tuple[str, date]:
    explicit_mode = explicit_mode.lower()
    if explicit_mode == "intraday":
        return "INTRADAY", bet_date
    if explicit_mode == "nextday":
        return "NEXT_DAY", bet_date + timedelta(days=1)

    # auto
    if results_df.empty or "DATE_ONLY" not in results_df.columns:
        return "INTRADAY", bet_date

    day_rows = results_df[results_df["DATE_ONLY"] == bet_date]
    if day_rows.empty:
        return "INTRADAY", bet_date

    row = day_rows.iloc[-1]
    closed = True
    for slot in SLOTS:
        value = row.get(slot)
        if pd.isna(value):
            closed = False
            break
    if closed:
        return "NEXT_DAY", bet_date + timedelta(days=1)
    return "INTRADAY", bet_date


def run_script(script_name: str, args: Optional[List[str]] = None, dry_run: bool = False) -> Optional[subprocess.CompletedProcess]:
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    if dry_run:
        print(f"DRY-RUN: would run {' '.join(cmd)}")
        return None
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    if result.returncode != 0:
        tail = (result.stderr or "").splitlines()[-3:]
        tail_text = " | ".join(tail)
        print(f"❌ {script_name} failed (code {result.returncode}): {tail_text}")
    return result


def run_intraday_pipeline(bet_date: date, dry_run: bool = False) -> None:
    run_script("slot_recalc_engine.py", ["--date", bet_date.isoformat()], dry_run=dry_run)
    run_script("pattern_intelligence_engine.py", dry_run=dry_run)
    run_script("pattern_intelligence_enhanced.py", dry_run=dry_run)
    run_script("pattern_packs_exporter.py", dry_run=dry_run)
    run_script("pattern_packs_automerge.py", dry_run=dry_run)
    run_script("bet_pnl_tracker.py", dry_run=dry_run)
    run_script("quant_pnl_summary.py", dry_run=dry_run)


def run_nextday_pipeline(target_date: date, dry_run: bool = False) -> None:
    run_script("deepseek_scr9.py", ["--speed-mode", "fast"], dry_run=dry_run)
    run_script("prediction_hit_memory.py", dry_run=dry_run)
    run_script("pattern_intelligence_engine.py", dry_run=dry_run)
    run_script("pattern_intelligence_enhanced.py", dry_run=dry_run)
    run_script("pattern_packs_exporter.py", dry_run=dry_run)
    run_script("pattern_packs_automerge.py", dry_run=dry_run)
    run_script("strategy_recommendation_engine.py", dry_run=dry_run)
    run_script("bet_pnl_tracker.py", dry_run=dry_run)
    run_script("quant_pnl_summary.py", dry_run=dry_run)
    run_script("reality_check_engine.py", dry_run=dry_run)
    run_script("money_manager.py", dry_run=dry_run)
    run_script("smart_fusion_weights.py", dry_run=dry_run)
    run_script("execution_readiness_engine.py", dry_run=dry_run)
    run_script("precise_bet_engine.py", dry_run=dry_run)
    run_script("bet_plan_enhancer.py", dry_run=dry_run)
    run_script("final_bet_plan_engine.py", dry_run=dry_run)
    run_script("live_bet_sheet_engine.py", dry_run=dry_run)


def _load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _load_slot_health_snapshot() -> Dict[str, Dict[str, object]]:
    try:
        base_dir = quant_paths.get_base_dir()
    except Exception:
        return {}
    path = Path(base_dir) / SLOT_HEALTH_PATH
    data = _load_json(path) or {}
    snapshot: Dict[str, Dict[str, object]] = {}
    for slot in SLOTS:
        record = data.get(slot, {}) if isinstance(data, dict) else {}
        snapshot[slot] = {
            "roi": float(record.get("roi_30", record.get("roi_percent", 0.0)) or 0.0),
            "slump": bool(record.get("slump", False)),
            "slot_level": str(record.get("slot_level", "MID")).upper(),
        }
    roi_values = {k: v.get("roi", 0.0) for k, v in snapshot.items()}
    for slot, entry in snapshot.items():
        roi_val = entry.get("roi", 0.0)
        max_other_roi = max([v for k, v in roi_values.items() if k != slot], default=0.0)
        if roi_val < -30.0 and max_other_roi > 50.0:
            entry["slot_level"] = "OFF"
        elif roi_val > 300.0:
            entry["slot_level"] = "HIGH"
    return snapshot


def load_plan_from_excel(plan_path: Path) -> Optional[PlanSummary]:
    if not plan_path.exists():
        return None
    try:
        xls = pd.ExcelFile(plan_path)
    except Exception:
        return None

    sheet_name = "bets" if "bets" in xls.sheet_names else xls.sheet_names[0]
    df = pd.read_excel(xls, sheet_name=sheet_name)
    if df.empty:
        return None
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    required_cols = {"slot", "layer_type", "number_or_digit", "tier", "stake"}
    if not required_cols.issubset(df.columns):
        return None

    slots: List[PlanSlot] = []
    open_slots: List[str] = []
    for slot in SLOTS:
        slot_df = df[df["slot"].str.upper() == slot]
        if slot_df.empty:
            continue
        open_slots.append(slot)
        main = slot_df[slot_df["layer_type"].str.upper() == "MAIN"]
        andar_df = slot_df[slot_df["layer_type"].str.upper() == "ANDAR"]
        bahar_df = slot_df[slot_df["layer_type"].str.upper() == "BAHAR"]

        main_numbers = []
        slot_total = 0.0
        for _, row in main.iterrows():
            number = row.get("number_or_digit")
            tier = row.get("tier")
            stake = float(row.get("stake", 0) or 0)
            slot_total += stake
            main_numbers.append(f"{int(number):02d}({tier} ₹{stake:.0f})")

        andar_digit = None
        andar_stake = 0.0
        if not andar_df.empty:
            andar_digit = str(andar_df.iloc[0].get("number_or_digit"))
            andar_stake = float(andar_df.iloc[0].get("stake", 0) or 0)
            slot_total += andar_stake

        bahar_digit = None
        bahar_stake = 0.0
        if not bahar_df.empty:
            bahar_digit = str(bahar_df.iloc[0].get("number_or_digit"))
            bahar_stake = float(bahar_df.iloc[0].get("stake", 0) or 0)
            slot_total += bahar_stake

        slots.append(
            PlanSlot(
                slot=slot,
                main_numbers=main_numbers,
                andar=andar_digit,
                andar_stake=andar_stake,
                bahar=bahar_digit,
                bahar_stake=bahar_stake,
                slot_stake=slot_total,
            )
        )

    total_stake = sum(slot.slot_stake for slot in slots)
    return PlanSummary(slots=slots, total_stake=total_stake, open_slots=open_slots)


def load_plan_for_mode(mode: str, bet_date: date, target_date: date, dry_run: bool = False) -> Optional[PlanSummary]:
    if mode == "INTRADAY":
        path = BET_ENGINE_DIR / f"bet_plan_intraday_{bet_date.strftime('%Y%m%d')}.xlsx"
        return load_plan_from_excel(path)
    path = quant_paths.get_final_bet_plan_path(target_date.strftime("%Y-%m-%d"))
    plan = load_plan_from_excel(path)
    if plan:
        return plan

    master_path = BET_ENGINE_DIR / f"bet_plan_master_{target_date.strftime('%Y%m%d')}.xlsx"
    plan = load_plan_from_excel(master_path)
    if plan or dry_run:
        return plan

    # Fallback: attempt to regenerate tomorrow plan using latest SCR9 predictions.
    run_script("deepseek_scr9.py", ["--speed-mode", "fast"], dry_run=dry_run)
    run_script("precise_bet_engine.py", ["--target", "tomorrow"], dry_run=dry_run)
    return load_plan_from_excel(master_path)


def _find_column(columns: Iterable[str], keywords: List[str], require_all: bool = False) -> Optional[str]:
    for col in columns:
        col_lower = str(col).lower()
        if require_all:
            if all(keyword in col_lower for keyword in keywords):
                return col
        else:
            if any(keyword in col_lower for keyword in keywords):
                return col
    return None


def _format_number(value: object) -> str:
    """
    Small helper: for clean display only.
    Behaviour must remain equivalent to the old version:
    - 0–99 → zero-padded 2-digit
    - other values → simple str(value)
    """
    try:
        # First try the 0–99 2-digit world using central helper
        if is_valid_2d_number(value):
            return to_2d_str(value)

        # Fallback to old behaviour style for anything else
        num = float(value)
        if num.is_integer():
            return str(int(num))
        return str(value)
    except Exception:
        return str(value)


def load_final_bet_plan_for_date(target_date: date) -> Optional[dict]:
    """
    Load final bet plan for the given date from predictions\bet_engine.
    Returns a structured dict with slot-wise info, or None if not found.
    """

    bet_dir = Path(BET_ENGINE_DIR)
    if not bet_dir.exists():
        return None

    exact_file = bet_dir / f"final_bet_plan_{target_date.strftime('%Y%m%d')}.xlsx"
    candidates: List[Path] = []
    if exact_file.exists():
        candidates.append(exact_file)
    else:
        candidates.extend(sorted(bet_dir.glob("final_bet_plan_*.xlsx"), reverse=True))

    if not candidates:
        return None

    for path in candidates:
        try:
            df = pd.read_excel(path)
        except Exception as exc:
            print(f"[WARN] Could not load final bet plan: {exc}")
            continue

        if df.empty:
            continue

        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

        date_col = _find_column(df.columns, ["date"])
        enforce_date_match = path != exact_file
        if date_col:
            with np.errstate(over="ignore", invalid="ignore"):
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df[date_col] = df[date_col].dt.date
            df = df[df[date_col] == target_date]
        elif enforce_date_match:
            continue

        if df.empty:
            continue

        slot_col = _find_column(df.columns, ["slot"])
        if not slot_col:
            continue

        number_col = _find_column(df.columns, ["number", "num"])
        tier_col = _find_column(df.columns, ["tier", "rank_tier", "tier_label"])
        stake_col = _find_column(df.columns, ["stake", "bet_stake", "unit_stake"])
        andar_digit_col = _find_column(df.columns, ["andar", "digit"], require_all=True) or _find_column(
            df.columns, ["andar", "num"], require_all=True
        )
        bahar_digit_col = _find_column(df.columns, ["bahar", "digit"], require_all=True) or _find_column(
            df.columns, ["bahar", "num"], require_all=True
        )
        andar_stake_col = _find_column(df.columns, ["andar", "stake"], require_all=True)
        bahar_stake_col = _find_column(df.columns, ["bahar", "stake"], require_all=True)
        slot_total_col = _find_column(df.columns, ["slot", "stake"], require_all=True) or _find_column(
            df.columns, ["total_stake"]
        )

        slots: Dict[str, dict] = {}

        for slot in SLOTS:
            slot_rows = df[df[slot_col].astype(str).str.upper() == slot]
            if slot_rows.empty:
                continue

            numbers = []
            computed_total = 0.0

            if number_col:
                for _, row in slot_rows.iterrows():
                    num_val = row.get(number_col)
                    if pd.isna(num_val):
                        continue
                    tier_val = row.get(tier_col) if tier_col else None
                    tier_label = "?"
                    if tier_val is not None and not pd.isna(tier_val):
                        tier_label = str(tier_val)
                    stake_val = float(row.get(stake_col, 0) or 0) if stake_col else 0.0
                    computed_total += stake_val
                    numbers.append({"num": _format_number(num_val), "tier": tier_label, "stake": stake_val})

            andar_digit = None
            bahar_digit = None
            andar_stake = None
            bahar_stake = None

            if andar_digit_col and not slot_rows[andar_digit_col].dropna().empty:
                andar_digit = _format_number(slot_rows[andar_digit_col].dropna().iloc[0])
            if bahar_digit_col and not slot_rows[bahar_digit_col].dropna().empty:
                bahar_digit = _format_number(slot_rows[bahar_digit_col].dropna().iloc[0])
            if andar_stake_col and not slot_rows[andar_stake_col].dropna().empty:
                andar_stake = float(slot_rows[andar_stake_col].dropna().iloc[0] or 0)
                computed_total += andar_stake
            if bahar_stake_col and not slot_rows[bahar_stake_col].dropna().empty:
                bahar_stake = float(slot_rows[bahar_stake_col].dropna().iloc[0] or 0)
                computed_total += bahar_stake

            slot_total = computed_total
            if slot_total_col and not slot_rows[slot_total_col].dropna().empty:
                slot_total = float(slot_rows[slot_total_col].dropna().iloc[0] or 0)

            slots[slot] = {
                "numbers": numbers,
                "andar_digit": andar_digit,
                "andar_stake": andar_stake,
                "bahar_digit": bahar_digit,
                "bahar_stake": bahar_stake,
                "slot_stake": slot_total,
            }

        if not slots:
            continue

        total_stake = sum(slot_data.get("slot_stake", 0) or 0 for slot_data in slots.values())
        return {"slots": slots, "total_stake": total_stake}

    return None


def load_execution_readiness() -> ExecutionReadiness:
    data = _load_json(PERFORMANCE_DIR / "execution_readiness_summary.json") or {}
    return ExecutionReadiness(
        mode=data.get("mode"),
        multiplier=data.get("stake_multiplier"),
        base_stake=data.get("base_final_total_stake") or data.get("final_plan_stake"),
        recommended_stake=data.get("recommended_real_total_stake"),
        environment_score=data.get("environment_score"),
    )


def _roi(pnl: float, stake: float) -> Optional[float]:
    if stake:
        return pnl / stake * 100
    return None


def _load_golden_benchmark() -> Tuple[Optional[float], Optional[float]]:
    data = _load_json(PERFORMANCE_DIR / "golden_days_snapshot.json") or {}
    return data.get("total_pnl"), data.get("roi_pct") or data.get("roi")


def load_pnl_snapshot() -> PnLSnapshot:
    data = _load_json(PERFORMANCE_DIR / "quant_reality_pnl.json") or {}
    overall = data.get("overall", {})
    daily = data.get("daily", [])
    by_slot = data.get("by_slot", [])

    def _window(days: int) -> Tuple[Optional[float], Optional[float]]:
        if not daily:
            return None, None
        records = []
        for item in daily:
            try:
                d = parse_date(item["date"])
            except Exception:
                continue
            records.append((d, item))
        if not records:
            return None, None
        latest = max(d for d, _ in records)
        cutoff = latest - timedelta(days=days - 1)
        window_items = [it for d, it in records if d >= cutoff]
        stake = sum(it.get("total_stake", 0) or 0 for it in window_items)
        pnl = sum(it.get("pnl", 0) or 0 for it in window_items)
        return pnl, _roi(pnl, stake)

    best_slot = None
    worst_slot = None
    if by_slot:
        sorted_slots = sorted(by_slot, key=lambda x: x.get("pnl", 0), reverse=True)
        best_slot = (sorted_slots[0].get("slot"), sorted_slots[0].get("pnl"))
        worst_slot = (sorted_slots[-1].get("slot"), sorted_slots[-1].get("pnl"))

    last7_pnl, last7_roi = _window(7)
    last30_pnl, last30_roi = _window(30)
    golden_pnl, golden_roi = _load_golden_benchmark()

    return PnLSnapshot(
        overall_pnl=overall.get("total_pnl"),
        overall_roi=overall.get("overall_roi"),
        last7_pnl=last7_pnl,
        last7_roi=last7_roi,
        last30_pnl=last30_pnl,
        last30_roi=last30_roi,
        best_slot=best_slot,
        worst_slot=worst_slot,
        golden_roi=golden_roi,
        golden_pnl=golden_pnl,
    )


def _empty_pattern_summary(notes: Optional[List[str]] = None) -> PatternSummary:
    notes = notes or []
    return PatternSummary(
        total_hits=0,
        s40={
            "hits": 0,
            "hit_rate": 0.0,
            "daily_rate": 0.0,
            "daily_days": 0,
            "total_days": 0,
        },
        fam_164950={
            "hits": 0,
            "hit_rate": 0.0,
            "daily_rate": 0.0,
            "daily_days": 0,
            "total_days": 0,
        },
        extra_family=None,
        notes=notes,
        per_slot={},
    )


def _extract_extra_family(summary_json: Dict) -> Optional[Dict[str, object]]:
    families = summary_json.get("families", {}) if isinstance(summary_json, dict) else {}
    best: Optional[Dict[str, object]] = None
    best_score = (-float("inf"), -float("inf"), -float("inf"))

    def _slot_score(info: Dict) -> Tuple[float, int, int]:
        rate = info.get("hit_rate") or info.get("hit_rate_exact") or info.get("hit_rate_overall")
        hits = info.get("hits") or info.get("exact_hits") or 0
        cover = (
            info.get("days_covered")
            or info.get("covered_days")
            or info.get("days")
            or info.get("active_days")
            or 0
        )
        rate_val = float(rate) if rate is not None else 0.0
        if rate is not None and rate_val <= 1:
            rate_val *= 100.0
        return rate_val, int(hits or 0), int(cover or 0)

    for fam_name, fam_block in families.items():
        key = str(fam_name).upper()
        if key in ("S40", "FAMILY_164950", "PACK_164950"):
            continue
        if not isinstance(fam_block, dict):
            continue

        hit_rate_val = (
            fam_block.get("hit_rate")
            or fam_block.get("hit_rate_exact")
            or fam_block.get("hit_rate_overall")
        )
        hits = fam_block.get("hits") or fam_block.get("exact_hits") or 0
        cover_block = fam_block.get("daily_cover") or fam_block.get("cover") or {}
        covered = (
            cover_block.get("covered_days")
            or cover_block.get("days")
            or fam_block.get("days_covered")
            or fam_block.get("active_days")
            or fam_block.get("days")
            or 0
        )
        total_days = cover_block.get("total_days") or cover_block.get("days_total") or fam_block.get("total_days") or covered
        hit_rate_pct = float(hit_rate_val) if hit_rate_val is not None else 0.0
        if hit_rate_val is not None and hit_rate_pct <= 1:
            hit_rate_pct *= 100.0

        per_slot = fam_block.get("per_slot", {}) if isinstance(fam_block.get("per_slot"), dict) else {}
        best_slot = None
        weak_slot = None
        if per_slot:
            slot_items = [(s, info) for s, info in per_slot.items() if isinstance(info, dict)]
            sorted_slots = sorted(slot_items, key=lambda kv: _slot_score(kv[1]), reverse=True)
            if sorted_slots:
                best_slot = sorted_slots[0][0]
                weak_slot = sorted(slot_items, key=lambda kv: _slot_score(kv[1]))[0][0]

        score = (hit_rate_pct, int(hits or 0), int(covered or 0))
        if score > best_score:
            best_score = score
            best = {
                "name": fam_name,
                "hit_rate": hit_rate_pct,
                "hits": hits,
                "covered_days": int(covered or 0),
                "total_days": int(total_days or 0),
                "best_slot": best_slot,
                "weak_slot": weak_slot,
            }

    return best


def load_pattern_summary_from_intel(window_days: int = 90) -> PatternSummary:
    """Build pattern summary directly from saved PatternIntel JSON outputs."""

    summary_json = quant_learning_core.load_pattern_summary_json(window_days=window_days)
    if not summary_json:
        return _empty_pattern_summary(["Pattern summary unavailable (missing file)"])

    patterns = summary_json.get("patterns", {})
    daily_cover = summary_json.get("daily_cover", {})
    notes: List[str] = []

    def _family_block(key: str, label: str) -> Dict[str, float]:
        fam_stats = patterns.get(key, {}) or {}
        cover = daily_cover.get("S40" if key == "S40" else "PACK_164950", {}) or {}
        rate = fam_stats.get("hit_rate_overall", fam_stats.get("hit_rate_exact", 0.0)) or 0.0
        return {
            "hits": int(fam_stats.get("exact_hits", 0) or 0),
            "hit_rate": float(rate) * 100.0,
            "daily_rate": (cover.get("covered_days", 0) or 0) / (cover.get("total_days", 0) or 1) * 100.0,
            "daily_days": int(cover.get("covered_days", 0) or 0),
            "total_days": int(cover.get("total_days", 0) or 0),
            "best_slot": fam_stats.get("best_slot"),
            "weak_slot": fam_stats.get("weak_slot"),
        }

    s40_block = _family_block("S40", "S40")
    fam_block = _family_block("PACK_164950", "164950")

    per_slot: Dict[str, Dict[str, Dict[str, float]]] = {}
    for slot in SLOTS:
        s40_slot = (patterns.get("S40", {}).get("by_slot", {}) or {}).get(slot, {})
        fam_slot = (patterns.get("PACK_164950", {}).get("by_slot", {}) or {}).get(slot, {})
        per_slot[slot] = {
            "S40": s40_slot,
            "PACK_164950": fam_slot,
        }

    extra_family = _extract_extra_family(summary_json)

    return PatternSummary(
        total_hits=int(summary_json.get("rows", 0) or 0),
        s40=s40_block,
        fam_164950=fam_block,
        extra_family=extra_family,
        notes=notes,
        per_slot=per_slot,
    )


def load_pattern_summary(window_days: int = 90) -> PatternSummary:
    return load_pattern_summary_from_intel(window_days=window_days)


def _pattern_slot_stats(window_days: int = 30) -> Dict[str, Dict[str, float]]:
    stats = compute_pack_hit_stats(window_days=window_days, base_dir=quant_paths.get_project_root())
    if not stats:
        return {}
    baseline_s40 = stats.get("S40", {}).get("hit_rate", 0.0)
    baseline_fam = stats.get("FAMILY_164950", {}).get("hit_rate", 0.0)
    per_slot = stats.get("per_slot", {})
    summary: Dict[str, Dict[str, float]] = {}
    for slot in SLOTS:
        slot_stats = per_slot.get(slot)
        if not slot_stats:
            continue
        summary[slot] = {
            "total": slot_stats.get("total", 0),
            "s40_hits": slot_stats.get("s40_hits", 0),
            "fam_hits": slot_stats.get("fam_hits", 0),
            "s40_rate": slot_stats.get("s40_rate", 0.0),
            "fam_rate": slot_stats.get("fam_rate", 0.0),
            "s40_baseline": baseline_s40,
            "fam_baseline": baseline_fam,
            "days_total": slot_stats.get("days_total", 0),
            "s40_days": slot_stats.get("s40_days", 0),
            "fam_days": slot_stats.get("fam_days", 0),
        }
    summary["baseline_s40"] = baseline_s40
    summary["baseline_fam"] = baseline_fam
    return summary


def _short_window_regimes(slot_stats: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    regimes: Dict[str, Dict[str, object]] = {}
    for slot in SLOTS:
        stats = slot_stats.get(slot)
        if not stats:
            continue
        total_days = max(1, int(stats.get("days_total", 0) or 0))

        def _regime(days_with: int) -> str:
            if days_with <= 0:
                return "OFF"
            ratio = days_with / total_days if total_days else 0.0
            if ratio >= 0.7:
                return "BOOST"
            return "NORMAL"

        s40_days = int(stats.get("s40_days", 0) or 0)
        fam_days = int(stats.get("fam_days", 0) or 0)
        regimes[slot] = {
            "total_days": total_days,
            "s40_days": s40_days,
            "fam_days": fam_days,
            "s40_regime": _regime(s40_days),
            "fam_regime": _regime(fam_days),
        }
    return regimes


def check_pattern_baseline(current_keys: Iterable[str]) -> Optional[str]:
    baseline_path = PERFORMANCE_DIR / "pattern_baseline.json"
    current_set = set(current_keys)
    baseline_data = _load_json(baseline_path)
    if baseline_data is None:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps({"baseline_keys": sorted(current_set)}))
        return "Baseline recorded for pattern families (first run)."
    baseline_keys = set(baseline_data.get("baseline_keys", []))
    missing = baseline_keys - current_set
    new_keys = current_set - baseline_keys
    if missing:
        return f"⚠️ Missing baseline families: {', '.join(sorted(missing))}"
    if new_keys:
        return f"Existing families preserved; {len(new_keys)} new families added since baseline."
    return "Existing pattern families preserved."


def load_strategy_summary() -> StrategySummary:
    data = _load_json(PERFORMANCE_DIR / "strategy_recommendation.json") or {}
    return StrategySummary(
        recommended=data.get("recommended_strategy"),
        top_family=data.get("top_family"),
        confidence=data.get("confidence_level"),
        risk_mode=data.get("risk_mode"),
    )


def load_money_manager() -> MoneyManagerSummary:
    data = _load_json(PERFORMANCE_DIR / "money_management_plan.json") or {}
    bankroll = data.get("bankroll_rules", {})
    return MoneyManagerSummary(
        risk_mode=bankroll.get("risk_mode") or data.get("risk_mode"),
        daily_cap=(bankroll.get("daily_caps") or {}).get("total") or (data.get("daily_limits") or {}).get("max_total_stake"),
        single_cap=(bankroll.get("daily_caps") or {}).get("single") or (data.get("daily_limits") or {}).get("max_single_stake"),
    )


def load_confidence_scores() -> ConfidenceSummary:
    data = _load_json(PERFORMANCE_DIR / "prediction_confidence.json") or {}
    scores = {}
    labels = {}
    reasons: Dict[str, str] = {}
    conf = data.get("confidence_scores", {})
    for slot, slot_data in conf.items():
        score = slot_data.get("confidence_score")
        cool_off = bool(slot_data.get("factors", {}).get("cool_off"))
        scores[slot] = min(score, 40) if (score is not None and cool_off) else score
        if score is None:
            continue
        if cool_off:
            labels[slot] = "COOL_OFF"
        elif score >= 80:
            labels[slot] = "VERY_HIGH"
        elif score >= 65:
            labels[slot] = "HIGH"
        elif score >= 50:
            labels[slot] = "MEDIUM"
        else:
            labels[slot] = "LOW"
        if cool_off:
            reasons[slot] = "Source confidence is already cool-off gated by prediction_confidence.json"

    return ConfidenceSummary(scores=scores, labels=labels, reasons=reasons)


def apply_roi_cool_off_gate(
    confidence: ConfidenceSummary, topn_insight: Optional[Dict]
) -> ConfidenceSummary:
    """
    Apply a short-window ROI → confidence cool-off guardrail.

    Rule (per slot): if P&L regime is SLUMP and Top1/Top5 ROI are deeply red
    with no green shoots (max Top1–Top5 ≤ 0), force the confidence down to LOW
    (score capped at 35).
    """

    if not topn_insight or not confidence.scores:
        return confidence

    per_slot_roi = topn_insight.get("per_slot", {}) or {}
    slot_pnl_map = _load_slot_roi()

    for slot in SLOTS:
        roi_map = per_slot_roi.get(slot, {})
        roi_by_n = roi_map.get("roi_by_N", {}) if isinstance(roi_map, dict) else {}

        roi1 = roi_by_n.get(1)
        roi5 = roi_by_n.get(5)
        roi_candidates = [roi_by_n.get(n) for n in range(1, 6) if roi_by_n.get(n) is not None]

        regime = _pnl_regime(slot_pnl_map.get(slot))

        if (
            regime == "SLUMP"
            and roi1 is not None
            and roi5 is not None
            and roi_candidates
            and roi1 <= -90.0
            and roi5 <= -50.0
            and max(roi_candidates) <= 0.0
        ):
            score = confidence.scores.get(slot)
            if score is not None:
                confidence.scores[slot] = min(score, 35)
            confidence.labels[slot] = "LOW"
            confidence.reasons[slot] = (
                f"Cool-off: {slot} is {regime} with Top1={roi1:+.1f}%, "
                f"Top5={roi5:+.1f}%, no green shoots (max Top1–5 ≤ 0)."
            )

    return confidence


def currency(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"₹{value:,.0f}"


def pct(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}%"


def print_plan_section(
    plan: Optional[PlanSummary], execution: ExecutionReadiness, mode: str, final_plan: Optional[dict] = None
) -> None:
    print("1️⃣ PREDICTION SNAPSHOT")
    if mode == "NEXT_DAY" and final_plan:
        for slot in SLOTS:
            if slot not in final_plan.get("slots", {}):
                continue
            slot_data = final_plan["slots"][slot]
            numbers = slot_data.get("numbers") or []
            mains = (
                ", ".join(f"{n['num']}({n.get('tier', '?')} ₹{n.get('stake', 0):.0f})" for n in numbers)
                if numbers
                else "-"
            )
            extras = []
            if slot_data.get("andar_digit") is not None:
                stake = slot_data.get("andar_stake")
                stake_val = stake if stake is not None else 0
                extras.append(f"ANDAR={slot_data['andar_digit']}(₹{stake_val:.0f})")
            if slot_data.get("bahar_digit") is not None:
                stake = slot_data.get("bahar_stake")
                stake_val = stake if stake is not None else 0
                extras.append(f"BAHAR={slot_data['bahar_digit']}(₹{stake_val:.0f})")
            if extras:
                print(
                    f"   {slot}: {mains} | {', '.join(extras)} → Slot stake: ₹{slot_data.get('slot_stake', 0):.0f}"
                )
            else:
                print(f"   {slot}: {mains} → Slot stake: ₹{slot_data.get('slot_stake', 0):.0f}")

        total_planned = final_plan.get("total_stake")
        if total_planned is None:
            total_planned = sum(s.get("slot_stake", 0) or 0 for s in final_plan.get("slots", {}).values())
        if execution.recommended_stake:
            print(
                f"   TOTAL planned stake: {currency(total_planned)} → Recommended live stake: {currency(execution.recommended_stake)}"
            )
        else:
            print(f"   TOTAL planned stake: {currency(total_planned)}")
        return

    if not plan or not plan.slots:
        print("   (Plan data not available)")
        return
    for slot in plan.slots:
        mains = ", ".join(slot.main_numbers) if slot.main_numbers else "-"
        andar = f"ANDAR={slot.andar}(₹{slot.andar_stake:.0f})" if slot.andar else "ANDAR=NA"
        bahar = f"BAHAR={slot.bahar}(₹{slot.bahar_stake:.0f})" if slot.bahar else "BAHAR=NA"
        print(f"   {slot.slot}: {mains} | {andar}, {bahar} → Slot stake: ₹{slot.slot_stake:.0f}")
    if execution.recommended_stake:
        print(f"   TOTAL planned stake: {currency(plan.total_stake)} → Recommended live stake: {currency(execution.recommended_stake)}")
    else:
        print(f"   TOTAL planned stake: {currency(plan.total_stake)}")


def print_pnl_section(pnl: PnLSnapshot) -> None:
    print("\n2️⃣ P&L SNAPSHOT")
    print(f"   Overall P&L      : {currency(pnl.overall_pnl)} (ROI {pct(pnl.overall_roi)})")
    print(f"   Last 7 days      : {currency(pnl.last7_pnl)} (ROI {pct(pnl.last7_roi)})")
    print(f"   Last 30 days     : {currency(pnl.last30_pnl)} (ROI {pct(pnl.last30_roi)})")
    best = f"{pnl.best_slot[0]} {currency(pnl.best_slot[1])}" if pnl.best_slot else "N/A"
    worst = f"{pnl.worst_slot[0]} {currency(pnl.worst_slot[1])}" if pnl.worst_slot else "N/A"
    print(f"   Best slot        : {best}")
    print(f"   Weak slot        : {worst}")
    if pnl.golden_roi is not None:
        print(f"   Golden days ROI  : {pct(pnl.golden_roi)} (P&L {currency(pnl.golden_pnl)})")


def print_pattern_section(patterns: PatternSummary) -> None:
    print("\n3️⃣ PATTERN & LEARNING (pattern window baseline)")
    if patterns.total_hits is not None:
        print(f"   Hits analyzed    : {patterns.total_hits}")
    if patterns.s40:
        hr = patterns.s40.get("hit_rate", 0.0) or 0.0
        hits = patterns.s40.get("hits")
        print(f"   S40 family       : {hr:.2f}% per-row, {hits} tagged hits")
        print(
            f"   S40 daily cover  : {patterns.s40.get('daily_rate', 0.0):.1f}% of days "
            f"({patterns.s40.get('daily_days', 0)}/{patterns.s40.get('total_days', 0)}) with ≥1 S40 result"
        )
    if patterns.fam_164950:
        hr = patterns.fam_164950.get("hit_rate", 0.0) or 0.0
        hits = patterns.fam_164950.get("hits")
        print(f"   164950 family    : {hr:.2f}% per-row, {hits} tagged hits")
        print(
            f"   164950 daily     : {patterns.fam_164950.get('daily_rate', 0.0):.1f}% of days "
            f"({patterns.fam_164950.get('daily_days', 0)}/{patterns.fam_164950.get('total_days', 0)}) with ≥1 164950 result"
        )
    if patterns.extra_family:
        fam = patterns.extra_family
        fam_name = fam.get("name") or fam.get("family")
        hit_rate = float(fam.get("hit_rate", 0.0) or 0.0)
        covered = fam.get("covered_days") or fam.get("daily_days") or 0
        total_days = fam.get("total_days") or fam.get("daily_total") or covered
        best_slot = fam.get("best_slot") or "n/a"
        weak_slot = fam.get("weak_slot") or "n/a"
        print(
            f"   - Extra family (best): {fam_name} – hit_rate={hit_rate:.2f}%, "
            f"daily_cover={covered}/{total_days} days, best_slot={best_slot}, weak_slot={weak_slot}"
        )
    best_s40 = patterns.s40.get("best_slot") if patterns.s40 else None
    weak_s40 = patterns.s40.get("weak_slot") if patterns.s40 else None
    best_164 = patterns.fam_164950.get("best_slot") if patterns.fam_164950 else None
    weak_164 = patterns.fam_164950.get("weak_slot") if patterns.fam_164950 else None
    if best_s40 or weak_s40:
        print(f"   S40 best={best_s40 or 'n/a'}, weak={weak_s40 or 'n/a'} (window 90d)")
    if best_164 or weak_164:
        print(f"   164950 best={best_164 or 'n/a'}, weak={weak_164 or 'n/a'} (window 90d)")

    if patterns.per_slot:
        print("   Per-slot regime (pattern window):")
        for slot in SLOTS:
            slot_s40 = patterns.per_slot.get(slot, {}).get("S40", {}) or {}
            slot_fam = patterns.per_slot.get(slot, {}).get("PACK_164950", {}) or {}

            def _fmt_block(block: Dict[str, float]) -> str:
                regime = quant_learning_core.slot_regime(block.get("hit_rate_exact"))
                active = int(block.get("active_days", 0) or 0)
                total = int(block.get("days_with_results", 0) or 0)
                return f"{regime} ({active}/{total})"

            s40_part = _fmt_block(slot_s40)
            fam_part = _fmt_block(slot_fam)
            print(f"   {slot}: S40 {s40_part}, 164950 {fam_part}")
    for note in patterns.notes:
        print(f"   {note}")


def _temperature(rate: float, baseline: float, margin: float = 0.05) -> str:
    if rate is None:
        return "NORMAL"
    if rate >= baseline + margin:
        return "HOT"
    if rate <= baseline - margin:
        return "COLD"
    return "NORMAL"


def print_pattern_slot_section(window_days: int = 30) -> None:
    slot_stats = _pattern_slot_stats(window_days=window_days)
    if not slot_stats:
        print("   Per-slot pattern layer warming up (no data).")
        return
    print(f"   Short-window regime (last {window_days}d):")
    regimes = _short_window_regimes(slot_stats)
    for slot in SLOTS:
        stats = regimes.get(slot)
        if not stats:
            continue
        print(
            f"   {slot}: S40 {stats['s40_regime']} ({stats['s40_days']}/{stats['total_days']}), "
            f"164950 {stats['fam_regime']} ({stats['fam_days']}/{stats['total_days']})"
        )


def _load_slot_roi() -> Dict[str, float]:
    snapshot = _load_slot_health_snapshot()
    roi_map: Dict[str, float] = {}
    for slot, entry in snapshot.items():
        roi_val = entry.get("roi") if isinstance(entry, dict) else None
        if roi_val is not None:
            try:
                roi_map[slot] = float(roi_val)
            except Exception:
                continue
    return roi_map


def _pnl_regime(value: Optional[float]) -> str:
    if value is None:
        return "OK"
    if value >= 15:
        return "STRONG"
    if value <= -10:
        return "SLUMP"
    if value < 0:
        return "WEAK"
    return "OK"


def print_regime_snapshot(window_days: int = 30) -> None:
    print("3️⃣ REGIME SNAPSHOT")
    slot_pattern_stats = _pattern_slot_stats(window_days=window_days)
    regimes = _short_window_regimes(slot_pattern_stats)
    slot_health = _load_slot_health_snapshot()
    roi_map = {slot: (slot_health.get(slot) or {}).get("roi") for slot in SLOTS}
    if not slot_pattern_stats and not roi_map:
        print("   Regime snapshot warming up (insufficient data).")
        return
    for slot in SLOTS:
        patterns = slot_pattern_stats.get(slot, {})
        slot_regime = regimes.get(slot, {})
        health_entry = slot_health.get(slot, {}) if isinstance(slot_health, dict) else {}
        pnl_label = "SLUMP" if health_entry.get("slump") else _pnl_regime(roi_map.get(slot))
        level_label = health_entry.get("slot_level", "MID")
        s40_label = slot_regime.get("s40_regime", "NORMAL")
        fam_label = slot_regime.get("fam_regime", "NORMAL")
        print(f"   {slot}: P&L={pnl_label}, Level={level_label}, S40={s40_label}, 164950={fam_label}")


def print_pattern_family_snapshot() -> None:
    patterns_root = quant_stats.get("patterns") if isinstance(quant_stats, dict) else None
    if not patterns_root:
        print("   (pattern family snapshot unavailable – run pattern_intelligence_enhanced.py first.)")
        return
    slots_block = patterns_root.get("slots", {}) if isinstance(patterns_root, dict) else {}
    if not slots_block:
        print("   (pattern family snapshot unavailable – run pattern_intelligence_enhanced.py first.)")
        return
    print("   27-family regimes:")
    for slot in SLOTS:
        slot_block = slots_block.get(slot, {}) if isinstance(slots_block, dict) else {}
        fam_block = slot_block.get("families", {}) if isinstance(slot_block, dict) else {}
        boosts = []
        offs = []
        for fam_name, fam_info in fam_block.items():
            regime = str(fam_info.get("regime_30d", fam_info.get("regime" ,"NORMAL"))).upper()
            hit_rate = fam_info.get("hit_rate_30d", fam_info.get("hit_rate", 0.0))
            if regime == "BOOST":
                boosts.append((fam_name, hit_rate))
            elif regime == "OFF":
                offs.append((fam_name, hit_rate))
        boosts = sorted(boosts, key=lambda x: (-(x[1] or 0), x[0]))[:3]
        offs = sorted(offs, key=lambda x: (x[1] or 0, x[0]))[:2]
        boost_tags = "{" + ", ".join([b[0] for b in boosts]) + "}" if boosts else "{}"
        off_tags = "{" + ", ".join([o[0] for o in offs]) + "}" if offs else "{}"
        print(f"   {slot}: BOOST={boost_tags} | OFF={off_tags}")

    print("\n   S40 & 164950 cross-slot snapshot:")
    for fam in ["S40", "FAMILY_164950"]:
        per_slot = {}
        for slot in SLOTS:
            fam_info = ((slots_block.get(slot, {}) or {}).get("families", {}) or {}).get(fam, {})
            per_slot[slot] = fam_info
        best_slot = max(per_slot.items(), key=lambda x: x[1].get("hit_rate_30d", 0.0) if isinstance(x[1], dict) else 0.0)[0]
        weak_slot = min(per_slot.items(), key=lambda x: x[1].get("hit_rate_30d", 0.0) if isinstance(x[1], dict) else 0.0)[0]
        parts = [f"{slot}={str((info or {}).get('regime_30d') or (info or {}).get('regime') or 'NORMAL')}" for slot, info in per_slot.items()]
        print(f"   {fam}: best={best_slot}, weak={weak_slot}, regimes: {', '.join(parts)}")


def print_risk_section(strategy: StrategySummary, money: MoneyManagerSummary, execution: ExecutionReadiness, confidence: ConfidenceSummary) -> None:
    print("\n4️⃣ RISK & EXECUTION")
    strat = strategy.recommended or "(strategy data NA)"
    risk_mode = strategy.risk_mode or money.risk_mode or "UNKNOWN"
    print(f"   Strategy         : {strat}")
    print(f"   Risk mode        : {risk_mode}")
    if execution.mode:
        mult = execution.multiplier if execution.multiplier is not None else 1.0
        print(f"   Execution mode   : {execution.mode} (x{mult})")
    if money.daily_cap or money.single_cap:
        print(f"   Money manager    : daily cap {currency(money.daily_cap)}, single-slot cap {currency(money.single_cap)}")
    if confidence.scores:
        parts = []
        for slot in SLOTS:
            if slot in confidence.scores:
                label = confidence.labels.get(slot, "-")
                parts.append(f"{slot} {label} ({confidence.scores[slot]:.0f})")
        if parts:
            print(f"   Confidence       : {', '.join(parts)}")
        cool_off_notes = [
            f"{slot}: {reason}"
            for slot, reason in sorted(confidence.reasons.items())
            if reason
        ]
        if cool_off_notes:
            print("   Cool-off notes   :")
            for note in cool_off_notes:
                print(f"      • {note}")


def print_script_performance_section(window_days: int = SCRIPT_METRICS_WINDOW_DAYS) -> None:
    print()
    # Script hit metrics use a shorter pattern window than ROI to keep recency sharp.
    print(f"5️⃣ SCRIPT PERFORMANCE (last {window_days} days)")
    metrics, summary = get_metrics_table(window_days=window_days, mode="per_slot")

    if metrics is None or summary is None or metrics.empty:
        print("   No script hit metrics available yet (script_hit_memory is empty).")
        return

    total_hits = 0
    for col in ("exact_hits", "mirror_hits", "neighbor_hits"):
        if col in metrics.columns:
            total_hits += int(metrics[col].sum())

    if total_hits == 0:
        print(f"   No script-level hits in the last {window_days} days – league not meaningful yet.")
        return

    league = build_script_league(metrics, min_predictions=10, min_hits_for_hero=1)
    if not league:
        print(f"   No script-level hits in the last {window_days} days – league not meaningful yet.")
        return

    def _fmt_scores(score: Optional[float], rate: Optional[float]) -> str:
        if score is None:
            return "score n/a"
        rate_val = rate if rate is not None else 0.0
        return f"score {score:+.2f} exact {rate_val:.1%}"

    heroes_df = hero_weak_table(metrics, min_predictions=10)
    hero_map = {row.get("slot"): row for _, row in heroes_df.iterrows()} if heroes_df is not None else {}

    for slot in SLOTS:
        hero_row = hero_map.get(slot) if hero_map else None
        hero_id = hero_row.get("hero_script") if hero_row is not None else None
        weak_id = hero_row.get("weak_script") if hero_row is not None else None
        hero_score = hero_row.get("hero_score") if hero_row is not None else None
        weak_score = hero_row.get("weak_score") if hero_row is not None else None
        hero_rate = hero_row.get("hero_hit_rate_exact") if hero_row is not None else None
        weak_rate = hero_row.get("weak_hit_rate_exact") if hero_row is not None else None
        hero_id = hero_id or "n/a"
        weak_id = weak_id or "n/a"
        print(
            f"   {slot}: hero {hero_id} ({_fmt_scores(hero_score, hero_rate)}) "
            f"| weak {weak_id} ({_fmt_scores(weak_score, weak_rate)})"
        )

    overall = league.get("overall") if isinstance(league, dict) else None
    if overall:
        hero = overall.get("hero", {})
        weak = overall.get("weak", {})
        hero_id = hero.get("script_id") or "n/a"
        weak_id = weak.get("script_id") or "n/a"
        hero_score = hero.get("score")
        weak_score = weak.get("score")
        hero_rate = hero.get("hit_rate_exact")
        weak_rate = weak.get("hit_rate_exact")
        print(
            f"   Overall hero: {hero_id} ({_fmt_scores(hero_score, hero_rate)}) "
            f"| overall weak: {weak_id} ({_fmt_scores(weak_score, weak_rate)})"
        )


def _derive_topn_best(insight: Optional[Dict]) -> Tuple[Optional[int], Dict[str, int]]:
    if not insight:
        return None, {}

    overall = insight.get("overall", {}) if isinstance(insight, dict) else {}
    per_slot = insight.get("per_slot", {}) if isinstance(insight, dict) else {}

    def _best_from_map(roi_map: Dict[int, float]) -> Optional[int]:
        if not roi_map:
            return None
        best_roi = max(roi_map.values())
        best_candidates = [n for n, roi_val in roi_map.items() if roi_val == best_roi]
        return min(best_candidates) if best_candidates else None

    overall_best = overall.get("best_N") if isinstance(overall, dict) else None
    if overall_best is None:
        overall_roi_map = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
        overall_best = _best_from_map(overall_roi_map)

    per_slot_best: Dict[str, int] = {}
    if isinstance(per_slot, dict):
        for slot, slot_map in per_slot.items():
            roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
            slot_best = slot_map.get("best_N") if isinstance(slot_map, dict) else None
            if slot_best is None:
                slot_best = _best_from_map(roi_map)
            if slot_best is not None:
                try:
                    per_slot_best[str(slot).upper()] = int(slot_best)
                except Exception:
                    continue

    return (int(overall_best) if overall_best is not None else None), per_slot_best


def _load_topn_best_profile(insight: Optional[Dict] = None) -> Tuple[Optional[int], Dict[str, int]]:
    overall_best, per_slot_best = _derive_topn_best(insight)
    if overall_best or per_slot_best:
        return overall_best, per_slot_best

    try:
        base_dir = quant_paths.get_base_dir()
    except Exception:
        return None, {}
    path = Path(base_dir) / "logs" / "performance" / "topn_roi_summary.json"
    profile = _load_json(path) or {}
    if not profile:
        return None, {}

    overall_block = profile.get("overall", {}) if isinstance(profile, dict) else {}
    overall_best = overall_block.get("best_N") or profile.get("overall_best_N")
    per_slot_block = profile.get("best_n_per_slot", {}) if isinstance(profile, dict) else {}
    per_slot_best = {}
    if isinstance(per_slot_block, dict):
        for slot, value in per_slot_block.items():
            try:
                per_slot_best[str(slot).upper()] = int(value)
            except Exception:
                continue
    return (int(overall_best) if overall_best is not None else None), per_slot_best


def print_topn_roi_insight(insight: Optional[Dict] = None) -> Optional[Dict]:
    window_days = TOPN_WINDOW_DAYS
    try:
        base_dir = quant_paths.get_base_dir()
    except Exception:
        base_dir = Path(".")
    topn_policy = _load_json(Path(base_dir) / TOPN_POLICY_PATH) or {}
    if insight is None:
        insight = quant_stats.get("topn") if isinstance(quant_stats, dict) else None
    if insight is None:
        insight = compute_topn_roi(window_days=window_days)
    if not insight:
        print("6️⃣ TOP-N ROI INSIGHT (requested 30d)\n   (Top-N ROI snapshot unavailable – run topn_roi_scanner.py first.)")
        return None
    overall = (insight.get("overall") if isinstance(insight, dict) else {}) or {}
    per_slot = (insight.get("slots") or insight.get("per_slot") or {}) if isinstance(insight, dict) else {}
    roi_by_n = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
    best_n = overall.get("best_N") if isinstance(overall, dict) else None
    best_roi = overall.get("best_roi") if isinstance(overall, dict) else None
    available_days = insight.get("available_days")
    effective_days = f"effective {available_days}d" if available_days else "effective window"
    print(f"6️⃣ TOP-N ROI INSIGHT (requested {window_days}d, {effective_days})")
    if best_n is not None:
        print(f"   Best N = {best_n} with overall ROI = {best_roi:+.1f}%")
    else:
        print("   No ROI data available.")

    print(f"   Per-slot ROI ({effective_days}):")
    for slot in SLOTS:
        roi_map = per_slot.get(slot, {}).get("roi_by_N", {}) if isinstance(per_slot.get(slot), dict) else {}
        parts = [f"Top{n}:{roi_map.get(n, 0.0):+.1f}%" for n in range(1, 11)]
        print(f"   {slot}: {' | '.join(parts)}")
        slot_best = per_slot.get(slot, {}).get("best_N") if isinstance(per_slot.get(slot), dict) else None
        slot_best_roi = per_slot.get(slot, {}).get("best_roi") if isinstance(per_slot.get(slot), dict) else None
        if roi_map:
            top_band = [roi_map.get(n) for n in range(1, 6) if roi_map.get(n) is not None]
            deep_positive = [(n, v) for n, v in roi_map.items() if n > 5 and v is not None and v > 0]
            if top_band and all((r is not None and r <= 0) for r in top_band) and deep_positive and slot_best:
                print(f"      {slot}: best_N={slot_best}, ROI={slot_best_roi:+.1f}% (Top1–5 red; using deeper N-band).")
            elif slot_best and slot_best <= 5 and slot_best_roi is not None and slot_best_roi > 0:
                print(f"      {slot}: best_N={slot_best}, ROI={slot_best_roi:+.1f}% (tight profitable band).")

    overall_best, per_slot_best = _load_topn_best_profile(insight)
    if overall_best is not None or per_slot_best:
        default_n = overall_best or 3
        print("\n   Top-N mode recommendation:")
        print(f"      Overall best N : {overall_best if overall_best is not None else default_n}")
        for slot in SLOTS:
            slot_best = per_slot_best.get(slot, default_n)
            print(f"      {slot} best N    : {slot_best}")

    if isinstance(topn_policy, dict) and topn_policy:
        print("\n   Top-N policy (final-best guidance):")
        for slot in SLOTS:
            policy_block = topn_policy.get(slot, {}) if isinstance(topn_policy, dict) else {}
            final_best_n = policy_block.get("final_best_n") if isinstance(policy_block, dict) else None
            roi_final_best = policy_block.get("roi_final_best") if isinstance(policy_block, dict) else None
            if final_best_n is None and roi_final_best is None:
                continue
            roi_text = f"{float(roi_final_best):+.1f}%" if roi_final_best is not None else "n/a"
            print(f"      {slot}: final_best_n={final_best_n if final_best_n is not None else '-'} | roi≈{roi_text}")
    return insight


def print_arjun_section(base_dir: Optional[Path] = None) -> None:
    print()
    print("7️⃣ ARJUN MODE (FOCUSED SHOT)")
    try:
        project_root = Path(base_dir) if base_dir else quant_paths.get_base_dir()
        arjun_path = project_root / "data" / "arjun_pick.json"
        if not arjun_path.exists():
            print("   No Arjun pick available (run quant_arjun_mode.py after bet_pnl_tracker & precise_bet_engine).")
            return

        data = json.loads(arjun_path.read_text())
        slot = data.get("slot") if isinstance(data, dict) else None
        number = data.get("number") if isinstance(data, dict) else None
        andar = data.get("andar") if isinstance(data, dict) else None
        bahar = data.get("bahar") if isinstance(data, dict) else None
        if not slot or not number:
            print("   No Arjun pick available (run quant_arjun_mode.py after bet_pnl_tracker & precise_bet_engine).")
            return

        sources = data.get("sources", {}) if isinstance(data, dict) else {}
        slot_health = sources.get("slot_health", {}) if isinstance(sources, dict) else {}
        hero_script = sources.get("hero_script") if isinstance(sources, dict) else None
        patterns = sources.get("patterns", {}) if isinstance(sources, dict) else {}
        topn = sources.get("topn", {}) if isinstance(sources, dict) else {}

        reason_parts: List[str] = []
        if slot_health:
            if not slot_health.get("slump"):
                reason_parts.append("non-slump")
            roi_val = slot_health.get("roi")
            if roi_val is not None:
                reason_parts.append(f"ROI {float(roi_val):+.1f}%")
        s40_regime = patterns.get("S40") if isinstance(patterns, dict) else None
        fam_regime = patterns.get("FAMILY_164950") if isinstance(patterns, dict) else None
        if s40_regime:
            reason_parts.append(f"S40 {s40_regime}")
        if fam_regime:
            reason_parts.append(f"164950 {fam_regime}")
        if isinstance(topn, dict) and topn.get("roi") is not None:
            reason_parts.append(f"TopN {float(topn.get('roi', 0.0)):+.1f}%")
        if hero_script:
            reason_parts.append(f"hero {hero_script}")

        print(f"   Slot: {slot} | Number: {number} (ANDAR={andar}, BAHAR={bahar})")
        if reason_parts:
            print(f"   Source: {', '.join(reason_parts)}")
    except Exception:
        print("   No Arjun pick available (run quant_arjun_mode.py after bet_pnl_tracker & precise_bet_engine).")


def print_header(bet_date: date, target_date: date, mode: str, strategy: StrategySummary, execution: ExecutionReadiness, plan: Optional[PlanSummary]):
    print("=" * 80)
    print(f"🎯 QUANT DAILY BRIEF – {target_date.isoformat()} (MODE: {mode})")
    print("=" * 80)
    print(f"Bet date          : {bet_date.isoformat()}")
    print(f"Target date       : {target_date.isoformat()}")
    strat = strategy.recommended or "(strategy NA)"
    print(f"Strategy          : {strat}")
    if execution.mode:
        mult = execution.multiplier if execution.multiplier is not None else 1.0
        print(f"Execution mode    : {execution.mode} ({mult}x)")
    if execution.base_stake or execution.recommended_stake or (plan and plan.total_stake):
        base = execution.base_stake or (plan.total_stake if plan else None)
        if base is not None:
            line = f"Final plan stake  : {currency(base)}"
            if execution.recommended_stake:
                line += f" → Recommended live stake: {currency(execution.recommended_stake)}"
            print(line)


def build_brief(mode: str, bet_date: date, target_date: date, dry_run: bool = False) -> None:
    if mode == "INTRADAY":
        run_intraday_pipeline(bet_date, dry_run=dry_run)
    else:
        run_nextday_pipeline(target_date, dry_run=dry_run)

    plan = load_plan_for_mode(mode, bet_date, target_date, dry_run=dry_run)
    final_plan = load_final_bet_plan_for_date(target_date) if mode == "NEXT_DAY" else None
    execution = load_execution_readiness()
    pnl = load_pnl_snapshot()
    patterns = load_pattern_summary_from_intel()
    strategy = load_strategy_summary()
    money = load_money_manager()
    topn_insight = compute_topn_roi(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    confidence = apply_roi_cool_off_gate(load_confidence_scores(), topn_insight)

    print_header(bet_date, target_date, mode, strategy, execution, plan)
    print_plan_section(plan, execution, mode, final_plan=final_plan)
    print_pnl_section(pnl)
    print_regime_snapshot(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    print_pattern_section(patterns)
    print_pattern_slot_section(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    print_pattern_family_snapshot()
    print_risk_section(strategy, money, execution, confidence)
    print_script_performance_section(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    topn_insight = print_topn_roi_insight(topn_insight)
    print_arjun_section()
    slot_health_snapshot = _load_slot_health_snapshot()
    slot_level_parts = [
        f"{slot}={slot_health_snapshot.get(slot, {}).get('slot_level', 'MID')}" for slot in SLOTS
    ]
    if slot_level_parts:
        print(f"Slot levels: {', '.join(slot_level_parts)}")
    print("=" * 80)
    best_roi = None
    best_n, _ = _load_topn_best_profile(topn_insight)
    if topn_insight:
        overall = topn_insight.get("overall", {}) if isinstance(topn_insight, dict) else {}
        best_roi = overall.get("best_roi") if isinstance(overall, dict) else None

    verdict = "Short verdict: "
    pattern_snapshot = quant_stats.get("patterns") if isinstance(quant_stats, dict) else {}
    has_off = False
    if pattern_snapshot:
        slots_block = pattern_snapshot.get("slots", {}) if isinstance(pattern_snapshot, dict) else {}
        for slot in SLOTS:
            fams = (slots_block.get(slot, {}) or {}).get("families", {}) if isinstance(slots_block, dict) else {}
            if any(str((fams.get(f) or {}).get("regime_30d", "")).upper() == "OFF" for f in fams):
                has_off = True
                break
    if pnl.last7_roi and pnl.last30_roi and pnl.last7_roi > 0 and pnl.last30_roi > 0:
        if best_roi is not None and best_roi < 0:
            verdict += "System profitable but Top-N band soft — keep stakes disciplined."
        elif has_off:
            verdict += "ROI healthy; some families OFF — stay balanced while waiting for fresh hits."
        else:
            verdict += "ROI healthy; focus on BOOST families and positive Top-N bands."
    else:
        verdict += "Recent ROI under pressure or cooling families — stay defensive until hit-rate recovers."

    print(verdict)
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Quant Daily Brief")
    parser.add_argument("--date", dest="date_str", help="Bet date (YYYY-MM-DD)")
    parser.add_argument("--mode", choices=["auto", "intraday", "nextday"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Do not run subprocesses; preview only")
    args = parser.parse_args()

    bet_date = parse_date(args.date_str) if args.date_str else find_latest_bet_date()
    results_df = load_results_df()
    mode, target_date = decide_mode(bet_date, args.mode, results_df)

    if not args.dry_run:
        refresh_script_hit_memory_and_metrics(window_days=SCRIPT_METRICS_WINDOW_DAYS)
        try:
            pattern_core.build_pattern_config(window_days=120)
        except Exception:
            pass

    build_brief(mode, bet_date, target_date, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
