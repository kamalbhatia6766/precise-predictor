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
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

import quant_data_core
import quant_paths
from script_hit_metrics import (
    build_script_weight_map,
    compute_pack_hit_stats,
    get_metrics_table,
    build_script_league,
    format_script_league,
)
from script_hit_memory_utils import load_script_weights
from utils_2digit import is_valid_2d_number, to_2d_str

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
PERFORMANCE_DIR = quant_paths.get_performance_logs_dir()
BET_ENGINE_DIR = quant_paths.get_bet_engine_dir()
SCRIPT_METRICS_WINDOW_DAYS = 30


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
    notes: List[str]


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
        df["DATE_ONLY"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
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


def load_plan_for_mode(mode: str, bet_date: date, target_date: date) -> Optional[PlanSummary]:
    if mode == "INTRADAY":
        path = BET_ENGINE_DIR / f"bet_plan_intraday_{bet_date.strftime('%Y%m%d')}.xlsx"
        return load_plan_from_excel(path)
    path = quant_paths.get_final_bet_plan_path(target_date.strftime("%Y-%m-%d"))
    return load_plan_from_excel(path)


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
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
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

    return PnLSnapshot(
        overall_pnl=overall.get("total_pnl"),
        overall_roi=overall.get("overall_roi"),
        last7_pnl=last7_pnl,
        last7_roi=last7_roi,
        last30_pnl=last30_pnl,
        last30_roi=last30_roi,
        best_slot=best_slot,
        worst_slot=worst_slot,
    )


def load_pattern_summary(window_days: int = 90) -> PatternSummary:
    stats = compute_pack_hit_stats(window_days=window_days, base_dir=quant_paths.get_project_root())
    notes: List[str] = []
    if not stats:
        return PatternSummary(total_hits=None, s40=None, fam_164950=None, notes=notes)

    per_slot = stats.get("per_slot", {})
    best_s40 = None
    worst_s40 = None
    if per_slot:
        ordered = sorted(per_slot.items(), key=lambda kv: kv[1].get("s40_rate", 0), reverse=True)
        best_s40 = ordered[0][0] if ordered else None
        worst_s40 = ordered[-1][0] if ordered else None
    if best_s40 and worst_s40:
        notes.append(f"S40 best={best_s40}, weak={worst_s40} (window {window_days}d)")

    return PatternSummary(
        total_hits=stats.get("total_rows"),
        s40=stats.get("S40"),
        fam_164950=stats.get("FAMILY_164950"),
        notes=notes,
    )


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
        }
    summary["baseline_s40"] = baseline_s40
    summary["baseline_fam"] = baseline_fam
    return summary


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
    conf = data.get("confidence_scores", {})
    for slot, slot_data in conf.items():
        score = slot_data.get("confidence_score")
        scores[slot] = score
        if score is None:
            continue
        if score >= 80:
            labels[slot] = "VERY_HIGH"
        elif score >= 65:
            labels[slot] = "HIGH"
        elif score >= 50:
            labels[slot] = "MEDIUM"
        else:
            labels[slot] = "LOW"
    return ConfidenceSummary(scores=scores, labels=labels)


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


def print_pattern_section(patterns: PatternSummary) -> None:
    print("\n3️⃣ PATTERN & LEARNING")
    if patterns.total_hits is not None:
        print(f"   Hits analyzed    : {patterns.total_hits}")
    if patterns.s40:
        hr = patterns.s40.get("hit_rate")
        hits = patterns.s40.get("hits")
        print(f"   S40 family       : {pct(hr)} hit rate, {hits} hits")
    if patterns.fam_164950:
        hr = patterns.fam_164950.get("hit_rate")
        hits = patterns.fam_164950.get("hits")
        print(f"   164950 family    : {pct(hr)} hit rate, {hits} hits")
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
    for slot in SLOTS:
        stats = slot_stats.get(slot)
        if not stats:
            continue
        s40_label = _temperature(stats.get("s40_rate", 0.0), stats.get("s40_baseline", 0.0))
        fam_label = _temperature(stats.get("fam_rate", 0.0), stats.get("fam_baseline", 0.0))
        print(
            f"   {slot}: S40 {s40_label} ({int(stats.get('s40_hits',0))}/{stats.get('total')}), "
            f"164950 {fam_label} ({int(stats.get('fam_hits',0))}/{stats.get('total')})"
        )


def _load_slot_roi() -> Dict[str, float]:
    data = _load_json(PERFORMANCE_DIR / "quant_reality_pnl.json") or {}
    by_slot = data.get("by_slot", [])
    roi_map: Dict[str, float] = {}
    for item in by_slot:
        slot = str(item.get("slot", "")).upper()
        roi = item.get("roi") or item.get("pnl")
        try:
            roi_map[slot] = float(roi)
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
    roi_map = _load_slot_roi()
    if not slot_pattern_stats and not roi_map:
        print("   Regime snapshot warming up (insufficient data).")
        return
    for slot in SLOTS:
        patterns = slot_pattern_stats.get(slot, {})
        pnl_label = _pnl_regime(roi_map.get(slot))
        baseline_s40 = slot_pattern_stats.get("baseline_s40", 0.0)
        baseline_fam = slot_pattern_stats.get("baseline_fam", 0.0)
        s40_rate = patterns.get("s40_rate", 0.0)
        fam_rate = patterns.get("fam_rate", 0.0)

        def _regime_label(rate: float, baseline: float) -> str:
            if pnl_label == "STRONG" and rate >= baseline + 0.10:
                return "HOT"
            if pnl_label == "SLUMP" and rate <= max(0.0, baseline - 0.10):
                return "COOL"
            return "NORMAL"

        s40_label = _regime_label(s40_rate, baseline_s40)
        fam_label = _regime_label(fam_rate, baseline_fam)
        print(f"   {slot}: P&L={pnl_label}, S40={s40_label}, 164950={fam_label}")


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


def print_script_performance_section(window_days: int = SCRIPT_METRICS_WINDOW_DAYS) -> None:
    print()
    print(f"5️⃣ SCRIPT PERFORMANCE (last {window_days} days)")
    metrics, summary = get_metrics_table(window_days=window_days)

    # Guard: if we truly have no usable data, show warming-up message
    if metrics is None or summary is None or metrics.empty:
        print("   Script performance layer warming up (no window data yet).")
        return

    def fmt_row(row) -> str:
        name = str(row.get("SCRIPT_ID", "") or row.get("script_name", "")).strip()
        total = int(row.get("EVENTS", row.get("total_predictions", 0)) or 0)
        exact_hits = int(row.get("EXACT", row.get("primary_hits", 0)) or 0)
        extended_hits = int(row.get("EXTENDED", row.get("total_hits", 0)) or 0)
        exact_rate = float(row.get("EXACT_PCT", row.get("hit_rate", 0.0)) or 0.0)
        extended_rate = float(row.get("EXTENDED_PCT", 0.0 if "EXTENDED_PCT" in row else extended_hits / total * 100 if total else 0.0) or 0.0)
        signal = str(row.get("SIGNAL", "-") or "-")

        return (
            "   "
            + f"{name}: EXACT {exact_hits}/{total}, EXT {extended_hits}/{total} "
            + f"→ EXACT {exact_rate:.1f}%, EXT {extended_rate:.1f}% (Signal: {signal})"
        )

    # Print one line per script from the metrics table
    for _, row in metrics.iterrows():
        print(fmt_row(row))

    # Build and print the league summary (heroes / weak scripts)
    league = build_script_league(metrics)
    league_text = format_script_league(league)
    for line in league_text.splitlines():
        print(f"   {line}")

    weight_map = build_script_weight_map(window_days=window_days)
    if not weight_map:
        print("   Script weights (30d): neutral (all 1.00).")
    else:
        slot_weights = load_script_weights(window_days=window_days)
        grouped: Dict[str, List[str]] = {slot: [] for slot in SLOTS}
        for (script, slot), wt in slot_weights.items():
            if slot not in grouped:
                continue
            grouped[slot].append(f"{script}={wt:.2f}")
        print("   Script weights (30d):")
        for slot in SLOTS:
            entries = grouped.get(slot) or []
            if not entries:
                continue
            preview = ", ".join(sorted(entries))
            print(f"     {slot}: {preview}")


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

    plan = load_plan_for_mode(mode, bet_date, target_date)
    final_plan = load_final_bet_plan_for_date(target_date) if mode == "NEXT_DAY" else None
    execution = load_execution_readiness()
    pnl = load_pnl_snapshot()
    patterns = load_pattern_summary()
    strategy = load_strategy_summary()
    money = load_money_manager()
    confidence = load_confidence_scores()

    print_header(bet_date, target_date, mode, strategy, execution, plan)
    print_plan_section(plan, execution, mode, final_plan=final_plan)
    print_pnl_section(pnl)
    print_regime_snapshot(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    print_pattern_section(patterns)
    print_pattern_slot_section(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    print_risk_section(strategy, money, execution, confidence)
    print_script_performance_section(window_days=SCRIPT_METRICS_WINDOW_DAYS)
    print("=" * 80)
    verdict = "Short verdict: System learning healthy; keep stakes disciplined."
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

    build_brief(mode, bet_date, target_date, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
