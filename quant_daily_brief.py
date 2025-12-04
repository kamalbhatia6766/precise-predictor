"""Central daily orchestrator for Precise Predictor.

Runs the core daily pipeline and prints a concise briefing
summarizing performance and the latest bet plan.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

SLOTS: tuple[str, ...] = ("FRBD", "GZBD", "GALI", "DSWR")


def run_step(name: str, args: list[str]) -> bool:
    print(f"\n[QUANT-DAILY] ▶ {name} ...")
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=False,
            text=True,
        )
        print(f"[QUANT-DAILY] ✅ {name} completed.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[QUANT-DAILY] ❌ {name} failed with code {e.returncode}")
        return False


def load_quant_pnl_summary(json_path: Path) -> Dict[str, Any]:
    if not json_path.exists():
        return {}
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data or {}
    except Exception:
        return {}


def _find_section(data: Dict[str, Any], keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    for key in keys:
        section = data.get(key)
        if isinstance(section, dict):
            return section
    return None


def _extract_number(container: Dict[str, Any], *names: str) -> Optional[float]:
    for name in names:
        value = container.get(name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _format_amount(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"₹{value:,.0f}" if abs(value) >= 100 else f"₹{value:,.2f}"


def _format_roi(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else "" if value < 0 else ""
    return f"{sign}{value:.2f}%"


def parse_pnl_brief(data: Dict[str, Any]) -> dict[str, Any]:
    overall_section = _find_section(data, ["OVERALL", "overall", "SUMMARY", "summary"])
    if not overall_section and all(k.isupper() for k in data.keys()):
        overall_section = data

    totals = {
        "stake": _extract_number(overall_section or {}, "TOTAL_STAKE", "TOTAL_BET", "TOTAL") if overall_section else None,
        "return": _extract_number(overall_section or {}, "TOTAL_RETURN", "RETURN") if overall_section else None,
        "pnl": _extract_number(overall_section or {}, "NET_PNL", "PNL", "TOTAL_PNL") if overall_section else None,
        "roi": _extract_number(overall_section or {}, "ROI_%", "ROI", "OVERALL_ROI") if overall_section else None,
        "date_from": None,
        "date_to": None,
    }

    date_range = _find_section(data, ["DATE_RANGE", "date_range", "DATES", "dates"])
    if isinstance(date_range, dict):
        totals["date_from"] = date_range.get("from") or date_range.get("start")
        totals["date_to"] = date_range.get("to") or date_range.get("end")
    else:
        totals["date_from"] = data.get("DATE_FROM") or data.get("START_DATE")
        totals["date_to"] = data.get("DATE_TO") or data.get("END_DATE")

    slots: dict[str, dict[str, Any]] = {}
    for slot in SLOTS:
        slot_data = data.get(slot) if isinstance(data.get(slot), dict) else None
        slots[slot] = {
            "roi": _extract_number(slot_data or {}, "ROI_%", "ROI"),
            "slump": slot_data.get("SLUMP") if isinstance(slot_data, dict) else None,
        }

    return {"totals": totals, "slots": slots}


def find_latest_bet_plan(plan_dir: Path) -> tuple[Optional[Path], Optional[str]]:
    candidates = list(plan_dir.glob("bet_plan_master_*.xlsx"))
    if not candidates:
        return None, None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    target_date = None
    match = re.search(r"bet_plan_master_(\d{8})", latest.name)
    if match:
        try:
            target_date = datetime.strptime(match.group(1), "%Y%m%d").date().isoformat()
        except ValueError:
            target_date = None

    return latest, target_date


def print_brief(summary: dict[str, Any], plan_path: Optional[Path], plan_date: Optional[str]) -> None:
    totals = summary.get("totals", {})
    date_from = totals.get("date_from") or "N/A"
    date_to = totals.get("date_to") or "N/A"

    print("=" * 70)
    print("📊 QUANT DAILY BRIEF")
    print("=" * 70)
    print(f"📅 Date Range: {date_from} → {date_to}")
    print(
        f"💰 Total Stake: {_format_amount(totals.get('stake'))} | "
        f"Return: {_format_amount(totals.get('return'))} | "
        f"P&L: {_format_amount(totals.get('pnl'))} | "
        f"ROI: {_format_roi(totals.get('roi'))}"
    )

    print("\nSLOT SNAPSHOT:")
    for slot in SLOTS:
        slot_info = summary.get("slots", {}).get(slot, {})
        roi_text = _format_roi(slot_info.get("roi"))
        slump = slot_info.get("slump")
        slump_text = "N/A" if slump is None else str(bool(slump))
        print(f"  {slot}: ROI={roi_text}, slump={slump_text}")

    print("\nNEXT BET PLAN:")
    if plan_path:
        print(f"  File : {plan_path.relative_to(Path.cwd())}")
        if plan_date:
            print(f"  Target Date: {plan_date}")
    else:
        print("  [QUANT-DAILY] ℹ No bet_plan_master_*.xlsx found yet.")
    print("=" * 70)


def main() -> int:
    try:
        project_root = Path(__file__).resolve().parent
        os.chdir(project_root)

        steps_ok = True
        steps_ok &= run_step(
            "Reality P&L Tracker",
            ["py", "-3.12", "bet_pnl_tracker.py", "--days", "30"],
        )

        steps_ok &= run_step(
            "Slot Health Signals",
            ["py", "-3.12", "quant_pnl_signals.py"],
        )

        steps_ok &= run_step(
            "ULTRA v5 Bet Engine",
            ["py", "-3.12", "precise_bet_engine.py"],
        )

        if not steps_ok:
            print("\n[QUANT-DAILY] ⚠ Some steps failed. Daily brief may be incomplete.")

        pnl_path = project_root / "logs" / "performance" / "quant_reality_pnl.json"
        pnl_data = load_quant_pnl_summary(pnl_path)
        if not pnl_data:
            print("[QUANT-DAILY] ℹ quant_reality_pnl.json not found or empty, skipping P&L brief.")
            summary = {"totals": {}, "slots": {}}
        else:
            summary = parse_pnl_brief(pnl_data)

        plan_dir = project_root / "predictions" / "bet_engine"
        plan_path, plan_date = find_latest_bet_plan(plan_dir)
        if plan_path is None:
            print("[QUANT-DAILY] ℹ No bet_plan_master_*.xlsx found yet.")

        print_brief(summary, plan_path, plan_date)
        return 0 if steps_ok else 1
    except Exception as exc:  # pragma: no cover - safety net
        print(f"[QUANT-DAILY] ❌ Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
