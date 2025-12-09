"""Quick combined review console leveraging existing summaries."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_json_safely(path: Path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_golden_overlay(golden_data):
    if not isinstance(golden_data, dict):
        return [], None

    hero_numbers = []

    raw_heroes = (
        golden_data.get("hero_numbers")
        or golden_data.get("HERO_NUMBERS")
        or (golden_data.get("digits") or {}).get("hero_numbers")
        or (golden_data.get("summary") or {}).get("hero_numbers")
    )

    if isinstance(raw_heroes, str):
        parts = [p.strip() for p in raw_heroes.replace(";", ",").split(",") if p.strip()]
        hero_numbers = [p.zfill(2) for p in parts if p.replace(" ", "").isdigit()]
    elif isinstance(raw_heroes, list):
        hero_numbers = [str(x).strip().zfill(2) for x in raw_heroes]

    top_cross = None
    summary = golden_data.get("summary") if isinstance(golden_data.get("summary"), dict) else {}
    cross_patterns = golden_data.get("cross_slot_pairs") if isinstance(golden_data.get("cross_slot_pairs"), dict) else {}
    pattern_key = summary.get("top_cross_pattern") or golden_data.get("top_cross_pattern")

    def _parse_pattern(pattern: str):
        if not isinstance(pattern, str):
            return None, None
        normalized = pattern.replace("→", "->")
        if "->" in normalized:
            from_slot, to_slot = [p.strip() for p in normalized.split("->", 1)]
            return from_slot or None, to_slot or None
        return None, None

    if isinstance(pattern_key, str) and pattern_key.upper() != "NONE":
        from_slot, to_slot = _parse_pattern(pattern_key)
        pattern_info = cross_patterns.get(pattern_key) if isinstance(cross_patterns, dict) else None
        hits = pattern_info.get("hits") if isinstance(pattern_info, dict) else None
        avg_rank = pattern_info.get("avg_rank") if isinstance(pattern_info, dict) else None

        if from_slot or to_slot or pattern_info:
            top_cross = {
                "from_slot": str(from_slot) if from_slot else None,
                "to_slot": str(to_slot) if to_slot else None,
                "hits": hits,
                "avg_rank": avg_rank,
            }

    return hero_numbers, top_cross


def extract_near_miss_pressure(near_miss_data, top_n: int = 10):
    if not isinstance(near_miss_data, dict):
        return []

    agg_section = None
    for key in ["aggregate", "AGGREGATE", "aggregate_candidates"]:
        if key in near_miss_data:
            agg_section = near_miss_data[key]
            break

    if agg_section is None:
        return []

    agg = agg_section
    if isinstance(agg_section, dict):
        agg = (
            agg_section.get("top_near_miss_candidates")
            or agg_section.get("candidates")
            or agg_section.get("top_candidates")
            or agg_section
        )

    result = []
    if isinstance(agg, list):
        for item in agg:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                num, cnt = item[0], item[1]
            elif isinstance(item, dict):
                num = item.get("number") or item.get("num") or item.get("N")
                cnt = item.get("count") or item.get("C") or item.get("freq")
            else:
                continue
            if num is None or cnt is None:
                continue
            try:
                num_str = str(int(num)).zfill(2)
                cnt_int = int(cnt)
            except Exception:
                continue
            result.append((num_str, cnt_int))

    result.sort(key=lambda x: x[1], reverse=True)
    return result[:top_n]


def build_today_numbers(plan_summary, final_plan):
    slots = ["FRBD", "GZBD", "GALI", "DSWR"]
    today_numbers = {slot: set() for slot in slots}

    if final_plan and isinstance(final_plan, dict):
        slot_data = final_plan.get("slots", {}) or {}
        for slot, data in slot_data.items():
            if slot not in today_numbers:
                continue
            numbers = data.get("numbers") or []
            for num_entry in numbers:
                if isinstance(num_entry, dict):
                    num_val = num_entry.get("num") or num_entry.get("number")
                    if num_val is not None:
                        today_numbers[slot].add(str(num_val).zfill(2))
            for digit_key in ["andar_digit", "bahar_digit"]:
                digit_val = data.get(digit_key)
                if digit_val is not None and str(digit_val).strip():
                    today_numbers[slot].add(str(digit_val).strip().zfill(2))
        return today_numbers

    if plan_summary and getattr(plan_summary, "slots", None):
        for slot in plan_summary.slots:
            for item in slot.main_numbers:
                if isinstance(item, str):
                    num_part = item.split("(", 1)[0].strip()
                    if num_part:
                        today_numbers[slot.slot].add(num_part.zfill(2))
            if slot.andar:
                today_numbers[slot.slot].add(str(slot.andar).zfill(2))
            if slot.bahar:
                today_numbers[slot.slot].add(str(slot.bahar).zfill(2))
    return today_numbers


def build_slot_overlay(today_numbers, hero_numbers, near_miss_pressure):
    hero_set = set(hero_numbers)
    pressure_set = {n for (n, _c) in near_miss_pressure}

    per_slot_hero = {}
    per_slot_pressure = {}
    for slot, nums in today_numbers.items():
        normalized = {str(x).zfill(2) for x in nums}
        per_slot_hero[slot] = sorted(normalized & hero_set)
        per_slot_pressure[slot] = sorted(normalized & pressure_set)
    return per_slot_hero, per_slot_pressure


def main():
    base_dir = Path(__file__).resolve().parent
    print("📝 Launching quant_daily_brief for snapshot...")
    subprocess.run([sys.executable, str(base_dir / "quant_daily_brief.py")], check=False)
    print("ℹ️ For ROI details see roi_summary.py; for pattern intelligence see pattern_intelligence_engine.py")

    try:
        import quant_daily_brief as qdb

        bet_date = qdb.find_latest_bet_date()
        results_df = qdb.load_results_df()
        mode, target_date = qdb.decide_mode(bet_date, "auto", results_df)
        plan_summary = qdb.load_plan_for_mode(mode, bet_date, target_date)
        final_plan = qdb.load_final_bet_plan_for_date(target_date) if mode == "NEXT_DAY" else None
    except Exception:
        bet_date = None
        target_date = None
        plan_summary = None
        final_plan = None

    perf_dir = base_dir / "logs" / "performance"
    golden_json_path = perf_dir / "golden_block_insights.json"
    near_miss_path = perf_dir / "near_miss_report.json"

    golden_data = load_json_safely(golden_json_path)
    near_miss_data = load_json_safely(near_miss_path)

    hero_numbers, top_cross = extract_golden_overlay(golden_data) if golden_data else ([], None)
    near_miss_pressure = extract_near_miss_pressure(near_miss_data) if near_miss_data else []

    today_numbers = build_today_numbers(plan_summary, final_plan)
    per_slot_hero, per_slot_pressure = build_slot_overlay(today_numbers, hero_numbers, near_miss_pressure)

    print("\n5️⃣ GOLDEN PHYSICS OVERLAY")
    print("   (Informational layer based on golden days + near-miss pressure)")
    print("------------------------------------------------------------------")

    if not golden_data:
        print("   ⚠️ Golden block insights file not found. Run golden_block_finder.py first.")
    else:
        if hero_numbers:
            print(f"   Hero numbers (global): {', '.join(hero_numbers)}")
        else:
            print("   Hero numbers (global): -")

        print("   Hero numbers present in today’s plan:")
        for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_heroes = per_slot_hero.get(slot, [])
            if slot_heroes:
                print(f"     • {slot}: {', '.join(slot_heroes)}")
            else:
                print(f"     • {slot}: -")

        if near_miss_pressure:
            top_pressure_str = ", ".join([f"{n}({c})" for (n, c) in near_miss_pressure])
            print(f"   Near-miss pressure (aggregate, last 30 days): {top_pressure_str}")
            print("   Near-miss numbers present in today’s plan:")
            for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                slot_p = per_slot_pressure.get(slot, [])
                if slot_p:
                    print(f"     • {slot}: {', '.join(slot_p)}")
                else:
                    print(f"     • {slot}: -")
        else:
            if near_miss_data is None:
                print("   Near-miss pressure: file not found. Run near_miss_analyzer.py.")
            else:
                print("   Near-miss pressure: no aggregate candidates detected.")

        if top_cross and (top_cross.get("from_slot") or top_cross.get("to_slot")):
            fs = top_cross.get("from_slot") or "?"
            ts = top_cross.get("to_slot") or "?"
            hits = top_cross.get("hits")
            avg_rank = top_cross.get("avg_rank")
            print("   Golden cross-slot pattern:")
            print(f"     • {fs} → {ts} | Hits: {hits}, Avg rank: {avg_rank}")
        else:
            print("   Golden cross-slot pattern: not available in JSON.")

    print("------------------------------------------------------------------")
    print("Note: Golden overlay is advisory only; core plan and stakes remain unchanged.")


if __name__ == "__main__":
    main()
