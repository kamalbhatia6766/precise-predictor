import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import quant_paths
import quant_data_core

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]

MAX_N = 40
UNIT_STAKE = 10
FULL_PAYOUT_PER_UNIT = 90
NEAR_HIT_WEIGHT = 0.3


def safe(value):
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def _to_number(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        num = int(str(value).strip()) % 100
        return f"{num:02d}"
    except Exception:
        return None


def _normalise_slot(value: object) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper()
    mapping = {"1": "FRBD", "2": "GZBD", "3": "GALI", "4": "DSWR"}
    normalised = mapping.get(text, text)
    return normalised if normalised in SLOTS else None


def _write_csv(summary: Dict) -> None:
    output_path = Path("logs/performance/topn_roi_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary.get("overall", {}) or {}
    roi_by_n = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
    per_slot = summary.get("per_slot", {}) or {}

    rows = []
    for n, roi in roi_by_n.items():
        rows.append({"N": n, "slot": "ALL", "ROI": safe(roi)})
    for slot, slot_map in per_slot.items():
        roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
        for n, roi in roi_map.items():
            rows.append({"N": n, "slot": slot, "ROI": safe(roi)})
    if not rows:
        return
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_profile_json(summary: Dict, target_window: int) -> None:
    output_path = Path("logs/performance/topn_roi_profile.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overall = summary.get("overall", {}) or {}
    roi_by_n = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
    per_slot = summary.get("per_slot", {}) or {}

    per_slot_best = {}
    per_slot_roi = {}
    for slot, slot_map in per_slot.items():
        roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
        if roi_map:
            best_roi = max(roi_map.values())
            best_n = min([n for n, roi in roi_map.items() if roi == best_roi])
            per_slot_best[slot] = best_n
            per_slot_roi[slot] = {
                "Top1": roi_map.get(1),
                "Top2": roi_map.get(2),
                "Top3": roi_map.get(3),
                "Top4": roi_map.get(4),
                "Top5": roi_map.get(5),
                "Top10": roi_map.get(10),
                "roi_by_N": {str(k): v for k, v in roi_map.items()},
            }

    payload = {
        "window_days": target_window,
        "best_N_overall": overall.get("best_N"),
        "overall_best_N": overall.get("best_N"),
        "best_roi_overall": overall.get("best_roi"),
        "per_N_roi": {str(k): v for k, v in roi_by_n.items()},
        "per_slot_best_N": per_slot_best,
        "best_n_per_slot": per_slot_best,
        "per_slot_roi": per_slot_roi,
        "window_start": summary.get("window_start"),
        "window_end": summary.get("window_end"),
        "available_days": summary.get("available_days"),
    }

    output_path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _json_default(obj):
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _write_best_roi_json(summary: Dict, target_window: int) -> None:
    try:
        base_dir = quant_paths.get_base_dir()
        output_path = Path(base_dir) / "logs" / "performance" / "topn_roi_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overall = summary.get("overall", {}) or {}
        per_slot_summary = summary.get("per_slot", {}) or {}
        roi_by_n_overall = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}

        overall_best_n = overall.get("best_N") if isinstance(overall, dict) else None
        overall_best_roi = overall.get("best_roi") if isinstance(overall, dict) else None

        per_slot: Dict[str, Dict[str, float]] = {}
        best_n_per_slot: Dict[str, int] = {}
        roi_per_slot: Dict[str, Dict[str, float]] = {}
        for slot, slot_map in per_slot_summary.items():
            roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
            if not roi_map:
                continue
            best_roi = max(roi_map.values())
            best_n = min([n for n, roi in roi_map.items() if roi == best_roi])
            per_slot[slot] = {"best_N": int(best_n), "roi": float(roi_map.get(best_n, 0.0))}
            best_n_per_slot[slot] = int(best_n)
            roi_per_slot[slot] = {str(k): v for k, v in roi_map.items()}

        payload = {
            "window_days": summary.get("available_days", target_window),
            "overall": {"best_N": overall_best_n, "roi": overall_best_roi},
            "per_slot": per_slot,
            "timestamp": datetime.now().isoformat(),
            "overall_best_N": overall_best_n,
            "best_n_per_slot": best_n_per_slot,
            "roi_per_n": {
                "overall": {str(k): v for k, v in roi_by_n_overall.items()},
                "per_slot": roi_per_slot,
            },
        }

        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    except Exception as exc:
        print(f"[topn_roi_scanner] Warning: unable to write topn_roi_summary.json: {exc}")


def _load_results_map() -> Dict[date, Dict[str, str]]:
    df = quant_data_core.load_results_dataframe()
    if df is None or df.empty:
        return {}

    df = df.copy()
    df["DATE"] = pd.to_datetime(df.get("DATE"), errors="coerce")
    df = df.dropna(subset=["DATE"])
    results: Dict[date, Dict[str, str]] = {}
    for _, row in df.iterrows():
        day = row["DATE"].date()
        slot_map: Dict[str, str] = results.setdefault(day, {})
        for slot in SLOTS:
            num = _to_number(row.get(slot))
            if num:
                slot_map[slot] = num
    return results


def _load_bet_plans(max_n: int) -> Dict[date, Dict[str, List[Tuple[str, float]]]]:
    base_dir = quant_paths.get_base_dir()
    bet_dir = Path(base_dir) / "predictions" / "bet_engine"
    plans: Dict[date, Dict[str, List[Tuple[str, float]]]] = {}
    if not bet_dir.exists():
        return plans

    for path in bet_dir.glob("bet_plan_master_*.xlsx"):
        try:
            date_str = path.stem.replace("bet_plan_master_", "")
            plan_date = datetime.strptime(date_str, "%Y%m%d").date()
        except Exception:
            continue

        try:
            bets = pd.read_excel(path, sheet_name="bets")
        except Exception:
            continue

        slot_map: Dict[str, List[Tuple[str, float]]] = plans.setdefault(plan_date, {})
        for idx, row in bets.iterrows():
            layer = str(row.get("layer_type", "")).strip().upper()
            if layer != "MAIN":
                continue
            slot = _normalise_slot(row.get("slot"))
            if not slot:
                continue
            number = _to_number(row.get("number") if "number" in row else row.get("number_or_digit"))
            if not number:
                continue
            try:
                stake_val = float(row.get("stake", 0) or 0)
            except Exception:
                stake_val = 0.0
            rank_val = row.get("source_rank")
            rank_num = None
            try:
                rank_num = float(rank_val)
            except Exception:
                rank_num = None
            slot_map.setdefault(slot, []).append((number, stake_val, rank_num, idx))

        # Stabilise ordering by explicit rank -> stake -> original index
        for slot, items in slot_map.items():
            items.sort(key=lambda t: (t[2] if t[2] is not None else float("inf"), -safe(t[1]), t[3]))
            slot_map[slot] = [(num, stake) for num, stake, _, _ in items][:max_n]
    return plans


def _write_numbers_summary(summary: Dict) -> None:
    try:
        base_dir = quant_paths.get_base_dir()
        output_path = Path(base_dir) / "logs" / "performance" / "topn_roi_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        window_start = summary.get("window_start")
        window_end = summary.get("window_end")
        available_days = summary.get("available_days")

        per_slot_summary = summary.get("per_slot", {}) or {}
        payload_slots: Dict[str, Dict] = {}
        for slot, slot_map in per_slot_summary.items():
            roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
            num_map = slot_map.get("numbers_by_N", {}) if isinstance(slot_map, dict) else {}
            best_n = slot_map.get("best_N")
            payload_slots[slot] = {
                "best_N": best_n,
                "best_roi": slot_map.get("best_roi"),
                "roi_by_N": {str(k): v for k, v in roi_map.items()},
                "numbers_by_N": {str(k): v for k, v in num_map.items()},
            }

        payload = {
            "window": {
                "start": window_start.isoformat() if hasattr(window_start, "isoformat") else window_start,
                "end": window_end.isoformat() if hasattr(window_end, "isoformat") else window_end,
                "days": int(available_days) if available_days is not None else None,
            },
            "slots": payload_slots,
        }

        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    except Exception as exc:
        print(f"[topn_roi_scanner] Warning: unable to write topn_roi_summary.json: {exc}")


def _write_scan_json(summary: Dict, target_window: int, max_n: int) -> None:
    try:
        base_dir = quant_paths.get_base_dir()
        output_path = Path(base_dir) / "logs" / "performance" / "topn_roi_scan.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overall = summary.get("overall", {}) or {}
        per_slot_summary = summary.get("per_slot", {}) or {}

        def _roi_map(raw: Dict[int, float]) -> Dict[str, float]:
            return {str(k): v for k, v in raw.items() if k <= max_n}

        numbers_by_slot = summary.get("numbers_by_slot", {}) if isinstance(summary, dict) else {}

        slots_payload: Dict[str, Dict] = {}
        for slot, slot_map in per_slot_summary.items():
            roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
            slots_payload[slot] = {
                "best_N": slot_map.get("best_N"),
                "best_roi": slot_map.get("best_roi"),
                "roi_by_N": _roi_map(roi_map),
                "numbers_by_N": {str(k): v for k, v in numbers_by_slot.get(slot, {}).items()},
            }

        payload = {
            "window_days_requested": target_window,
            "window_days_used": summary.get("window_days_used") or summary.get("available_days"),
            "window_start": summary.get("window_start"),
            "window_end": summary.get("window_end"),
            "overall": {
                "best_N": overall.get("best_N"),
                "best_roi": overall.get("best_roi"),
                "roi_by_N": _roi_map(overall.get("roi_by_N", {})),
            },
            "slots": slots_payload,
        }

        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    except Exception as exc:
        print(f"[topn_roi_scanner] Warning: unable to write topn_roi_scan.json: {exc}")


def _write_topn_policy(summary: Dict, max_n: int) -> None:
    payload, debug_rows = _prepare_topn_policy_data(summary, max_n=max_n)
    try:
        base_dir = quant_paths.get_base_dir()
        output_path = Path(base_dir) / "data" / "topn_policy.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        print(f"Saved Top-N policy to {output_path}")
    except Exception as exc:
        print(f"[topn_roi_scanner] Warning: unable to write topn_policy.json: {exc}")
    return debug_rows


def _prepare_topn_policy_data(summary: Dict, max_n: int) -> Tuple[Dict[str, Dict[str, object]], List[Dict[str, object]]]:
    per_slot = summary.get("per_slot", {}) or {}
    overall = summary.get("overall", {}) or {}
    payload: Dict[str, Dict[str, object]] = {}
    debug_rows: List[Dict[str, object]] = []
    window_end = summary.get("window_end")
    window_days = summary.get("window_days_used") or summary.get("available_days") or summary.get("window_days_requested")

    def _parse_roi_map(raw_map: Dict) -> Dict[int, float]:
        return {int(k): v for k, v in raw_map.items()} if isinstance(raw_map, dict) else {}

    for slot, slot_map in per_slot.items():
        roi_map = _parse_roi_map(slot_map.get("roi_by_N", {}))
        days_map = _parse_roi_map(slot_map.get("days_by_N", {}))
        hits_map = _parse_roi_map(slot_map.get("hits_by_N", {}))
        near_hits_map = _parse_roi_map(slot_map.get("near_hits_by_N", {}))
        entry, slot_debug = _compute_slot_policy(
            slot,
            roi_map,
            days_map,
            hits_map,
            near_hits_map,
            max_n,
            window_end,
        )
        if roi_map:
            entry["roi_by_N"] = {str(k): v for k, v in roi_map.items()}
        payload[slot] = entry
        debug_rows.extend(slot_debug)

    roi_map_all = _parse_roi_map(overall.get("roi_by_N", {}))
    days_map_all = _parse_roi_map(overall.get("days_by_N", {}))
    hits_map_all = _parse_roi_map(overall.get("hits_by_N", {}))
    near_hits_map_all = _parse_roi_map(overall.get("near_hits_by_N", {}))
    overall_entry, overall_debug = _compute_slot_policy(
        "ALL", roi_map_all, days_map_all, hits_map_all, near_hits_map_all, max_n, window_end
    )
    if roi_map_all:
        overall_entry["roi_by_N"] = {str(k): v for k, v in roi_map_all.items()}
    payload["ALL"] = overall_entry
    debug_rows.extend(overall_debug)

    payload["meta"] = {
        "as_of": window_end.isoformat() if hasattr(window_end, "isoformat") else window_end,
        "window_days": window_days,
        "generated": datetime.now().isoformat(),
        "overall_roi_by_N": {str(k): v for k, v in roi_map_all.items()},
    }

    return payload, debug_rows


def _compute_slot_policy(
    slot: str,
    roi_map: Dict[int, float],
    days_map: Dict[int, float],
    hits_map: Dict[int, float],
    near_hits_map: Dict[int, float],
    max_n: int,
    window_end,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    roi_curve_exact: Dict[int, float] = {}
    roi_curve_near: Dict[int, float] = {}
    debug_rows: List[Dict[str, object]] = []
    date_label = window_end.isoformat() if hasattr(window_end, "isoformat") else str(window_end or "")

    for n in range(1, max_n + 1):
        days_played = int(days_map.get(n, 0) or 0)
        hits_in_top_n = int(hits_map.get(n, 0) or 0)
        near_hits_in_top_n = int(near_hits_map.get(n, 0) or 0)

        total_stake = days_played * n * UNIT_STAKE
        total_return_exact = hits_in_top_n * (FULL_PAYOUT_PER_UNIT * UNIT_STAKE)
        near_return = near_hits_in_top_n * (NEAR_HIT_WEIGHT * FULL_PAYOUT_PER_UNIT * UNIT_STAKE)
        total_return_nearaware = total_return_exact + near_return

        roi_exact = ((total_return_exact - total_stake) / total_stake) * 100.0 if total_stake else 0.0
        roi_nearaware = ((total_return_nearaware - total_stake) / total_stake) * 100.0 if total_stake else 0.0

        roi_curve_exact[n] = roi_exact
        roi_curve_near[n] = roi_nearaware

        debug_rows.append(
            {
                "date": date_label,
                "slot": slot,
                "N": n,
                "band_type": "EXACT",
                "total_bet": total_stake,
                "total_return": total_return_exact,
                "net_pnl": total_return_exact - total_stake,
                "roi_pct": roi_exact,
            }
        )
        debug_rows.append(
            {
                "date": date_label,
                "slot": slot,
                "N": n,
                "band_type": "NEARAWARE",
                "total_bet": total_stake,
                "total_return": total_return_nearaware,
                "net_pnl": total_return_nearaware - total_stake,
                "roi_pct": roi_nearaware,
            }
        )

    best_n_exact, roi_best_exact = _select_best_band(roi_curve_exact)
    best_n_near, roi_best_near = _select_best_band(roi_curve_near)

    if roi_best_near is not None and roi_best_near > (roi_best_exact or float("-inf")) and roi_best_near > 0:
        final_best_n = best_n_near
        roi_final_best = roi_best_near
    else:
        final_best_n = best_n_exact
        roi_final_best = roi_best_exact

    entry = {
        "roi_curve_exact": {str(k): v for k, v in roi_curve_exact.items()},
        "roi_curve_nearaware": {str(k): v for k, v in roi_curve_near.items()},
        "best_n_exact": best_n_exact,
        "roi_best_exact": roi_best_exact,
        "best_n_nearaware": best_n_near,
        "roi_best_nearaware": roi_best_near,
        "final_best_n": final_best_n,
        "roi_final_best": roi_final_best,
    }

    return entry, debug_rows


def _select_best_band(roi_curve: Dict[int, float]) -> Tuple[int, float]:
    best_n = None
    best_roi = None
    for n, roi in roi_curve.items():
        if best_roi is None or roi > best_roi or (roi == best_roi and (best_n is None or n < best_n)):
            best_roi = roi
            best_n = n
    return best_n, best_roi


def _write_debug_csv(rows: List[Dict[str, object]]) -> None:
    output_path = Path("logs/performance/topn_roi_debug.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("date,slot,N,band_type,total_bet,total_return,net_pnl,roi_pct\n")
        return
    fieldnames = ["date", "slot", "N", "band_type", "total_bet", "total_return", "net_pnl", "roi_pct"]
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _compute_roi(window_days: int, max_n: int) -> Dict:
    results_map = _load_results_map()
    bet_plans = _load_bet_plans(max_n=max_n)
    matched_dates = sorted(set(results_map.keys()) & set(bet_plans.keys()))
    if not matched_dates:
        return {}

    latest = max(matched_dates)
    cutoff = latest - timedelta(days=window_days - 1)
    window_dates = [d for d in matched_dates if d >= cutoff]

    per_slot: Dict[str, Dict[str, Dict[int, float]]] = {}
    per_slot_numbers: Dict[str, Dict[int, Dict[str, int]]] = {}
    overall_stake: Dict[int, float] = {n: 0.0 for n in range(1, max_n + 1)}
    overall_return: Dict[int, float] = {n: 0.0 for n in range(1, max_n + 1)}
    overall_days: Dict[int, int] = {n: 0 for n in range(1, max_n + 1)}
    overall_hits: Dict[int, int] = {n: 0 for n in range(1, max_n + 1)}

    for slot in SLOTS:
        per_slot[slot] = {
            "roi_by_N": {},
            "days_by_N": {n: 0 for n in range(1, max_n + 1)},
            "hits_by_N": {n: 0 for n in range(1, max_n + 1)},
            "near_hits_by_N": {n: 0 for n in range(1, max_n + 1)},
        }
        per_slot_numbers[slot] = {n: {} for n in range(1, max_n + 1)}

    for day in window_dates:
        results = results_map.get(day, {})
        bets_for_day = bet_plans.get(day, {})
        for slot in SLOTS:
            picks = bets_for_day.get(slot, [])
            if not picks:
                continue
            actual = results.get(slot)
            for n in range(1, max_n + 1):
                chosen = picks[:n]
                if not chosen:
                    continue
                stake_sum = sum(stake for _, stake in chosen)
                if stake_sum <= 0:
                    continue
                per_slot[slot]["days_by_N"][n] += 1
                overall_days[n] += 1
                overall_stake[n] += stake_sum

                hit_return = 0.0
                if actual:
                    hit_return = sum(stake for num, stake in chosen if num == actual) * FULL_PAYOUT_PER_UNIT
                    if hit_return > 0:
                        per_slot[slot]["hits_by_N"][n] += 1
                        overall_hits[n] += 1

                per_slot[slot].setdefault("stake_by_N", {n: 0 for n in range(1, max_n + 1)})[n] += stake_sum
                per_slot[slot].setdefault("return_by_N", {n: 0 for n in range(1, max_n + 1)})[n] += hit_return
                overall_return[n] += hit_return

                freq_map = per_slot_numbers[slot].setdefault(n, {})
                for num, _ in chosen:
                    freq_map[num] = freq_map.get(num, 0) + 1

    for slot, slot_maps in per_slot.items():
        roi_by_n: Dict[int, float] = {}
        best_n = None
        best_roi = None
        for n in range(1, max_n + 1):
            stake_val = slot_maps.get("stake_by_N", {}).get(n, 0.0)
            ret_val = slot_maps.get("return_by_N", {}).get(n, 0.0)
            roi_val = ((ret_val - stake_val) / stake_val * 100.0) if stake_val else 0.0
            roi_by_n[n] = safe(roi_val)
            if best_roi is None or roi_val > best_roi or (roi_val == best_roi and (best_n is None or n < best_n)):
                best_roi = roi_val
                best_n = n
            slot_maps["roi_by_N"][n] = roi_by_n[n]
        slot_maps["best_N"] = best_n
        slot_maps["best_roi"] = safe(best_roi) if best_roi is not None else None
        numbers_by_n: Dict[int, List[str]] = {}
        for n, freq in per_slot_numbers[slot].items():
            ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
            numbers_by_n[n] = [num for num, _ in ranked][:n]
        slot_maps["numbers_by_N"] = numbers_by_n

    roi_overall: Dict[int, float] = {}
    best_overall_n = None
    best_overall_roi = None
    for n in range(1, max_n + 1):
        stake_val = overall_stake[n]
        ret_val = overall_return[n]
        roi_val = ((ret_val - stake_val) / stake_val * 100.0) if stake_val else 0.0
        roi_overall[n] = safe(roi_val)
        if best_overall_roi is None or roi_val > best_overall_roi or (roi_val == best_overall_roi and (best_overall_n is None or n < best_overall_n)):
            best_overall_roi = roi_val
            best_overall_n = n

    return {
        "window_start": min(window_dates),
        "window_end": latest,
        "available_days": len(window_dates),
        "window_days_used": len(window_dates),
        "overall": {
            "roi_by_N": roi_overall,
            "best_N": best_overall_n,
            "best_roi": safe(best_overall_roi) if best_overall_roi is not None else None,
            "days_by_N": overall_days,
            "hits_by_N": overall_hits,
        },
        "per_slot": per_slot,
        "numbers_by_slot": {slot: maps.get("numbers_by_N", {}) for slot, maps in per_slot.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan ROI performance across Top-N buckets.")
    parser.add_argument("--window_days", type=int, default=30, help="Window (in days) to evaluate ROI.")
    parser.add_argument("--max_n", type=int, default=20, help="Maximum N to scan (inclusive).")
    args = parser.parse_args()

    target_window = args.window_days
    max_n = min(max(args.max_n, 1), MAX_N)
    summary = _compute_roi(window_days=target_window, max_n=max_n)
    if not summary:
        print("No data available for ROI scan.")
        return 0

    _write_csv(summary)
    _write_profile_json(summary, target_window=target_window)

    start = summary.get("window_start")
    end = summary.get("window_end")
    days_used = summary.get("available_days", 0)
    window_note = "" if days_used >= target_window else f" (only {days_used} days available)"
    effective_label = f"effective {days_used}d" if days_used else "effective window"
    print(f"=== TOP-N ROI SCANNER (requested {target_window}d) ===")
    print(f"Window: {start} to {end}, days={days_used}{window_note}")

    overall = summary.get("overall", {}) or {}
    roi_map = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
    display_ns: List[int] = list(range(1, max_n + 1))
    roi_values = [v for v in roi_map.values() if v is not None]
    min_days_required = 15
    all_zero_roi = days_used < min_days_required or not roi_values

    if all_zero_roi:
        warm_note = (
            "Top-N ROI module warming up – insufficient matched bet/results days. "
            f"Matched days: {days_used} (need >= {min_days_required})."
        )
        print(warm_note)
    else:
        for n in display_ns:
            roi_val = roi_map.get(n)
            if roi_val is None:
                continue
            print(f"N={n}  → ROI = {safe(roi_val):+.1f}%")

    per_slot = summary.get("per_slot", {}) or {}
    if not all_zero_roi and per_slot:
        print(f"\nPer-slot ROI ({effective_label}):")
        for slot, n_map in sorted(per_slot.items()):
            roi_by_n = n_map.get("roi_by_N", {}) if isinstance(n_map, dict) else {}
            nums_by_n = n_map.get("numbers_by_N", {}) if isinstance(n_map, dict) else {}
            parts = []
            for n in range(1, min(max_n, 10) + 1):
                if n in roi_by_n:
                    parts.append(f"Top{n}:{safe(roi_by_n[n]):+.1f}%")
            if parts:
                print(f"{slot}: {' | '.join(parts)}")

            best_n = n_map.get("best_N")
            if best_n:
                print(f"    Best N: {best_n}")
            if nums_by_n:
                normalized_keys = []
                for k in nums_by_n.keys():
                    try:
                        normalized_keys.append(int(k))
                    except Exception:
                        continue
                key_n = best_n if best_n in normalized_keys else (min(normalized_keys) if normalized_keys else None)
                if key_n:
                    top_numbers = nums_by_n.get(key_n) or nums_by_n.get(str(key_n), [])
                    print(f"    Top{key_n} numbers (last 30d): {', '.join(top_numbers)}")

    best_n = overall.get("best_N") if isinstance(overall, dict) else None
    best_roi = overall.get("best_roi") if isinstance(overall, dict) else None
    if not all_zero_roi and best_n is not None:
        print(f"Best N = {best_n} with ROI = {best_roi:+.1f}%")
    _write_best_roi_json(summary, target_window=target_window)
    _write_numbers_summary(summary)
    _write_scan_json(summary, target_window=target_window, max_n=max_n)
    debug_rows = _write_topn_policy(summary, max_n=max_n)
    _write_debug_csv(debug_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

