import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
import argparse
import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

from quant_stats_core import compute_topn_roi
import quant_paths

MAX_N = 40


def _write_csv(summary: Dict) -> None:
    output_path = Path("logs/performance/topn_roi_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overall = summary.get("overall", {}) or {}
    roi_by_n = overall.get("roi_by_N", {}) if isinstance(overall, dict) else {}
    per_slot = summary.get("per_slot", {}) or {}

    rows = []
    for n, roi in roi_by_n.items():
        rows.append({"N": n, "slot": "ALL", "ROI": roi})
    for slot, slot_map in per_slot.items():
        roi_map = slot_map.get("roi_by_N", {}) if isinstance(slot_map, dict) else {}
        for n, roi in roi_map.items():
            rows.append({"N": n, "slot": slot, "ROI": roi})
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan ROI performance across Top-N buckets.")
    parser.add_argument("--window_days", type=int, default=30, help="Window (in days) to evaluate ROI.")
    parser.add_argument("--max_n", type=int, default=20, help="Maximum N to scan (inclusive).")
    args = parser.parse_args()

    target_window = args.window_days
    max_n = min(max(args.max_n, 1), MAX_N)
    summary = compute_topn_roi(window_days=target_window, max_n=max_n)
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
    for n in display_ns:
        roi_val = roi_map.get(n)
        if roi_val is None:
            continue
        print(f"N={n}  → ROI = {roi_val:+.1f}%")

    per_slot = summary.get("per_slot", {}) or {}
    if per_slot:
        print(f"\nPer-slot ROI ({effective_label}):")
        for slot, n_map in sorted(per_slot.items()):
            roi_by_n = n_map.get("roi_by_N", {}) if isinstance(n_map, dict) else {}
            nums_by_n = n_map.get("numbers_by_N", {}) if isinstance(n_map, dict) else {}
            parts = []
            for n in range(1, min(max_n, 10) + 1):
                if n in roi_by_n:
                    parts.append(f"Top{n}:{roi_by_n[n]:+.1f}%")
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
    if best_n is not None:
        print(f"Best N = {best_n} with ROI = {best_roi:+.1f}%")
    _write_best_roi_json(summary, target_window=target_window)
    _write_numbers_summary(summary)
    _write_scan_json(summary, target_window=target_window, max_n=max_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

