from typing import Dict

import csv
import json
from datetime import date, datetime
from pathlib import Path

from quant_stats_core import compute_topn_roi

TOP_N_VALUES = list(range(1, 11))


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
            best_n = max(roi_map, key=lambda k: roi_map.get(k, float("-inf")))
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
        "best_roi_overall": overall.get("best_roi"),
        "per_N_roi": {str(k): v for k, v in roi_by_n.items()},
        "per_slot_best_N": per_slot_best,
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


def main() -> int:
    target_window = 30
    summary = compute_topn_roi(window_days=target_window)
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
    for n in TOP_N_VALUES:
        roi_val = roi_map.get(n)
        if roi_val is None:
            continue
        print(f"N={n}  → ROI = {roi_val:+.1f}%")

    per_slot = summary.get("per_slot", {}) or {}
    if per_slot:
        print(f"\nPer-slot ROI ({effective_label}):")
        for slot, n_map in sorted(per_slot.items()):
            roi_by_n = n_map.get("roi_by_N", {}) if isinstance(n_map, dict) else {}
            parts = []
            for n in TOP_N_VALUES:
                if n in roi_by_n:
                    parts.append(f"Top{n}:{roi_by_n[n]:+.1f}%")
            if parts:
                print(f"{slot}: {' | '.join(parts)}")

    best_n = overall.get("best_N") if isinstance(overall, dict) else None
    best_roi = overall.get("best_roi") if isinstance(overall, dict) else None
    if best_n is not None:
        print(f"Best N = {best_n} with ROI = {best_roi:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

