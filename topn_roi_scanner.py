from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from quant_core.data_core import load_results_dataframe
from utils_2digit import to_2d_str

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]
TOP_N_VALUES = [2, 3, 4, 5, 6, 10]


def _normalise_number(value) -> str:
    if value is None:
        return ""
    try:
        return to_2d_str(int(value))
    except Exception:
        text = str(value).strip()
        if text.upper() == "XX" or text == "":
            return ""
        if text.isdigit():
            return to_2d_str(int(text))
        return ""


def _load_recent_results(days: int = 30) -> pd.DataFrame:
    df = load_results_dataframe()
    if df is None or df.empty:
        return pd.DataFrame(columns=["DATE", *SLOTS])
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    df = df.dropna(subset=["DATE"])
    unique_dates = sorted(df["DATE"].unique())
    if len(unique_dates) > days:
        cutoff_dates = set(unique_dates[-days:])
        df = df[df["DATE"].isin(cutoff_dates)]
    for slot in SLOTS:
        df[slot] = df[slot].apply(_normalise_number)
    return df


def _parse_predictions(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        results: List[str] = []
        for item in value:
            results.extend(_parse_predictions(item))
        return results
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    parts = [p for p in (
        text.replace(";", ",").replace("|", ",").replace("/", ",").split(",")
    ) if p.strip()]
    numbers: List[str] = []
    for part in parts:
        part = part.strip()
        if part.isdigit():
            numbers.append(to_2d_str(int(part)))
    return numbers


def _load_predictions() -> Dict:
    pred_dir = Path("predictions/deepseek_scr9")
    predictions: Dict = {}
    if not pred_dir.exists():
        return predictions
    files = sorted(pred_dir.glob("ultimate_predictions_*.xlsx"), key=lambda p: p.stat().st_mtime)
    for path in files:
        try:
            df = pd.read_excel(path)
        except Exception:
            continue
        df.columns = [str(c).strip().upper() for c in df.columns]
        date_col = next((c for c in df.columns if "DATE" in c), None)
        if not date_col:
            continue
        slot_columns = {
            slot: [c for c in df.columns if slot in c and "OPP" not in c and "DATE" not in c]
            for slot in SLOTS
        }
        for _, row in df.iterrows():
            date_val = pd.to_datetime(row.get(date_col), errors="coerce")
            if pd.isna(date_val):
                continue
            date_key = date_val.date()
            predictions.setdefault(date_key, {})
            for slot, cols in slot_columns.items():
                preds: List[str] = []
                for col in sorted(cols):
                    preds.extend(_parse_predictions(row.get(col)))
                # Remove duplicates while preserving order
                seen = set()
                ordered = []
                for num in preds:
                    if num in seen:
                        continue
                    seen.add(num)
                    ordered.append(num)
                if ordered:
                    predictions[date_key][slot] = ordered
    return predictions


def _summarise_roi(results_df: pd.DataFrame, predictions: Dict) -> pd.DataFrame:
    records: List[Dict] = []
    if results_df.empty:
        return pd.DataFrame(columns=["N", "slot", "total_days", "total_stake", "total_profit", "ROI"])

    result_map = {
        (row.DATE, slot): row[slot]
        for _, row in results_df.iterrows()
        for slot in SLOTS
        if isinstance(row[slot], str) and row[slot] != ""
    }
    dates = sorted({d for d in results_df["DATE"].unique() if d in predictions})

    for N in TOP_N_VALUES:
        overall_stake = overall_profit = hits_all = days_all = 0
        for slot in SLOTS:
            stake = profit = hits = days = 0
            for day in dates:
                pred_list = predictions.get(day, {}).get(slot, [])
                if not pred_list:
                    continue
                actual = result_map.get((day, slot))
                if not actual:
                    continue
                top_preds = pred_list[:N]
                stake_day = len(top_preds)
                if stake_day == 0:
                    continue
                days += 1
                stake += stake_day
                if actual in top_preds:
                    profit_day = 90 - stake_day
                    hits += 1
                else:
                    profit_day = -stake_day
                profit += profit_day
            roi = profit / stake if stake else 0.0
            records.append(
                {
                    "N": N,
                    "slot": slot,
                    "total_days": days,
                    "total_stake": stake,
                    "total_profit": profit,
                    "ROI": roi,
                }
            )
            overall_stake += stake
            overall_profit += profit
            hits_all += hits
            days_all += days
        overall_roi = overall_profit / overall_stake if overall_stake else 0.0
        records.append(
            {
                "N": N,
                "slot": "ALL",
                "total_days": days_all,
                "total_stake": overall_stake,
                "total_profit": overall_profit,
                "ROI": overall_roi,
            }
        )
    return pd.DataFrame(records)


def main() -> int:
    results_df = _load_recent_results(days=30)
    predictions = _load_predictions()

    summary_df = _summarise_roi(results_df, predictions)
    output_path = Path("logs/performance/topn_roi_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print("=== TOP-N ROI SCANNER (last 30 days) ===")
    if summary_df.empty:
        print("No data available for ROI scan.")
        return 0

    for N in TOP_N_VALUES:
        subset = summary_df[(summary_df["N"] == N) & (summary_df["slot"] == "ALL")]
        roi_val = subset["ROI"].iloc[0] if not subset.empty else 0.0
        days = int(subset["total_days"].iloc[0]) if not subset.empty else 0
        hits = 0
        for slot in SLOTS:
            slot_row = summary_df[(summary_df["N"] == N) & (summary_df["slot"] == slot)]
            if not slot_row.empty:
                hits += int((slot_row["total_profit"] > -slot_row["total_stake"]).sum())
        print(f"N={N}: overall ROI={roi_val:+.1%} (days={days})")
        for slot in SLOTS:
            slot_row = summary_df[(summary_df["N"] == N) & (summary_df["slot"] == slot)]
            roi_slot = slot_row["ROI"].iloc[0] if not slot_row.empty else 0.0
            days_slot = int(slot_row["total_days"].iloc[0]) if not slot_row.empty else 0
            print(f"  {slot}: ROI={roi_slot:+.1%} (days={days_slot})")
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
