"""Merged predictions view built from the latest SCR9 output."""
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import re

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def find_latest_predictions(base_dir: Path) -> Optional[Path]:
    pred_dir = base_dir / "predictions" / "deepseek_scr9"
    if not pred_dir.exists():
        return None
    files = sorted(pred_dir.glob("ultimate_predictions_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _parse_numbers(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_parse_numbers(item))
        return values

    value_str = str(value)
    if not value_str.strip():
        return []

    digits = re.findall(r"\d{1,2}", value_str)
    return [int(d) % 100 for d in digits]


def _extract_slot_numbers(df: pd.DataFrame, slot: str) -> list:
    slot_upper = slot.upper()
    columns_upper = {str(col).upper(): col for col in df.columns}

    # 1) Direct wide column (e.g., FRBD with comma-separated numbers)
    if slot_upper in columns_upper:
        col_name = columns_upper[slot_upper]
        for val in reversed(df[col_name].dropna().tolist()):
            parsed = _parse_numbers(val)
            if parsed:
                return parsed

    # 2) Split columns like FRBD_1, FRBD_2, FRBD_3
    sub_cols = [name for upper, name in columns_upper.items() if upper.startswith(f"{slot_upper}_")]
    if sub_cols:
        values = []
        for name in sorted(sub_cols):
            values.extend(_parse_numbers(df[name].dropna().iloc[-1] if not df[name].dropna().empty else None))
        if values:
            return values

    # 3) Long format with slot/number columns
    if {"SLOT", "NUMBER"}.issubset(columns_upper):
        slot_col = columns_upper["SLOT"]
        number_col = columns_upper["NUMBER"]
        slot_df = df[df[slot_col].astype(str).str.upper() == slot_upper]
        if "RANK" in columns_upper:
            rank_col = columns_upper["RANK"]
            slot_df = slot_df.sort_values(rank_col)
        nums = []
        for val in slot_df[number_col].tolist():
            nums.extend(_parse_numbers(val))
        if nums:
            return nums

    return []


def summarize_predictions(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        print(f"⚠️ Unable to read {file_path.name}: {exc}")
        return pd.DataFrame()

    summary_rows = []
    for slot in SLOTS:
        numbers = _extract_slot_numbers(df, slot)
        summary_rows.append({
            "slot": slot,
            "top_numbers": numbers[:3]
        })
    return pd.DataFrame(summary_rows)


def main():
    base_dir = Path(__file__).resolve().parent
    latest_file = find_latest_predictions(base_dir)

    if not latest_file:
        print("⚠️ No SCR9 prediction file found. Attempting to generate one...")
        subprocess.run([sys.executable, str(base_dir / "deepseek_scr9.py")], check=False)
        latest_file = find_latest_predictions(base_dir)

    if not latest_file:
        print("❌ No prediction files available after fallback.")
        return

    print(f"✅ Using prediction file: {latest_file.name}")
    summary_df = summarize_predictions(latest_file)
    if summary_df.empty:
        print("⚠️ Could not build a merged view from the file")
        return

    print("\n📊 Merged Prediction Snapshot:")
    for _, row in summary_df.iterrows():
        numbers = ", ".join(f"{int(n):02d}" for n in row["top_numbers"])
        print(f" • {row['slot']}: {numbers}")

    output_path = base_dir / "predictions" / "prediction_merge_latest.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        wide_data = {}
        for _, row in summary_df.iterrows():
            nums = row["top_numbers"]
            for idx in range(3):
                key = f"{row['slot']}_{idx + 1}"
                wide_data[key] = nums[idx] if idx < len(nums) else None

        export_df = summary_df.copy()
        export_df["numbers_str"] = export_df["top_numbers"].apply(lambda lst: ",".join(f"{int(n):02d}" for n in lst))

        with pd.ExcelWriter(output_path) as writer:
            export_df.to_excel(writer, sheet_name="by_slot", index=False)
            pd.DataFrame([wide_data]).to_excel(writer, sheet_name="wide", index=False)
        print(f"💾 Summary saved to {output_path}")
    except Exception as exc:
        print(f"⚠️ Unable to save merged summary: {exc}")


if __name__ == "__main__":
    main()
