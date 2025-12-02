"""Explore prediction hit history for a specific date."""
import argparse
from pathlib import Path

import pandas as pd

MEMORY_FILE = Path(__file__).resolve().parent / "logs" / "performance" / "script_hit_memory.xlsx"


def main():
    parser = argparse.ArgumentParser(description="Explore prediction hit memory by date")
    parser.add_argument("--date", help="Date to inspect (YYYY-MM-DD)")
    args = parser.parse_args()

    if not MEMORY_FILE.exists():
        print("❌ script_hit_memory.xlsx not found. Run prediction_hit_memory.py first.")
        return

    df = pd.read_excel(MEMORY_FILE)
    df.columns = [str(col).upper() for col in df.columns]
    if df.empty:
        print("⚠️ Hit memory file is empty")
        return

    if "DATE" not in df.columns:
        print("⚠️ DATE column missing; cannot filter by date")
        return

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    target_date = None
    if args.date:
        try:
            target_date = pd.to_datetime(args.date).date()
        except Exception:
            print("⚠️ Invalid date provided; using latest available")
    if target_date is None:
        target_date = df["DATE"].dropna().max()

    day_df = df[df["DATE"] == target_date]
    if day_df.empty:
        print(f"⚠️ No records found for {target_date}")
        return

    print(f"✅ Showing hits for {target_date}:")
    if {"SCRIPT", "SLOT"}.issubset(day_df.columns):
        grouped = day_df.groupby(["SCRIPT", "SLOT"])
        for (script, slot), group in grouped:
            print(f"\n{script} - {slot} ({len(group)} hits)")
            cols = [col for col in ["HIT_TYPE", "NUMBER", "PREDICTION"] if col in group.columns]
            print(group[cols].head())
    else:
        print(day_df.head())


if __name__ == "__main__":
    main()
