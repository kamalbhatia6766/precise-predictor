"""Generic data summary utility for real results."""
from pathlib import Path

import pandas as pd

import quant_data_core

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def main():
    try:
        df = quant_data_core.load_results_dataframe()
    except Exception as exc:
        print(f"❌ Unable to load results: {exc}")
        return

    if df is None or df.empty:
        print("⚠️ No real results available")
        return

    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df.dropna(subset=["DATE"])
    date_range = (df["DATE"].min().date(), df["DATE"].max().date())

    print(f"📅 Date range: {date_range[0]} → {date_range[1]} ({df['DATE'].nunique()} days)")

    for slot in SLOTS:
        if slot not in df.columns:
            continue
        numbers = pd.to_numeric(df[slot], errors="coerce").dropna().astype(int) % 100
        print(f"\n🎯 {slot} — {len(numbers)} records")
        bins = [0, 25, 50, 75, 100]
        labels = ["00-24", "25-49", "50-74", "75-99"]
        dist = pd.cut(numbers, bins=bins, labels=labels, right=False, include_lowest=True).value_counts().sort_index()
        for label, count in dist.items():
            print(f"   {label}: {count}")


if __name__ == "__main__":
    main()
