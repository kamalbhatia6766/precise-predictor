"""Core access to prediction hit memory."""
import json
from pathlib import Path

import pandas as pd

MEMORY_FILE = Path(__file__).resolve().parent / "logs" / "performance" / "script_hit_memory.xlsx"


def main():
    if not MEMORY_FILE.exists():
        print("❌ script_hit_memory.xlsx not found. Run prediction_hit_memory.py first.")
        return

    try:
        df = pd.read_excel(MEMORY_FILE)
    except Exception as exc:
        print(f"⚠️ Unable to read hit memory file: {exc}")
        return

    df.columns = [str(col).upper() for col in df.columns]
    total_rows = len(df)
    print(f"✅ Loaded hit memory: {total_rows} rows")

    hit_type_counts = {}
    script_type_counts = {}
    slot_type_counts = {}

    if "HIT_TYPE" in df.columns:
        hit_type_counts = df["HIT_TYPE"].value_counts().to_dict()
        print("📊 Hit counts by type:")
        for hit_type, count in hit_type_counts.items():
            print(f" • {hit_type}: {count}")

        if {"SCRIPT", "HIT_TYPE"}.issubset(df.columns):
            script_type_counts = (
                df.groupby(["SCRIPT", "HIT_TYPE"]).size().unstack(fill_value=0)
            )
            print("\n📌 Hits by script and type:")
            for script, row in script_type_counts.iterrows():
                details = ", ".join([f"{t}:{row[t]}" for t in row.index])
                print(f" • {script}: {details}")

        if {"REAL_SLOT", "HIT_TYPE"}.issubset(df.columns):
            slot_type_counts = (
                df.groupby(["REAL_SLOT", "HIT_TYPE"]).size().unstack(fill_value=0)
            )
            print("\n🎯 Hits by slot and type:")
            for slot, row in slot_type_counts.iterrows():
                details = ", ".join([f"{t}:{row[t]}" for t in row.index])
                print(f" • {slot}: {details}")
    else:
        print("⚠️ HIT_TYPE column missing; showing raw columns instead.")
        print(df.head())

    summary_path = MEMORY_FILE.with_name("prediction_memory_summary.json")
    try:
        summary = {
            "total_rows": total_rows,
            "hit_type_counts": hit_type_counts,
            "script_type_counts": script_type_counts.to_dict() if hasattr(script_type_counts, "to_dict") else {},
            "slot_type_counts": slot_type_counts.to_dict() if hasattr(slot_type_counts, "to_dict") else {},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to {summary_path}")
    except Exception as exc:
        print(f"⚠️ Unable to save summary: {exc}")


if __name__ == "__main__":
    main()
