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

    total_rows = len(df)
    print(f"✅ Loaded hit memory: {total_rows} rows")

    if "HIT_TYPE" in df.columns:
        counts = df["HIT_TYPE"].value_counts().to_dict()
        print("📊 Hit counts by type:")
        for hit_type, count in counts.items():
            print(f" • {hit_type}: {count}")
    else:
        print("⚠️ HIT_TYPE column missing; showing raw columns instead.")
        print(df.head())

    summary_path = MEMORY_FILE.with_name("prediction_memory_summary.json")
    try:
        summary = {
            "total_rows": total_rows,
            "hit_type_counts": counts if "counts" in locals() else {},
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to {summary_path}")
    except Exception as exc:
        print(f"⚠️ Unable to save summary: {exc}")


if __name__ == "__main__":
    main()
