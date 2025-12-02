"""Merged predictions view built from the latest SCR9 output."""
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def find_latest_predictions(base_dir: Path) -> Optional[Path]:
    pred_dir = base_dir / "predictions" / "deepseek_scr9"
    if not pred_dir.exists():
        return None
    files = sorted(pred_dir.glob("ultimate_predictions_*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def summarize_predictions(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_excel(file_path)
    except Exception as exc:
        print(f"⚠️ Unable to read {file_path.name}: {exc}")
        return pd.DataFrame()

    summary_rows = []
    for slot in SLOTS:
        if slot in df.columns:
            numbers = pd.to_numeric(df[slot], errors="coerce").dropna().astype(int).tolist()
            summary_rows.append({"slot": slot, "top_numbers": numbers[:10]})
        elif {"slot", "number"}.issubset(df.columns):
            slot_df = df[df["slot"].astype(str).str.upper() == slot]
            nums = pd.to_numeric(slot_df.get("number"), errors="coerce").dropna().astype(int).tolist()
            summary_rows.append({"slot": slot, "top_numbers": nums[:10]})
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
        summary_df.to_excel(output_path, index=False)
        print(f"💾 Summary saved to {output_path}")
    except Exception as exc:
        print(f"⚠️ Unable to save merged summary: {exc}")


if __name__ == "__main__":
    main()
