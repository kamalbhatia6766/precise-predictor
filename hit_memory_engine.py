"""Wrapper for prediction hit memory analysis with schema cleanup."""
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _clean_hit_memory(memory_path: Path) -> None:
    """Ensure the hit memory file has a single, clean HIT_TYPE column."""
    if not memory_path.exists():
        return

    try:
        df = pd.read_excel(memory_path)
    except Exception as exc:  # pragma: no cover - defensive for real data runs
        print(f"⚠️ Unable to clean hit memory file: {exc}")
        return

    # Normalize column names and drop duplicates
    df.columns = [str(col).strip() for col in df.columns]

    # Handle legacy lower-case column while preferring upper-case
    if "hit_type" in df.columns and "HIT_TYPE" not in df.columns:
        df.rename(columns={"hit_type": "HIT_TYPE"}, inplace=True)
    elif "hit_type" in df.columns and "HIT_TYPE" in df.columns:
        df["HIT_TYPE"] = df["HIT_TYPE"].fillna(df["hit_type"])
        df.drop(columns=["hit_type"], inplace=True)

    # Remove duplicate column names (keeping the first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    if "HIT_TYPE" in df.columns:
        df["HIT_TYPE"] = df["HIT_TYPE"].astype(str).str.strip()

    try:
        df.to_excel(memory_path, index=False)
        print("🔄 HIT_TYPE column normalized in hit memory file.")
    except Exception as exc:  # pragma: no cover - defensive for real data runs
        print(f"⚠️ Unable to save cleaned hit memory file: {exc}")


def main():
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "prediction_hit_memory.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    subprocess.run(cmd, check=True)

    # Clean up the generated hit memory file to enforce schema stability
    memory_path = base_dir / "logs" / "performance" / "script_hit_memory.xlsx"
    _clean_hit_memory(memory_path)


if __name__ == "__main__":
    main()
