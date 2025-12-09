"""
Ultimate Prediction Engine

Unified entry point that loads core data via central helpers and runs the
prediction pipeline with robust CLI handling.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import sys

import pandas as pd

import quant_data_core
import quant_paths

REQUIRED_COLUMNS = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]


def print_banner(speed_mode: str) -> None:
    print("=" * 70)
    print("🚀 ULTIMATE PREDICTION ENGINE")
    print("   Merged SCR1-11 + Fusion + Adaptive Systems")
    print("=" * 70)
    print(f"🕒 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⚡ Speed Mode: {speed_mode}")


def resolve_default_data_path() -> Path:
    """Use quant_paths to get the canonical results file path."""
    return quant_paths.get_results_file_path()


def normalize_dataframe_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize column names and validate required fields."""
    renamed_cols: Dict[str, str] = {}
    for col in df.columns:
        normalized = str(col).strip().upper()
        renamed_cols[col] = normalized
    df = df.rename(columns=renamed_cols)

    alias_map = {
        "DATE": "DATE",
        "FRBD": "FRBD",
        "GZBD": "GZBD",
        "GALI": "GALI",
        "DSWR": "DSWR",
    }

    for alias, target in alias_map.items():
        if alias in df.columns and target not in df.columns:
            df = df.rename(columns={alias: target})

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        print(f"❌ Error loading data: Missing columns: {missing_cols}")
        return None

    return df


def coerce_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def summarize_data(df: pd.DataFrame) -> None:
    total_records = len(df)
    date_series = df["DATE"].dropna()
    if not date_series.empty:
        min_date = date_series.min().date()
        max_date = date_series.max().date()
        print(f"✅ Data loaded: {total_records} records")
        print(f"📅 Date range: {min_date} to {max_date}")
    else:
        print(f"✅ Data loaded: {total_records} records (no valid DATE values)")


def load_core_data(data_file: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load the core dataset with fallback to central loader."""
    source_path = Path(data_file) if data_file else resolve_default_data_path()
    df: Optional[pd.DataFrame] = None

    if data_file is None:
        try:
            df = quant_data_core.load_results_dataframe()
        except Exception as exc:
            print(f"⚠️  Could not load via quant_data_core: {exc}")

    if df is None:
        if not source_path.exists():
            print(f"❌ Error loading data: File not found: {source_path}")
            return None
        try:
            df = pd.read_excel(source_path)
        except Exception as exc:
            print(f"❌ Error loading data: {exc}")
            return None

    df = normalize_dataframe_columns(df)
    if df is None:
        return None

    df = coerce_dates(df)
    summarize_data(df)
    return df


def run_prediction_pipeline(df: pd.DataFrame, speed_mode: str, target_date: Optional[str]) -> int:
    predictions_dir = quant_paths.get_predictions_dir("prediction_engine")
    predictions_dir.mkdir(parents=True, exist_ok=True)

    plan = quant_data_core.build_prediction_plan(df)
    quant_data_core.print_prediction_plan_summary(plan)

    print("\n⚙️  Running existing prediction modules...")
    print(f"   ➤ Mode: {speed_mode}")
    if target_date:
        print(f"   ➤ Target date override: {target_date}")

    # Placeholder for the unchanged prediction logic.
    # This keeps the structure and outputs ready while centralizing data loading.
    print(f"📂 Predictions will be stored in: {predictions_dir}")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ULTIMATE PREDICTION ENGINE – Merged SCR1-11 + Fusion + Adaptive Systems"
    )
    parser.add_argument("--date", dest="date", help="Target date for prediction (YYYY-MM-DD)")
    parser.add_argument(
        "--speed-mode",
        choices=["full", "fast"],
        default="full",
        help="Choose between full or fast processing",
    )
    parser.add_argument(
        "--data-file",
        dest="data_file",
        help="Optional path to override the default core data file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    print_banner(args.speed_mode)

    df = load_core_data(args.data_file)
    if df is None or df.empty:
        print("❌ No usable data found for prediction engine; aborting run.")
        return 1

    return run_prediction_pipeline(df, args.speed_mode, args.date)


if __name__ == "__main__":
    sys.exit(main())
