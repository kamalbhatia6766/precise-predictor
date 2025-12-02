"""Utility to dump ML features used by deepseek_scr6."""
from pathlib import Path

import pandas as pd

import quant_excel_loader
from deepseek_scr6 import build_ml_feature_dataframe

SLOT_NAMES = {1: "FRBD", 2: "GZBD", 3: "GALI", 4: "DSWR"}


def main():
    base_dir = Path(__file__).resolve().parent
    data = quant_excel_loader.load_results_excel()
    if data.empty:
        print("❌ No data available for feature building")
        return

    output_path = base_dir / "logs" / "performance" / "ml_features_latest.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for slot_id, slot_name in SLOT_NAMES.items():
            slot_series = data[data["slot"] == slot_id]["number"].dropna().astype(int).tolist()
            feature_df = build_ml_feature_dataframe(slot_series)
            if feature_df.empty:
                print(f"⚠️ Not enough history to build features for {slot_name}")
                continue
            feature_df.to_excel(writer, sheet_name=slot_name, index=False)
            print(f"✅ Built {len(feature_df)} feature rows for {slot_name}")

    print(f"💾 Feature dump saved to {output_path}")


if __name__ == "__main__":
    main()
