"""Sanity check wrapper combining audit and ROI summaries."""
import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
AUDIT_FILE = BASE_DIR / "logs" / "performance" / "system_audit_report.xlsx"
PANEL_FILE = BASE_DIR / "logs" / "performance" / "quant_reality_pnl.json"


def load_audit_summary():
    if not AUDIT_FILE.exists():
        print("⚠️ Audit report missing")
        return None
    try:
        return pd.read_excel(AUDIT_FILE, sheet_name="summary")
    except Exception as exc:
        print(f"⚠️ Unable to read audit report: {exc}")
        return None


def load_roi_snapshot():
    if not PANEL_FILE.exists():
        print("⚠️ quant_reality_pnl.json missing")
        return {}
    try:
        with open(PANEL_FILE, "r") as f:
            return json.load(f) or {}
    except Exception as exc:
        print(f"⚠️ Unable to read ROI snapshot: {exc}")
        return {}


def main():
    audit_df = load_audit_summary()
    if audit_df is not None and not audit_df.empty:
        print("✅ Audit summary loaded")
        fields = {col: audit_df[col].iloc[0] for col in audit_df.columns if not audit_df[col].empty}
        print(f"   Total audited dates: {fields.get('total_audited_dates', 'N/A')}")
        print(f"   Full OK days      : {fields.get('full_ok_days', 'N/A')}")
        print(f"   Missing final plan: {fields.get('missing_final_plan_days', 'N/A')}")
    else:
        print("⚠️ No audit data available")

    roi_data = load_roi_snapshot()
    overall = roi_data.get("overall", {})
    if overall:
        print(f"💰 Overall ROI: {overall.get('overall_roi', 0):+.1f}% | P&L: ₹{overall.get('total_pnl', 0):+,.0f}")
    else:
        print("⚠️ No ROI snapshot available")


if __name__ == "__main__":
    main()
