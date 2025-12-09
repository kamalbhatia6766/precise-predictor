import json
from pathlib import Path
import sys
from datetime import timedelta
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def fail(message: str) -> None:
    print(message)
    sys.exit(0)


def load_quant_reality_pnl():
    """Load quant_reality_pnl.json as the single source of truth."""
    perf_dir = BASE_DIR / "logs" / "performance"
    pnl_file = perf_dir / "quant_reality_pnl.json"
    if not pnl_file.exists():
        fail("No P&L data found. Run bet_pnl_tracker.py first.")
    try:
        with open(pnl_file, "r") as f:
            data = json.load(f)
    except Exception as exc:
        fail(f"Failed to read quant_reality_pnl.json: {exc}")
    return data


def build_daily_df(data):
    """Return a DataFrame with DATE, STAKE, RETURN, PNL from quant_reality_pnl data."""
    daily_entries = data.get("daily") or []
    records = data.get("records") or []

    rows = []
    source_rows = daily_entries if daily_entries else records
    for entry in source_rows:
        date_val = pd.to_datetime(entry.get("date") or entry.get("DATE"), errors="coerce")
        if pd.isna(date_val):
            continue

        stake = float(entry.get("total_stake", entry.get("stake", 0)) or 0)
        ret = float(entry.get("total_return", entry.get("return", 0)) or 0)
        pnl = entry.get("pnl")
        pnl = float(pnl) if pnl is not None else ret - stake

        rows.append({"DATE": date_val.normalize(), "STAKE": stake, "RETURN": ret, "PNL": pnl})

    daily_df = pd.DataFrame(rows)
    if not daily_df.empty:
        daily_df = daily_df.groupby("DATE")[["STAKE", "RETURN", "PNL"]].sum().reset_index()
    return daily_df


def summarize_windows(daily_df):
    max_date = daily_df["DATE"].max()
    min_date = daily_df["DATE"].min()
    total_days = (max_date - min_date).days + 1

    def window_stats(days):
        unique_dates = sorted(daily_df["DATE"].unique())
        window_dates = unique_dates[-days:] if days else unique_dates
        window_df = daily_df[daily_df["DATE"].isin(window_dates)]
        stake = window_df["STAKE"].sum()
        pnl = window_df["PNL"].sum()
        roi = (pnl / stake * 100) if stake > 0 else 0
        unique_days = len(window_dates)
        return pnl, roi, unique_days

    overall_pnl = daily_df["PNL"].sum()
    overall_stake = daily_df["STAKE"].sum()
    overall_roi = (overall_pnl / overall_stake * 100) if overall_stake > 0 else 0
    overall = (overall_pnl, overall_roi, daily_df["DATE"].nunique())
    last7 = window_stats(7)
    last30 = window_stats(30)
    return (min_date, max_date, total_days), overall, last7, last30


def compute_drawdown(daily_df):
    daily_pnl = daily_df.groupby("DATE")["PNL"].sum().reset_index().sort_values("DATE")
    cumulative = daily_pnl["PNL"].cumsum()
    peak = cumulative.iloc[0] if not cumulative.empty else 0
    worst_dd = 0
    for val in cumulative:
        peak = max(peak, val)
        drawdown = val - peak
        worst_dd = min(worst_dd, drawdown)
    peak_base = peak if peak != 0 else max(cumulative.max(), 1)
    dd_pct = abs(worst_dd) / peak_base * 100 if peak_base else 0
    return worst_dd, dd_pct


def slot_roi_from_quant(data):
    slot_rows = data.get("by_slot") or []
    if not slot_rows:
        return None
    return {row.get("slot", "").upper(): row for row in slot_rows}


def format_currency(value):
    return f"₹{value:,.0f}" if value == value else "₹0"


def main():
    quant_data = load_quant_reality_pnl()
    daily = build_daily_df(quant_data)
    if daily.empty:
        fail("No P&L data found. Run bet_pnl_tracker.py first.")

    window_info, overall, last7, last30 = summarize_windows(daily)

    overall_pnl, overall_roi, _ = overall

    print("=== ROI SUMMARY ===")
    print(f"Window: {window_info[0].date()} → {window_info[1].date()} ({window_info[2]} days)")
    print("")
    print("1) P&L SNAPSHOT")
    print(f"   Overall      : P&L {format_currency(overall_pnl)} (ROI {overall_roi:.1f}%)")
    print(f"   Last 7 days  : P&L {format_currency(last7[0])} (ROI {last7[1]:.1f}%, days={last7[2]}/7)")
    print(f"   Last 30 days : P&L {format_currency(last30[0])} (ROI {last30[1]:.1f}%, days={last30[2]}/30)")

    worst_dd, dd_pct = compute_drawdown(daily)
    print("\n2) RISK STATS (full window)")
    print(f"   Worst drawdown : {format_currency(worst_dd)} ({dd_pct:.1f}%)")

    slot_summary = slot_roi_from_quant(quant_data)
    if slot_summary is not None:
        print("\n3) SLOT-WISE ROI (full window)")
        for slot in SLOTS:
            row = slot_summary.get(slot, {})
            if not row:
                continue
            stake_val = row.get("total_stake", 0)
            ret_val = row.get("total_return", 0)
            roi_val = row.get("roi_pct")
            if roi_val is None:
                pnl_val = row.get("total_pnl", ret_val - stake_val)
                roi_val = (pnl_val / stake_val * 100) if stake_val > 0 else 0
            print(f"   {slot}: stake {format_currency(stake_val)}, return {format_currency(ret_val)}, ROI {roi_val:.1f}%")


if __name__ == "__main__":
    main()
