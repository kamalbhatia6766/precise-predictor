from pathlib import Path
import sys
import re
from datetime import timedelta
import pandas as pd
import pattern_packs


BASE_DIR = Path(__file__).resolve().parent
SLOTS = ["FRBD", "GZBD", "GALI", "DSWR"]


def fail(message):
    print(message)
    sys.exit(0)


def load_pnl_data():
    perf_dir = BASE_DIR / "logs" / "performance"
    pnl_file = None
    for ext in ("xlsx", "csv"):
        candidate = perf_dir / f"quant_pnl_summary.{ext}"
        if candidate.exists():
            pnl_file = candidate
            break
    if not pnl_file:
        fail("quant_pnl_summary not found, cannot compute ROI summary.")
    if pnl_file.suffix == ".csv":
        df = pd.read_csv(pnl_file)
    else:
        df = pd.read_excel(pnl_file)
    return df


def pick_date_column(df):
    for col in df.columns:
        if "date" in str(col).lower():
            return col
    return None


def build_stake_return(df):
    stake_cols = [c for c in df.columns if "stake" in str(c).lower()]
    return_cols = [c for c in df.columns if "return" in str(c).lower()]
    stake_series = df[stake_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if stake_cols else pd.Series([0] * len(df))
    return_series = df[return_cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) if return_cols else pd.Series([0] * len(df))
    return stake_series, return_series


def summarize_windows(daily):
    max_date = daily["DATE"].max()
    min_date = daily["DATE"].min()
    total_days = (max_date - min_date).days + 1

    def window_stats(days):
        if days is None:
            window_df = daily
        else:
            if daily["DATE"].nunique() < days:
                window_df = daily
            else:
                start_date = max_date - timedelta(days=days - 1)
                window_df = daily[daily["DATE"] >= start_date]
        stake = window_df["STAKE"].sum()
        ret = window_df["RETURN"].sum()
        pnl = ret - stake
        roi = (ret / stake - 1) * 100 if stake > 0 else 0
        return pnl, roi

    overall = window_stats(None)
    last7 = window_stats(7)
    last30 = window_stats(30)
    return (min_date, max_date, total_days), overall, last7, last30


def slot_roi(df, date_col):
    slot_col = None
    for col in df.columns:
        if "slot" == str(col).lower():
            slot_col = col
            break
    if slot_col is None:
        return None
    stakes, returns = build_stake_return(df)
    temp = pd.DataFrame({"slot": df[slot_col].astype(str).str.upper(), "stake": stakes, "return": returns, "date": pd.to_datetime(df[date_col], errors="coerce")})
    temp = temp[temp["date"].notna()]
    summary = temp.groupby("slot").agg({"stake": "sum", "return": "sum"}).reset_index()
    return summary


def parse_filename_date(stem):
    match = re.search(r"(20\d{6})", stem)
    if match:
        text = match.group(1)
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    return pd.NaT


def load_real_results():
    results_file = BASE_DIR / "number prediction learn.xlsx"
    if not results_file.exists():
        fail("number prediction learn.xlsx not found, cannot compute ROI summary.")
    df = pd.read_excel(results_file)
    df.columns = [str(c).strip() for c in df.columns]
    date_col = pick_date_column(df)
    if not date_col:
        fail("DATE column missing in real results, cannot compute ROI summary.")
    df["DATE"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df["DATE"].notna()]
    slot_cols = {slot: col for slot in SLOTS for col in df.columns if str(col).upper() == slot}
    if not slot_cols:
        fail("Slot columns missing in real results, cannot compute ROI summary.")
    return df[["DATE"] + list(slot_cols.values())].rename(columns={v: k for k, v in slot_cols.items()})


def collect_bets():
    bet_dir = BASE_DIR / "predictions" / "bet_engine"
    bet_files = sorted(bet_dir.glob("bet_plan_master_*.xlsx"))
    if not bet_files:
        fail("bet_plan_master files not found, cannot compute ROI summary.")
    records = []
    for file in bet_files:
        try:
            bets_df = pd.read_excel(file, sheet_name="bets")
        except Exception:
            try:
                bets_df = pd.read_excel(file)
            except Exception:
                continue
        bets_df.columns = [str(c).strip().lower() for c in bets_df.columns]
        date_val = bets_df["date"].apply(pd.to_datetime, errors="coerce") if "date" in bets_df.columns else pd.Series([parse_filename_date(file.stem)] * len(bets_df))
        slot_series = bets_df["slot"] if "slot" in bets_df.columns else pd.Series([None] * len(bets_df))
        layer_series = bets_df["layer_type"].str.upper() if "layer_type" in bets_df.columns else pd.Series(["MAIN"] * len(bets_df))
        number_series = bets_df["number_or_digit"] if "number_or_digit" in bets_df.columns else bets_df["number"] if "number" in bets_df.columns else pd.Series([None] * len(bets_df))
        stake_series = bets_df["stake"] if "stake" in bets_df.columns else pd.Series([10] * len(bets_df))
        for d, s, layer, num, stake in zip(date_val, slot_series, layer_series, number_series, stake_series):
            if pd.isna(d) or pd.isna(s) or pd.isna(num):
                continue
            slot = str(s).strip().upper()
            if slot not in SLOTS:
                continue
            if str(layer).upper() != "MAIN":
                continue
            num_str = str(int(float(num))) if str(num).replace(".", "", 1).isdigit() else str(num).strip()
            if not num_str.isdigit() or len(num_str) > 2:
                continue
            num_str = num_str.zfill(2)
            try:
                stake_val = float(stake)
            except Exception:
                continue
            records.append({"DATE": pd.to_datetime(d), "slot": slot, "number": num_str, "stake": stake_val})
    return pd.DataFrame(records)


def s40_performance(results_df, bets_df):
    if bets_df.empty:
        return None
    merged = bets_df.merge(results_df, on="DATE", how="left", suffixes=("_bet", ""))
    merged = merged[merged["slot"].isin(SLOTS)]
    def evaluate(row):
        res = row.get(row["slot"], None)
        if pd.isna(res):
            return False, 0
        hit = str(res).zfill(2) == row["number"]
        ret = row["stake"] * 90 if hit else 0
        return hit, ret
    hit_flags = []
    returns = []
    for _, row in merged.iterrows():
        hit, ret = evaluate(row)
        hit_flags.append(hit)
        returns.append(ret)
    merged["hit"] = hit_flags
    merged["return"] = returns
    merged["group"] = merged["number"].apply(lambda x: "S40" if pattern_packs.is_s40(str(x)) else "NON")
    summary = {}
    for label in ["S40", "NON"]:
        subset = merged[merged["group"] == label]
        stake_total = subset["stake"].sum()
        ret_total = subset["return"].sum()
        hits = subset["hit"].sum()
        roi = (ret_total / stake_total - 1) * 100 if stake_total > 0 else 0
        summary[label] = {"hits": int(hits), "stake": stake_total, "return": ret_total, "roi": roi}
    return summary


def script_hits():
    hit_file = BASE_DIR / "logs" / "performance" / "script_hit_memory.xlsx"
    if not hit_file.exists():
        fail("script_hit_memory.xlsx not found, cannot compute ROI summary.")
    df = pd.read_excel(hit_file)
    if df.empty:
        return []
    df.columns = [str(c).strip().lower() for c in df.columns]
    script_col = "script_name" if "script_name" in df.columns else None
    if not script_col:
        return []
    family_col = "hit_family" if "hit_family" in df.columns else None
    df[script_col] = df[script_col].astype(str)
    if family_col:
        df[family_col] = df[family_col].astype(str)
    grouped = []
    for name, sub in df.groupby(script_col):
        if family_col:
            cross = sub[sub[family_col].str.contains("CROSS", case=False, na=False)].shape[0]
            direct = sub.shape[0] - cross
            grouped.append((name, sub.shape[0], cross, direct))
        else:
            grouped.append((name, sub.shape[0], None, None))
    grouped.sort(key=lambda x: x[1], reverse=True)
    return grouped


def format_currency(value):
    return f"₹{value:,.0f}" if value == value else "₹0"


def main():
    pnl_df = load_pnl_data()
    date_col = pick_date_column(pnl_df)
    if not date_col:
        fail("DATE column missing in P&L data, cannot compute ROI summary.")
    pnl_df["DATE"] = pd.to_datetime(pnl_df[date_col], errors="coerce")
    pnl_df = pnl_df[pnl_df["DATE"].notna()]
    stakes, returns = build_stake_return(pnl_df)
    daily = pd.DataFrame({"DATE": pnl_df["DATE"], "STAKE": stakes, "RETURN": returns}).groupby("DATE").sum().reset_index()
    if daily.empty:
        fail("No valid P&L data to summarize.")
    window_info, overall, last7, last30 = summarize_windows(daily)

    print("=== ROI SUMMARY ===")
    print(f"Window: {window_info[0].date()} → {window_info[1].date()} ({window_info[2]} days)")
    print("")
    print("1) P&L SNAPSHOT")
    print(f"   Overall      : P&L {format_currency(overall[0])} (ROI {overall[1]:.1f}%)")
    print(f"   Last 7 days  : P&L {format_currency(last7[0])} (ROI {last7[1]:.1f}%)")
    print(f"   Last 30 days : P&L {format_currency(last30[0])} (ROI {last30[1]:.1f}%)")

    slot_summary = slot_roi(pnl_df, "DATE")
    if slot_summary is not None:
        print("\n2) SLOT-WISE ROI (full window)")
        for slot in SLOTS:
            row = slot_summary[slot_summary["slot"] == slot]
            if row.empty:
                continue
            stake_val = row["stake"].values[0]
            ret_val = row["return"].values[0]
            roi_val = (ret_val / stake_val - 1) * 100 if stake_val > 0 else 0
            print(f"   {slot}: stake {format_currency(stake_val)}, return {format_currency(ret_val)}, ROI {roi_val:.1f}%")

    results_df = load_real_results()
    bets_df = collect_bets()
    perf = s40_performance(results_df, bets_df)
    if perf:
        print("\n3) S40 vs NON-S40 (2-digit bets only, full window)")
        s40 = perf.get("S40", {})
        non = perf.get("NON", {})
        print(f"   S40     : hits {s40.get('hits',0)}, stake {format_currency(s40.get('stake',0))}, return {format_currency(s40.get('return',0))}, ROI {s40.get('roi',0):.1f}%")
        print(f"   Non-S40 : hits {non.get('hits',0)}, stake {format_currency(non.get('stake',0))}, return {format_currency(non.get('return',0))}, ROI {non.get('roi',0):.1f}%")

    hits = script_hits()
    if hits:
        print("\n4) SCRIPT PERFORMANCE (from hit memory)")
        for name, total, cross, direct in hits:
            if cross is None:
                print(f"   {name}: {total} hits")
            else:
                print(f"   {name}: {total} hits ({cross} CROSS, {direct} DIRECT)")


if __name__ == "__main__":
    main()
