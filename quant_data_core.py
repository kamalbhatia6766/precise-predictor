# quant_data_core.py - ENHANCED VERSION
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import quant_paths

def load_results_dataframe():
    """
    Load central results file in a robust way.

    - Handles files with or without a header row.
    - Forces columns: DATE, FRBD, GZBD, GALI, DSWR.
    - Parses DATE from Excel serials, timestamps, or strings.
    - Ensures slot columns are numeric.
    """
    results_file = quant_paths.get_results_file_path()

    try:
        df_raw = pd.read_excel(results_file, header=None)
        print(f"Found columns: {df_raw.columns.tolist()}")
        print(f"Raw shape: {df_raw.shape}")
    except Exception as e:
        print(f"❌ Error loading real results: {e}")
        return pd.DataFrame()

    if df_raw.empty:
        print("❌ Real results file is empty")
        return pd.DataFrame()

    def _is_datetime_like(value):
        if pd.isna(value):
            return False
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            return True
        # Excel serials or other numeric encodings
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if isinstance(value, str):
            try:
                parsed = pd.to_datetime(value, errors="raise")
                return not pd.isna(parsed)
            except Exception:
                return False
        return False

    # Decide if first row is header or data
    first_row = df_raw.iloc[0]
    first_cell = first_row.iloc[0]
    header_is_data = _is_datetime_like(first_cell)

    if header_is_data:
        print("ℹ️  Detected first row as data (no header row present)")
        df = df_raw.iloc[:, :5].copy()
    else:
        print("ℹ️  Detected header row; normalizing column names")
        inferred_columns = [str(col).strip().upper() for col in first_row]
        df = df_raw.iloc[1:, :5].copy()
        df.columns = inferred_columns

    # Force final columns regardless of path
    df = df.iloc[:, :5].copy()
    df.columns = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]

    # Robust DATE parsing
    excel_origin = datetime(1899, 12, 30).date()

    def _parse_date_value(value):
        if pd.isna(value):
            return None

        # Already datetime-like
        if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
            try:
                return pd.to_datetime(value, errors="coerce").date()
            except Exception:
                return None

        # Excel serial / numeric
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return excel_origin + timedelta(days=float(value))
            except Exception:
                return None

        # String
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            try:
                return pd.to_datetime(value, errors="raise").date()
            except Exception:
                return None

        return None

    parsed_dates = []
    invalid_date_values = []

    for raw_val in df["DATE"].tolist():
        parsed = _parse_date_value(raw_val)
        parsed_dates.append(parsed)
        if parsed is None:
            invalid_date_values.append(raw_val)

    if invalid_date_values:
        sample_values = invalid_date_values[:5]
        print(f"⚠️  Failed to parse {len(invalid_date_values)} DATE entries. Samples: {sample_values}")

    df["DATE"] = pd.to_datetime(parsed_dates, errors="coerce")
    invalid_after_parse = df["DATE"].isna().sum()

    if invalid_after_parse:
        print(f"⚠️  Dropping {invalid_after_parse} rows with unparseable DATE values")
        df = df.dropna(subset=["DATE"])

    # If still nothing valid, bail out gracefully
    if df.empty:
        print("❌ No valid DATE values found after parsing; exiting gracefully")
        return pd.DataFrame()

    # Drop obviously bogus historical dates (e.g. Excel serial 0 -> 1899-12-31)
    min_valid_date = datetime(2000, 1, 1)
    bogus_mask = df["DATE"] < pd.Timestamp(min_valid_date)
    if bogus_mask.any():
        dropped_bogus = int(bogus_mask.sum())
        print(f"⚠️  Dropping {dropped_bogus} rows with implausible DATE < {min_valid_date.date()}")
        df = df.loc[~bogus_mask].copy()

    if df.empty:
        print("❌ No valid DATE values left after DATE sanity filter; exiting gracefully")
        return pd.DataFrame()

    # Ensure slot columns exist and are numeric
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        if slot not in df.columns:
            print(f"⚠️  Slot column '{slot}' not found, creating with NaN values")
            df[slot] = np.nan
        df.loc[:, slot] = pd.to_numeric(df[slot], errors="coerce")

    # Final sanity + logging
    total_rows = len(df)
    if not pd.api.types.is_datetime64_any_dtype(df["DATE"]):
        # Even if dtype is 'object', as long as values are datetime-like, we can still work.
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    unique_dates = df["DATE"].dt.date.dropna().unique()
    if len(unique_dates) == 0:
        print("❌ No valid DATE values found after final DATE conversion")
        return pd.DataFrame()

    df = df.sort_values("DATE").reset_index(drop=True)

    print(f"✅ Loaded results data: {total_rows} records with columns: {df.columns.tolist()}")
    print(f"📅 DATE range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")

    return df

def get_latest_result_date(df):
    """Get the latest result date from DataFrame - ROBUST VERSION"""
    if df.empty or 'DATE' not in df.columns:
        return None
    
    # Make a copy and ensure DATE is datetime
    df_temp = df.copy()
    df_temp['DATE'] = pd.to_datetime(df_temp['DATE'], errors='coerce')
    df_temp = df_temp.dropna(subset=['DATE'])
    
    if df_temp.empty:
        return None
    
    # Convert to date only
    df_temp['DATE_ONLY'] = df_temp['DATE'].dt.date
    today = datetime.now().date()
    
    # Filter to past or today dates
    past_or_today = df_temp[df_temp['DATE_ONLY'] <= today]
    
    if not past_or_today.empty:
        return past_or_today['DATE_ONLY'].max()
    
    # Fallback: if all dates are in future, use the most recent one anyway
    return df_temp['DATE_ONLY'].max()

def get_slot_fill_status_for_date(df, date):
    """Check which slots are filled for a given date"""
    if df.empty or 'DATE' not in df.columns:
        return {}
    
    date_data = df[df['DATE'] == date]
    if date_data.empty:
        return {}
    
    slot_status = {}
    slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
    
    for slot in slots:
        if slot in date_data.columns:
            value = date_data[slot].iloc[0]
            # Check if slot has valid result (not NaN and not empty string)
            slot_status[slot] = pd.notna(value) and value != ''
        else:
            slot_status[slot] = False
    
    return slot_status

def build_prediction_plan(df):
    """Build prediction plan based on latest results - ROBUST VERSION"""
    latest_date = get_latest_result_date(df)
    today = datetime.now().date()
    
    # Fallback if no valid date found
    if not latest_date:
        latest_date = today
        print("⚠️  No valid result data found, using today as fallback")
    
    slot_status = get_slot_fill_status_for_date(df, latest_date)
    
    # Determine if we're dealing with today or a past day
    if latest_date == today:
        # Today - check which slots are filled
        filled_slots = [slot for slot, filled in slot_status.items() if filled]
        all_slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
        remaining_slots = [slot for slot in all_slots if slot not in filled_slots]
        
        if remaining_slots:
            # Partial day - predict remaining slots for today
            is_partial_day = True
            today_slots_to_predict = remaining_slots
            next_date = today
            mode = "partial_today"
        else:
            # Full day completed - predict next day
            is_partial_day = False
            today_slots_to_predict = []
            next_date = today + timedelta(days=1)
            mode = "next_day"
    else:
        # Past day - predict next day
        is_partial_day = False
        today_slots_to_predict = []
        next_date = latest_date + timedelta(days=1)
        mode = "next_day"
    
    plan = {
        'latest_result_date': latest_date,
        'is_partial_day': is_partial_day,
        'today_slots_to_predict': today_slots_to_predict,
        'next_date': next_date,
        'mode': mode,
        'slot_status': slot_status
    }
    
    return plan

def print_prediction_plan_summary(plan):
    """Print human-readable prediction plan summary"""
    print("\n🎯 PREDICTION PLAN SUMMARY")
    print("=" * 40)
    print(f"Latest result date: {plan['latest_result_date']}")
    print(f"Plan mode: {plan['mode']}")
    
    if plan['is_partial_day']:
        print(f"Same-day slots to predict: {', '.join(plan['today_slots_to_predict'])}")
    else:
        print("Same-day slots: None")
        
    print(f"Next prediction date: {plan['next_date']}")
    
    if plan['slot_status']:
        print("Slot status:")
        for slot, filled in plan['slot_status'].items():
            status = "✅ Filled" if filled else "❌ Missing"
            print(f"  {slot}: {status}")

# Utility function for date handling
def get_date_range(days_back=30):
    """Get date range for analysis"""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    return start_date, end_date

def filter_data_by_date(df, start_date, end_date):
    """Filter DataFrame by date range"""
    if 'DATE' not in df.columns:
        return df
    
    mask = (df['DATE'] >= start_date) & (df['DATE'] <= end_date)
    return df[mask]