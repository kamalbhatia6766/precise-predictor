# prediction_hit_memory.py - UPDATED WITH FULL PACK UNIVERSE TAGS
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import re
from collections import defaultdict
from utils_2digit import is_valid_2d_number, to_2d_str
from script_hit_memory_utils import (
    append_script_hit_row,
    rebuild_script_hit_memory,
    load_script_hit_memory,
)
import warnings
warnings.filterwarnings('ignore')

# Import pattern packs with full universe
import pattern_packs
import quant_data_core

class PredictionHitMemory:
    def __init__(self):
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.script_patterns = {
            'SCR1': 'scr1_precise_predictions_*.xlsx',
            'SCR2': 'scr2_predictions_*.xlsx', 
            'SCR3': 'scr3_predictions_*.xlsx',
            'SCR4': 'scr4_predictions_*.xlsx',
            'SCR5': 'scr5_predictions_*.xlsx',
            'SCR6': 'ultimate_predictions_*.xlsx',
            'SCR7': 'advanced_predictions_*.xlsx',
            'SCR8': 'scr10_predictions_*.xlsx',
            'SCR9': 'ultimate_predictions_*.xlsx'
        }
        

    def load_real_results(self, file_path):
        """Load real results from Excel file using the same logic as other scripts."""
        try:
            df = quant_data_core.load_results_dataframe()

            slot_cols = ["FRBD", "GZBD", "GALI", "DSWR"]
            missing_cols = [col for col in ["DATE", *slot_cols] if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Results DataFrame missing required columns: {missing_cols}")

            rows = []
            for _, row in df.iterrows():
                date_val = row["DATE"]
                if pd.isna(date_val):
                    continue

                try:
                    date_val = pd.to_datetime(date_val)
                except Exception:
                    continue

                for slot in slot_cols:
                    raw_val = row.get(slot)
                    if pd.isna(raw_val):
                        continue

                    s = str(raw_val).strip()
                    if not s or s.upper() == "XX":
                        continue

                    try:
                        num = int(float(s)) % 100
                    except Exception:
                        continue

                    rows.append({
                        "date": date_val,
                        "slot": slot.upper(),
                        "number": num,
                    })

            real_df = pd.DataFrame(rows)
            if real_df.empty:
                raise ValueError("No valid data found in Excel file")

            real_df["date"] = pd.to_datetime(real_df["date"])
            real_df = real_df.sort_values(["date", "slot"]).reset_index(drop=True)

            min_date = real_df["date"].min().strftime("%Y-%m-%d")
            max_date = real_df["date"].max().strftime("%Y-%m-%d")
            print(f"📅 Real results loaded: {len(real_df)} rows from {min_date} to {max_date}")

            return real_df

        except Exception as e:
            raise ValueError(f"Error loading real results: {e}")

    def clean_number(self, x):
        """Convert to integer number (same as other scripts)."""
        try:
            s = str(x).strip()
            digits = ''.join([c for c in s if c.isdigit()])
            if not digits:
                return None
            num = int(digits)
            return num % 100  # Ensure 2-digit number
        except:
            return None
    
    def get_target_date(self, real_df):
        """Get latest date with all four slots filled."""
        # Check if we have all four slots for each date
        date_counts = real_df.groupby('date')['slot'].nunique()
        complete_dates = date_counts[date_counts == 4].index
        
        if len(complete_dates) == 0:
            raise ValueError("No dates found with all four slots filled")
        
        target_date = max(complete_dates)
        return target_date
    
    def find_latest_prediction_file(self, script_name, pattern):
        """Find latest prediction file for a script."""
        script_dir = Path(__file__).resolve().parent / "predictions" / f"deepseek_{script_name.lower()}"
        
        if not script_dir.exists():
            print(f"⚠️  Directory not found: {script_dir}")
            return None
        
        files = list(script_dir.glob(pattern))
        if not files:
            print(f"⚠️  No files found for {script_name} with pattern {pattern}")
            return None
        
        # Sort by modification time (newest first)
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        return latest_file
    
    def load_predictions_file(self, file_path, script_name):
        """Load and normalize predictions file."""
        try:
            df = pd.read_excel(file_path)
            df.columns = [str(col).strip().lower() for col in df.columns]
            
            print(f"   📁 {script_name}: {file_path.name} ({len(df)} rows)")
            
            # Try to detect format and convert to long format
            return self._convert_to_long_format(df, script_name)
            
        except Exception as e:
            print(f"   ❌ Error loading {script_name}: {e}")
            return None
    
    def _convert_to_long_format(self, df, script_name):
        """Convert predictions to long format (date, slot, numbers)."""
        long_data = []
        
        # Check if already in long format (has slot column)
        slot_candidates = [col for col in df.columns if 'slot' in col]
        if slot_candidates:
            slot_col = slot_candidates[0]
            # Find numbers column
            for col in df.columns:
                if col != slot_col and not any(x in col for x in ['date', 'opp']):
                    sample_val = df[col].iloc[0] if len(df) > 0 else ''
                    if isinstance(sample_val, str) and ',' in sample_val:
                        for _, row in df.iterrows():
                            numbers = self._parse_numbers_str(row[col])
                            if numbers:
                                long_data.append({
                                    'date': row.get('date', ''),
                                    'slot': row[slot_col].upper(),
                                    'numbers': numbers
                                })
                        break
        else:
            # Wide format - look for slot columns
            slot_columns = []
            for col in df.columns:
                col_upper = col.upper()
                if any(slot in col_upper for slot in self.slots) and '_OPP' not in col_upper:
                    slot_columns.append(col)
            
            date_col = 'date'
            if 'date' not in df.columns and len(df.columns) > 0:
                date_col = df.columns[0]  # Assume first column is date
            
            for _, row in df.iterrows():
                for slot_col in slot_columns:
                    numbers_str = row[slot_col]
                    if pd.notna(numbers_str):
                        numbers = self._parse_numbers_str(str(numbers_str))
                        if numbers:
                            # Extract slot name from column name
                            slot_name = None
                            for slot in self.slots:
                                if slot in slot_col.upper():
                                    slot_name = slot
                                    break
                            if slot_name:
                                long_data.append({
                                    'date': row[date_col],
                                    'slot': slot_name,
                                    'numbers': numbers
                                })
        
        return long_data
    
    def _parse_numbers_str(self, numbers_str):
        """Parse comma-separated numbers string to list of integers."""
        if pd.isna(numbers_str) or numbers_str == '':
            return []
        
        numbers = []
        for num_str in str(numbers_str).split(','):
            num_str = num_str.strip()
            if num_str.isdigit():
                try:
                    numbers.append(int(num_str))
                except ValueError:
                    continue
        
        return numbers
    
    def analyze_hits(self, real_df, target_date):
        """Analyze hits across all scripts for target date with time-shift tracking."""
        all_hits = []
        
        # Get real results for target date and surrounding dates for time-shift analysis
        real_results = {}
        for date_offset in [-1, 0, 1]:  # Previous day, same day, next day
            check_date = target_date + timedelta(days=date_offset)
            date_data = real_df[real_df['date'] == check_date]
            for slot in self.slots:
                slot_data = date_data[date_data['slot'] == slot]
                if not slot_data.empty:
                    real_results[(check_date, slot)] = slot_data['number'].iloc[0]
        
        print(f"\n🎯 Analyzing hits for {target_date} (with time-shift tracking):")
        for (date, slot), real_num in real_results.items():
            days_diff = (date - target_date).days
            time_shift = "SAME_DAY" if days_diff == 0 else "NEXT_DAY" if days_diff == 1 else "PREV_DAY"
            print(f"   {date.strftime('%Y-%m-%d')} {slot}: {real_num} ({time_shift})")
        
        # Check each script
        for script_name, pattern in self.script_patterns.items():
            print(f"\n🔍 Checking {script_name}...")
            file_path = self.find_latest_prediction_file(script_name, pattern)
            if not file_path:
                continue
            
            predictions = self.load_predictions_file(file_path, script_name)
            if not predictions:
                continue
            
            # Filter predictions for target date and surrounding dates
            target_predictions = []
            for pred in predictions:
                pred_date = pred['date']
                # Handle different date formats
                for check_date in [target_date - timedelta(days=1), target_date, target_date + timedelta(days=1)]:
                    if isinstance(pred_date, datetime):
                        if pred_date.date() == check_date.date():
                            target_predictions.append({**pred, 'prediction_date': check_date})
                    elif isinstance(pred_date, str):
                        try:
                            if pd.to_datetime(pred_date).date() == check_date.date():
                                target_predictions.append({**pred, 'prediction_date': check_date})
                        except:
                            # Try string comparison as fallback
                            if str(pred_date) == str(check_date):
                                target_predictions.append({**pred, 'prediction_date': check_date})
                    else:
                        if str(pred_date) == str(check_date):
                            target_predictions.append({**pred, 'prediction_date': check_date})
            
            for pred in target_predictions:
                pred_date = pred['prediction_date']
                pred_slot = pred['slot']
                pred_numbers = pred['numbers']
                
                for (real_date, real_slot), real_num in real_results.items():
                    if real_num in pred_numbers:
                        # Determine time shift and hit family
                        days_delta = (real_date - pred_date).days
                        if days_delta == 0:
                            time_shift = "SAME_DAY"
                            if pred_slot == real_slot:
                                slot_shift = "SAME_SLOT"
                                hit_family = "DIRECT"
                            else:
                                slot_shift = "CROSS_SLOT" 
                                hit_family = "CROSS_SAME_DAY"
                        elif days_delta == 1:
                            time_shift = "NEXT_DAY"
                            slot_shift = "CROSS_SLOT" if pred_slot != real_slot else "SAME_SLOT"
                            hit_family = "CROSS_NEXT_DAY"
                        elif days_delta == -1:
                            time_shift = "PREV_DAY"
                            slot_shift = "CROSS_SLOT" if pred_slot != real_slot else "SAME_SLOT"
                            hit_family = "CROSS_PREV_DAY"
                        else:
                            continue
                        
                        rank = pred_numbers.index(real_num) + 1
                        list_size = len(pred_numbers)
                        
                        # ✅ UPDATED: Get FULL pattern tags from new pattern_packs
                        is_s40 = pattern_packs.is_s40(real_num)
                        digit_tags = pattern_packs.get_digit_pack_tags(real_num)
                        digit_tags_str = ",".join(digit_tags)
                        
                        hit_type = "SAME_DAY"
                        if days_delta == 0 and pred_slot != real_slot:
                            hit_type = "CROSS_SAME_DAY"
                        elif days_delta == 1:
                            hit_type = "CROSS_NEXT_DAY"
                        elif days_delta == -1:
                            hit_type = "CROSS_PREV_DAY"

                        slot_shift_detail = f"{pred_slot}→{real_slot}" if pred_slot != real_slot else "SAME_SLOT"

                        hit_data = {
                            'date': real_date,
                            'real_slot': real_slot,
                            'real_number': real_num,
                            'script': script_name,
                            'predicted_slot': pred_slot,
                            'prediction_date': pred_date,
                            'rank': rank,
                            'list_size': list_size,
                            'HIT_TYPE': hit_type,
                            'is_s40': is_s40,
                            'digit_pack_tags': digit_tags_str,
                            'source_file': file_path.name,
                            # ✅ PHASE 3: NEW TIME-SHIFT FIELDS
                            'time_shift': time_shift,
                            'slot_shift': slot_shift_detail,
                            'hit_family': hit_family,
                            'days_delta': days_delta,
                            'source_strategy': 'BASE'  # Default, can be enhanced later
                        }
                        
                        all_hits.append(hit_data)
                        print(f"     ✅ {real_date.strftime('%Y-%m-%d')} {real_slot}={real_num}: {hit_family} in {pred_slot} (rank {rank}, {time_shift})")
        
        return all_hits
    
    def save_hits_to_memory(self, hits_data):
        """Save hits data to memory file with new time-shift columns."""
        output_dir = Path(__file__).resolve().parent / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "script_hit_memory.xlsx"
        
        # Create DataFrame
        hits_df = pd.DataFrame(hits_data)
        
        if hits_df.empty:
            print("ℹ️ No hits data to save (normal if predictions missed real results)")
            return output_file
        
        # If file exists, append (avoiding duplicates)
        if output_file.exists():
            existing_df = pd.read_excel(output_file)
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, hits_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(
                subset=['date', 'real_slot', 'real_number', 'script', 'predicted_slot', 'prediction_date']
            )
            
            # ✅ PHASE 3: Prune old data (keep last 90 days)
            if 'date' in combined_df.columns:
                combined_df['date'] = pd.to_datetime(combined_df['date'])
                cutoff_date = datetime.now().date() - timedelta(days=90)
                combined_df = combined_df[combined_df['date'].dt.date >= cutoff_date]
        else:
            combined_df = hits_df
        
        # Ensure required columns exist and ordered for downstream consumers
        required_cols = [
            'date', 'real_slot', 'real_number', 'script', 'predicted_slot', 'rank',
            'prediction_date', 'time_shift', 'slot_shift', 'hit_family',
            'days_delta', 'source_strategy', 'HIT_TYPE'
        ]
        for col in required_cols:
            if col not in combined_df.columns:
                combined_df[col] = pd.NA

        combined_df = combined_df[required_cols + [c for c in combined_df.columns if c not in required_cols]]

        # Save to Excel
        combined_df.to_excel(output_file, index=False)
        print(f"💾 Saved {len(combined_df)} hits to {output_file} (with time-shift tracking)")
        
        return output_file
    
    def print_summary(self, hits_data):
        """Print summary of hits with time-shift analysis."""
        if not hits_data:
            print("ℹ️ No hits found for this target date (normal when predictions don't match) – nothing to append.")
            return
        
        # Group by hit family
        hit_families = {}
        for hit in hits_data:
            family = hit['hit_family']
            if family not in hit_families:
                hit_families[family] = []
            hit_families[family].append(hit)
        
        print(f"\n📈 HIT SUMMARY (with Time-Shift Analysis):")
        for family, hits in hit_families.items():
            print(f"   {family}: {len(hits)} hits")
        
        # Per-script summary
        script_summary = defaultdict(list)
        for hit in hits_data:
            script_summary[hit['script']].append(hit)
        
        print(f"\n🔧 PER-SCRIPT SUMMARY:")
        for script, hits in script_summary.items():
            hit_details = []
            for hit in hits:
                detail = f"{hit['real_slot']}({hit['hit_family']} rank{hit['rank']})"
                hit_details.append(detail)
            
            print(f"   {script}: {', '.join(hit_details)}")


def _parse_date_value(val):
    try:
        return pd.to_datetime(val).date()
    except Exception:
        return None


def _build_final_shortlist_map(target_date):
    import quant_paths

    try:
        bet_plan_path = quant_paths.get_bet_plan_master_path(target_date.strftime("%Y-%m-%d"))
        if not bet_plan_path.exists():
            return {}
        sheets = pd.read_excel(bet_plan_path, sheet_name=None)
        bets_df = None
        if isinstance(sheets, dict):
            bets_df = sheets.get("bets") or next(iter(sheets.values()))
        else:
            bets_df = sheets
        if bets_df is None or bets_df.empty:
            return {}
        bets_df.columns = [str(c).strip().lower() for c in bets_df.columns]
        slot_col = next((c for c in bets_df.columns if "slot" in c), None)
        num_col = next((c for c in bets_df.columns if "number" in c), None)
        layer_col = next((c for c in bets_df.columns if "layer" in c), None)
        if slot_col is None or num_col is None:
            return {}
        filtered = bets_df.copy()
        if layer_col and "main" in set(filtered[layer_col].astype(str).str.lower()):
            filtered = filtered[filtered[layer_col].astype(str).str.lower() == "main"]
        shortlist_map = defaultdict(list)
        for _, row in filtered.iterrows():
            slot_val = str(row.get(slot_col, "")).strip().upper()
            number_val = row.get(num_col)
            if pd.isna(number_val):
                continue
            try:
                number_int = int(float(str(number_val).strip())) % 100
                shortlist_map[slot_val].append(number_int)
            except Exception:
                continue
        return shortlist_map
    except Exception:
        return {}


def _collect_script_predictions(memory: PredictionHitMemory, target_date):
    predictions = defaultdict(dict)
    for script_name, pattern in memory.script_patterns.items():
        file_path = memory.find_latest_prediction_file(script_name, pattern)
        if not file_path:
            continue
        rows = memory.load_predictions_file(file_path, script_name)
        if not rows:
            continue
        for entry in rows:
            pred_date = _parse_date_value(entry.get('date'))
            if pred_date != target_date:
                continue
            slot = str(entry.get('slot', '')).upper()
            nums = entry.get('numbers') or []
            if slot and nums:
                predictions[script_name][slot] = nums
    return predictions


def _build_script_hit_rows(memory: PredictionHitMemory, target_date, real_df):
    if real_df is None or real_df.empty:
        return []
    real_df['date'] = pd.to_datetime(real_df['date']).dt.date
    date_df = real_df[real_df['date'] == target_date]
    if date_df.empty:
        return []

    shortlist_map = _build_final_shortlist_map(target_date)
    predictions = _collect_script_predictions(memory, target_date)
    rows = []

    for slot in memory.slots:
        slot_real = date_df[date_df['slot'] == slot]
        if slot_real.empty:
            continue
        real_number = int(slot_real['number'].iloc[0]) % 100
        slot_preds = {k: v.get(slot, []) for k, v in predictions.items() if slot in v}
        real_in_any = any(real_number in vals for vals in slot_preds.values())
        final_shortlist = shortlist_map.get(slot, [])

        for script_name, pred_list in slot_preds.items():
            is_in_shortlist = bool(set([n % 100 for n in pred_list]) & set(final_shortlist))
            has_hit = real_number in [n % 100 for n in pred_list]
            if not real_in_any:
                hit_flag = "BLIND_MISS"
            elif has_hit and real_number in final_shortlist:
                hit_flag = "FINAL_HIT"
            elif has_hit:
                hit_flag = "SCRIPT_HIT_BUT_NOT_FINAL"
            else:
                hit_flag = "BLIND_MISS"

            hit_type = "EXACT" if has_hit else "NONE"
            rows.append({
                'date': target_date.strftime('%Y-%m-%d'),
                'slot': slot,
                'script_name': script_name,
                'real_number': to_2d_str(real_number),
                'top_predictions': '|'.join([to_2d_str(n) for n in pred_list]),
                'is_in_final_shortlist': is_in_shortlist,
                'hit_flag': hit_flag,
                'hit_type': hit_type,
                'created_at': datetime.now().isoformat(),
            })

    return rows


def _rebuild_script_hit_memory(window_days: int):
    print(f"Rebuilding script hit memory CSV for last {window_days} days...")
    memory = PredictionHitMemory()
    real_df = memory.load_real_results(Path(__file__).resolve().parent / "number prediction learn.xlsx")
    real_df['date'] = pd.to_datetime(real_df['date']).dt.date
    max_date = real_df['date'].max()
    min_date = max_date - timedelta(days=window_days - 1)
    rows = []
    for date_val in sorted(real_df['date'].unique()):
        if date_val < min_date:
            continue
        rows.extend(_build_script_hit_rows(memory, date_val, real_df))
    path = rebuild_script_hit_memory(rows)
    print(f"Script hit memory rebuilt at {path}")


def _update_latest_script_hit_memory():
    print("Updating script hit memory for latest completed date...")
    memory = PredictionHitMemory()
    real_df = memory.load_real_results(Path(__file__).resolve().parent / "number prediction learn.xlsx")
    target_date = memory.get_target_date(real_df)
    rows = _build_script_hit_rows(memory, target_date, real_df)
    existing = load_script_hit_memory()
    if not existing.empty and 'date' in existing.columns:
        existing = existing[existing['date'] != target_date.strftime('%Y-%m-%d')]
    updated_rows = existing.to_dict(orient='records') + rows
    path = rebuild_script_hit_memory(updated_rows)
    print(f"Latest script hit memory updated at {path}")

def main():
    parser = argparse.ArgumentParser(description="Prediction hit memory toolkit")
    parser.add_argument("--mode", choices=["legacy", "rebuild", "update-latest"], default="legacy")
    parser.add_argument("--window", type=int, default=30, help="Window in days for rebuild mode")
    args = parser.parse_args()

    if args.mode == "rebuild":
        _rebuild_script_hit_memory(args.window)
        return 0
    if args.mode == "update-latest":
        _update_latest_script_hit_memory()
        return 0

    try:
        print("=== PREDICTION HIT MEMORY - FULL PACK UNIVERSE ===")
        print("📊 Loading real results with full central pack registry (168,129 packs, legacy 837-pack subset included)...")

        memory = PredictionHitMemory()

        real_file = Path(__file__).resolve().parent / "number prediction learn.xlsx"
        real_df = memory.load_real_results(real_file)
        target_date = memory.get_target_date(real_df)
        print(f"🎯 Target date: {target_date}")

        hits_data = memory.analyze_hits(real_df, target_date)
        output_file = memory.save_hits_to_memory(hits_data)
        memory.print_summary(hits_data)

        print(f"\n✅ Enhanced hit memory analysis completed! (Central pack registry: 168,129 packs; legacy 837-pack subset included)")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
