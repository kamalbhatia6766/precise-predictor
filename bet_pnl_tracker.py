# bet_pnl_tracker.py - ENHANCED REALITY P&L TRACKER WITH DEFENSIVE COLUMN HANDLING
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import sys
from typing import Dict, List, Optional, Tuple
import json

warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths
import quant_data_core

class BetPnLTracker:
    def __init__(self):
        self.base_dir = quant_paths.get_base_dir()
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.unit = 10  # ₹10 per bet
        
        # 🆕 Payout multipliers
        self.PAYOUT_MULTIPLIERS = {
            'Main': 90,    # 90x for exact 2-digit hit
            'ANDAR': 9,    # 9x for ANDAR digit hit  
            'BAHAR': 9,    # 9x for BAHAR digit hit
            'S36': 0,      # Not implemented yet
            'PackCore': 0, # Not implemented yet
            'PackBooster': 0 # Not implemented yet
        }
        
    def load_all_bet_plans(self) -> Dict:
        """Load all bet plan files from predictions/bet_engine"""
        bet_plans_dir = quant_paths.get_bet_engine_dir()
        bet_plan_files = list(bet_plans_dir.glob("bet_plan_master_*.xlsx"))
        bet_plans = {}
        
        print(f"🔍 Found {len(bet_plan_files)} bet plan files")
        
        for file in bet_plan_files:
            try:
                # 🆕 Use central path helper to parse date
                date_from_file = quant_paths.parse_date_from_filename(file.stem)
                if date_from_file:
                    bet_plans[date_from_file] = file
                    print(f"   ✅ {date_from_file}: {file.name}")
                else:
                    print(f"   ⚠️  Could not parse date from: {file.name}")
            except Exception as e:
                print(f"   ❌ Error processing {file.name}: {e}")
                continue
                
        return bet_plans

    def safe_column_access(self, df, column_name, default_value=None):
        """🆕 Safe column access with defensive defaults"""
        if column_name in df.columns:
            return df[column_name]
        else:
            print(f"⚠️  Column '{column_name}' not found, using default: {default_value}")
            if default_value is not None:
                return pd.Series([default_value] * len(df))
            else:
                return pd.Series([0] * len(df))

    def parse_bet_plan(self, file_path: Path) -> Dict:
        """Parse a bet plan file and extract structured bet data - DEFENSIVE VERSION"""
        try:
            # Try to read bets sheet
            bets_df = pd.read_excel(file_path, sheet_name='bets')
        except Exception as e:
            print(f"⚠️  Error reading bets sheet from {file_path}: {e}")
            return {}

        # Normalize column names
        bets_df.columns = [str(col).strip().lower() for col in bets_df.columns]
        
        # 🆕 DEFENSIVE: Check required columns exist
        required_columns = ['slot', 'layer_type', 'number_or_digit']
        for col in required_columns:
            if col not in bets_df.columns:
                print(f"❌ Required column '{col}' missing in {file_path}")
                return {}
        
        # 🆕 Extract date from filename if not in data
        file_date = quant_paths.parse_date_from_filename(file_path.stem)
        
        slot_bets = {}
        for slot in self.slots:
            slot_data = bets_df[bets_df['slot'] == slot]
            
            if slot_data.empty:
                continue
                
            # Process Main bets
            main_bets = slot_data[slot_data['layer_type'].str.upper() == 'MAIN']
            main_numbers = []
            main_stake_total = 0
            
            for _, row in main_bets.iterrows():
                num_val = row['number_or_digit']
                stake_val = row.get('stake', self.unit)  # Default to unit if missing
                
                if pd.notna(num_val) and pd.notna(stake_val):
                    try:
                        if isinstance(num_val, str):
                            num_str = num_val.strip()
                            if num_str.isdigit():
                                number = int(num_str)
                                main_numbers.append(number)
                                main_stake_total += float(stake_val)
                        else:
                            number = int(float(num_val))
                            main_numbers.append(number)
                            main_stake_total += float(stake_val)
                    except (ValueError, TypeError):
                        continue

            # Process ANDAR bets
            andar_bets = slot_data[slot_data['layer_type'].str.upper() == 'ANDAR']
            andar_digit = None
            andar_stake = 0
            
            if not andar_bets.empty:
                andar_row = andar_bets.iloc[0]
                andar_val = andar_row['number_or_digit']
                andar_stake = andar_row.get('stake', self.unit)
                
                if pd.notna(andar_val):
                    try:
                        if isinstance(andar_val, str):
                            andar_digit = int(andar_val.strip())
                        else:
                            andar_digit = int(float(andar_val))
                    except (ValueError, TypeError):
                        andar_digit = None

            # Process BAHAR bets
            bahar_bets = slot_data[slot_data['layer_type'].str.upper() == 'BAHAR']
            bahar_digit = None
            bahar_stake = 0
            
            if not bahar_bets.empty:
                bahar_row = bahar_bets.iloc[0]
                bahar_val = bahar_row['number_or_digit']
                bahar_stake = bahar_row.get('stake', self.unit)
                
                if pd.notna(bahar_val):
                    try:
                        if isinstance(bahar_val, str):
                            bahar_digit = int(bahar_val.strip())
                        else:
                            bahar_digit = int(float(bahar_val))
                    except (ValueError, TypeError):
                        bahar_digit = None

            slot_bets[slot] = {
                'main_numbers': main_numbers,
                'main_stake': main_stake_total,
                'andar_digit': andar_digit,
                'andar_stake': andar_stake,
                'bahar_digit': bahar_digit,
                'bahar_stake': bahar_stake,
                'total_stake': main_stake_total + andar_stake + bahar_stake
            }

        return slot_bets

    def load_real_results(self) -> pd.DataFrame:
        """🆕 Load real results using central data core - DEFENSIVE VERSION"""
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
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return True
            if isinstance(value, str):
                try:
                    parsed = pd.to_datetime(value, errors='raise')
                    return not pd.isna(parsed)
                except Exception:
                    return False
            return False

        # Detect whether the first row is data or header
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

        # Force final columns regardless of detection path
        df = df.iloc[:, :5].copy()
        df.columns = ["DATE", "FRBD", "GZBD", "GALI", "DSWR"]

        # Robust DATE parsing
        excel_origin = datetime(1899, 12, 30).date()

        def _parse_date_value(value):
            if pd.isna(value):
                return None
            if isinstance(value, (pd.Timestamp, datetime, np.datetime64)):
                try:
                    return pd.to_datetime(value, errors='coerce').date()
                except Exception:
                    return None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    return (excel_origin + timedelta(days=float(value)))
                except Exception:
                    return None
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    return None
                try:
                    return pd.to_datetime(value, errors='raise').date()
                except Exception:
                    return None
            return None

        parsed_dates = []
        invalid_date_values = []

        for idx, raw_val in enumerate(df['DATE'].tolist()):
            parsed = _parse_date_value(raw_val)
            parsed_dates.append(parsed)
            if parsed is None:
                invalid_date_values.append(raw_val)

        if invalid_date_values:
            sample_values = invalid_date_values[:5]
            print(f"⚠️  Failed to parse {len(invalid_date_values)} DATE entries. Samples: {sample_values}")

        df['DATE'] = pd.to_datetime(parsed_dates, errors='coerce')
        invalid_after_parse = df['DATE'].isna().sum()
        if invalid_after_parse:
            print(f"⚠️  Dropping {invalid_after_parse} rows with unparseable DATE values")
            df = df.dropna(subset=['DATE'])

        # Ensure slot columns exist and are numeric
        for slot in self.slots:
            if slot not in df.columns:
                print(f"⚠️  Slot column '{slot}' not found, creating with NaN values")
                df[slot] = np.nan
            df[slot] = pd.to_numeric(df[slot], errors='coerce')

        # Self-checks and logging
        if not pd.api.types.is_datetime64_any_dtype(df['DATE']):
            print("❌ DATE column could not be converted to datetime64[ns]; exiting gracefully")
            return pd.DataFrame()

        total_rows = len(df)
        unique_dates = df['DATE'].dt.date.dropna().unique()
        if len(unique_dates) == 0:
            print("❌ No valid DATE values found after parsing; exiting gracefully")
            return pd.DataFrame()

        print(f"✅ Loaded {total_rows} real result records; columns={df.columns.tolist()}")
        print(f"📅 Unique DATE count: {len(unique_dates)}; sample: {list(unique_dates)[:5]}")
        print(f"📅 DATE range: {df['DATE'].min().date()} to {df['DATE'].max().date()}")

        return df

    def compute_hits_and_returns(self, slot_bets: Dict, real_number: int) -> Dict:
        """Compute hits and returns for a slot given real result"""
        if real_number is None or pd.isna(real_number):
            return {
                'main_hit': 0, 'main_return': 0,
                'andar_hit': 0, 'andar_return': 0, 
                'bahar_hit': 0, 'bahar_return': 0,
                'total_return': 0
            }
        
        # Extract tens and ones digits
        tens_digit = real_number // 10
        ones_digit = real_number % 10
        
        main_hit = 1 if real_number in slot_bets.get('main_numbers', []) else 0
        main_return = main_hit * slot_bets.get('main_stake', 0) * self.PAYOUT_MULTIPLIERS['Main']
        
        andar_hit = 1 if (slot_bets.get('andar_digit') is not None and 
                          tens_digit == slot_bets['andar_digit']) else 0
        andar_return = andar_hit * slot_bets.get('andar_stake', 0) * self.PAYOUT_MULTIPLIERS['ANDAR']
        
        bahar_hit = 1 if (slot_bets.get('bahar_digit') is not None and 
                          ones_digit == slot_bets['bahar_digit']) else 0
        bahar_return = bahar_hit * slot_bets.get('bahar_stake', 0) * self.PAYOUT_MULTIPLIERS['BAHAR']
        
        total_return = main_return + andar_return + bahar_return
        
        return {
            'main_hit': main_hit,
            'main_return': main_return,
            'andar_hit': andar_hit, 
            'andar_return': andar_return,
            'bahar_hit': bahar_hit,
            'bahar_return': bahar_return,
            'total_return': total_return
        }

    def compute_pnl(self, bet_plans: Dict, real_results_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute P&L for all dates with both bet plans and real results - DEFENSIVE VERSION"""
        slot_level_data = []
        layer_level_data = []

        # Get unique dates from bet_plans
        dates = sorted(bet_plans.keys())
        processed_dates = []

        # Normalize real results date column once
        real_results_df = real_results_df.copy()
        real_results_df['DATE_ONLY'] = pd.to_datetime(real_results_df['DATE'], errors='coerce').dt.date

        def _normalize_bet_plan_date(date_value):
            if isinstance(date_value, datetime):
                return date_value.date()
            if hasattr(date_value, 'date'):
                try:
                    return date_value.date()
                except Exception:
                    pass
            if isinstance(date_value, str):
                for fmt in ['%Y-%m-%d', '%Y%m%d']:
                    try:
                        return datetime.strptime(date_value, fmt).date()
                    except ValueError:
                        continue
            if isinstance(date_value, (pd.Timestamp, np.datetime64)):
                try:
                    return pd.to_datetime(date_value).date()
                except Exception:
                    return None
            return date_value if hasattr(date_value, 'year') and hasattr(date_value, 'month') and hasattr(date_value, 'day') else None

        def _find_matching_results(date_obj):
            tried_dates = []
            candidates = [date_obj, date_obj - timedelta(days=1), date_obj + timedelta(days=1)]
            for cand in candidates:
                tried_dates.append(cand)
                subset = real_results_df[real_results_df['DATE_ONLY'] == cand]
                if not subset.empty:
                    return subset, tried_dates, cand
            return None, tried_dates, None

        matched_dates_data = []
        for date in dates:
            date_obj = _normalize_bet_plan_date(date)
            if not date_obj:
                print(f"⚠️  Invalid date format: {date}, skipping")
                continue

            date_results, tried_dates, matched_date = _find_matching_results(date_obj)
            if date_results is None:
                print(f"⚠️  No real results found for bet plan date {date_obj}; tried dates: {[d.isoformat() for d in tried_dates]}")
                continue

            if matched_date != date_obj:
                print(f"ℹ️  Using real results from {matched_date} for bet plan date {date_obj}")

            matched_dates_data.append({
                'date_obj': date_obj,
                'date_str': date_obj.isoformat(),
                'bet_file': bet_plans[date],
                'date_results': date_results
            })

        print(f"📅 Bet plan dates with real result matches: {len(matched_dates_data)} / {len(dates)}")
        if not matched_dates_data:
            return pd.DataFrame(slot_level_data), pd.DataFrame(layer_level_data)

        for match in matched_dates_data:
            date_obj = match['date_obj']
            date = match['date_str']
            date_results = match['date_results']

            slot_bets = self.parse_bet_plan(match['bet_file'])
            if not slot_bets:
                print(f"⚠️  No valid bet data for date: {date}")
                continue
                
            processed_dates.append(date)
            day_stake_total = 0
            day_return_total = 0
            
            for slot in self.slots:
                if slot not in slot_bets:
                    continue
                    
                bets = slot_bets[slot]
                real_number = None
                
                # 🆕 DEFENSIVE: Find real result for this slot and date
                if not date_results.empty and slot in date_results.columns:
                    valid_results = date_results[date_results[slot].notna()]
                    if not valid_results.empty:
                        real_number = valid_results[slot].iloc[0]
                        # 🆕 Ensure real_number is integer
                        try:
                            real_number = int(float(real_number))
                        except (ValueError, TypeError):
                            real_number = None
                
                # Compute hits and returns
                returns_data = self.compute_hits_and_returns(bets, real_number)
                
                stake_total = bets['total_stake']
                return_total = returns_data['total_return']
                pnl = return_total - stake_total
                roi_pct = (return_total / stake_total - 1) * 100 if stake_total > 0 else 0
                
                # Slot-level data
                slot_level_data.append({
                    'date': date,
                    'slot': slot,
                    'result_number': real_number if real_number is not None else 'MISSING',
                    'total_stake': stake_total,
                    'total_return': return_total,
                    'pnl': pnl,
                    'roi_pct': roi_pct,
                    'main_hit': returns_data['main_hit'],
                    'andar_hit': returns_data['andar_hit'],
                    'bahar_hit': returns_data['bahar_hit']
                })
                
                # Layer-level data
                # Main layer
                if bets['main_stake'] > 0:
                    main_roi = (returns_data['main_return'] / bets['main_stake'] - 1) * 100 if bets['main_stake'] > 0 else 0
                    layer_level_data.append({
                        'date': date,
                        'slot': slot,
                        'layer_type': 'Main',
                        'stake': bets['main_stake'],
                        'return': returns_data['main_return'],
                        'pnl': returns_data['main_return'] - bets['main_stake'],
                        'roi_pct': main_roi,
                        'hit': returns_data['main_hit']
                    })
                
                # ANDAR layer
                if bets['andar_stake'] > 0 and bets['andar_digit'] is not None:
                    andar_roi = (returns_data['andar_return'] / bets['andar_stake'] - 1) * 100 if bets['andar_stake'] > 0 else 0
                    layer_level_data.append({
                        'date': date,
                        'slot': slot,
                        'layer_type': 'ANDAR',
                        'stake': bets['andar_stake'],
                        'return': returns_data['andar_return'],
                        'pnl': returns_data['andar_return'] - bets['andar_stake'],
                        'roi_pct': andar_roi,
                        'hit': returns_data['andar_hit']
                    })
                
                # BAHAR layer
                if bets['bahar_stake'] > 0 and bets['bahar_digit'] is not None:
                    bahar_roi = (returns_data['bahar_return'] / bets['bahar_stake'] - 1) * 100 if bets['bahar_stake'] > 0 else 0
                    layer_level_data.append({
                        'date': date,
                        'slot': slot,
                        'layer_type': 'BAHAR',
                        'stake': bets['bahar_stake'],
                        'return': returns_data['bahar_return'],
                        'pnl': returns_data['bahar_return'] - bets['bahar_stake'],
                        'roi_pct': bahar_roi,
                        'hit': returns_data['bahar_hit']
                    })
                
                day_stake_total += stake_total
                day_return_total += return_total
            
            # Add day-level summary row
            day_pnl = day_return_total - day_stake_total
            day_roi = (day_return_total / day_stake_total - 1) * 100 if day_stake_total > 0 else 0
            
            slot_level_data.append({
                'date': date,
                'slot': 'DAY_TOTAL',
                'result_number': '',
                'total_stake': day_stake_total,
                'total_return': day_return_total,
                'pnl': day_pnl,
                'roi_pct': day_roi,
                'main_hit': '',
                'andar_hit': '',
                'bahar_hit': ''
            })
        
        print(f"✅ Processed {len(processed_dates)} dates with complete data")
        
        # 🆕 DEFENSIVE: Create empty DataFrames if no data
        if not slot_level_data:
            slot_level_data = [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'slot': 'NO_DATA',
                'result_number': 'MISSING',
                'total_stake': 0,
                'total_return': 0,
                'pnl': 0,
                'roi_pct': 0,
                'main_hit': 0,
                'andar_hit': 0,
                'bahar_hit': 0
            }]
        
        if not layer_level_data:
            layer_level_data = [{
                'date': datetime.now().strftime('%Y-%m-%d'),
                'slot': 'NO_DATA',
                'layer_type': 'Main',
                'stake': 0,
                'return': 0,
                'pnl': 0,
                'roi_pct': 0,
                'hit': 0
            }]
        
        return pd.DataFrame(slot_level_data), pd.DataFrame(layer_level_data)

    def build_summary_tables(self, slot_pnl_df: pd.DataFrame, layer_pnl_df: pd.DataFrame) -> Dict:
        """Build comprehensive summary tables - DEFENSIVE VERSION"""
        # 🆕 DEFENSIVE: Handle empty DataFrames
        if slot_pnl_df.empty or layer_pnl_df.empty:
            return {
                'overall': {
                    'total_stake': 0,
                    'total_return': 0,
                    'total_pnl': 0,
                    'overall_roi': 0,
                    'days_processed': 0,
                    'date_range': {'start': 'N/A', 'end': 'N/A'}
                },
                'by_slot': [],
                'by_layer': [],
                'daily': []
            }
        
        # Filter out day total rows for slot analysis
        slot_analysis_df = slot_pnl_df[slot_pnl_df['slot'] != 'DAY_TOTAL']
        
        # Overall summary
        total_stake = slot_analysis_df['total_stake'].sum()
        total_return = slot_analysis_df['total_return'].sum()
        total_pnl = total_return - total_stake
        overall_roi = (total_return / total_stake - 1) * 100 if total_stake > 0 else 0
        
        # By slot summary
        slot_summary = slot_analysis_df.groupby('slot').agg({
            'total_stake': 'sum',
            'total_return': 'sum',
            'pnl': 'sum',
            'main_hit': 'sum',
            'andar_hit': 'sum', 
            'bahar_hit': 'sum'
        }).reset_index()
        
        slot_summary['roi_pct'] = (slot_summary['total_return'] / slot_summary['total_stake'] - 1) * 100
        slot_summary = slot_summary.round(2)
        
        # By layer type summary
        layer_summary = layer_pnl_df.groupby('layer_type').agg({
            'stake': 'sum',
            'return': 'sum',
            'pnl': 'sum',
            'hit': 'sum'
        }).reset_index()
        
        layer_summary['roi_pct'] = (layer_summary['return'] / layer_summary['stake'] - 1) * 100
        layer_summary = layer_summary.round(2)
        
        # Daily performance
        daily_summary = slot_pnl_df[slot_pnl_df['slot'] == 'DAY_TOTAL'][['date', 'total_stake', 'total_return', 'pnl', 'roi_pct']]
        daily_summary = daily_summary.round(2)
        
        return {
            'overall': {
                'total_stake': total_stake,
                'total_return': total_return,
                'total_pnl': total_pnl,
                'overall_roi': overall_roi,
                'days_processed': len(daily_summary),
                'date_range': {
                    'start': slot_pnl_df['date'].min(),
                    'end': slot_pnl_df['date'].max()
                }
            },
            'by_slot': slot_summary.to_dict('records'),
            'by_layer': layer_summary.to_dict('records'),
            'daily': daily_summary.to_dict('records')
        }

    def export_quant_pnl_summary(self, slot_pnl_df: pd.DataFrame) -> None:
        """Export per-day, per-slot stake/return summary for ROI analysis"""
        if slot_pnl_df.empty:
            return

        daily_records = []
        slot_df = slot_pnl_df[slot_pnl_df['slot'] != 'DAY_TOTAL']

        for date_val, group in slot_df.groupby('date'):
            record = {'DATE': date_val}
            total_stake = 0
            total_return = 0

            for slot in self.slots:
                slot_rows = group[group['slot'] == slot]
                stake_val = float(slot_rows['total_stake'].sum()) if not slot_rows.empty else 0.0
                return_val = float(slot_rows['total_return'].sum()) if not slot_rows.empty else 0.0
                record[f"{slot}_stake"] = stake_val
                record[f"{slot}_return"] = return_val
                total_stake += stake_val
                total_return += return_val

            record['TOTAL_STAKE'] = total_stake
            record['TOTAL_RETURN'] = total_return
            daily_records.append(record)

        if not daily_records:
            return

        df_summary = pd.DataFrame(daily_records)
        df_summary = df_summary.sort_values('DATE')

        output_path = self.base_dir / "logs" / "performance" / "quant_pnl_summary.xlsx"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_summary.to_excel(output_path, index=False)

    def save_pnl_master_file(self, slot_pnl_df: pd.DataFrame, layer_pnl_df: pd.DataFrame, summary_data: Dict):
        """Save master P&L file with all details"""
        output_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.xlsx"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 🆕 Load existing data if available for idempotent behavior
        if output_file.exists():
            try:
                existing_slot = pd.read_excel(output_file, sheet_name='daily_slot_pnl')
                existing_layer = pd.read_excel(output_file, sheet_name='daily_layer_pnl')
                
                # Remove existing rows for dates we're about to write
                dates_to_update = slot_pnl_df['date'].unique()
                existing_slot = existing_slot[~existing_slot['date'].isin(dates_to_update)]
                existing_layer = existing_layer[~existing_layer['date'].isin(dates_to_update)]
                
                # Combine old and new data
                slot_pnl_df = pd.concat([existing_slot, slot_pnl_df]).sort_values('date')
                layer_pnl_df = pd.concat([existing_layer, layer_pnl_df]).sort_values('date')
                
                print(f"💾 Updated existing P&L file, preserved {len(existing_slot)} existing slot records")
            except Exception as e:
                print(f"⚠️  Error reading existing P&L file, creating new: {e}")
        
        # Create summary DataFrame
        summary_df = pd.DataFrame([summary_data['overall']])
        slot_summary_df = pd.DataFrame(summary_data['by_slot'])
        layer_summary_df = pd.DataFrame(summary_data['by_layer'])
        daily_summary_df = pd.DataFrame(summary_data['daily'])
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            slot_pnl_df.to_excel(writer, sheet_name='daily_slot_pnl', index=False)
            layer_pnl_df.to_excel(writer, sheet_name='daily_layer_pnl', index=False)
            summary_df.to_excel(writer, sheet_name='summary_overall', index=False)
            slot_summary_df.to_excel(writer, sheet_name='summary_by_slot', index=False)
            layer_summary_df.to_excel(writer, sheet_name='summary_by_layer', index=False)
            daily_summary_df.to_excel(writer, sheet_name='summary_daily', index=False)
        
        print(f"💾 Master P&L saved: {output_file}")
        
        # 🆕 Also save JSON for easy consumption by other scripts
        json_output = output_file.with_suffix('.json')
        with open(json_output, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f"💾 JSON summary saved: {json_output}")

    def print_console_summary(self, summary_data: Dict):
        """Print comprehensive console summary"""
        overall = summary_data['overall']
        
        print("\n" + "="*70)
        print("📊 QUANT REALITY P&L - MASTER SUMMARY")
        print("="*70)
        
        print(f"📅 Date Range: {overall['date_range']['start']} to {overall['date_range']['end']}")
        print(f"📈 Days Processed: {overall['days_processed']}")
        print(f"💰 Total Stake: ₹{overall['total_stake']:.0f}")
        print(f"🎯 Total Return: ₹{overall['total_return']:.0f}")
        print(f"💵 Total P&L: ₹{overall['total_pnl']:+.0f}")
        print(f"📊 Overall ROI: {overall['overall_roi']:+.1f}%")
        
        if summary_data['by_slot']:
            print(f"\n🏆 SLOT PERFORMANCE:")
            slot_performance = sorted(summary_data['by_slot'], key=lambda x: x['roi_pct'], reverse=True)
            for slot in slot_performance:
                trend = "🔴" if slot['roi_pct'] < 0 else "🟢"
                print(f"   {trend} {slot['slot']}: {slot['roi_pct']:+.1f}% (₹{slot['pnl']:+.0f})")
        
        if summary_data['by_layer']:
            print(f"\n🎯 LAYER PERFORMANCE:")
            for layer in summary_data['by_layer']:
                trend = "🔴" if layer['roi_pct'] < 0 else "🟢"
                hit_rate = (layer['hit'] / (layer['stake'] / self.unit) * 100) if layer['stake'] > 0 else 0
                print(f"   {trend} {layer['layer_type']}: {layer['roi_pct']:+.1f}% (Hit Rate: {hit_rate:.1f}%)")
        
        # Find best and worst performing slots
        if summary_data['by_slot']:
            best_slot = max(summary_data['by_slot'], key=lambda x: x['roi_pct'])
            worst_slot = min(summary_data['by_slot'], key=lambda x: x['roi_pct'])
            print(f"\n⭐ BEST PERFORMER: {best_slot['slot']} ({best_slot['roi_pct']:+.1f}%)")
            print(f"📉 WORST PERFORMER: {worst_slot['slot']} ({worst_slot['roi_pct']:+.1f}%)")
        
        print("="*70)

    def run(self, days_back: Optional[int] = None):
        """Main function to run the enhanced P&L tracker"""
        print("🔍 QUANT REALITY P&L TRACKER - ENHANCED & DEFENSIVE")
        print("="*50)
        
        # Step 1: Load bet plans
        bet_plans = self.load_all_bet_plans()
        if not bet_plans:
            print("❌ No bet plan files found.")
            return False
        
        # Step 2: Load real results
        real_results_df = self.load_real_results()
        if real_results_df.empty:
            print("❌ No real results found.")
            return False
        
        # Step 3: Filter by days_back if specified
        if days_back:
            cutoff_date = datetime.now().date() - timedelta(days=days_back)
            def _ensure_date(date_val):
                if isinstance(date_val, datetime):
                    return date_val.date()
                if hasattr(date_val, 'year') and hasattr(date_val, 'month') and hasattr(date_val, 'day'):
                    return date_val
                if isinstance(date_val, str):
                    for fmt in ['%Y-%m-%d', '%Y%m%d']:
                        try:
                            return datetime.strptime(date_val, fmt).date()
                        except ValueError:
                            continue
                return None

            bet_plans = {date: file for date, file in bet_plans.items()
                         if _ensure_date(date) and _ensure_date(date) >= cutoff_date}
            print(f"📅 Filtered to last {days_back} days: {len(bet_plans)} bet plans")
        
        # Step 4: Compute P&L
        slot_pnl_df, layer_pnl_df = self.compute_pnl(bet_plans, real_results_df)
        
        if slot_pnl_df.empty:
            print("❌ No P&L data computed. Check if dates in bet plans and real results match.")
            return False
        
        # Step 5: Build summaries
        summary_data = self.build_summary_tables(slot_pnl_df, layer_pnl_df)

        # Step 6: Save master file
        self.save_pnl_master_file(slot_pnl_df, layer_pnl_df, summary_data)

        # Step 6b: Export compact quant P&L summary for ROI consumers
        self.export_quant_pnl_summary(slot_pnl_df)
        
        # Step 7: Print console summary
        self.print_console_summary(summary_data)
        
        print("\n✅ Enhanced P&L tracking completed successfully!")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Reality P&L Tracker')
    parser.add_argument('--days-back', type=int, help='Process only last N days of data')
    parser.add_argument('--all', action='store_true', help='Process all available data (default)')
    
    args = parser.parse_args()
    
    tracker = BetPnLTracker()
    
    if args.days_back:
        success = tracker.run(days_back=args.days_back)
    else:
        success = tracker.run()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
