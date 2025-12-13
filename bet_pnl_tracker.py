# bet_pnl_tracker.py - ENHANCED REALITY P&L TRACKER WITH DEFENSIVE COLUMN HANDLING
# Additions:
# - Slot-wise forensic summaries with configurable window (--days)
# - FRBD-focused slump diagnostics and optional debug alignment tracer (--debug-frbd)
# - Forensic CSV export under output/ for offline review
# Examples:
#   py -3.12 bet_pnl_tracker.py --days 45
#   py -3.12 bet_pnl_tracker.py --days 30 --debug-frbd
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
        self.SLOT_COLUMNS = {
            "FRBD": "FRBD",
            "GZBD": "GZBD",
            "GALI": "GALI",
            "DSWR": "DSWR"
        }
        self.unit = 10  # ₹10 per bet
        self.matched_dates_data = []
        
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
        
        for file in bet_plan_files:
            try:
                # 🆕 Use central path helper to parse date
                date_from_file = quant_paths.parse_date_from_filename(file.stem)
                if date_from_file:
                    bet_plans[date_from_file] = file
                else:
                    print(
                        f"   ℹ️  Skipping non-standard file (intra run): {file.name} – date not parsed (expected for experimental files)."
                    )
            except Exception as e:
                print(f"   ❌ Error processing {file.name}: {e}")
                continue

        print(f"Loaded {len(bet_plan_files)} bet plan files (showing first 5):")
        for f in bet_plan_files[:5]:
            print("   •", f.name)
        print("   …")

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

    def normalize_date_value(self, date_value):
        """Normalize various date formats to date objects"""
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
            try:
                parsed = pd.to_datetime(date_value, errors='coerce')
                if not pd.isna(parsed):
                    return parsed.date()
            except Exception:
                return None
        if isinstance(date_value, (pd.Timestamp, np.datetime64)):
            try:
                return pd.to_datetime(date_value).date()
            except Exception:
                return None
        if hasattr(date_value, 'year') and hasattr(date_value, 'month') and hasattr(date_value, 'day'):
            return date_value
        return None

    def normalize_result(self, value: Optional[object]) -> Optional[int]:
        """Normalize a result value to an integer between 0-99 or return None if invalid/missing"""
        if value is None:
            return None
        try:
            if isinstance(value, float) and np.isnan(value):
                return None
        except Exception:
            pass

        if isinstance(value, str):
            value = value.strip()
            if value.upper() in {"MISSING", "XX", "NAN", ""}:
                return None
            if not value.isdigit():
                return None
            try:
                value = int(value)
            except Exception:
                return None

        try:
            int_val = int(float(value))
        except Exception:
            return None

        if 0 <= int_val <= 99:
            return int_val
        return None

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
        expected_columns = ["DATE"] + [self.SLOT_COLUMNS[slot] for slot in self.slots]
        df.columns = expected_columns

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
        max_real_date = None
        if 'DATE_ONLY' in real_results_df.columns:
            valid_dates = real_results_df['DATE_ONLY'].dropna()
            if not valid_dates.empty:
                max_real_date = valid_dates.max()

        matched_dates_data = []
        for date in dates:
            date_obj = self.normalize_date_value(date)
            if not date_obj:
                print(f"⚠️  Invalid date format: {date}, skipping")
                continue

            if max_real_date and date_obj > max_real_date:
                print(f"ℹ️  Skipping bet plan date {date_obj}: no real result available yet (max real date: {max_real_date})")
                continue

            date_results = real_results_df[real_results_df['DATE_ONLY'] == date_obj]
            if date_results.empty:
                print(f"⚠️  No real results found for bet plan date {date_obj}; skipping")
                continue

            matched_dates_data.append({
                'date_obj': date_obj,
                'date_str': date_obj.isoformat(),
                'bet_file': bet_plans[date],
                'date_results': date_results
            })

        print(f"📅 Bet plan dates with real result matches: {len(matched_dates_data)} / {len(dates)}")
        self.matched_dates_data = matched_dates_data
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
                    'bahar_hit': returns_data['bahar_hit'],
                    'bet_numbers': bets.get('main_numbers', [])
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
                'bahar_hit': '',
                'bet_numbers': []
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
                'bahar_hit': 0,
                'bet_numbers': []
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
        slot_analysis_df = slot_pnl_df[slot_pnl_df['slot'] != 'DAY_TOTAL'].copy()

        # Normalize bet_numbers and compute validity flags for win/loss tracking
        slot_analysis_df['bet_numbers'] = slot_analysis_df['bet_numbers'].apply(
            lambda x: x if isinstance(x, list) else []
        )
        slot_analysis_df['has_bet'] = slot_analysis_df['bet_numbers'].apply(lambda bets: len(bets) > 0)
        slot_analysis_df['norm_result'] = slot_analysis_df['result_number'].apply(self.normalize_result)
        slot_analysis_df['valid_row'] = slot_analysis_df['has_bet'] & slot_analysis_df['norm_result'].notna()
        slot_analysis_df['win_flag'] = slot_analysis_df[['main_hit', 'andar_hit', 'bahar_hit']].sum(axis=1) > 0
        
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

        valid_rows = slot_analysis_df[slot_analysis_df['valid_row']]
        wins_losses = valid_rows.groupby('slot').agg(
            wins=('win_flag', 'sum'),
            total_valid=('win_flag', 'size')
        ).reset_index()
        wins_losses['losses'] = wins_losses['total_valid'] - wins_losses['wins']
        wins_losses = wins_losses[['slot', 'wins', 'losses']]

        slot_summary = slot_summary.merge(wins_losses, on='slot', how='left')
        slot_summary[['wins', 'losses']] = slot_summary[['wins', 'losses']].fillna(0).astype(int)
        slot_summary['hit_rate'] = slot_summary.apply(
            lambda row: (row['wins'] / (row['wins'] + row['losses'])) if (row['wins'] + row['losses']) > 0 else 0.0,
            axis=1
        )

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

    def compute_slot_forensics(self, slot_pnl_df: pd.DataFrame, days_window: int = 30) -> Optional[Dict]:
        """Build slot-wise forensic stats for the last N days"""
        if slot_pnl_df.empty:
            print("⚠️  Skipping forensic computation: empty P&L frame")
            return None

        slot_df = slot_pnl_df[slot_pnl_df['slot'] != 'DAY_TOTAL'].copy()
        slot_df['date_dt'] = pd.to_datetime(slot_df['date'], errors='coerce').dt.date

        if slot_df['date_dt'].isna().all():
            print("⚠️  Skipping forensic computation: no parsable dates")
            return None

        max_date = slot_df['date_dt'].max()
        if pd.isna(max_date):
            print("⚠️  Skipping forensic computation: unable to detect max date")
            return None

        cutoff = max_date - timedelta(days=days_window - 1)
        slot_df = slot_df[slot_df['date_dt'] >= cutoff]

        slot_summaries = []
        frbd_daily = []
        per_slot_diag = {}

        def _has_bet(numbers):
            if numbers is None:
                return False
            if isinstance(numbers, (list, tuple, set)):
                return len(numbers) > 0
            return bool(numbers)

        for slot in self.slots:
            slot_data = slot_df[slot_df['slot'] == slot].copy()
            slot_data['bet_numbers'] = slot_data['bet_numbers'].apply(lambda x: x if isinstance(x, list) else [])
            slot_data['has_bet'] = slot_data['bet_numbers'].apply(_has_bet)
            slot_data['norm_result'] = slot_data['result_number'].apply(self.normalize_result)

            slot_data_valid = slot_data[slot_data['has_bet'] & slot_data['norm_result'].notna()].copy()

            total_bet = slot_data_valid['total_stake'].sum()
            total_return = slot_data_valid['total_return'].sum()
            net_pnl = total_return - total_bet
            roi_percent = (net_pnl / total_bet * 100) if total_bet > 0 else 0

            slot_data_valid['win_flag'] = slot_data_valid[['main_hit', 'andar_hit', 'bahar_hit']].sum(axis=1) > 0
            wins = int(slot_data_valid['win_flag'].sum())
            losses = int(len(slot_data_valid) - wins)
            hit_rate = (wins / (wins + losses)) if (wins + losses) > 0 else 0

            slot_summary = {
                'slot': slot,
                'days': days_window,
                'total_bet': total_bet,
                'total_return': total_return,
                'net_pnl': net_pnl,
                'roi_percent': roi_percent,
                'wins': wins,
                'losses': losses,
                'hit_rate': hit_rate
            }

            slot_summaries.append(slot_summary)

            per_slot_diag[slot] = {
                **self._compute_slot_streaks(slot_data),
                'roi_percent': roi_percent,
                'days_window': days_window
            }

            if slot == 'FRBD':
                for _, row in slot_data.iterrows():
                    frbd_daily.append({
                        'date': row['date_dt'],
                        'total_bet': row['total_stake'],
                        'total_return': row['total_return'],
                        'net_pnl': row['total_return'] - row['total_stake'],
                        'hit_flag': 'HIT' if row[['main_hit', 'andar_hit', 'bahar_hit']].sum() > 0 else 'MISS',
                        'status': 'VALID' if row['has_bet'] and pd.notna(row['norm_result']) else ('SKIP_NO_BET' if not row['has_bet'] else 'SKIP_NO_RESULT')
                    })

        frbd_summary = next((s for s in slot_summaries if s['slot'] == 'FRBD'), None)
        others = [s for s in slot_summaries if s['slot'] != 'FRBD' and s['total_bet'] > 0]
        avg_roi_others = np.mean([s['roi_percent'] for s in others]) if others else 0

        frbd_diag = per_slot_diag.get('FRBD', {})
        if frbd_diag:
            frbd_diag.update({
                'roi_percent': frbd_summary['roi_percent'] if frbd_summary else 0,
                'avg_roi_others': avg_roi_others,
                'days_window': days_window
            })

        return {
            'slot_summaries': slot_summaries,
            'frbd_diag': frbd_diag,
            'frbd_daily': sorted(frbd_daily, key=lambda x: x['date'] or datetime.min.date()),
            'per_slot_diag': per_slot_diag
        }

    def build_slot_slump_diagnostics(self, forensic_data: Dict) -> Dict:
        """Convert slump diagnostics into JSON-safe structure"""
        if not forensic_data:
            return {}

        slot_summaries = forensic_data.get('slot_summaries', [])
        per_slot_diag = forensic_data.get('per_slot_diag', {})
        frbd_diag = forensic_data.get('frbd_diag', {})

        if not slot_summaries or not per_slot_diag:
            return {}

        slot_roi_map = {row['slot']: row.get('roi_percent', 0.0) for row in slot_summaries}
        diagnostics = {}

        for slot in self.slots:
            diag = per_slot_diag.get(slot, {})
            if not diag:
                continue

            slot_roi = float(diag.get('roi_percent', 0.0))

            if slot == 'FRBD':
                avg_other = float(frbd_diag.get('avg_roi_others', 0.0))
            else:
                other_rois = [roi for s, roi in slot_roi_map.items() if s != slot]
                avg_other = float(np.mean(other_rois)) if other_rois else 0.0

            diagnostics[slot] = {
                'last_hit_date': diag.get('last_hit_date').isoformat() if diag.get('last_hit_date') else None,
                'longest_losing_streak': int(diag.get('longest_losing_streak', 0) or 0),
                'current_losing_streak': int(diag.get('current_losing_streak', 0) or 0),
                'slot_roi_pct': slot_roi,
                'others_avg_roi_pct': avg_other,
                'roi_diff_vs_others_pct': slot_roi - avg_other
            }

        return diagnostics

    def _merge_slump_into_summary(self, summary_data: Dict, forensic_data: Dict) -> Dict:
        """Embed streak/slump signals into the per-slot summary for JSON persistence"""
        if not summary_data or not summary_data.get('by_slot'):
            return summary_data

        per_slot_diag = forensic_data.get('per_slot_diag', {}) if forensic_data else {}
        if not per_slot_diag:
            return summary_data

        slot_map = {
            str(entry.get('slot')).upper(): entry
            for entry in summary_data.get('by_slot', [])
            if isinstance(entry, dict) and entry.get('slot')
        }

        for slot, diag in per_slot_diag.items():
            entry = slot_map.get(slot)
            if not entry:
                continue

            current_streak = int(diag.get('current_losing_streak', 0) or 0)
            longest_streak = int(diag.get('longest_losing_streak', 0) or 0)
            roi_pct = float(diag.get('roi_percent', entry.get('roi_percent', entry.get('roi_pct', 0.0))))

            in_slump = (roi_pct <= -30.0) and (current_streak >= 2)

            entry['current_losing_streak'] = current_streak
            entry['longest_losing_streak'] = longest_streak
            entry['in_slump'] = in_slump

        return summary_data

    def _compute_slot_streaks(self, slot_df: pd.DataFrame) -> Dict:
        """Compute slot streak metrics for any slot"""
        if slot_df.empty:
            return {
                'last_hit_date': None,
                'longest_losing_streak': 0,
                'current_losing_streak': 0
            }

        slot_df = slot_df.sort_values('date_dt').copy()
        slot_df['bet_numbers'] = slot_df['bet_numbers'].apply(lambda x: x if isinstance(x, list) else [])
        slot_df['has_bet'] = slot_df['bet_numbers'].apply(lambda x: len(x) > 0)
        slot_df['norm_result'] = slot_df['result_number'].apply(self.normalize_result)
        slot_df = slot_df[slot_df['has_bet'] & slot_df['norm_result'].notna()]

        if slot_df.empty:
            return {
                'last_hit_date': None,
                'longest_losing_streak': 0,
                'current_losing_streak': 0
            }

        slot_df['win_flag'] = slot_df[['main_hit', 'andar_hit', 'bahar_hit']].sum(axis=1) > 0
        dates = slot_df['date_dt'].tolist()
        wins = slot_df['win_flag'].tolist()

        last_hit_date = None
        for d, w in zip(dates, wins):
            if w:
                last_hit_date = d

        longest = 0
        current = 0
        streak = 0
        for w in wins:
            if not w:
                streak += 1
            else:
                longest = max(longest, streak)
                streak = 0
        longest = max(longest, streak)

        for w in reversed(wins):
            if not w:
                current += 1
            else:
                break

        return {
            'last_hit_date': last_hit_date,
            'longest_losing_streak': longest,
            'current_losing_streak': current
        }

    def print_slot_forensic_table(self, forensic_data: Dict):
        """Print slot-wise forensic summary table"""
        if not forensic_data:
            return

        slot_rows = forensic_data.get('slot_summaries', [])
        if not slot_rows:
            return

        print("\nSLOT  DAYS  TOTAL_BET  TOTAL_RETURN  NET_PNL  ROI_%  WINS  LOSSES  HIT_RATE")
        for row in slot_rows:
            print(f"{row['slot']:<4} {row['days']:<4} {row['total_bet']:<10.2f} {row['total_return']:<12.2f} "
                  f"{row['net_pnl']:<8.2f} {row['roi_percent']:<6.2f} {row['wins']:<5} {row['losses']:<7} "
                  f"{row['hit_rate']:.2f}")

    def print_frbd_slump_diagnostics(self, forensic_data: Dict):
        """Print FRBD slump diagnostic block"""
        diag = forensic_data.get('frbd_diag', {}) if forensic_data else {}
        if not diag:
            return

        days_window = diag.get('days_window', 30)
        print(f"\n=== FRBD SLUMP DIAGNOSTICS (last {days_window} days) ===")
        last_hit = diag.get('last_hit_date')
        last_hit_text = last_hit.isoformat() if last_hit else "no hits in window"
        print(f"Last FRBD hit date: {last_hit_text}")
        print(f"Longest losing streak: {diag.get('longest_losing_streak', 0)} days")
        print(f"Current losing streak: {diag.get('current_losing_streak', 0)} days")
        frbd_roi = diag.get('roi_percent', None)
        avg_others = diag.get('avg_roi_others', 0.0)
        print(f"FRBD ROI: {frbd_roi:+.2f}%")
        print(f"Other slots average ROI: {avg_others:+.2f}%")
        if frbd_roi is not None:
            delta = frbd_roi - avg_others
            if delta < -1e-6:
                print(f"FRBD is underperforming vs others by {abs(delta):.2f}% over last {days_window} days")
            elif delta > 1e-6:
                print(f"FRBD is outperforming others by {delta:.2f}% over last {days_window} days")
            else:
                print(f"FRBD is performing in line with other slots over last {days_window} days")

    def print_all_slot_slump_diagnostics(self, forensic_data: Dict):
        """Print slump diagnostics block for all slots"""
        if not forensic_data:
            return

        per_slot_diag = forensic_data.get('per_slot_diag', {})
        slot_summaries = forensic_data.get('slot_summaries', [])
        if not per_slot_diag:
            return

        slot_roi_map = {row['slot']: row.get('roi_percent', 0) for row in slot_summaries}

        for slot in self.slots:
            # 🆕 Skip FRBD here because it already has a dedicated block
            if slot == "FRBD":
                continue

            diag = per_slot_diag.get(slot, {})
            if not diag:
                continue

            days_window = diag.get('days_window', 30)
            print(f"\n=== {slot} SLUMP DIAGNOSTICS (last {days_window} days) ===")
            last_hit = diag.get('last_hit_date')
            last_hit_text = last_hit.isoformat() if last_hit else "no hits in window"
            print(f"Last {slot} hit date: {last_hit_text}")
            print(f"Longest losing streak: {diag.get('longest_losing_streak', 0)} days")
            print(f"Current losing streak: {diag.get('current_losing_streak', 0)} days")
            print(f"{slot} ROI: {diag.get('roi_percent', 0):+.2f}%")

            other_rois = [roi for s, roi in slot_roi_map.items() if s != slot]
            if other_rois:
                avg_other = float(np.mean(other_rois))
                print(f"Other slots average ROI: {avg_other:+.2f}%")
                slot_roi = diag.get('roi_percent', 0)
                delta = slot_roi - avg_other
                if delta < -1e-6:
                    print(f"{slot} is underperforming vs others by {abs(delta):.2f}% over last {days_window} days")
                elif delta > 1e-6:
                    print(f"{slot} is outperforming others by {delta:.2f}% over last {days_window} days")
                else:
                    print(f"{slot} is performing in line with other slots over last {days_window} days")

    def print_slot_debug_lines(self, slot_name: str, bet_plans: Dict, real_results_df: pd.DataFrame, debug_days: int = 10):
        """Optional debug tracer for any slot alignment"""
        if not self.matched_dates_data:
            print("⚠️  Debug trace unavailable: no matched dates data")
            return

        slot_key = slot_name.upper()
        if slot_key not in self.SLOT_COLUMNS:
            print(f"⚠️  Invalid slot '{slot_name}'. Valid options: {', '.join(self.slots)}")
            return

        sorted_matches = sorted(self.matched_dates_data, key=lambda x: x['date_obj'])
        debug_slice = sorted_matches[-debug_days:]

        print("\nDATE       | SLOT | RESULT | BET_NUMBERS | STATUS")
        for match in debug_slice:
            date_obj = match['date_obj']
            bet_file = match['bet_file']
            date_results = match['date_results']

            slot_bets = self.parse_bet_plan(bet_file).get(slot_key, {})
            result_number = None
            slot_column = self.SLOT_COLUMNS[slot_key]
            if not date_results.empty and slot_column in date_results.columns:
                valid_results = date_results[date_results[slot_column].notna()]
                if not valid_results.empty:
                    try:
                        result_number = int(float(valid_results[slot_column].iloc[0]))
                    except Exception:
                        result_number = None

            bet_numbers = slot_bets.get('main_numbers', [])
            norm_result = self.normalize_result(result_number)

            if not bet_numbers:
                status = "SKIP_NO_BET"
            elif norm_result is None:
                status = "SKIP_NO_RESULT"
            else:
                hits = self.compute_hits_and_returns(slot_bets, norm_result)
                status = "HIT" if hits.get('main_hit') or hits.get('andar_hit') or hits.get('bahar_hit') else "MISS"

            display_result = norm_result if norm_result is not None else (result_number if result_number is not None else 'MISSING')
            print(f"{date_obj} | {slot_key} | {display_result:>6} | {bet_numbers} | {status}")

    def print_frbd_debug_lines(self, bet_plans: Dict, real_results_df: pd.DataFrame, debug_days: int = 10):
        """Backward-compatible FRBD debug tracer"""
        self.print_slot_debug_lines("FRBD", bet_plans, real_results_df, debug_days=debug_days)

    def export_forensic_report(self, forensic_data: Dict):
        """Export forensic tables to CSV under output/"""
        if not forensic_data:
            return

        output_dir = self.base_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frbd_forensic_report_{datetime.now().strftime('%Y%m%d')}.csv"
        output_path = output_dir / filename

        rows = []
        for row in forensic_data.get('slot_summaries', []):
            rows.append({
                'section': 'SLOT_SUMMARY',
                **row
            })

        diag = forensic_data.get('frbd_diag', {})
        if diag:
            rows.append({
                'section': 'FRBD_DIAGNOSTICS',
                'last_hit_date': diag.get('last_hit_date'),
                'longest_losing_streak': diag.get('longest_losing_streak'),
                'current_losing_streak': diag.get('current_losing_streak'),
                'roi_percent': diag.get('roi_percent'),
                'avg_roi_others': diag.get('avg_roi_others'),
                'days_window': diag.get('days_window')
            })

        for row in forensic_data.get('frbd_daily', []):
            rows.append({
                'section': 'FRBD_DAILY',
                **row
            })

        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            print(f"💾 Forensic report saved: {output_path}")

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

        # Ensure optional validator columns are present on the first sheet
        alias_map = {
            'TOTAL_BET': 'total_stake',
            'TOTAL_RETURN': 'total_return',
            'NET_PNL': 'pnl',
            'ROI_%': 'roi_pct',
        }
        for alias, source in alias_map.items():
            if source in slot_pnl_df.columns and alias not in slot_pnl_df.columns:
                slot_pnl_df[alias] = slot_pnl_df[source]
        
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

    def run(self, days_back: Optional[int] = None, forensic_days: int = 30, debug_frbd: bool = False,
            debug_slot: Optional[str] = None, debug_all: bool = False):
        """Main function to run the enhanced P&L tracker with optional forensics"""
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

        # Step 6: Forensic slot/FRBD summaries (used for slump diagnostics)
        forensic_data = self.compute_slot_forensics(slot_pnl_df, days_window=forensic_days)
        if forensic_data:
            summary_data['slot_slump_diagnostics'] = self.build_slot_slump_diagnostics(forensic_data)
            summary_data = self._merge_slump_into_summary(summary_data, forensic_data)

        # Step 7: Save master file
        self.save_pnl_master_file(slot_pnl_df, layer_pnl_df, summary_data)

        # Step 7b: Export compact quant P&L summary for ROI consumers
        self.export_quant_pnl_summary(slot_pnl_df)

        # Step 8: Forensic slot/FRBD summaries
        if forensic_data:
            self.print_slot_forensic_table(forensic_data)
            self.print_frbd_slump_diagnostics(forensic_data)
            self.print_all_slot_slump_diagnostics(forensic_data)
            self.export_forensic_report(forensic_data)

        # Step 9: Optional slot debug alignment tracer
        if debug_all:
            for slot in self.slots:
                self.print_slot_debug_lines(slot, bet_plans, real_results_df, debug_days=min(10, forensic_days))
        elif debug_slot:
            self.print_slot_debug_lines(debug_slot.upper(), bet_plans, real_results_df, debug_days=min(10, forensic_days))
        elif debug_frbd:
            self.print_frbd_debug_lines(bet_plans, real_results_df, debug_days=min(10, forensic_days))

        # Step 10: Print console summary
        self.print_console_summary(summary_data)

        print("\n✅ Enhanced P&L tracking completed successfully!")
        return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Reality P&L Tracker')
    parser.add_argument('--days-back', type=int, help='Process only last N days of data')
    parser.add_argument('--days', type=int, default=30, help='Forensic analysis window in days (default: 30)')
    parser.add_argument('--debug-frbd', action='store_true', help='Enable FRBD alignment debug trace for recent days')
    parser.add_argument('--debug-slot', type=str,
                        help='Enable alignment debug trace for a specific slot (FRBD/GZBD/GALI/DSWR)')
    parser.add_argument('--debug-all', action='store_true', help='Enable alignment debug trace for all slots')
    parser.add_argument('--all', action='store_true', help='Process all available data (default)')
    
    args = parser.parse_args()
    
    tracker = BetPnLTracker()
    
    run_kwargs = {
        'days_back': args.days_back,
        'forensic_days': args.days,
        'debug_frbd': args.debug_frbd,
        'debug_slot': args.debug_slot,
        'debug_all': args.debug_all
    }

    if args.days_back:
        success = tracker.run(**run_kwargs)
    else:
        run_kwargs['days_back'] = None
        success = tracker.run(**run_kwargs)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
