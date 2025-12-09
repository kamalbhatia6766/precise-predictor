# precise_daily_runner.py - ROCKET MODE - UPDATED WITH CENTRAL HELPERS + PARTIAL-DAY LOGIC
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import subprocess
import sys
import time
import argparse
import warnings
import os
warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths
import quant_data_core

def get_last_real_date():
    """Get last real result date from number prediction learn.xlsx"""
    try:
        # 🆕 Use central data core instead of custom logic
        df = quant_data_core.load_results_dataframe()
        if df.empty:
            return datetime.now().date()
        
        latest_date = quant_data_core.get_latest_result_date(df)
        return latest_date
        
    except Exception as e:
        print(f"⚠️  Error getting last real date: {e}")
        return datetime.now().date()

def determine_auto_target_date():
    """Determine target date for AUTO mode using central helpers"""
    try:
        # 🆕 Use central prediction planning
        df = quant_data_core.load_results_dataframe()
        plan = quant_data_core.build_prediction_plan(df)
        
        latest_date = plan['latest_result_date']
        system_today = datetime.now().date()
        
        # If we have same-day predictions, use latest_date, else use next_date
        if plan['is_partial_day']:
            target_date = latest_date
            mode = "SAME_DAY_PARTIAL"
        else:
            target_date = plan['next_date'] 
            mode = "NEXT_DAY_FULL"
        
        return target_date, mode, latest_date
        
    except Exception as e:
        print(f"⚠️  Error in auto target date: {e}")
        # Fallback to old logic
        last_real_date = get_last_real_date()
        system_today = datetime.now().date()
        
        if last_real_date <= system_today:
            target_date = last_real_date + timedelta(days=1)
            mode = "TOMORROW"
        else:
            target_date = last_real_date
            mode = "TODAY"
        
        return target_date, mode, last_real_date

def determine_target_date(mode):
    """Determine target date based on mode"""
    if mode == 'auto':
        return determine_auto_target_date()
    elif mode == 'today':
        last_real_date = get_last_real_date()
        return last_real_date, "TODAY", last_real_date
    elif mode == 'tomorrow':
        last_real_date = get_last_real_date()
        return last_real_date + timedelta(days=1), "TOMORROW", last_real_date
    else:
        last_real_date = get_last_real_date()
        return last_real_date + timedelta(days=1), "TOMORROW", last_real_date

def run_scr9_full():
    """Run SCR9 in full mode - FIXED ARGUMENTS"""
    try:
        print("🎯 RUNNING SCR9 (FULL MODE)...")
        start_time = time.time()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        # ✅ FIXED: Use correct arguments for deepseek_scr9.py
        result = subprocess.run([
            sys.executable, "deepseek_scr9.py",
            "--speed-mode", "full"  # ✅ ONLY this argument
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5000, env=env)
        
        if result.returncode == 0:
            scr9_time = time.time() - start_time
            print(f"✅ SCR9 Engine (full) completed ({scr9_time:.1f}s)")
            return True, scr9_time
        else:
            print(f"❌ SCR9 Engine failed: {result.stderr[:200]}")
            return False, 0
    except subprocess.TimeoutExpired:
        print("❌ SCR9 Engine timeout")
        return False, 0
    except Exception as e:
        print(f"❌ SCR9 Engine error: {e}")
        return False, 0

def run_bet_engine(target_mode, source):
    """Run bet engine"""
    try:
        print("💰 RUNNING BET ENGINE (Optimized)...")
        start_time = time.time()

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        result = subprocess.run([
            sys.executable, "precise_bet_engine.py",
            "--target", target_mode,
            "--source", source
        ], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300, env=env)
        
        if result.returncode == 0:
            bet_time = time.time() - start_time
            print(f"✅ Bet Engine completed ({bet_time:.1f}s)")
            
            # Extract summary from bet engine output
            for line in result.stdout.split('\n'):
                if "GRAND TOTAL:" in line:
                    print(f"   {line.strip()}")
                if "ULTRA v5 QUANTUM" in line:
                    print(f"   {line.strip()}")
            
            return True, bet_time
        else:
            print(f"❌ Bet Engine failed: {result.stderr[:200]}")
            return False, 0
    except Exception as e:
        print(f"❌ Bet Engine error: {e}")
        return False, 0

def get_latest_predictions_file():
    """Get latest predictions file using central paths"""
    try:
        predictions_dir = quant_paths.get_scr9_predictions_dir()
        if predictions_dir.exists():
            ultimate_files = list(predictions_dir.glob("ultimate_predictions_*.xlsx"))
            if ultimate_files:
                return max(ultimate_files, key=lambda x: x.stat().st_mtime)
        return None
    except Exception as e:
        print(f"⚠️  Error getting latest predictions: {e}")
        return None

def print_brief_summary(target_date, total_time, speed_mode, target_mode):
    """Print brief prediction summary"""
    print("\n" + "="*60)
    print("🎯 PREDICTION SUMMARY - ROCKET MODE")
    print("="*60)
    
    # 🆕 Get prediction plan for context
    try:
        df = quant_data_core.load_results_dataframe()
        plan = quant_data_core.build_prediction_plan(df)
        
        print(f"📅 Latest result date: {plan['latest_result_date']}")
        print(f"🎯 Plan mode: {plan['mode']}")
        if plan['is_partial_day']:
            print(f"🔴 Same-day slots: {', '.join(plan['today_slots_to_predict'])}")
        print(f"🟢 Next-day date: {plan['next_date']}")
        
    except Exception as e:
        print(f"⚠️  Could not load prediction plan: {e}")
    
    # Get latest bet plan to show numbers
    bet_engine_dir = quant_paths.get_bet_plans_dir()
    bet_files = list(bet_engine_dir.glob("bet_plan_master_*.xlsx"))
    
    if bet_files:
        latest_bet = max(bet_files, key=lambda x: x.stat().st_mtime)
        try:
            df = pd.read_excel(latest_bet, sheet_name='bets')
            
            slots_data = {}
            for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                slot_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'Main')]
                if not slot_bets.empty:
                    # ✅ FIXED: Convert numbers to strings to avoid TypeError
                    numbers = [str(row['number_or_digit']) for _, row in slot_bets.iterrows()]
                    andar_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'ANDAR')]
                    bahar_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'BAHAR')]
                    
                    andar_digit = andar_bets['number_or_digit'].iloc[0] if not andar_bets.empty else "None"
                    bahar_digit = bahar_bets['number_or_digit'].iloc[0] if not bahar_bets.empty else "None"
                    total_stake = slot_bets['stake'].sum() + (andar_bets['stake'].iloc[0] if not andar_bets.empty else 0) + (bahar_bets['stake'].iloc[0] if not bahar_bets.empty else 0)
                    
                    slots_data[slot] = {
                        'numbers': numbers,
                        'andar': andar_digit,
                        'bahar': bahar_digit,
                        'stake': total_stake
                    }
            
            # Print slot-wise summary
            for slot, data in slots_data.items():
                numbers_str = ", ".join(data['numbers'])
                print(f"  {slot}: {numbers_str} | A:{data['andar']}/B:{data['bahar']} | ₹{data['stake']}")
                
        except Exception as e:
            print(f"⚠️  Error reading bet plan: {e}")
    
    print(f"\n⏰ Total time: {total_time:.1f}s | Speed: {speed_mode.upper()}")
    print(f"🎯 Target Date: {target_date}")
    print(f"📊 Target Mode: {target_mode}")

def main():
    parser = argparse.ArgumentParser(description='Precise Daily Runner - Rocket Mode')
    parser.add_argument('--mode', choices=['predict', 'backtest'], default='predict')
    parser.add_argument('--target', choices=['today', 'tomorrow', 'auto'], default='auto')
    parser.add_argument('--source', choices=['scr9', 'fusion'], default='scr9')
    parser.add_argument('--speed-mode', choices=['fast', 'full'], default='full')
    
    args = parser.parse_args()
    
    try:
        # 🆕 PRINT PREDICTION PLAN AT START
        print("🚀 PRECISE DAILY RUNNER - ROCKET MODE")
        print("="*50)
        
        # Load data and build prediction plan
        df = quant_data_core.load_results_dataframe()
        plan = quant_data_core.build_prediction_plan(df)
        quant_data_core.print_prediction_plan_summary(plan)
        
        # ✅ ROCKET MODE: Clear date mapping at start  
        system_today = datetime.now().date()
        target_date, target_mode_display, last_real_date = determine_target_date(args.target)
        
        print(f"\n🎯 EXECUTION TARGET:")
        print(f"   Target Prediction Date: {target_date}")
        print(f"   Target Mode: {args.target.upper()} ({target_mode_display})")
        print(f"   Source: {args.source.upper()}")
        print(f"   Speed Mode: {args.speed_mode.upper()}")
        
        total_start_time = time.time()
        
        if args.mode == 'predict':
            # 🆕 TODO: Implement partial-day execution logic
            # For now, maintain existing behavior but with better context
            
            # Run SCR9
            scr9_success, scr9_time = run_scr9_full()
            
            if scr9_success:
                # Run Bet Engine
                bet_success, bet_time = run_bet_engine(args.target, args.source)
                
                if bet_success:
                    total_time = time.time() - total_start_time
                    
                    # ✅ ROCKET MODE: Brief summary
                    print_brief_summary(target_date, total_time, args.speed_mode, args.target)
                    
                    print(f"\n✅ ROCKET MODE completed in {total_time:.1f}s")
                    
                    # 🆕 Print prediction plan reminder
                    if plan['is_partial_day']:
                        print(f"💡 REMINDER: {len(plan['today_slots_to_predict'])} same-day slots need predictions!")
                    
                    return 0
                else:
                    print("❌ Bet Engine failed")
                    return 1
            else:
                print("❌ SCR9 failed")
                return 1
        else:
            print("❌ Only predict mode supported in Rocket Mode")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
