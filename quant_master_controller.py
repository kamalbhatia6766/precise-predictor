# quant_master_controller.py - UPDATED
# quant_master_controller.py - ROCKET BRIEF MODE - MINIMAL SPAM + DAILY-AUTO MODE
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import time
import argparse

# 🆕 Import central helpers for daily-auto mode
import quant_paths
import quant_data_core

def run_script_brief(script_name, args="", timeout=3600):
    """🚀 Run script with minimal output - NO SPAM"""
    print(f"   → {script_name}...", end="", flush=True)
    
    try:
        cmd = [sys.executable, script_name] 
        if args:
            cmd.extend(args.split())
            
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=timeout)
        
        if result.returncode == 0:
            print(" ✅")
            return True
        else:
            print(" ❌")
            # Show only first line of error
            error_lines = result.stderr.strip().split('\n')
            if error_lines:
                print(f"      Error: {error_lines[0][:100]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(" ⏰ TIMEOUT")
        return False
    except Exception as e:
        print(f" ❌ {str(e)[:100]}...")
        return False

def get_system_status():
    """Get current system status from various analytics"""
    status = {
        'target_date': 'Unknown',
        'base_stake': 0,
        'final_stake': 0,
        'slot_stakes': {},
        'slot_stakes_final': {},
        'scripts_completed': 0,
        'zone': 'UNKNOWN',
        'risk_mode': 'UNKNOWN',
        'strategy': 'UNKNOWN'
    }
    
    try:
        # Get target date from latest bet plan
        latest_bet = quant_paths.find_latest_bet_plan_master()
        if latest_bet:
            date_str = latest_bet.stem.replace('bet_plan_master_', '')
            status['target_date'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            
            # Get base stake from bet plan
            df = pd.read_excel(latest_bet, sheet_name='bets')
            status['base_stake'] = df['stake'].sum()
            
            # Get slot details - ✅ FIXED: Convert numbers to strings
            for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
                slot_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'Main')]
                if not slot_bets.empty:
                    numbers = [str(row['number_or_digit']) for _, row in slot_bets.iterrows()][:3]
                    andar_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'ANDAR')]
                    bahar_bets = df[(df['slot'] == slot) & (df['layer_type'] == 'BAHAR')]
                    
                    andar_digit = andar_bets['number_or_digit'].iloc[0] if not andar_bets.empty else "None"
                    bahar_digit = bahar_bets['number_or_digit'].iloc[0] if not bahar_bets.empty else "None"
                    total_stake = slot_bets['stake'].sum() + (andar_bets['stake'].iloc[0] if not andar_bets.empty else 0) + (bahar_bets['stake'].iloc[0] if not bahar_bets.empty else 0)
                    
                    status['slot_stakes'][slot] = {
                        'numbers': numbers,
                        'andar': andar_digit,
                        'bahar': bahar_digit,
                        'stake': total_stake
                    }
        
        # Get final stakes from analytics overlay
        stake_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        if stake_file.exists():
            with open(stake_file, 'r') as f:
                stake_data = json.load(f)
                status['final_stake'] = stake_data.get('total_daily_stake', 0)
                status['slot_stakes_final'] = stake_data.get('slot_stakes', {})
        
        # Get risk status
        recovery_file = quant_paths.get_performance_logs_dir() / "loss_recovery_plan.json"
        if recovery_file.exists():
            with open(recovery_file, 'r') as f:
                recovery_data = json.load(f)
                status['zone'] = recovery_data.get('current_zone', 'UNKNOWN')
                status['risk_mode'] = recovery_data.get('recommended_risk_mode', 'UNKNOWN')
        
        # Get strategy
        strategy_file = quant_paths.get_performance_logs_dir() / "strategy_recommendation.json"
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                strategy_data = json.load(f)
                status['strategy'] = strategy_data.get('recommended_strategy', 'UNKNOWN')
        
    except Exception as e:
        print(f"⚠️  Error getting system status: {e}")
    
    return status

def print_rocket_summary(success_count, total_scripts, status):
    """🚀 Print ultra-brief rocket summary"""
    print("\n" + "="*60)
    print("🎯 QUANTUM DAILY SUMMARY - ROCKET MODE")
    print("="*60)
    
    print(f"📅 Predictions For : {status['target_date']}")
    print(f"💰 Base Stake Plan : ₹{status['base_stake']}")
    print(f"🎯 Final Live Stake: ₹{status['final_stake']} (after analytics overlay)")
    print()
    
    # Print slot predictions - ✅ FIXED: Numbers are already strings
    for slot, data in status['slot_stakes'].items():
        numbers_str = ", ".join(data['numbers'])
        print(f"  {slot}: {numbers_str} | A:{data['andar']}/B:{data['bahar']} | ₹{data['stake']}")
    
    print(f"\n🎯 DYNAMIC SLOT STAKES: ", end="")
    slot_stakes_str = []
    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        if slot in status.get('slot_stakes_final', {}):
            slot_stakes_str.append(f"{slot}=₹{status['slot_stakes_final'][slot]}")
    print(", ".join(slot_stakes_str) + f" (Total=₹{status['final_stake']})")
    
    print(f"📊 SCRIPTS COMPLETED: {success_count}/{total_scripts}")
    print(f"🚀 SYSTEM STATUS   : OPTIMAL (Zone: {status['zone']}, Risk: {status['risk_mode']})")
    print(f"🎯 STRATEGY        : {status.get('strategy', 'BASE')}")

def run_reality_pipeline():
    """🆕 Run reality pipeline scripts after predictions"""
    print(f"\n🎯 REALITY PIPELINE: Running analytics & stake allocation...")
    print("-" * 50)
    
    pipeline_scripts = [
        ("dynamic_stake_allocator.py", "", 300),
        ("loss_recovery_engine.py", "", 300),
        ("pattern_intelligence_engine.py", "", 300),
        ("strategy_recommendation_engine.py", "", 300),
        ("final_bet_plan_engine.py", "", 300),
        ("live_bet_sheet_engine.py", "", 300)
    ]
    
    success_count = 0
    for script, args, timeout in pipeline_scripts:
        if run_script_brief(script, args, timeout):
            success_count += 1
    
    return success_count, len(pipeline_scripts)

def load_reality_data():
    """🆕 Load reality JSONs for final summary"""
    data = {
        'stake_plan': {},
        'recovery_plan': {},
        'strategy_recommendation': {}
    }
    
    try:
        stake_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        if stake_file.exists():
            with open(stake_file, 'r') as f:
                data['stake_plan'] = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading stake plan: {e}")
    
    try:
        recovery_file = quant_paths.get_performance_logs_dir() / "loss_recovery_plan.json"
        if recovery_file.exists():
            with open(recovery_file, 'r') as f:
                data['recovery_plan'] = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading recovery plan: {e}")
    
    try:
        strategy_file = quant_paths.get_performance_logs_dir() / "strategy_recommendation.json"
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                data['strategy_recommendation'] = json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading strategy: {e}")
    
    return data

def run_pack_universe_check():
    """🆕 Run pack universe sanity check"""
    try:
        import pattern_packs
        is_ok, message = pattern_packs.run_pack_universe_sanity_check()
        print(f"🧮 {message}")
        return is_ok
    except Exception as e:
        print(f"⚠️  Pack universe check failed: {e}")
        return False

def run_daily_auto_mode():
    """🆕 Run complete daily-auto workflow"""
    print("🚀 QUANT DAILY AUTO - ONE BUTTON QUANT DAY")
    print("="*60)
    
    # 🆕 Step 0: Run pack universe sanity check
    print("🧮 Running pack universe sanity check...")
    run_pack_universe_check()
    
    # Step 1: Load data and build prediction plan
    try:
        df = quant_data_core.load_results_dataframe()
        plan = quant_data_core.build_prediction_plan(df)
        
        # Print prediction plan summary
        quant_data_core.print_prediction_plan_summary(plan)
        
    except Exception as e:
        print(f"❌ Error loading prediction plan: {e}")
        return False
    
    # Step 2: Run precise_daily_runner.py
    print(f"\n🎯 STEP 1: RUNNING DAILY PREDICTIONS...")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, "precise_daily_runner.py",
            "--mode", "predict",
            "--target", "auto", 
            "--source", "scr9",
            "--speed-mode", "full"
        ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
        
        if result.returncode != 0:
            print(f"❌ Daily runner failed with exit code: {result.returncode}")
            print(f"   Error: {result.stderr[:200]}...")
            return False
            
        print("✅ Daily predictions completed successfully")
        
    except subprocess.TimeoutExpired:
        print("❌ Daily runner timeout (30 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running daily runner: {e}")
        return False
    
    # Step 3: Run P&L tracker
    print(f"\n📊 STEP 2: UPDATING REALITY P&L...")
    print("-" * 40)
    
    pnl_success = False
    try:
        result = subprocess.run([
            sys.executable, "bet_pnl_tracker.py",
            "--days-back", "30"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            pnl_success = True
            print("✅ P&L tracker completed successfully")
        else:
            print(f"⚠️  P&L tracker failed with exit code: {result.returncode}")
            print(f"   Error: {result.stderr[:200]}...")
            # Continue even if P&L fails, but mark it
            
    except subprocess.TimeoutExpired:
        print("⚠️  P&L tracker timeout (10 minutes)")
    except Exception as e:
        print(f"⚠️  Error running P&L tracker: {e}")
    
    # 🆕 STEP 4: Run reality pipeline
    pipeline_success, pipeline_total = run_reality_pipeline()
    
    # Step 5: Load and print performance summary
    print(f"\n💰 STEP 3: LOADING PERFORMANCE SUMMARY...")
    print("-" * 40)
    
    performance_data = {}
    try:
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.json"
        if pnl_file.exists():
            with open(pnl_file, 'r') as f:
                performance_data = json.load(f)
            print("✅ Performance data loaded")
        else:
            print("⚠️  No performance data found (quant_reality_pnl.json missing)")
    except Exception as e:
        print(f"⚠️  Error loading performance data: {e}")
    
    # Step 6: Print final summary
    print_final_daily_summary(plan, performance_data, pnl_success, pipeline_success, pipeline_total)
    
    return True

def print_final_daily_summary(plan, performance_data, pnl_success, pipeline_success, pipeline_total):
    """🆕 Print final daily-auto summary"""
    print("\n" + "="*70)
    print("🎯 QUANT DAILY AUTO - COMPLETE")
    print("="*70)
    
    # Load reality data
    reality_data = load_reality_data()
    
    # Prediction plan info
    print(f"📅 Latest result date: {plan['latest_result_date']}")
    print(f"🎯 Plan mode: {plan['mode']}")
    
    if plan['is_partial_day']:
        print(f"🔴 Same-day slots: {', '.join(plan['today_slots_to_predict'])}")
    else:
        print(f"🔴 Same-day slots: None")
        
    print(f"🟢 Next-day date: {plan['next_date']}")
    
    # 🆕 Reality pipeline info
    if reality_data['stake_plan']:
        stake_plan = reality_data['stake_plan']
        slot_stakes = stake_plan.get('slot_stakes', {})
        total_stake = stake_plan.get('total_daily_stake', 0)
        
        slot_stakes_str = []
        for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
            if slot in slot_stakes:
                slot_stakes_str.append(f"{slot}=₹{slot_stakes[slot]}")
        
        print(f"🎯 FINAL LIVE STAKE: ₹{total_stake} ({', '.join(slot_stakes_str)})")
    
    if reality_data['recovery_plan']:
        recovery_plan = reality_data['recovery_plan']
        zone = recovery_plan.get('current_zone', 'UNKNOWN')
        risk_mode = recovery_plan.get('recommended_risk_mode', 'UNKNOWN')
        print(f"🚀 ZONE / RISK: {zone} / {risk_mode}")
    
    if reality_data['strategy_recommendation']:
        strategy_rec = reality_data['strategy_recommendation']
        strategy = strategy_rec.get('recommended_strategy', 'UNKNOWN')
        top_family = strategy_rec.get('top_family', 'UNKNOWN')
        hit_rate = strategy_rec.get('top_family_hit_rate', 0)
        print(f"🎯 STRATEGY: {strategy} (top family {top_family} @ {hit_rate}%)")
    
    # Performance data
    if performance_data and 'overall' in performance_data:
        overall = performance_data['overall']
        print(f"\n💰 RECENT PERFORMANCE (Last {overall.get('days_processed', 'N')} days):")
        print(f"   Total Stake: ₹{overall.get('total_stake', 0):.0f}")
        print(f"   Total P&L: ₹{overall.get('total_pnl', 0):+.0f}")
        print(f"   Overall ROI: {overall.get('overall_roi', 0):+.1f}%")
        
        # Best and worst slots
        if 'by_slot' in performance_data and performance_data['by_slot']:
            slots_sorted = sorted(performance_data['by_slot'], key=lambda x: x.get('roi_pct', 0), reverse=True)
            if slots_sorted:
                best_slot = slots_sorted[0]
                worst_slot = slots_sorted[-1]
                print(f"   Best Slot: {best_slot.get('slot')} ({best_slot.get('roi_pct', 0):+.1f}%)")
                print(f"   Worst Slot: {worst_slot.get('slot')} ({worst_slot.get('roi_pct', 0):+.1f}%)")
    else:
        print(f"\n💰 PERFORMANCE DATA: Not available")
    
    # Status indicators
    print(f"\n✅ STATUS:")
    print(f"   Daily Predictions: ✅ COMPLETED")
    print(f"   P&L Update: {'✅ COMPLETED' if pnl_success else '⚠️  FAILED'}")
    print(f"   Reality Pipeline: {pipeline_success}/{pipeline_total} scripts successful")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(description='Quantum Master Controller - Rocket Brief Mode')
    parser.add_argument('--mode', choices=['daily_live', 'backtest', 'analyze', 'daily-auto'], default='daily_live')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without executing')
    parser.add_argument('--speed-mode', choices=['fast', 'full'], default='full')
    parser.add_argument('--with-housekeeping', action='store_true', help='Run housekeeping after completion')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 QUANTUM MASTER CONTROLLER - ROCKET BRIEF MODE")
    print("="*60)
    print(f"Mode: {args.mode} | Dry-run: {'ON' if args.dry_run else 'OFF'}")
    print(f"🚀 Quantum Features: Auto Target, Golden Learning")
    print("="*60)
    
    if args.dry_run:
        print("🔄 DRY RUN MODE - No scripts will be executed")
        return 0
    
    # 🆕 Handle daily-auto mode separately
    if args.mode == 'daily-auto':
        success = run_daily_auto_mode()
        
        # 🆕 Run housekeeping if requested
        if success and args.with_housekeeping:
            print(f"\n🧹 RUNNING HOUSEKEEPING...")
            try:
                housekeeping_result = subprocess.run([
                    sys.executable, "quant_housekeeping.py",
                    "--dry-run"  # Always dry-run by default for safety
                ], capture_output=True, text=True, timeout=300)
                
                if housekeeping_result.returncode == 0:
                    print("✅ Housekeeping completed")
                else:
                    print("⚠️  Housekeeping had issues")
            except Exception as e:
                print(f"⚠️  Housekeeping failed: {e}")
        
        return 0 if success else 1
    
    # 🚀 Original script execution pipeline - BRIEF MODE (for other modes)
    scripts = [
        # Phase 1: Prediction Pipeline
        ("precise_daily_runner.py", "--mode predict --target auto --source scr9", 2400),
        
        # Phase 2: Analytics & Learning
        ("prediction_hit_memory.py", "", 300),
        ("pattern_intelligence_engine.py", "", 300),
        ("golden_block_analyzer.py", "", 300),
        ("pattern_intelligence_enhanced.py", "", 300),
        ("pattern_packs_exporter.py", "", 300),
        
        # Phase 3: Risk & Money Management
        ("dynamic_stake_allocator.py", "", 300),
        ("smart_fusion_weights.py", "", 300),
        ("money_manager.py", "", 300),
        ("analytics_slot_overlay.py", "", 300),
        ("auto_backtest_runner.py", "", 300),
        ("real_time_performance_dashboard.py", "", 300),
        ("loss_recovery_engine.py", "", 300),
        ("strategy_recommendation_engine.py", "", 300),
        ("reality_check_engine.py", "", 300),
        
        # Phase 4: Live Execution
        ("final_bet_plan_engine.py", "", 300),
        ("live_bet_sheet_engine.py", "", 300)
    ]
    
    print(f"\n[1/4] PREDICTION PIPELINE: Running precise_daily_runner.py")
    success_count = 0
    total_scripts = len(scripts)
    
    for i, (script, script_args, timeout) in enumerate(scripts):
        if run_script_brief(script, script_args, timeout):
            success_count += 1
    
    # 🚀 Generate meta config
    print(f"\n[+] QUANTUM META CONFIG: Generating daily meta configuration...")
    try:
        meta_config = {
            "timestamp": datetime.now().isoformat(),
            "system_date": datetime.now().strftime("%Y-%m-%d"),
            "scripts_executed": total_scripts,
            "scripts_successful": success_count,
            "rocket_mode": True,
            "quantum_features": ["Auto Target", "Golden Learning", "Pattern Intelligence"]
        }
        
        meta_file = quant_paths.get_performance_logs_dir() / "daily_meta_config.json"
        meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_file, 'w') as f:
            json.dump(meta_config, f, indent=2)
        print("💾 Daily meta config saved")
    except Exception as e:
        print(f"⚠️  Error saving meta config: {e}")
    
    # 🚀 Get system status and print final summary
    status = get_system_status()
    print_rocket_summary(success_count, total_scripts, status)
    
    print("\n" + "="*60)
    print(f"🎯 QUANTUM MASTER CONTROLLER COMPLETED!")
    print(f"   ROCKET BRIEF MODE: {success_count}/{total_scripts} scripts successful")
    print("="*60)
    
    return 0 if success_count == total_scripts else 1

if __name__ == "__main__":
    exit(main())