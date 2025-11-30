# live_bet_sheet_engine.py - UPDATED
# live_bet_sheet_engine.py - ROCKET SYNC - CONSISTENT STAKES WITH FINAL PLAN
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths
import quant_data_core
import pattern_packs

def get_target_date_from_stake_plan():
    """🆕 Get target date from dynamic stake plan"""
    try:
        stake_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        if stake_file.exists():
            with open(stake_file, 'r') as f:
                data = json.load(f)
                return data.get('target_date', get_latest_bet_plan_date())
        return get_latest_bet_plan_date()
    except:
        return get_latest_bet_plan_date()

def get_latest_bet_plan_date():
    """Get date from latest bet plan master file"""
    try:
        latest_file = quant_paths.find_latest_bet_plan_master()
        if latest_file:
            date_str = latest_file.stem.replace('bet_plan_master_', '')
            return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        return datetime.now().strftime("%Y-%m-%d")
    except:
        return datetime.now().strftime("%Y-%m-%d")

def load_execution_readiness():
    """Load execution readiness score"""
    try:
        readiness_file = quant_paths.get_performance_logs_dir() / "execution_readiness.json"
        if readiness_file.exists():
            with open(readiness_file, 'r') as f:
                data = json.load(f)
                return data.get('readiness_score', 50), data.get('readiness_level', 'UNKNOWN')
        return 50, 'UNKNOWN'
    except:
        return 50, 'UNKNOWN'

def load_final_stakes_from_final_plan(target_date):
    """🆕 Load final slot stakes from final bet plan (single source of truth)"""
    try:
        final_plan_file = quant_paths.get_final_bet_plan_path(target_date.replace('-', ''))
        if final_plan_file.exists():
            df = pd.read_excel(final_plan_file, sheet_name='final_slot_plan')
            # Filter out TOTAL row and get slot stakes
            slot_data = df[df['slot'].isin(['FRBD', 'GZBD', 'GALI', 'DSWR'])]
            stakes = {}
            for _, row in slot_data.iterrows():
                stakes[row['slot']] = float(row['final_slot_stake'])
            return stakes
        return {}
    except Exception as e:
        print(f"⚠️  Error loading final stakes: {e}")
        return {}

def load_base_stakes():
    """Load base stakes from bet plan"""
    try:
        latest_file = quant_paths.find_latest_bet_plan_master()
        if latest_file:
            df = pd.read_excel(latest_file, sheet_name='bets')
            return df['stake'].sum()
        return 0
    except:
        return 0

def load_reality_data():
    """🆕 Load reality JSONs for bet sheet generation"""
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

def scale_stakes_to_target(base_bet_plan, target_stakes):
    """🆕 Scale bets to match target stakes exactly with rounding adjustment (NaN-safe)"""
    if base_bet_plan.empty:
        return base_bet_plan

    result_df = base_bet_plan.copy()

    # Initialize columns
    result_df['base_stake'] = result_df['stake']
    result_df['weight'] = result_df['stake'].astype(float)

    # Fill NaN with 0 so scaling math doesn't break
    result_df['base_stake'] = result_df['base_stake'].fillna(0.0)
    result_df['weight'] = result_df['weight'].fillna(0.0)

    result_df['final_stake'] = 0.0

    for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
        slot_mask = result_df['slot'] == slot
        slot_rows = result_df[slot_mask]

        if slot_rows.empty:
            continue

        target_slot_stake = target_stakes.get(slot, 0)

        # If target stake is zero, set all stakes to zero
        if target_slot_stake == 0:
            result_df.loc[slot_mask, 'final_stake'] = 0.0
            continue

        base_slot_total = slot_rows['base_stake'].sum()

        # Skip scaling if no base stake
        if base_slot_total == 0:
            result_df.loc[slot_mask, 'final_stake'] = result_df.loc[slot_mask, 'base_stake']
            continue

        # Calculate scale factor and apply initial scaling
        total_weight = result_df.loc[slot_mask, 'weight'].sum()
        if total_weight > 0:
            scale_factor = target_slot_stake / total_weight
        else:
            scale_factor = 0.0

        # Apply scaling and round to nearest ₹5 (NaN-safe per row)
        for idx in slot_rows.index:
            weight = float(result_df.loc[idx, 'weight'])

            # If weight is 0 or negative, keep stake zero
            if weight <= 0:
                result_df.loc[idx, 'final_stake'] = 0.0
                continue

            weighted_stake = weight * scale_factor

            # Extra safety: if somehow NaN aa bhi gaya to skip
            if pd.isna(weighted_stake):
                result_df.loc[idx, 'final_stake'] = 0.0
                continue

            # Round to nearest ₹5
            rounded_stake = round(weighted_stake / 5) * 5

            # Minimum stake protection for non-zero base stakes
            if result_df.loc[idx, 'base_stake'] >= 10 and rounded_stake < 5:
                rounded_stake = 5

            result_df.loc[idx, 'final_stake'] = rounded_stake

        # Fix rounding discrepancies slot-level
        total_after_round = result_df.loc[slot_mask, 'final_stake'].sum()
        diff = target_slot_stake - total_after_round

        if diff != 0 and total_after_round > 0:
            # Find the row with largest stake to adjust
            max_stake_idx = result_df.loc[slot_mask, 'final_stake'].idxmax()
            current_max = result_df.loc[max_stake_idx, 'final_stake']
            new_stake = current_max + diff

            # Ensure minimum stake
            if new_stake < 5 and result_df.loc[max_stake_idx, 'base_stake'] >= 10:
                new_stake = 5

            result_df.loc[max_stake_idx, 'final_stake'] = new_stake

            # Recalculate total to verify
            final_total = result_df.loc[slot_mask, 'final_stake'].sum()
            if abs(final_total - target_slot_stake) > 1:  # Allow small floating point differences
                print(
                    f"⚠️  Slot {slot} stake mismatch: target ₹{target_slot_stake}, actual ₹{final_total}"
                )

    return result_df


def apply_s40_boost_and_scaling(base_bet_plan, slot_stakes, strategy):
    """🆕 Apply S40 boost and stake scaling to base bet plan"""
    if base_bet_plan.empty:
        return base_bet_plan
    
    result_df = base_bet_plan.copy()
    
    # Add columns for tracking
    result_df['base_stake'] = result_df['stake']
    result_df['is_s40'] = 'No'
    result_df['weight'] = result_df['stake'].astype(float)
    
    # Apply S40 boost if strategy requires it
    is_s40_boost = strategy.get('recommended_strategy') == 'STRAT_S40_BOOST'
    
    if is_s40_boost:
        for idx, row in result_df.iterrows():
            if (row['layer_type'] == 'Main' and 
                pattern_packs.is_s40(row['number_or_digit'])):
                result_df.at[idx, 'weight'] = row['weight'] * 1.2
                result_df.at[idx, 'is_s40'] = 'Yes'
    
    # Scale stakes to match target exactly
    result_df = scale_stakes_to_target(result_df, slot_stakes)
    
    return result_df

def generate_live_bet_sheet():
    """Generate live bet sheet with consistent stakes from final plan"""
    try:
        # 🚀 Get all required data
        target_date = get_target_date_from_stake_plan()
        readiness_score, readiness_level = load_execution_readiness()
        
        # 🆕 Load final stakes from final bet plan (single source of truth)
        final_stakes = load_final_stakes_from_final_plan(target_date)
        base_stake_total = load_base_stakes()
        
        # 🆕 Load reality data
        reality_data = load_reality_data()
        recovery_plan = reality_data.get('recovery_plan', {})
        strategy_rec = reality_data.get('strategy_recommendation', {})
        
        # Calculate total from final stakes
        final_total = sum(final_stakes.values()) if final_stakes else 0
        
        print("🎯 LIVE BET SHEET - ROCKET SYNC")
        print("="*50)
        print(f"📅 Target Date     : {target_date}")
        print(f"🚀 Mode            : {readiness_level} (score: {readiness_score})")
        print(f"💰 Base Stake Plan : ₹{base_stake_total} (from bet_plan_master)")
        
        # 🆕 Try to load final bet plan first, then fall back to base bet plan
        bet_engine_dir = quant_paths.get_bet_engine_dir()
        final_bet_plan_file = quant_paths.get_final_bet_plan_path(target_date.replace('-', ''))
        base_bet_plan_file = quant_paths.get_bet_plan_master_path(target_date.replace('-', ''))
        
        if final_bet_plan_file.exists():
            print("✅ Using FINAL bet plan as input")
            df_final = pd.read_excel(final_bet_plan_file, sheet_name='final_slot_plan')
            df_meta = pd.read_excel(final_bet_plan_file, sheet_name='meta_summary')
            
            # Use meta data for strategy info
            if not df_meta.empty:
                meta_strategy = df_meta.iloc[0].get('meta_strategy', 'UNKNOWN')
                numeric_strategy = df_meta.iloc[0].get('numeric_strategy', 'UNKNOWN')
                risk_mode = df_meta.iloc[0].get('risk_mode', 'UNKNOWN')
                zone = df_meta.iloc[0].get('zone', 'UNKNOWN')
            else:
                meta_strategy = strategy_rec.get('recommended_strategy', 'UNKNOWN')
                numeric_strategy = 'UNKNOWN'
                risk_mode = recovery_plan.get('recommended_risk_mode', 'UNKNOWN')
                zone = recovery_plan.get('zone', 'UNKNOWN')
            
            # Load base bet plan for the actual bet numbers
            if base_bet_plan_file.exists():
                base_bet_df = pd.read_excel(base_bet_plan_file, sheet_name='bets')
            else:
                print("❌ Base bet plan not found, cannot generate live bets")
                return False
        elif base_bet_plan_file.exists():
            print("✅ Using BASE bet plan as input (no final plan found)")
            base_bet_df = pd.read_excel(base_bet_plan_file, sheet_name='bets')
            meta_strategy = strategy_rec.get('recommended_strategy', 'BASE')
            numeric_strategy = 'BASE'
            risk_mode = recovery_plan.get('recommended_risk_mode', 'UNKNOWN')
            zone = recovery_plan.get('zone', 'UNKNOWN')
        else:
            print("❌ No bet plan files found")
            return False
        
        # 🆕 Apply S40 boost and stake scaling
        live_bet_df = apply_s40_boost_and_scaling(base_bet_df, final_stakes, strategy_rec)
        
        # Print slot summary
        print(f"🎯 Final Slot Stakes (from final_bet_plan):")
        
        slot_base_totals = {}
        slot_final_totals = {}
        
        for slot in ["FRBD", "GZBD", "GALI", "DSWR"]:
            slot_base = live_bet_df[live_bet_df['slot'] == slot]['base_stake'].sum()
            slot_final = live_bet_df[live_bet_df['slot'] == slot]['final_stake'].sum()
            slot_base_totals[slot] = slot_base
            slot_final_totals[slot] = slot_final
            
            target = final_stakes.get(slot, 0)
            print(f"  {slot}: base ₹{slot_base:.0f} → final ₹{slot_final:.0f} (target ₹{target})")
        
        # Print reality info
        print(f"🚀 Zone {zone} / Risk {risk_mode} / Strategy {meta_strategy} ({numeric_strategy})")
        print(f"💵 TOTAL EXECUTION STAKE: ₹{final_total}")
        
        # Verify stake consistency
        calculated_total = live_bet_df['final_stake'].sum()
        if abs(calculated_total - final_total) > 1:
            print(f"⚠️  STAKE MISMATCH: Final plan ₹{final_total} vs Live sheet ₹{calculated_total}")
        else:
            print("✅ Stake consistency: PERFECT")
        
        # 🚀 Create the live bet sheet Excel file
        output_file = quant_paths.get_live_bet_sheet_path(target_date)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create summary sheet data
        summary_data = [{
            'target_date': target_date,
            'zone': zone,
            'risk_mode': risk_mode,
            'meta_strategy': meta_strategy,
            'numeric_strategy': numeric_strategy,
            'top_family': strategy_rec.get('top_family', 'UNKNOWN'),
            'top_family_hit_rate': strategy_rec.get('top_family_hit_rate', 0),
            'slot_stakes_frbd': final_stakes.get('FRBD', 0),
            'slot_stakes_gzbd': final_stakes.get('GZBD', 0),
            'slot_stakes_gali': final_stakes.get('GALI', 0),
            'slot_stakes_dswr': final_stakes.get('DSWR', 0),
            'total_daily_stake': final_total,
            'total_final_stake': calculated_total,
            'readiness_level': readiness_level,
            'readiness_score': readiness_score,
            'stake_consistency': 'PERFECT' if abs(calculated_total - final_total) <= 1 else 'MISMATCH'
        }]
        
        summary_df = pd.DataFrame(summary_data)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Live bet sheet with all bets
            live_output_cols = ['slot', 'layer_type', 'number_or_digit', 'tier', 
                              'base_stake', 'final_stake', 'is_s40']
            # Only include columns that exist
            available_cols = [col for col in live_output_cols if col in live_bet_df.columns]
            live_bet_df[available_cols].to_excel(writer, sheet_name='LIVE_BETS', index=False)
            
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='SUMMARY', index=False)
            
            # Keep original detailed bets for reference
            base_bet_df.to_excel(writer, sheet_name='DETAILED_BETS_ORIGINAL', index=False)
        
        print(f"📁 Output File: {output_file.name}")
        print("✅ LIVE BET SHEET READY FOR EXECUTION")
        
        return True
        
    except Exception as e:
        print(f"❌ Error generating live bet sheet: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🚀 Starting Live Bet Sheet Generation...")
    
    success = generate_live_bet_sheet()
    
    if success:
        print("✅ Live Bet Sheet completed successfully!")
        return 0
    else:
        print("❌ Live Bet Sheet generation failed!")
        return 1

if __name__ == "__main__":
    exit(main())
