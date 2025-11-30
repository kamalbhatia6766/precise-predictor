# final_bet_plan_engine.py - UPDATED
"""
FINAL BET PLAN ENGINE - Consolidated Betting Strategy Generator

PURPOSE:
Generate a single consolidated final bet plan based on meta-strategy analysis,
combining BASE, DYNAMIC, and CONVICTION strategies with reality JSONs.

USAGE:
py -3.12 final_bet_plan_engine.py
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import sys

# 🆕 Import central helpers
import quant_paths
import quant_data_core

# Strategy mapping from meta to numeric
META_TO_NUMERIC_STRATEGY = {
    "STRAT_S40_BOOST": "DYNAMIC",
    "STRAT_164950_CORE": "DYNAMIC", 
    "STRAT_MID_PACK_FOCUS": "DYNAMIC",
    "STRAT_BALANCED_CORE": "DYNAMIC",
    "BASE": "BASE",
    "DYNAMIC": "DYNAMIC", 
    "CONVICTION": "CONVICTION"
}

class FinalBetPlanEngine:
    def __init__(self):
        self.base_dir = quant_paths.get_project_root()
        self.target_date = None
        self.strategy_data = {}
        self.confidence_data = {}
        self.dynamic_stakes = {}
        self.recovery_plan = {}
        self.bet_plans = {}
        
    def discover_latest_files(self):
        """Discover latest bet plan files and determine target date"""
        bet_engine_dir = quant_paths.get_bet_engine_dir()
        
        if not bet_engine_dir.exists():
            print("❌ Bet engine directory not found")
            return False
        
        # Find all bet plan files and extract dates
        file_patterns = {
            'master': "bet_plan_master_*.xlsx",
            'enhanced': "enhanced_bet_plan_*.xlsx", 
            'conviction': "high_conviction_bet_plan_*.xlsx"
        }
        
        found_dates = set()
        
        for plan_type, pattern in file_patterns.items():
            files = list(bet_engine_dir.glob(pattern))
            if files:
                # Extract date from filename
                for file in files:
                    date_str = file.stem.split('_')[-1]
                    if date_str.isdigit() and len(date_str) == 8:
                        found_dates.add(date_str)
                        self.bet_plans[plan_type] = file
                        print(f"✅ Found {plan_type} plan: {file.name}")
        
        if not found_dates:
            print("❌ No bet plan files found")
            return False
        
        # Use the latest date
        self.target_date = max(found_dates)
        print(f"📅 Detected latest date: {self.target_date}")
        
        return True
    
    def load_strategy_recommendation(self):
        """Load strategy recommendation data"""
        strategy_file = quant_paths.get_performance_logs_dir() / "strategy_recommendation.json"
        
        if not strategy_file.exists():
            print("⚠️ Strategy recommendation not found, using default: DYNAMIC")
            self.strategy_data = {
                'recommended_strategy': 'DYNAMIC',
                'confidence_level': 'LOW',
                'risk_mode': 'DEFENSIVE',
                'reason': 'Strategy recommendation missing, defaulting to DYNAMIC'
            }
            return True
            
        try:
            with open(strategy_file, 'r') as f:
                data = json.load(f)
            
            meta_strategy = data.get('recommended_strategy', 'DYNAMIC')
            numeric_strategy = META_TO_NUMERIC_STRATEGY.get(meta_strategy, 'DYNAMIC')
            
            self.strategy_data = {
                'meta_strategy': meta_strategy,
                'numeric_strategy': numeric_strategy,
                'confidence_level': data.get('confidence_level', 'LOW'),
                'risk_mode': data.get('risk_mode', 'DEFENSIVE'),
                'reason': data.get('reason', 'No reason provided'),
                'window_days': data.get('window_days', 10),
                'strategies': data.get('strategies', {})
            }
            
            print(f"✅ Loaded strategy: {meta_strategy} → {numeric_strategy}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading strategy recommendation: {e}, using default: DYNAMIC")
            self.strategy_data = {
                'meta_strategy': 'DYNAMIC',
                'numeric_strategy': 'DYNAMIC',
                'confidence_level': 'LOW', 
                'risk_mode': 'DEFENSIVE',
                'reason': f'Error loading strategy: {e}'
            }
            return True
    
    def load_dynamic_stakes(self):
        """Load dynamic stake plan"""
        stake_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        
        if not stake_file.exists():
            print("⚠️ Dynamic stake plan not found")
            return True
            
        try:
            with open(stake_file, 'r') as f:
                data = json.load(f)
            
            self.dynamic_stakes = data.get('slot_stakes', {})
            print(f"✅ Loaded dynamic stakes: {self.dynamic_stakes}")
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading dynamic stakes: {e}")
            return True

    def load_loss_recovery_plan(self):
        """🆕 Load loss recovery plan for risk context"""
        recovery_file = quant_paths.get_performance_logs_dir() / "loss_recovery_plan.json"
        
        if not recovery_file.exists():
            print("⚠️ Loss recovery plan not found")
            return True
            
        try:
            with open(recovery_file, 'r') as f:
                data = json.load(f)
            
            self.recovery_plan = {
                'zone': data.get('zone', 'UNKNOWN'),
                'risk_mode': data.get('recommended_risk_mode', 'DEFENSIVE'),
                'rolling_roi': data.get('rolling_roi', 0)
            }
            print(f"✅ Loaded recovery plan: {self.recovery_plan['zone']} zone")
            return True
            
        except Exception as e:
            print(f"⚠️ Error loading recovery plan: {e}")
            return True
    
    def _get_confidence_level(self, score):
        """Map confidence score to level"""
        if score >= 80:
            return "VERY_HIGH"
        elif score >= 65:
            return "HIGH"
        elif score >= 50:
            return "MEDIUM" 
        elif score >= 35:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _find_column_case_insensitive(self, df, column_name):
        """Find column name case-insensitively"""
        column_name_lower = column_name.lower()
        for col in df.columns:
            if col.lower() == column_name_lower:
                return col
        return None
    
    def extract_slot_stakes(self, file_path, plan_type):
        """Extract slot stakes from bet plan files"""
        if not file_path.exists():
            print(f"⚠️ {plan_type} plan file not found: {file_path}")
            return {}
        
        try:
            # Handle different file structures
            if plan_type == 'master':
                # Try summary sheet first, then main sheet
                sheet_name = None
                excel_file = pd.ExcelFile(file_path)
                if 'summary' in [sheet.lower() for sheet in excel_file.sheet_names]:
                    sheet_name = [sheet for sheet in excel_file.sheet_names if sheet.lower() == 'summary'][0]
                elif 'bets' in [sheet.lower() for sheet in excel_file.sheet_names]:
                    sheet_name = [sheet for sheet in excel_file.sheet_names if sheet.lower() == 'bets'][0]
                
                if not sheet_name:
                    sheet_name = excel_file.sheet_names[0]  # Fallback to first sheet
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Find columns case-insensitively
                slot_col = self._find_column_case_insensitive(df, 'slot')
                stake_col = self._find_column_case_insensitive(df, 'stake')
                
                stakes = {}
                slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
                
                if slot_col and stake_col:
                    for slot in slots:
                        slot_rows = df[df[slot_col] == slot]
                        if not slot_rows.empty:
                            total_stake = slot_rows[stake_col].sum()
                            stakes[slot] = float(total_stake)
                
                # If we couldn't extract, use default base stake of 55
                if not stakes:
                    print(f"   Using default base stake (₹55) for {plan_type} plan")
                    stakes = {slot: 55 for slot in slots}
                
                return stakes
                
            elif plan_type == 'enhanced':
                # Use reality-driven dynamic stake JSON as the enhanced layer
                print(f"   Using dynamic stake plan JSON for {plan_type} plan")
                # Agar JSON khali ho gaya to safe fallback: 55 per slot
                if not self.dynamic_stakes:
                    return {slot: 55 for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']}
                return self.dynamic_stakes.copy()
                
            elif plan_type == 'conviction':
                # Conviction plan has high_conviction_plan sheet
                sheet_name = None
                excel_file = pd.ExcelFile(file_path)
                if 'high_conviction_plan' in [sheet.lower() for sheet in excel_file.sheet_names]:
                    sheet_name = [sheet for sheet in excel_file.sheet_names if sheet.lower() == 'high_conviction_plan'][0]
                
                if not sheet_name:
                    sheet_name = excel_file.sheet_names[0]
                
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                
                # Find columns case-insensitively
                slot_col = self._find_column_case_insensitive(df, 'slot')
                recommended_stake_col = self._find_column_case_insensitive(df, 'recommended_stake')
                
                stakes = {}
                slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
                
                if slot_col and recommended_stake_col:
                    for slot in slots:
                        slot_rows = df[df[slot_col] == slot]
                        if not slot_rows.empty:
                            stake_value = slot_rows[recommended_stake_col].iloc[0]
                            if pd.notna(stake_value):
                                stakes[slot] = float(stake_value)
                
                # Fallback to dynamic stakes if extraction failed
                if not stakes:
                    print(f"   Using dynamic stakes for {plan_type} plan")
                    stakes = self.dynamic_stakes.copy()
                
                return stakes
                
        except Exception as e:
            print(f"⚠️ Error extracting {plan_type} stakes: {e}")
            # Return appropriate fallback
            if plan_type == 'master':
                return {slot: 55 for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']}
            else:
                return self.dynamic_stakes.copy()
    
    def generate_final_plan(self):
        """Generate the final bet plan"""
        # Extract stakes from all plan types
        base_stakes = self.extract_slot_stakes(self.bet_plans.get('master'), 'master')
        dynamic_stakes = self.extract_slot_stakes(self.bet_plans.get('enhanced'), 'enhanced')
        conviction_stakes = self.extract_slot_stakes(self.bet_plans.get('conviction'), 'conviction')
        
        # Determine strategy
        meta_strategy = self.strategy_data.get('meta_strategy', 'DYNAMIC')
        numeric_strategy = self.strategy_data.get('numeric_strategy', 'DYNAMIC')
        
        # Generate final slot plan
        final_plan_data = []
        slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
        
        print(f"\n🎯 Strategy: {meta_strategy} → {numeric_strategy}")
        print("📊 Per-slot comparison:")
        print("-" * 60)
        
        for slot in slots:
            # Get stakes for all strategies
            base_stake = base_stakes.get(slot, 55)
            dynamic_stake = dynamic_stakes.get(slot, 55)
            conviction_stake = conviction_stakes.get(slot, 55)
            
            # Choose final stake based on numeric strategy
            if numeric_strategy == 'BASE':
                final_stake = base_stake
            elif numeric_strategy == 'DYNAMIC':
                final_stake = dynamic_stake
            else:  # CONVICTION
                final_stake = conviction_stake
            
            # Add to plan data
            final_plan_data.append({
                'date': f"{self.target_date[:4]}-{self.target_date[4:6]}-{self.target_date[6:8]}",
                'slot': slot,
                'meta_strategy': meta_strategy,
                'numeric_strategy': numeric_strategy,
                'meta_confidence_level': self.strategy_data.get('confidence_level', 'LOW'),
                'risk_mode': self.strategy_data.get('risk_mode', 'DEFENSIVE'),
                'zone': self.recovery_plan.get('zone', 'UNKNOWN'),
                'base_slot_stake': base_stake,
                'dynamic_slot_stake': dynamic_stake,
                'conviction_slot_stake': conviction_stake,
                'final_slot_stake': final_stake
            })
            
            # Print comparison
            print(f"  {slot}: base=₹{base_stake:.0f}, dynamic=₹{dynamic_stake:.0f}, "
                  f"conviction=₹{conviction_stake:.0f} → FINAL=₹{final_stake:.0f} ({numeric_strategy})")
        
        # Add total row
        total_final_stake = sum(item['final_slot_stake'] for item in final_plan_data)
        final_plan_data.append({
            'date': 'TOTAL',
            'slot': '',
            'meta_strategy': '',
            'numeric_strategy': '',
            'meta_confidence_level': '',
            'risk_mode': '',
            'zone': '',
            'base_slot_stake': sum(item['base_slot_stake'] for item in final_plan_data[:-1]),
            'dynamic_slot_stake': sum(item['dynamic_slot_stake'] for item in final_plan_data[:-1]),
            'conviction_slot_stake': sum(item['conviction_slot_stake'] for item in final_plan_data[:-1]),
            'final_slot_stake': total_final_stake
        })
        
        return final_plan_data, total_final_stake
    
    def generate_meta_summary(self, final_plan_data):
        """Generate meta summary sheet"""
        strategy_stats = self.strategy_data.get('strategies', {})
        
        meta_data = [{
            'target_date': f"{self.target_date[:4]}-{self.target_date[4:6]}-{self.target_date[6:8]}",
            'meta_strategy': self.strategy_data.get('meta_strategy', 'DYNAMIC'),
            'numeric_strategy': self.strategy_data.get('numeric_strategy', 'DYNAMIC'),
            'meta_confidence_level': self.strategy_data.get('confidence_level', 'LOW'),
            'risk_mode': self.strategy_data.get('risk_mode', 'DEFENSIVE'),
            'zone': self.recovery_plan.get('zone', 'UNKNOWN'),
            'rolling_roi': self.recovery_plan.get('rolling_roi', 0),
            'window_days': self.strategy_data.get('window_days', 10),
            'base_total_stake': strategy_stats.get('BASE', {}).get('total_stake', 0),
            'base_total_profit': strategy_stats.get('BASE', {}).get('total_profit', 0),
            'base_roi': strategy_stats.get('BASE', {}).get('roi_percent', 0),
            'dynamic_total_stake': strategy_stats.get('DYNAMIC', {}).get('total_stake', 0),
            'dynamic_total_profit': strategy_stats.get('DYNAMIC', {}).get('total_profit', 0),
            'dynamic_roi': strategy_stats.get('DYNAMIC', {}).get('roi_percent', 0),
            'conviction_total_stake': strategy_stats.get('CONVICTION', {}).get('total_stake', 0),
            'conviction_total_profit': strategy_stats.get('CONVICTION', {}).get('total_profit', 0),
            'conviction_roi': strategy_stats.get('CONVICTION', {}).get('roi_percent', 0),
            'strategy_reason': self.strategy_data.get('reason', 'No reason provided'),
            'final_total_stake': final_plan_data[-1]['final_slot_stake']  # Total from final plan
        }]
        
        return meta_data
    
    def save_final_plan(self, final_plan_data, meta_summary_data):
        """Save final bet plan to Excel"""
        output_file = quant_paths.get_final_bet_plan_path(self.target_date)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrames
        df_final_plan = pd.DataFrame(final_plan_data)
        df_meta_summary = pd.DataFrame(meta_summary_data)
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_final_plan.to_excel(writer, sheet_name='final_slot_plan', index=False)
            df_meta_summary.to_excel(writer, sheet_name='meta_summary', index=False)
        
        return output_file
    
    def run_engine(self):
        """Run complete final bet plan engine"""
        print("🧠 FINAL BET PLAN ENGINE - Consolidated Strategy Generator")
        print("=" * 60)
        
        # Step 1: Discover files and target date
        if not self.discover_latest_files():
            return False
        
        # Step 2: Load strategy recommendation
        if not self.load_strategy_recommendation():
            return False
        
        # Step 3: Load dynamic stakes
        if not self.load_dynamic_stakes():
            return False

        # Step 4: 🆕 Load loss recovery plan
        if not self.load_loss_recovery_plan():
            return False
        
        # Step 5: Generate final plan
        final_plan_data, total_final_stake = self.generate_final_plan()
        
        # Step 6: Generate meta summary
        meta_summary_data = self.generate_meta_summary(final_plan_data)
        
        # Step 7: Save final plan
        output_file = self.save_final_plan(final_plan_data, meta_summary_data)
        
        # Final summary
        print("\n🎉 FINAL SUMMARY:")
        print("-" * 40)
        print(f"  TOTAL final stake: ₹{total_final_stake:.0f}")
        print(f"  Meta strategy: {self.strategy_data.get('meta_strategy', 'DYNAMIC')}")
        print(f"  Numeric strategy: {self.strategy_data.get('numeric_strategy', 'DYNAMIC')}")
        print(f"  Zone: {self.recovery_plan.get('zone', 'UNKNOWN')}")
        print(f"  Risk mode: {self.strategy_data.get('risk_mode', 'DEFENSIVE')}")
        print(f"  Output file: {output_file}")
        
        return True

def main():
    engine = FinalBetPlanEngine()
    success = engine.run_engine()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
