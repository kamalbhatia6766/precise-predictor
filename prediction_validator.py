# prediction_validator.py
"""
PREDICTION VALIDATOR - Historical Strategy Performance Analyzer

PURPOSE:
Compare historical performance of three strategies:
1. BASE: Equal stakes (₹55 per slot)
2. DYNAMIC: Dynamic stake allocation  
3. HIGH_CONVICTION: Dynamic + conviction-based filtering

USAGE:
py -3.12 prediction_validator.py --days 10
"""

import pandas as pd
import json
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import sys

class PredictionValidator:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.results = {
            'base': {'stake': 0, 'return': 0, 'profit': 0},
            'dynamic': {'stake': 0, 'return': 0, 'profit': 0}, 
            'conviction': {'stake': 0, 'return': 0, 'profit': 0}
        }
        
    def load_pnl_history(self, days=10):
        """Load P&L history from Excel file"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        
        if not pnl_file.exists():
            print("❌ P&L history file not found")
            return None
            
        try:
            df = pd.read_excel(pnl_file, sheet_name='day_level')
            # Get last N days
            df = df.tail(days)
            return df
        except Exception as e:
            print(f"❌ Error loading P&L history: {e}")
            return None
    
    def load_current_plans(self):
        """Load current dynamic stake and conviction plans"""
        plans = {}
        
        # Load dynamic stake plan
        stake_file = self.base_dir / "logs" / "performance" / "dynamic_stake_plan.json"
        if stake_file.exists():
            try:
                with open(stake_file, 'r') as f:
                    stake_data = json.load(f)
                    # Extract only the stakes dictionary
                    plans['dynamic_stake'] = stake_data.get('stakes', {})
                    print(f"✅ Loaded dynamic stakes: {plans['dynamic_stake']}")
            except Exception as e:
                print(f"⚠️ Could not load dynamic stake plan: {e}")
                plans['dynamic_stake'] = {}
        
        # Find latest high conviction bet plan
        bet_plan_dir = self.base_dir / "predictions" / "bet_engine"
        conviction_files = list(bet_plan_dir.glob("high_conviction_bet_plan_*.xlsx"))
        if conviction_files:
            latest_file = max(conviction_files, key=lambda x: x.stat().st_mtime)
            try:
                df = pd.read_excel(latest_file, sheet_name='high_conviction_plan')
                plans['conviction'] = df
                print(f"✅ Loaded conviction plan: {latest_file.name}")
            except Exception as e:
                print(f"⚠️ Could not load conviction plan: {e}")
        
        return plans
    
    def calculate_strategy_performance(self, pnl_df, plans):
        """Calculate performance for all three strategies"""
        base_stake_per_slot = 55
        
        for day_index, day_row in pnl_df.iterrows():
            # Extract day data
            try:
                base_stake_total = float(day_row.get('stake_total', 0))
                base_return_total = float(day_row.get('return_total', 0))
                base_profit_total = float(day_row.get('profit_total', 0))
                
                if base_stake_total == 0:
                    continue
                    
                # Strategy 1: BASE (equal stakes)
                self.results['base']['stake'] += base_stake_total
                self.results['base']['return'] += base_return_total  
                self.results['base']['profit'] += base_profit_total
                
                # Strategy 2: DYNAMIC (scaled by dynamic stake)
                if 'dynamic_stake' in plans and plans['dynamic_stake']:
                    stake_map = plans['dynamic_stake']
                    dynamic_total_stake = sum(stake_map.values())
                    
                    if base_stake_total > 0:
                        scale_factor = dynamic_total_stake / base_stake_total
                        
                        dynamic_return = base_return_total * scale_factor
                        dynamic_profit = base_profit_total * scale_factor
                        
                        self.results['dynamic']['stake'] += dynamic_total_stake
                        self.results['dynamic']['return'] += dynamic_return
                        self.results['dynamic']['profit'] += dynamic_profit
                
                # Strategy 3: HIGH_CONVICTION (dynamic + filtering)
                if 'conviction' in plans and 'dynamic_stake' in plans and plans['dynamic_stake']:
                    conviction_df = plans['conviction']
                    stake_map = plans['dynamic_stake']
                    
                    conviction_total_stake = 0
                    included_slots = []
                    
                    # Calculate conviction-based inclusion
                    for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                        slot_data = conviction_df[conviction_df['slot'] == slot]
                        if not slot_data.empty:
                            final_rec = slot_data.iloc[0].get('final_recommendation', 'PLAY_NORMAL')
                            conviction_flag = slot_data.iloc[0].get('conviction_flag', 'MEDIUM')
                            
                            # Include slots that are not avoided
                            if 'AVOID' not in final_rec.upper():
                                slot_stake = stake_map.get(slot, 0)
                                conviction_total_stake += slot_stake
                                included_slots.append(slot)
                    
                    if base_stake_total > 0 and conviction_total_stake > 0:
                        conviction_scale_factor = conviction_total_stake / base_stake_total
                        
                        conviction_return = base_return_total * conviction_scale_factor
                        conviction_profit = base_profit_total * conviction_scale_factor
                        
                        self.results['conviction']['stake'] += conviction_total_stake
                        self.results['conviction']['return'] += conviction_return
                        self.results['conviction']['profit'] += conviction_profit
                    
            except Exception as e:
                print(f"⚠️ Error processing day {day_row.get('date', 'unknown')}: {e}")
                continue
    
    def generate_report(self, days):
        """Generate validation report"""
        report_data = []
        
        for strategy, data in self.results.items():
            stake = data['stake']
            return_amt = data['return']
            profit = data['profit']
            
            roi = (profit / stake * 100) if stake > 0 else 0
            
            report_data.append({
                'Strategy': strategy.upper(),
                'Total_Stake': stake,
                'Total_Return': return_amt,
                'Total_Profit': profit,
                'ROI_Percent': roi
            })
        
        # Save to Excel
        report_file = self.base_dir / "logs" / "performance" / "prediction_validator_report.xlsx"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        df_report = pd.DataFrame(report_data)
        df_report.to_excel(report_file, index=False)
        
        return df_report
    
    def run_validation(self, days=10):
        """Run complete validation analysis"""
        print("🔍 PREDICTION VALIDATOR - Historical Strategy Analysis")
        print("=" * 60)
        print(f"Analyzing last {days} days of performance data...")
        print()
        
        # Load data
        pnl_df = self.load_pnl_history(days)
        if pnl_df is None or pnl_df.empty:
            print("❌ No P&L data available for analysis")
            return False
        
        plans = self.load_current_plans()
        
        # Calculate performance
        self.calculate_strategy_performance(pnl_df, plans)
        
        # Generate report
        report_df = self.generate_report(days)
        
        # Display summary
        print("📊 STRATEGY COMPARISON SUMMARY:")
        print("-" * 50)
        
        for _, row in report_df.iterrows():
            strategy = row['Strategy']
            stake = row['Total_Stake']
            profit = row['Total_Profit']
            roi = row['ROI_Percent']
            
            print(f"{strategy:15}: stake=₹{stake:,.0f}, profit=₹{profit:+,.0f}, ROI={roi:+.1f}%")
        
        print()
        print("💡 INTERPRETATION:")
        print("   BASE        : Equal ₹55 stakes for all slots")
        print("   DYNAMIC     : Current dynamic stake allocation")  
        print("   CONVICTION  : Dynamic + high conviction filtering")
        print()
        print(f"💾 Full report saved to: {self.base_dir / 'logs' / 'performance' / 'prediction_validator_report.xlsx'}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Prediction Validator - Historical Strategy Analysis')
    parser.add_argument('--days', type=int, default=10, help='Number of days to analyze (default: 10)')
    
    args = parser.parse_args()
    
    validator = PredictionValidator()
    success = validator.run_validation(args.days)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())