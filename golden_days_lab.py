import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

class GoldenDaysLab:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def load_pnl_data(self):
        """Load P&L data"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if not pnl_file.exists():
            print("❌ P&L file not found")
            return None
        try:
            day_df = pd.read_excel(pnl_file, sheet_name='day_level')
            day_df['date'] = pd.to_datetime(day_df['date']).dt.date
            return day_df
        except Exception as e:
            print(f"❌ Error reading P&L: {e}")
            return None

    def load_real_results(self):
        """Load real results from number prediction learn.xlsx"""
        results_file = self.base_dir / "number prediction learn.xlsx"
        if not results_file.exists():
            print("❌ Results file not found")
            return {}
            
        try:
            df = pd.read_excel(results_file, sheet_name=None, header=None)
            all_results = {}
            current_year, current_month = 2025, 1
            
            for sheet_name, sheet_df in df.items():
                for idx, row in sheet_df.iterrows():
                    if pd.isna(row[0]): continue
                    row0_str = str(row[0]).strip()
                    
                    if '2025' in row0_str:
                        try:
                            parts = row0_str.split('-')
                            if len(parts) >= 2: current_month = int(parts[1])
                        except: pass
                        continue
                    
                    if row0_str.isdigit():
                        day_num = int(row0_str)
                        for slot_idx, slot_name in enumerate(self.slots, 1):
                            if len(row) > slot_idx and pd.notna(row[slot_idx]):
                                cell_val = str(row[slot_idx]).strip()
                                if cell_val and cell_val not in ['XX', '']:
                                    try:
                                        number = int(float(cell_val)) % 100
                                        date_obj = datetime(current_year, current_month, day_num).date()
                                        all_results[(date_obj, slot_name)] = number
                                    except: continue
            return all_results
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return {}

    def analyze_golden_days(self, pnl_df, real_results):
        """Analyze high-profit days"""
        if pnl_df is None or not real_results:
            return [], []
            
        # Find golden days (profit >= 2500)
        golden_days = pnl_df[pnl_df['profit_total'] >= 2500].copy()
        golden_days = golden_days.sort_values('date')
        
        golden_data = []
        slot_hits_data = []
        
        for _, day_row in golden_days.iterrows():
            date = day_row['date']
            
            golden_data.append({
                'date': date,
                'profit_total': day_row['profit_total'],
                'cumulative_profit': day_row.get('cum_profit', day_row['profit_total']),
                'stake_total': day_row['stake_total'],
                'return_total': day_row['return_total']
            })
            
            # Analyze each slot for this day
            for slot in self.slots:
                result = real_results.get((date, slot))
                if result is None:
                    continue
                    
                # Basic pattern analysis
                tens_digit = result // 10
                ones_digit = result % 10
                is_s40 = result in [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
                
                slot_hits_data.append({
                    'date': date,
                    'slot': slot,
                    'hit_number': result,
                    's40_flag': is_s40,
                    'tens_digit': tens_digit,
                    'ones_digit': ones_digit,
                    'pack_tags': self.get_pack_tags(result)
                })
                
        return golden_data, slot_hits_data

    def get_pack_tags(self, number):
        """Get pack tags for a number"""
        packs = []
        num_str = f"{number:02d}"
        
        # Simple pack detection
        digits = [int(d) for d in num_str]
        
        # Check for sequential packs
        if len(digits) >= 2:
            if digits[1] == digits[0] + 1:  # Sequential like 23, 34
                packs.append(f"seq_{digits[0]}{digits[1]}")
                
        return ",".join(packs) if packs else "none"

    def generate_summary(self, slot_hits_data):
        """Generate pattern summary"""
        if not slot_hits_data:
            return []
            
        df = pd.DataFrame(slot_hits_data)
        
        summary = []
        # S40 analysis
        s40_count = len(df[df['s40_flag'] == True])
        s40_pct = (s40_count / len(df)) * 100 if len(df) > 0 else 0
        summary.append({'pattern': 'S40_Numbers', 'count': s40_count, 'percentage': s40_pct})
        
        # Tens digit analysis
        for tens in range(10):
            count = len(df[df['tens_digit'] == tens])
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            summary.append({'pattern': f'Tens_{tens}', 'count': count, 'percentage': pct})
            
        # Ones digit analysis  
        for ones in range(10):
            count = len(df[df['ones_digit'] == ones])
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            summary.append({'pattern': f'Ones_{ones}', 'count': count, 'percentage': pct})
            
        return summary

    def print_console_report(self, golden_data, slot_hits_data, summary_data):
        """Print console report"""
        print("\n" + "="*80)
        print("🎯 GOLDEN DAYS LAB - HIGH PROFIT DAYS ANALYSIS")
        print("="*80)
        
        print(f"\n💰 GOLDEN DAYS FOUND: {len(golden_data)}")
        print("-" * 60)
        for day in golden_data:
            print(f"📅 {day['date']}: Profit ₹{day['profit_total']:+,.0f} | Stake ₹{day['stake_total']} | Return ₹{day['return_total']}")
        
        print(f"\n🎯 TOTAL HITS ANALYZED: {len(slot_hits_data)}")
        print("-" * 40)
        
        if summary_data:
            print("\n📊 PATTERN SUMMARY:")
            print("-" * 40)
            for pattern in summary_data:
                if pattern['count'] > 0:
                    print(f"   {pattern['pattern']:15} : {pattern['count']:2} hits ({pattern['percentage']:5.1f}%)")

    def save_analysis(self, golden_data, slot_hits_data, summary_data):
        """Save analysis to Excel"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "golden_days_analysis.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            if golden_data:
                pd.DataFrame(golden_data).to_excel(writer, sheet_name='golden_days', index=False)
            if slot_hits_data:
                pd.DataFrame(slot_hits_data).to_excel(writer, sheet_name='slot_hits', index=False)
            if summary_data:
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='pattern_summary', index=False)
                
        print(f"\n💾 Analysis saved to: {output_file}")

    def run(self):
        """Main execution"""
        print("🔍 GOLDEN DAYS LAB - Analyzing high-profit patterns...")
        
        # Load data
        pnl_df = self.load_pnl_data()
        real_results = self.load_real_results()
        
        if pnl_df is None or not real_results:
            return False
            
        # Analyze
        golden_data, slot_hits_data = self.analyze_golden_days(pnl_df, real_results)
        summary_data = self.generate_summary(slot_hits_data)
        
        # Output
        self.print_console_report(golden_data, slot_hits_data, summary_data)
        self.save_analysis(golden_data, slot_hits_data, summary_data)
        
        return True

def main():
    lab = GoldenDaysLab()
    success = lab.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())