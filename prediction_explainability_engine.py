# prediction_explainability_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
from collections import Counter
import json

class PredictionExplainabilityEngine:
    """
    PREDICTION EXPLAINABILITY ENGINE
    ✓ Reads SCR9 predictions + analytics data
    ✓ Generates human-readable reasons for each number
    ✓ Creates explainability reports
    """
    
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def find_latest_scr9_files(self):
        """Find the latest SCR9 prediction files"""
        scr9_dir = os.path.join(self.base_dir, "predictions", "deepseek_scr9")
        
        if not os.path.exists(scr9_dir):
            print("❌ SCR9 predictions directory not found")
            return None, None, None
        
        # Find latest prediction files
        pred_files = glob.glob(os.path.join(scr9_dir, "ultimate_predictions_*.xlsx"))
        detail_files = glob.glob(os.path.join(scr9_dir, "ultimate_detailed_*.xlsx"))
        diag_files = glob.glob(os.path.join(scr9_dir, "ultimate_diagnostic_*.xlsx"))
        
        if not pred_files:
            print("❌ No SCR9 prediction files found")
            return None, None, None
        
        # Get latest files by modification time
        latest_pred = max(pred_files, key=os.path.getmtime)
        latest_detail = max(detail_files, key=os.path.getmtime) if detail_files else None
        latest_diag = max(diag_files, key=os.path.getmtime) if diag_files else None
        
        return latest_pred, latest_detail, latest_diag
    
    def load_analytics_data(self):
        """Load analytics data from pattern intelligence - TASK 4 FIX: Dual file support"""
        # TASK 4 FIX: Try both pattern intelligence filenames
        pattern_files = [
            os.path.join(self.base_dir, "logs", "performance", "pattern_intelligence_summary.json"),
            os.path.join(self.base_dir, "logs", "performance", "pattern_intelligence.json")
        ]
        
        analytics_data = {}
        
        # Load pattern intelligence (try both files)
        pattern_loaded = False
        for pattern_file in pattern_files:
            if os.path.exists(pattern_file):
                try:
                    with open(pattern_file, 'r') as f:
                        analytics_data['pattern_intelligence'] = json.load(f)
                    print(f"✅ Loaded pattern intelligence: {os.path.basename(pattern_file)}")
                    pattern_loaded = True
                    break
                except Exception as e:
                    print(f"⚠️ Could not load {pattern_file}: {e}")
        
        if not pattern_loaded:
            print("⚠️ No pattern intelligence files found")
        
        # Load performance data
        performance_file = os.path.join(self.base_dir, "logs", "performance", "ultimate_performance.csv")
        if os.path.exists(performance_file):
            try:
                analytics_data['performance'] = pd.read_csv(performance_file)
                print(f"✅ Loaded performance data: ultimate_performance.csv")
            except Exception as e:
                print(f"⚠️ Could not load performance data: {e}")
        else:
            print("⚠️ Performance file not found: ultimate_performance.csv")
        
        return analytics_data
    
    def analyze_number_patterns(self, number, slot, analytics_data):
        """Analyze patterns for a specific number"""
        reasons = []
        
        # Check pattern intelligence data
        if 'pattern_intelligence' in analytics_data:
            patterns = analytics_data['pattern_intelligence']
            
            # Check slot-specific patterns
            slot_key = slot.lower()
            if slot_key in patterns:
                slot_patterns = patterns[slot_key]
                
                # Check core numbers
                if 'core_numbers' in slot_patterns:
                    cores = slot_patterns['core_numbers']
                    if number in cores:
                        reasons.append(f"Core number (appears in {cores[number]} recent patterns)")
                
                # Check hot numbers
                if 'hot_numbers' in slot_patterns:
                    hots = slot_patterns['hot_numbers']
                    if number in hots:
                        reasons.append(f"Hot number (high frequency)")
                
                # Check gap analysis
                if 'gap_analysis' in slot_patterns:
                    gaps = slot_patterns['gap_analysis']
                    if str(number) in gaps:
                        gap_info = gaps[str(number)]
                        reasons.append(f"Optimal gap timing (last seen {gap_info} days ago)")
        
        # Check performance history
        if 'performance' in analytics_data:
            perf_df = analytics_data['performance']
            slot_perf = perf_df[perf_df['slot'] == slot]
            
            # Check if number has hit recently
            recent_hits = slot_perf[slot_perf['actual'] == number]
            if not recent_hits.empty:
                last_hit = recent_hits['date'].max()
                reasons.append(f"Recent hit on {last_hit}")
            
            # Check prediction accuracy for this number
            pred_hits = slot_perf[slot_perf['predictions'].str.contains(str(number), na=False)]
            if not pred_hits.empty:
                hit_rate = len(pred_hits) / len(slot_perf) * 100
                reasons.append(f"Good prediction history ({hit_rate:.1f}% inclusion rate)")
        
        return reasons
    
    def generate_opposite_reasons(self, number):
        """Generate reasons for opposite numbers"""
        opposite = self.get_opposite(number)
        return [
            f"Opposite of {number:02d}",
            "Complementary number pairing",
            "Mirror number strategy"
        ]
    
    def get_opposite(self, n):
        """Get opposite number"""
        if n < 10:
            return n * 10
        else:
            return (n % 10) * 10 + (n // 10)
    
    def analyze_frequency_patterns(self, number, slot, diagnostic_data):
        """Analyze frequency-based reasons"""
        reasons = []
        
        if diagnostic_data is not None:
            slot_diag = diagnostic_data[diagnostic_data['slot'] == slot]
            if not slot_diag.empty:
                # Check hot numbers
                hot_numbers_str = slot_diag['hot_numbers'].iloc[0]
                if pd.notna(hot_numbers_str):
                    hot_numbers = [int(x.strip()) for x in hot_numbers_str.split(',')]
                    if number in hot_numbers:
                        reasons.append("High frequency (hot number)")
                
                # Check recent numbers
                recent_numbers_str = slot_diag['recent_numbers'].iloc[0]
                if pd.notna(recent_numbers_str):
                    recent_numbers = [int(x.strip()) for x in recent_numbers_str.split(',')]
                    if number in recent_numbers:
                        reasons.append("Recently appeared")
                
                # Total records for context
                total_records = slot_diag['total_records'].iloc[0]
                reasons.append(f"Based on {total_records} historical records")
        
        return reasons
    
    def generate_ensemble_reasons(self, number, rank, all_predictions):
        """Generate reasons based on ensemble scoring"""
        reasons = []
        
        # Count how many scripts predicted this number
        script_count = 0
        for script_name, predictions in all_predictions.items():
            if number in predictions.get('all_numbers', []):
                script_count += 1
        
        if script_count >= 3:
            reasons.append(f"Multi-model consensus ({script_count} scripts)")
        
        if rank <= 3:
            reasons.append(f"High ensemble rank (#{rank})")
        elif rank <= 8:
            reasons.append(f"Medium ensemble rank (#{rank})")
        else:
            reasons.append(f"Supporting ensemble number (#{rank})")
        
        return reasons
    
    def create_explainability_report(self, predictions_file, detail_file, diagnostic_file):
        """Create comprehensive explainability report"""
        print("🎯 Generating Prediction Explainability Report...")
        
        # Load prediction data
        pred_df = pd.read_excel(predictions_file)
        detail_df = pd.read_excel(detail_file) if detail_file else None
        diag_df = pd.read_excel(diagnostic_file) if diagnostic_file else None
        
        # Load analytics data
        analytics_data = self.load_analytics_data()
        
        explainability_data = []
        
        # Process each date and slot
        for _, row in pred_df.iterrows():
            date = row['date']
            pred_type = row['type']
            
            for slot in self.slots:
                if slot in row and pd.notna(row[slot]):
                    numbers_str = row[slot]
                    numbers = [int(x.strip()) for x in numbers_str.split(',')]
                    
                    # Get opposites if available
                    opposites = []
                    opp_key = f'{slot}_OPP'
                    if opp_key in row and pd.notna(row[opp_key]):
                        opposites = [int(x.strip()) for x in row[opp_key].split(',')]
                    
                    # Analyze each number
                    for rank, number in enumerate(numbers, 1):
                        reasons = []
                        
                        # 1. Rank-based reason
                        if rank == 1:
                            reasons.append("🏆 TOP PREDICTION - Highest ensemble score")
                        elif rank <= 5:
                            reasons.append(f"🎯 Top-tier prediction (rank #{rank})")
                        elif rank <= 10:
                            reasons.append(f"📊 Mid-tier prediction (rank #{rank})")
                        else:
                            reasons.append(f"🔍 Supporting prediction (rank #{rank})")
                        
                        # 2. Pattern intelligence reasons
                        pattern_reasons = self.analyze_number_patterns(number, slot, analytics_data)
                        reasons.extend(pattern_reasons)
                        
                        # 3. Frequency analysis reasons
                        freq_reasons = self.analyze_frequency_patterns(number, slot, diag_df)
                        reasons.extend(freq_reasons)
                        
                        # 4. Opposite number reasons (if applicable)
                        if number in opposites[:3]:  # Only for top 3 opposites
                            opp_reasons = self.generate_opposite_reasons(self.get_opposite(number))
                            reasons.extend(opp_reasons)
                        
                        # 5. Range analysis
                        if 0 <= number <= 33:
                            reasons.append("Low range (0-33)")
                        elif 34 <= number <= 66:
                            reasons.append("Mid range (34-66)")
                        else:
                            reasons.append("High range (67-99)")
                        
                        # Add to explainability data
                        explainability_data.append({
                            'date': date,
                            'type': pred_type,
                            'slot': slot,
                            'number': f"{number:02d}",
                            'rank': rank,
                            'reasons': ' | '.join(reasons) if reasons else "General ensemble prediction",
                            'reason_count': len(reasons)
                        })
        
        explain_df = pd.DataFrame(explainability_data)
        
        # Save explainability report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.base_dir, "logs", "performance", f"prediction_explainability_{timestamp}.xlsx")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create summary by slot
        summary_data = []
        for slot in self.slots:
            slot_data = explain_df[explain_df['slot'] == slot]
            top_numbers = slot_data[slot_data['rank'] <= 5]['number'].unique()
            
            summary_data.append({
                'slot': slot,
                'top_numbers': ', '.join(top_numbers[:5]),
                'avg_reasons_per_number': slot_data['reason_count'].mean(),
                'total_unique_reasons': len(slot_data['reasons'].unique())
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            explain_df.to_excel(writer, sheet_name='Detailed_Explanations', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create human-readable text report
        text_report = os.path.join(self.base_dir, "logs", "performance", f"explainability_summary_{timestamp}.txt")
        
        with open(text_report, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("           PREDICTION EXPLAINABILITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source: {os.path.basename(predictions_file)}\n\n")
            
            for slot in self.slots:
                f.write(f"\n{slot} PREDICTIONS:\n")
                f.write("-" * 40 + "\n")
                
                slot_data = explain_df[explain_df['slot'] == slot]
                top_predictions = slot_data[slot_data['rank'] <= 5]
                
                for _, pred in top_predictions.iterrows():
                    f.write(f"\n#{pred['rank']}: {pred['number']}\n")
                    f.write(f"   Reasons: {pred['reasons']}\n")
            
            f.write(f"\n" + "=" * 80 + "\n")
            f.write("SUMMARY:\n")
            f.write("-" * 40 + "\n")
            for _, summary in summary_df.iterrows():
                f.write(f"{summary['slot']}: {summary['top_numbers']} | Avg reasons: {summary['avg_reasons_per_number']:.1f}\n")
        
        print(f"✅ Explainability report saved: {output_file}")
        print(f"✅ Text summary saved: {text_report}")
        
        return explain_df, output_file, text_report
    
    def run(self):
        """Main execution"""
        print("=" * 70)
        print("  🧠 PREDICTION EXPLAINABILITY ENGINE")
        print("  ✓ Human-readable prediction reasons")
        print("  ✓ Pattern intelligence integration")
        print("=" * 70)
        
        # Find latest SCR9 files
        pred_file, detail_file, diag_file = self.find_latest_scr9_files()
        
        if not pred_file:
            print("❌ No SCR9 prediction files found. Run SCR9 first.")
            return
        
        print(f"📊 Using prediction file: {os.path.basename(pred_file)}")
        if detail_file:
            print(f"📋 Using detail file: {os.path.basename(detail_file)}")
        if diag_file:
            print(f"🔍 Using diagnostic file: {os.path.basename(diag_file)}")
        
        # Generate explainability report
        explain_df, excel_file, text_file = self.create_explainability_report(pred_file, detail_file, diag_file)
        
        print("\n" + "=" * 70)
        print("📊 EXPLAINABILITY RESULTS")
        print("=" * 70)
        
        # Show top predictions with reasons
        for slot in self.slots:
            slot_data = explain_df[explain_df['slot'] == slot]
            top3 = slot_data[slot_data['rank'] <= 3]
            
            print(f"\n{slot} - Top 3 Predictions:")
            for _, pred in top3.iterrows():
                print(f"  #{pred['rank']}: {pred['number']} - {pred['reasons'][:60]}...")
        
        print(f"\n✅ Files generated:")
        print(f"   - {os.path.relpath(excel_file)}")
        print(f"   - {os.path.relpath(text_file)}")
        print("=" * 70)

if __name__ == "__main__":
    explainer = PredictionExplainabilityEngine()
    explainer.run()