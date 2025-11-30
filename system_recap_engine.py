# system_recap_engine.py - UPDATED WITH HUD SNAPSHOT
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob
import json
from pathlib import Path

class SystemRecapEngine:
    """
    SYSTEM RECAP ENGINE - AI BRAIN REPORT
    ✓ Creates visible learning snapshot
    ✓ Shows ROI summary, slot performance, pattern intelligence
    ✓ Generates human-readable recap files
    ✓ Provides HUD snapshot data
    """
    
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def load_performance_data(self):
        """Load ultimate performance data"""
        perf_file = self.base_dir / "logs" / "performance" / "ultimate_performance.csv"
        
        if not perf_file.exists():
            print("❌ Performance file not found")
            return None
            
        try:
            df = pd.read_csv(perf_file)
            print(f"✅ Loaded performance data: {len(df)} records")
            return df
        except Exception as e:
            print(f"❌ Error loading performance data: {e}")
            return None
    
    def load_pnl_data(self):
        """Load P&L history"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        
        if not pnl_file.exists():
            print("❌ P&L file not found")
            return None, None
            
        try:
            slot_df = pd.read_excel(pnl_file, sheet_name='slot_level')
            day_df = pd.read_excel(pnl_file, sheet_name='day_level')
            print(f"✅ Loaded P&L data: {len(slot_df)} slot records, {len(day_df)} days")
            return slot_df, day_df
        except Exception as e:
            print(f"❌ Error loading P&L data: {e}")
            return None, None
    
    def load_pattern_intelligence(self):
        """Load pattern intelligence data"""
        pattern_files = [
            self.base_dir / "logs" / "performance" / "pattern_intelligence_summary.json",
            self.base_dir / "logs" / "performance" / "pattern_intelligence.json"
        ]
        
        for pattern_file in pattern_files:
            if pattern_file.exists():
                try:
                    with open(pattern_file, 'r') as f:
                        data = json.load(f)
                    print(f"✅ Loaded pattern intelligence: {pattern_file.name}")
                    return data
                except Exception as e:
                    print(f"❌ Error loading {pattern_file}: {e}")
        
        print("⚠️ No pattern intelligence files found")
        return None
    
    def load_fusion_weights(self):
        """Load smart fusion weights"""
        weights_file = self.base_dir / "logs" / "performance" / "smart_fusion_weights.json"
        
        if weights_file.exists():
            try:
                with open(weights_file, 'r') as f:
                    weights = json.load(f)
                print(f"✅ Loaded fusion weights")
                return weights
            except Exception as e:
                print(f"❌ Error loading fusion weights: {e}")
        
        return None
    
    def load_strategy_recommendation(self):
        """Load strategy recommendations"""
        strategy_file = self.base_dir / "logs" / "performance" / "strategy_recommendation.json"
        
        if strategy_file.exists():
            try:
                with open(strategy_file, 'r') as f:
                    strategy = json.load(f)
                print(f"✅ Loaded strategy recommendations")
                return strategy
            except Exception as e:
                print(f"❌ Error loading strategy: {e}")
        
        return None
    
    def calculate_roi_summary(self, day_df):
        """Calculate ROI summary from day-level data"""
        if day_df is None or day_df.empty:
            return {}
            
        try:
            latest_day = day_df.iloc[-1]
            roi_data = {
                'total_stake': latest_day.get('stake_total', 0),
                'total_return': latest_day.get('return_total', 0),
                'total_profit': latest_day.get('profit_total', 0),
                'roi_percent': latest_day.get('roi_total', 0),
                'cumulative_profit': day_df['profit_total'].sum() if 'profit_total' in day_df.columns else 0
            }
            return roi_data
        except Exception as e:
            print(f"❌ Error calculating ROI: {e}")
            return {}
    
    def calculate_slot_performance(self, slot_df):
        """Calculate slot-wise performance"""
        if slot_df is None or slot_df.empty:
            return {}
            
        slot_performance = {}
        
        for slot in self.slots:
            slot_data = slot_df[slot_df['slot'] == slot]
            if not slot_data.empty:
                latest = slot_data.iloc[-1]
                slot_performance[slot] = {
                    'total_stake': latest.get('stake_total', 0),
                    'total_return': latest.get('return_total', 0),
                    'profit': latest.get('profit', 0),
                    'main_hits': latest.get('main_hits', 0),
                    'andar_hits': latest.get('andar_hits', 0),
                    'bahar_hits': latest.get('bahar_hits', 0)
                }
        
        return slot_performance
    
    def analyze_pattern_insights(self, pattern_data):
        """Analyze pattern intelligence insights"""
        if not pattern_data:
            return {}
            
        insights = {
            'total_patterns': pattern_data.get('total_patterns_analyzed', 0),
            'high_performance_patterns': pattern_data.get('high_performance_patterns', 0),
            'recommendations': pattern_data.get('top_recommendations', [])
        }
        
        return insights
    
    def get_hud_snapshot(self):
        """
        TASK 8: Lightweight HUD data for controller
        returns dict with:
          - roi_percent (float)
          - slot_profits: dict slot -> profit
          - patterns_analyzed (int)
        """
        print("📊 Generating HUD Snapshot...")
        
        # Reuse existing load functions
        slot_df, day_df = self.load_pnl_data()
        pattern_data = self.load_pattern_intelligence()
        
        # Calculate metrics using existing functions
        roi_summary = self.calculate_roi_summary(day_df)
        slot_performance = self.calculate_slot_performance(slot_df)
        pattern_insights = self.analyze_pattern_insights(pattern_data)
        
        hud_data = {
            "roi_percent": roi_summary.get("roi_percent", 0.0),
            "slot_profits": {slot: data.get("profit", 0) for slot, data in slot_performance.items()},
            "patterns_analyzed": pattern_insights.get("total_patterns", 0),
        }
        
        print(f"✅ HUD Snapshot: ROI {hud_data['roi_percent']:.1f}%, Patterns: {hud_data['patterns_analyzed']}")
        return hud_data

    def generate_recap_report(self):
        """Generate comprehensive AI Brain Report"""
        print("🧠 GENERATING SYSTEM RECAP - AI BRAIN REPORT")
        print("=" * 70)
        
        # Load all data sources
        perf_df = self.load_performance_data()
        slot_df, day_df = self.load_pnl_data()
        pattern_data = self.load_pattern_intelligence()
        fusion_weights = self.load_fusion_weights()
        strategy_data = self.load_strategy_recommendation()
        
        # Calculate metrics
        roi_summary = self.calculate_roi_summary(day_df)
        slot_performance = self.calculate_slot_performance(slot_df)
        pattern_insights = self.analyze_pattern_insights(pattern_data)
        
        # Generate text report
        self.create_text_recap(roi_summary, slot_performance, pattern_insights, fusion_weights, strategy_data)
        
        print(f"✅ System Recap completed: {self.timestamp}")
    
    def create_text_recap(self, roi_summary, slot_performance, pattern_insights, fusion_weights, strategy_data):
        """Create human-readable text recap"""
        recap_file = self.base_dir / "logs" / "performance" / f"system_recap_{self.timestamp}.txt"
        
        # Ensure directory exists
        recap_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(recap_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("              🤖 AI BRAIN REPORT - SYSTEM RECAP\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("System: PRECISE PREDICTOR - Phase 2.1\n")
            f.write("Slots: FRBD, GZBD, GALI, DSWR\n\n")
            
            # ROI SUMMARY SECTION
            f.write("💰 ROI & PERFORMANCE SUMMARY\n")
            f.write("-" * 40 + "\n")
            if roi_summary:
                f.write(f"Total Stake: ₹{roi_summary.get('total_stake', 0):.0f}\n")
                f.write(f"Total Return: ₹{roi_summary.get('total_return', 0):.0f}\n")
                f.write(f"Total Profit: ₹{roi_summary.get('total_profit', 0):.0f}\n")
                f.write(f"ROI: {roi_summary.get('roi_percent', 0):.1f}%\n")
                f.write(f"Cumulative Profit: ₹{roi_summary.get('cumulative_profit', 0):.0f}\n")
            else:
                f.write("No ROI data available\n")
            f.write("\n")
            
            # SLOT PERFORMANCE SECTION
            f.write("🎯 SLOT PERFORMANCE BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            for slot, data in slot_performance.items():
                f.write(f"{slot}:\n")
                f.write(f"  Stake: ₹{data.get('total_stake', 0):.0f} | ")
                f.write(f"Return: ₹{data.get('total_return', 0):.0f} | ")
                f.write(f"Profit: ₹{data.get('profit', 0):.0f}\n")
                f.write(f"  Hits: Main({data.get('main_hits', 0)}) | ")
                f.write(f"Andar({data.get('andar_hits', 0)}) | ")
                f.write(f"Bahar({data.get('bahar_hits', 0)})\n")
            f.write("\n")
            
            # PATTERN INTELLIGENCE SECTION
            f.write("🔮 PATTERN INTELLIGENCE INSIGHTS\n")
            f.write("-" * 40 + "\n")
            if pattern_insights:
                f.write(f"Patterns Analyzed: {pattern_insights.get('total_patterns', 0)}\n")
                f.write(f"High Performance: {pattern_insights.get('high_performance_patterns', 0)}\n")
                f.write("Top Recommendations:\n")
                for rec in pattern_insights.get('recommendations', [])[:5]:
                    f.write(f"  • {rec.get('action', '')} ({rec.get('priority', '')})\n")
            else:
                f.write("No pattern intelligence data\n")
            f.write("\n")
            
            # FUSION WEIGHTS SECTION
            f.write("⚖️ SMART FUSION WEIGHTS\n")
            f.write("-" * 40 + "\n")
            if fusion_weights:
                for slot, weight in fusion_weights.items():
                    f.write(f"{slot}: {weight:.2f}\n")
            else:
                f.write("No fusion weights data\n")
            f.write("\n")
            
            # STRATEGY RECOMMENDATIONS SECTION
            f.write("🎲 STRATEGY RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            if strategy_data:
                f.write(f"Recommended: {strategy_data.get('recommended_strategy', 'N/A')}\n")
                f.write(f"Confidence: {strategy_data.get('confidence_level', 'N/A')}\n")
                f.write(f"Risk Mode: {strategy_data.get('risk_mode', 'N/A')}\n")
                f.write(f"Reason: {strategy_data.get('reason', 'N/A')}\n")
                f.write("Notes:\n")
                for note in strategy_data.get('notes', []):
                    f.write(f"  • {note}\n")
            else:
                f.write("No strategy recommendations\n")
            f.write("\n")
            
            # SYSTEM LEARNING SNAPSHOT
            f.write("📈 SYSTEM LEARNING SNAPSHOT\n")
            f.write("-" * 40 + "\n")
            f.write("✓ Real-time performance tracking\n")
            f.write("✓ Pattern intelligence integration\n")
            f.write("✓ Dynamic stake allocation\n")
            f.write("✓ Multi-model ensemble predictions\n")
            f.write("✓ Explainability and transparency\n")
            f.write("✓ Continuous learning from 120-day window\n")
            f.write("✓ HUD snapshot for quick monitoring\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF AI BRAIN REPORT\n")
            f.write("=" * 80 + "\n")
        
        # Also print to console
        self.print_console_summary(roi_summary, slot_performance, pattern_insights)
        
        print(f"✅ AI Brain Report saved: {recap_file}")
    
    def print_console_summary(self, roi_summary, slot_performance, pattern_insights):
        """Print summary to console"""
        print("\n" + "=" * 70)
        print("🤖 AI BRAIN REPORT - LEARNING SNAPSHOT")
        print("=" * 70)
        
        if roi_summary:
            print(f"💰 ROI: {roi_summary.get('roi_percent', 0):.1f}% | Profit: ₹{roi_summary.get('total_profit', 0):.0f}")
        
        print(f"🎯 Slot Performance:")
        for slot, data in slot_performance.items():
            profit = data.get('profit', 0)
            profit_str = f"₹{profit:+.0f}" if profit != 0 else "₹0"
            print(f"   {slot}: {profit_str}")
        
        if pattern_insights:
            print(f"🔮 Patterns: {pattern_insights.get('total_patterns', 0)} analyzed, {pattern_insights.get('high_performance_patterns', 0)} high-performance")
        
        print(f"⏰ Generated: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)

def main():
    """Main execution"""
    print("🧠 SYSTEM RECAP ENGINE - AI BRAIN REPORT GENERATOR")
    print("✓ ROI Summary ✓ Slot Performance ✓ Pattern Intelligence")
    print("✓ Stakes & Weights ✓ Strategy Recommendations ✓ HUD Snapshot")
    
    engine = SystemRecapEngine()
    engine.generate_recap_report()
    
    return 0

if __name__ == "__main__":
    exit(main())