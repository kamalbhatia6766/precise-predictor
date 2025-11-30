# adaptive_fusion_2.0.py (FIXED VERSION)
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

class AdaptiveFusion2:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.sources = ['fusion', 'scr9']
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def load_performance_data(self):
        """Load all performance data"""
        data = {}
        
        # Load backtest results
        backtest_file = self.base_dir / "logs" / "performance" / "auto_backtest_results.xlsx"
        if backtest_file.exists():
            data['backtest'] = pd.read_excel(backtest_file)
            data['backtest']['date'] = pd.to_datetime(data['backtest']['date']).dt.date
        
        # Load P&L history
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if pnl_file.exists():
            data['pnl'] = pd.read_excel(pnl_file, sheet_name='day_level')
            data['pnl']['date'] = pd.to_datetime(data['pnl']['date']).dt.date
            
        return data
    
    def calculate_source_weights(self, data):
        """Calculate optimal source weights"""
        source_weights = {}
        
        if 'backtest' not in data:
            print("❌ No backtest data found")
            return source_weights
            
        backtest_df = data['backtest']
        
        for source in self.sources:
            source_data = backtest_df[backtest_df['source'] == source]
            
            if len(source_data) > 0:
                total_profit = float(source_data['profit_total'].sum())  # Convert to float
                total_stake = float(source_data['stake_total'].sum())
                roi = float((total_profit / total_stake) * 100) if total_stake > 0 else 0.0
                avg_profit = float(source_data['profit_total'].mean())
                
                # Calculate weight based on performance metrics
                base_weight = 1.0
                
                # ROI bonus
                if roi > 200:
                    base_weight += 0.5
                elif roi > 150:
                    base_weight += 0.3
                elif roi > 100:
                    base_weight += 0.1
                    
                # Consistency bonus (low variance)
                profit_std = float(source_data['profit_total'].std())
                if profit_std < 50:
                    base_weight += 0.2
                    
                source_weights[source] = {
                    'weight': round(base_weight, 2),
                    'roi': roi,
                    'total_profit': total_profit,
                    'avg_profit': avg_profit,
                    'consistency': 'HIGH' if profit_std < 50 else 'MEDIUM' if profit_std < 100 else 'LOW'
                }
        
        return source_weights
    
    def calculate_slot_source_optimization(self, data):
        """Calculate slot-specific source optimization"""
        slot_optimization = {}
        
        if 'pnl' not in data:
            return slot_optimization
            
        pnl_df = data['pnl']
        
        for slot in self.slots:
            profit_col = f'profit_{slot.lower()}'
            if profit_col in pnl_df.columns:
                slot_profit = float(pnl_df[profit_col].sum())  # Convert to float
                slot_optimization[slot] = {
                    'total_profit': slot_profit,
                    'preferred_source': self.determine_preferred_source(slot, data),
                    'performance_tier': 'HIGH' if slot_profit > 1000 else 'MEDIUM' if slot_profit > 0 else 'LOW'
                }
        
        return slot_optimization
    
    def determine_preferred_source(self, slot, data):
        """Determine preferred source for each slot"""
        # This would require more detailed slot-source performance data
        # For now, use overall source weights
        if slot in ['FRBD', 'DSWR']:
            return 'fusion'
        elif slot in ['GALI', 'GZBD']:
            return 'scr9'
        else:
            return 'fusion'
    
    def generate_adaptive_weights(self, source_weights, slot_optimization):
        """Generate adaptive weight recommendations"""
        recommendations = []
        
        # Source-level recommendations
        if source_weights:
            best_source = max(source_weights.items(), key=lambda x: x[1]['weight'])
            worst_source = min(source_weights.items(), key=lambda x: x[1]['weight'])
            
            recommendations.append({
                'type': 'SOURCE_OPTIMIZATION',
                'priority': 'HIGH',
                'action': f"Prioritize {best_source[0]} source",
                'reason': f"ROI: {best_source[1]['roi']:.1f}% vs {worst_source[1]['roi']:.1f}%",
                'weight_adjustment': f"Increase {best_source[0]} weight by 0.3"
            })
        
        # Slot-level recommendations
        high_perf_slots = [s for s, data in slot_optimization.items() if data['performance_tier'] == 'HIGH']
        for slot in high_perf_slots:
            recommendations.append({
                'type': 'SLOT_FOCUS',
                'priority': 'HIGH', 
                'action': f"Maximize focus on {slot}",
                'reason': f"Total profit: ₹{slot_optimization[slot]['total_profit']:.0f}",
                'weight_adjustment': f"Double stake in {slot}"
            })
        
        return recommendations
    
    def save_adaptive_plan(self, source_weights, slot_optimization, recommendations):
        """Save adaptive fusion plan"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create adaptive plan
        adaptive_plan = {
            'timestamp': datetime.now().isoformat(),
            'source_weights': source_weights,
            'slot_optimization': slot_optimization,
            'overall_recommendations': recommendations,
            'optimal_configuration': {
                'primary_source': max(source_weights.items(), key=lambda x: x[1]['weight'])[0] if source_weights else 'fusion',
                'high_focus_slots': [s for s, data in slot_optimization.items() if data['performance_tier'] == 'HIGH'],
                'risk_adjustment': 'AGGRESSIVE' if len([s for s, data in slot_optimization.items() if data['performance_tier'] == 'HIGH']) >= 2 else 'MODERATE'
            }
        }
        
        plan_file = output_dir / "adaptive_fusion_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(adaptive_plan, f, indent=2, default=str)  # Added default=str for serialization
            
        # Save Excel report
        excel_file = output_dir / "adaptive_fusion_report.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            if source_weights:
                source_df = pd.DataFrame([{**{'source': k}, **v} for k, v in source_weights.items()])
                source_df.to_excel(writer, sheet_name='source_weights', index=False)
            
            if slot_optimization:
                slot_df = pd.DataFrame([{**{'slot': k}, **v} for k, v in slot_optimization.items()])
                slot_df.to_excel(writer, sheet_name='slot_optimization', index=False)
            
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                rec_df.to_excel(writer, sheet_name='recommendations', index=False)
        
        print(f"💾 Adaptive fusion plan saved to: {plan_file}")
        print(f"💾 Detailed report saved to: {excel_file}")
    
    def print_console_report(self, source_weights, slot_optimization, recommendations):
        """Print console report"""
        print("\n" + "="*80)
        print("🔄 ADAPTIVE FUSION 2.0 - REAL-TIME OPTIMIZATION")
        print("="*80)
        
        print(f"\n📊 SOURCE PERFORMANCE ANALYSIS:")
        print("-" * 50)
        for source, metrics in source_weights.items():
            print(f"   {source.upper():6}: Weight {metrics['weight']:.2f} | ROI {metrics['roi']:6.1f}% | Profit ₹{metrics['total_profit']:+.0f}")
        
        print(f"\n🎯 SLOT OPTIMIZATION:")
        print("-" * 35)
        for slot, metrics in slot_optimization.items():
            print(f"   {slot}: ₹{metrics['total_profit']:+.0f} profit | {metrics['performance_tier']:6} | Source: {metrics['preferred_source']}")
        
        print(f"\n💡 ADAPTIVE RECOMMENDATIONS:")
        print("-" * 45)
        high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
        for rec in high_priority:
            print(f"   🚀 {rec['action']}")
            print(f"      💡 {rec['reason']}")
    
    def run(self):
        """Main execution"""
        print("🔄 ADAPTIVE FUSION 2.0 - Calculating real-time optimizations...")
        
        # Load data
        data = self.load_performance_data()
        
        if 'backtest' not in data:
            print("❌ No performance data found")
            return False
            
        # Calculate optimizations
        source_weights = self.calculate_source_weights(data)
        slot_optimization = self.calculate_slot_source_optimization(data)
        recommendations = self.generate_adaptive_weights(source_weights, slot_optimization)
        
        # Output results
        self.print_console_report(source_weights, slot_optimization, recommendations)
        self.save_adaptive_plan(source_weights, slot_optimization, recommendations)
        
        return True

def main():
    fusion = AdaptiveFusion2()
    success = fusion.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())