import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

class SmartFusionWeights:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]

    def _normalize_weights(self, weights):
        """Ensure weights is always a dictionary for downstream callers."""
        if weights in [None, False, True]:
            print("⚠️ Weight payload missing or boolean; normalizing to empty dict")
            return {}
        if isinstance(weights, dict):
            return weights
        try:
            return dict(weights)
        except Exception:
            print("⚠️ Unable to interpret weights structure; using empty fallback")
            return {}
        
    def calculate_optimal_weights(self):
        """Calculate optimal weights based on performance"""
        print("🧠 SMART FUSION WEIGHTS - Calculating optimal weights...")
        
        # Load performance data
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if not pnl_file.exists():
            print("⚠️ No P&L data found; returning empty weights")
            return {}
            
        try:
            day_df = pd.read_excel(pnl_file, sheet_name='day_level')
            day_df['date'] = pd.to_datetime(day_df['date']).dt.date
            
            # Calculate slot-wise performance
            slot_performance = {}
            for slot in self.slots:
                profit_col = f'profit_{slot.lower()}'
                if profit_col in day_df.columns:
                    total_profit = day_df[profit_col].sum()
                    slot_performance[slot] = total_profit
                else:
                    slot_performance[slot] = 0
                    
            # Calculate weights (normalize to 0-1 range)
            min_profit = min(slot_performance.values())
            max_profit = max(slot_performance.values())
            
            weights = {}
            if max_profit != min_profit:
                for slot, profit in slot_performance.items():
                    # Normalize and scale to 0.5-2.0 range
                    normalized = (profit - min_profit) / (max_profit - min_profit)
                    weight = 0.5 + (normalized * 1.5)  # 0.5 to 2.0
                    weights[slot] = round(weight, 2)
            else:
                # Equal weights if all same
                for slot in self.slots:
                    weights[slot] = 1.0
                    
            return weights
            
        except Exception as e:
            print(f"❌ Error calculating weights: {e}")
            return {}

    def generate_recommendations(self, weights):
        """Generate betting recommendations"""
        if not weights:
            print("⚠️ No weights available to generate recommendations")
            return
            
        print("\n🎯 SMART BETTING RECOMMENDATIONS:")
        print("="*50)
        
        # Sort slots by weight (highest first)
        sorted_slots = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for slot, weight in sorted_slots:
            if weight >= 1.5:
                recommendation = "🚀 HIGH CONFIDENCE - Increase stake"
                stake_multiplier = 1.5
            elif weight >= 1.0:
                recommendation = "✅ MEDIUM CONFIDENCE - Normal stake"  
                stake_multiplier = 1.0
            else:
                recommendation = "⚠️ LOW CONFIDENCE - Reduce stake"
                stake_multiplier = 0.7
                
            print(f"   {slot}: Weight {weight:.2f} | {recommendation}")

    def save_weights(self, weights):
        """Save weights to JSON file"""
        if weights is None:
            return
            
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "smart_fusion_weights.json"
        
        with open(output_file, 'w') as f:
            json.dump(weights, f, indent=2)
            
        print(f"\n💾 Weights saved to: {output_file}")
        return output_file

    def run(self):
        """Main execution"""
        weights = self._normalize_weights(self.calculate_optimal_weights())

        if weights is None:
            return False
            
        print("\n" + "="*60)
        print("🧠 SMART FUSION WEIGHTS - OPTIMAL ALLOCATION")
        print("="*60)
        
        print("\n📊 CALCULATED WEIGHTS (Higher = Better Performance):")
        print("-" * 45)
        for slot, weight in weights.items():
            print(f"   {slot}: {weight:.2f}")
            
        self.generate_recommendations(weights)
        self.save_weights(weights)
        
        return True

def main():
    optimizer = SmartFusionWeights()
    success = optimizer.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
