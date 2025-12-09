# high_conviction_filter.py
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class HighConvictionFilter:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        
    def load_enhanced_bet_plan(self):
        """Load the latest enhanced bet plan"""
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        enhanced_plans = list(bet_engine_dir.glob("enhanced_bet_plan_*.xlsx"))
        
        if not enhanced_plans:
            print("❌ No enhanced bet plan found")
            return None
            
        latest_plan = max(enhanced_plans, key=lambda x: x.stat().st_mtime)
        df = pd.read_excel(latest_plan, sheet_name='enhanced_bet_plan')
        print(f"✅ Loaded enhanced bet plan: {latest_plan.name}")
        return df, latest_plan
    
    def load_confidence_data(self):
        """Load confidence scores"""
        confidence_file = self.base_dir / "logs" / "performance" / "prediction_confidence.json"
        
        if not confidence_file.exists():
            print("❌ No confidence data found - Run confidence_scorer.py first")
            return None
            
        with open(confidence_file, 'r') as f:
            data = json.load(f)
            
        return data.get('confidence_scores', {})
    
    def apply_conviction_filter(self, enhanced_df, confidence_data):
        """Apply conviction filtering to enhanced bet plan"""
        conviction_data = []
        
        for _, row in enhanced_df.iterrows():
            slot = row['slot']
            slot_confidence = confidence_data.get(slot, {})
            confidence_score = slot_confidence.get('confidence_score', 50)
            
            # Determine conviction level
            conviction_flag = self.get_conviction_flag(confidence_score)
            final_recommendation = self.get_final_recommendation(confidence_score, row)
            
            conviction_row = {
                'slot': slot,
                'original_stake': row['original_stake'],
                'recommended_stake': row['recommended_stake'],
                'stake_change': row['stake_change'],
                'confidence_level': row['confidence_level'],
                'performance_tier': row['performance_tier'],
                'slot_confidence': confidence_score,
                'conviction_flag': conviction_flag,
                'final_recommendation': final_recommendation,
                'risk_adjustment': self.get_risk_adjustment(conviction_flag),
                'betting_priority': self.get_betting_priority(conviction_flag),
                'advisory_notes': row['advisory_notes'],
                'confidence_reasons': ', '.join(slot_confidence.get('reasons', []))
            }
            
            conviction_data.append(conviction_row)
        
        return pd.DataFrame(conviction_data)
    
    def get_conviction_flag(self, confidence_score):
        """Get conviction flag based on confidence score"""
        if confidence_score >= 80:
            return "VERY_HIGH"
        elif confidence_score >= 65:
            return "HIGH"
        elif confidence_score >= 50:
            return "MEDIUM"
        elif confidence_score >= 35:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def get_final_recommendation(self, confidence_score, row):
        """Get final betting recommendation"""
        stake_change = row['stake_change']
        
        if confidence_score >= 75 and stake_change > 0:
            return "PLAY_AGGRESSIVE"
        elif confidence_score >= 60:
            return "PLAY_NORMAL"
        elif confidence_score >= 45:
            return "PLAY_LIGHT"
        elif confidence_score >= 30:
            return "PLAY_CAUTIOUS"
        else:
            return "AVOID_SLOT"
    
    def get_risk_adjustment(self, conviction_flag):
        """Get risk adjustment based on conviction"""
        adjustments = {
            "VERY_HIGH": "INCREASE_RISK",
            "HIGH": "MAINTAIN_RISK", 
            "MEDIUM": "REDUCE_RISK_SLIGHTLY",
            "LOW": "REDUCE_RISK",
            "VERY_LOW": "MINIMAL_RISK"
        }
        return adjustments.get(conviction_flag, "MAINTAIN_RISK")
    
    def get_betting_priority(self, conviction_flag):
        """Get betting priority"""
        priorities = {
            "VERY_HIGH": 1,
            "HIGH": 2,
            "MEDIUM": 3,
            "LOW": 4,
            "VERY_LOW": 5
        }
        return priorities.get(conviction_flag, 3)
    
    def save_high_conviction_plan(self, conviction_df, original_path):
        """Save high conviction bet plan"""
        # Create high conviction filename
        original_name = original_path.stem
        conviction_name = original_name.replace("enhanced_bet_plan", "high_conviction_bet_plan")
        conviction_path = original_path.parent / f"{conviction_name}.xlsx"
        
        # Save conviction plan
        with pd.ExcelWriter(conviction_path, engine='openpyxl') as writer:
            conviction_df.to_excel(writer, sheet_name='high_conviction_plan', index=False)
            
            # Add conviction summary
            summary_data = self.generate_conviction_summary(conviction_df)
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='conviction_summary', index=False)
        
        print(f"💾 High conviction bet plan saved to: {conviction_path}")
        return conviction_path
    
    def generate_conviction_summary(self, conviction_df):
        """Generate conviction summary"""
        summary = []
        
        total_original = conviction_df['original_stake'].sum()
        total_recommended = conviction_df['recommended_stake'].sum()
        total_change = total_recommended - total_original
        
        summary.append({
            'metric': 'TOTAL_STAKE_ADJUSTMENT',
            'original_stake': f"₹{total_original}",
            'recommended_stake': f"₹{total_recommended}",
            'net_change': f"₹{total_change:+.0f}",
            'impact': 'HIGH' if abs(total_change) > 50 else 'MEDIUM'
        })
        
        # High conviction slots
        high_conviction = conviction_df[conviction_df['conviction_flag'].isin(['VERY_HIGH', 'HIGH'])]
        if len(high_conviction) > 0:
            summary.append({
                'metric': 'HIGH_CONVICTION_SLOTS',
                'slots': ', '.join(high_conviction['slot'].tolist()),
                'count': len(high_conviction),
                'avg_confidence': high_conviction['slot_confidence'].mean(),
                'recommendation': 'FOCUS_MAXIMUM'
            })
        
        # Low conviction slots
        low_conviction = conviction_df[conviction_df['conviction_flag'].isin(['LOW', 'VERY_LOW'])]
        if len(low_conviction) > 0:
            summary.append({
                'metric': 'LOW_CONVICTION_SLOTS',
                'slots': ', '.join(low_conviction['slot'].tolist()),
                'count': len(low_conviction),
                'avg_confidence': low_conviction['slot_confidence'].mean(),
                'recommendation': 'REDUCE_EXPOSURE'
            })
        
        # Risk assessment
        if len(high_conviction) >= 3:
            risk_level = "AGGRESSIVE"
        elif len(high_conviction) >= 2:
            risk_level = "MODERATE_HIGH"
        elif len(low_conviction) >= 2:
            risk_level = "CONSERVATIVE"
        else:
            risk_level = "MODERATE"
            
        summary.append({
            'metric': 'OVERALL_RISK_PROFILE',
            'risk_level': risk_level,
            'high_conviction_count': len(high_conviction),
            'low_conviction_count': len(low_conviction)
        })
        
        return summary
    
    def print_conviction_report(self, conviction_df):
        """Print conviction report to console"""
        print("\n" + "="*80)
        print("🎯 HIGH CONVICTION FILTER - CONFIDENCE-BASED BETTING")
        print("="*80)
        
        total_original = conviction_df['original_stake'].sum()
        total_recommended = conviction_df['recommended_stake'].sum()
        total_change = total_recommended - total_original
        
        print(f"\n💰 STAKE OPTIMIZATION SUMMARY:")
        print("-" * 45)
        print(f"   Original: ₹{total_original} → Conviction: ₹{total_recommended}")
        print(f"   Net Change: ₹{total_change:+.0f}")
        
        high_conviction = conviction_df[conviction_df['conviction_flag'].isin(['VERY_HIGH', 'HIGH'])]
        low_conviction = conviction_df[conviction_df['conviction_flag'].isin(['LOW', 'VERY_LOW'])]
        
        print(f"   High Conviction Slots: {len(high_conviction)}")
        print(f"   Low Conviction Slots: {len(low_conviction)}")
        
        print(f"\n🎯 CONVICTION-BASED RECOMMENDATIONS:")
        print("-" * 55)
        
        for _, row in conviction_df.iterrows():
            conviction_icon = "🚀" if row['conviction_flag'] == 'VERY_HIGH' else "✅" if row['conviction_flag'] == 'HIGH' else "⚡" if row['conviction_flag'] == 'MEDIUM' else "⚠️" if row['conviction_flag'] == 'LOW' else "🔴"
            rec_icon = "🎯" if row['final_recommendation'] == 'PLAY_AGGRESSIVE' else "✅" if 'PLAY_NORMAL' in row['final_recommendation'] else "🔶" if 'PLAY_LIGHT' in row['final_recommendation'] else "🚫"
            
            print(f"   {row['slot']}: {conviction_icon} {row['conviction_flag']:10} | {rec_icon} {row['final_recommendation']:15} | ₹{row['recommended_stake']:3}")
        
        print(f"\n💡 KEY INSIGHTS:")
        print("-" * 30)
        if len(high_conviction) > 0:
            print(f"   🎯 Focus on: {', '.join(high_conviction['slot'].tolist())}")
        if len(low_conviction) > 0:
            print(f"   ⚠️  Reduce: {', '.join(low_conviction['slot'].tolist())}")
    
    def run(self):
        """Main execution"""
        print("🎯 HIGH CONVICTION FILTER - Applying confidence-based filtering...")
        
        # Load enhanced bet plan
        enhanced_data, enhanced_path = self.load_enhanced_bet_plan()
        if enhanced_data is None:
            return False
        
        # Load confidence data
        confidence_data = self.load_confidence_data()
        if not confidence_data:
            return False
        
        # Apply conviction filter
        conviction_df = self.apply_conviction_filter(enhanced_data, confidence_data)
        
        # Save and display results
        conviction_path = self.save_high_conviction_plan(conviction_df, enhanced_path)
        self.print_conviction_report(conviction_df)
        
        print(f"\n💡 NEXT STEPS:")
        print("-" * 30)
        print("   1. Review high_conviction_bet_plan_YYYYMMDD.xlsx")
        print("   2. Focus betting on HIGH/Very High conviction slots")
        print("   3. Reduce exposure to LOW conviction slots")
        print("   4. Monitor performance of conviction-based strategy")
        
        return True

def main():
    filter = HighConvictionFilter()
    success = filter.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())