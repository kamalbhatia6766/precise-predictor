# bet_plan_enhancer.py
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class BetPlanEnhancer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        
    def load_latest_bet_plan(self):
        """Find and load latest bet plan - COMPATIBLE WITH ACTUAL STRUCTURE"""
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        bet_plans = list(bet_engine_dir.glob("bet_plan_master_*.xlsx"))

        if not bet_plans:
            print("❌ No bet plan found")
            return None, None
            
        latest_plan = max(bet_plans, key=lambda x: x.stat().st_mtime)
        print(f"✅ Loading: {latest_plan.name}")
        
        try:
            # Read SUMMARY sheet (actual structure)
            df = pd.read_excel(latest_plan, sheet_name="summary")
            print(f"✅ Loaded summary sheet with {len(df)} slots")
            return df, latest_plan
        except Exception as e:
            print(f"❌ Error reading bet plan: {e}")
            return None, None
    
    def load_advisory_data(self):
        """Load all advisory data from analytics scripts"""
        advisory_data = {}
        performance_dir = self.base_dir / "logs" / "performance"
        
        files_to_load = {
            'stakes': 'dynamic_stake_plan.json',
            'fusion': 'adaptive_fusion_plan.json', 
            'patterns': 'pattern_intelligence_summary.json',
            'money': 'money_management_plan.json',
            'weights': 'smart_fusion_weights.json'
        }
        
        for key, filename in files_to_load.items():
            file_path = performance_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        advisory_data[key] = json.load(f)
                    print(f"✅ Loaded {filename}")
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")
            else:
                print(f"⚠️ File not found: {filename}")
                
        return advisory_data
    
    def enhance_bet_plan(self, bet_plan_df, advisory_data):
        """Enhance bet plan with advisory data - COMPATIBLE VERSION"""
        enhanced_data = []
        
        for _, row in bet_plan_df.iterrows():
            slot = row['slot']  # ACTUAL COLUMN NAME (lowercase)
            original_stake = row['total_stake']  # ACTUAL COLUMN NAME
            
            # Get recommended stake from advisory
            recommended_stake = self.get_recommended_stake(slot, advisory_data, original_stake)
            
            enhanced_row = {
                'slot': slot,
                'original_stake': original_stake,
                'recommended_stake': recommended_stake,
                'stake_change': recommended_stake - original_stake,
                'confidence_level': self.get_confidence_level(recommended_stake, original_stake),
                'performance_tier': self.get_performance_tier(slot, advisory_data),
                'pattern_insights': self.get_pattern_insights(advisory_data),
                'risk_recommendation': self.get_risk_recommendation(advisory_data),
                'source_preference': self.get_source_preference(slot, advisory_data),
                'advisory_notes': self.generate_notes(slot, recommended_stake, original_stake, advisory_data)
            }
            
            enhanced_data.append(enhanced_row)
        
        return pd.DataFrame(enhanced_data)
    
    def get_recommended_stake(self, slot, advisory_data, original_stake):
        """Get recommended stake for slot"""
        if 'stakes' in advisory_data and 'stakes' in advisory_data['stakes']:
            return advisory_data['stakes']['stakes'].get(slot, original_stake)
        return original_stake
    
    def get_confidence_level(self, recommended_stake, original_stake):
        """Get confidence level based on stake change"""
        change = recommended_stake - original_stake
        if change >= 30:
            return 'VERY_HIGH'
        elif change >= 15:
            return 'HIGH'
        elif change <= -20:
            return 'LOW'
        elif change <= -10:
            return 'MEDIUM_LOW'
        else:
            return 'MEDIUM'
    
    def get_performance_tier(self, slot, advisory_data):
        """Get performance tier for slot"""
        if 'fusion' in advisory_data and 'slot_optimization' in advisory_data['fusion']:
            slot_data = advisory_data['fusion']['slot_optimization'].get(slot, {})
            return slot_data.get('performance_tier', 'UNKNOWN')
        return 'UNKNOWN'
    
    def get_pattern_insights(self, advisory_data):
        """Get pattern insights"""
        if 'patterns' in advisory_data and 'top_recommendations' in advisory_data['patterns']:
            recs = advisory_data['patterns']['top_recommendations']
            insights = [rec.get('action', '') for rec in recs[:2]]
            return "; ".join(insights) if insights else 'No pattern insights'
        return 'No pattern data'
    
    def get_risk_recommendation(self, advisory_data):
        """Get risk recommendation"""
        if 'money' in advisory_data and 'recommendations' in advisory_data['money']:
            recs = advisory_data['money']['recommendations']
            if recs:
                return recs[0].get('action', 'STANDARD_RISK')
        return 'STANDARD_RISK'
    
    def get_source_preference(self, slot, advisory_data):
        """Get source preference"""
        if 'fusion' in advisory_data and 'slot_optimization' in advisory_data['fusion']:
            slot_data = advisory_data['fusion']['slot_optimization'].get(slot, {})
            return slot_data.get('preferred_source', 'FUSION').upper()
        return 'FUSION'
    
    def generate_notes(self, slot, recommended_stake, original_stake, advisory_data):
        """Generate advisory notes"""
        notes = []
        
        # Stake change note
        stake_change = recommended_stake - original_stake
        if stake_change > 0:
            notes.append(f"STAKE_UP +₹{stake_change}")
        elif stake_change < 0:
            notes.append(f"STAKE_DOWN {stake_change}₹")
        else:
            notes.append("STAKE_HOLD")
        
        # Performance note
        perf_tier = self.get_performance_tier(slot, advisory_data)
        if perf_tier != 'UNKNOWN':
            notes.append(f"PERF_{perf_tier}")
        
        # Source preference note
        source_pref = self.get_source_preference(slot, advisory_data)
        notes.append(f"SOURCE_{source_pref}")
        
        return " | ".join(notes)
    
    def save_enhanced_plan(self, enhanced_df, original_path):
        """Save enhanced bet plan"""
        # Create enhanced filename
        original_name = original_path.stem
        enhanced_name = original_name.replace("bet_plan_master", "enhanced_bet_plan")
        enhanced_path = original_path.parent / f"{enhanced_name}.xlsx"
        
        # Save enhanced plan
        with pd.ExcelWriter(enhanced_path, engine='openpyxl') as writer:
            enhanced_df.to_excel(writer, sheet_name='enhanced_bet_plan', index=False)
            
            # Add advisory summary
            summary_data = self.generate_advisory_summary(enhanced_df)
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='advisory_summary', index=False)
        
        print(f"💾 Enhanced bet plan saved to: {enhanced_path}")
        return enhanced_path
    
    def generate_advisory_summary(self, enhanced_df):
        """Generate advisory summary"""
        summary = []
        
        total_original = enhanced_df['original_stake'].sum()
        total_recommended = enhanced_df['recommended_stake'].sum()
        total_change = total_recommended - total_original
        change_percent = (total_change / total_original) * 100 if total_original > 0 else 0
        
        summary.append({
            'metric': 'TOTAL_STAKE_CHANGE',
            'original': f"₹{total_original}",
            'recommended': f"₹{total_recommended}",
            'change': f"₹{total_change:+.0f}",
            'change_percent': f"{change_percent:+.1f}%",
            'impact': 'HIGH' if abs(total_change) > 50 else 'MEDIUM'
        })
        
        # High confidence slots
        high_conf_slots = enhanced_df[enhanced_df['confidence_level'].isin(['VERY_HIGH', 'HIGH'])]
        if len(high_conf_slots) > 0:
            summary.append({
                'metric': 'HIGH_CONFIDENCE_SLOTS',
                'value': ", ".join(high_conf_slots['slot'].tolist()),
                'count': len(high_conf_slots),
                'impact': 'HIGH'
            })
        
        # Low confidence slots  
        low_conf_slots = enhanced_df[enhanced_df['confidence_level'].isin(['LOW', 'MEDIUM_LOW'])]
        if len(low_conf_slots) > 0:
            summary.append({
                'metric': 'LOW_CONFIDENCE_SLOTS',
                'value': ", ".join(low_conf_slots['slot'].tolist()),
                'count': len(low_conf_slots),
                'impact': 'MEDIUM'
            })
        
        # Risk assessment
        if total_recommended > 250:
            risk_level = "HIGH_RISK"
        elif total_recommended > 180:
            risk_level = "MEDIUM_RISK" 
        else:
            risk_level = "LOW_RISK"
            
        summary.append({
            'metric': 'RISK_ASSESSMENT',
            'value': risk_level,
            'total_stake': f"₹{total_recommended}",
            'impact': 'HIGH'
        })
        
        return summary
    
    def print_enhancement_report(self, enhanced_df):
        """Print enhancement report"""
        print("\n" + "="*80)
        print("🎯 BET PLAN ENHANCER - SMART STAKE OPTIMIZATION")
        print("="*80)
        
        total_original = enhanced_df['original_stake'].sum()
        total_recommended = enhanced_df['recommended_stake'].sum()
        total_change = total_recommended - total_original
        change_percent = (total_change / total_original) * 100 if total_original > 0 else 0
        
        print(f"\n📊 ENHANCEMENT SUMMARY:")
        print("-" * 50)
        print(f"   Total Stake: ₹{total_original} → ₹{total_recommended}")
        print(f"   Net Change: ₹{total_change:+.0f} ({change_percent:+.1f}%)")
        
        high_conf_count = len(enhanced_df[enhanced_df['confidence_level'].isin(['VERY_HIGH', 'HIGH'])])
        low_conf_count = len(enhanced_df[enhanced_df['confidence_level'].isin(['LOW', 'MEDIUM_LOW'])])
        
        print(f"   High Confidence Slots: {high_conf_count}")
        print(f"   Low Confidence Slots: {low_conf_count}")
        
        print(f"\n🎯 SLOT-WISE OPTIMIZATIONS:")
        print("-" * 60)
        for _, row in enhanced_df.iterrows():
            stake_change = row['stake_change']
            change_icon = "🔼" if stake_change > 0 else "🔽" if stake_change < 0 else "➡️"
            confidence_icon = "🚀" if row['confidence_level'] in ['VERY_HIGH', 'HIGH'] else "⚠️" if row['confidence_level'] in ['LOW', 'MEDIUM_LOW'] else "✅"
            
            print(f"   {row['slot']}: {change_icon} ₹{row['recommended_stake']:3} | {confidence_icon} {row['confidence_level']:12} | {row['advisory_notes']}")
    
    def run(self):
        """Main execution"""
        print("🎯 BET PLAN ENHANCER - Applying Smart Analytics to Bet Plan...")
        
        # Load latest bet plan
        bet_plan_data, bet_plan_path = self.load_latest_bet_plan()
        if bet_plan_data is None:
            print("⚠️ Bet plan enhancer: no plan found; skipping enhancement stage")
            return True
        
        # Load advisory data
        advisory_data = self.load_advisory_data()
        if not advisory_data:
            print("⚠️ No advisory data found - skipping enhancement")
            return True
        
        # Enhance bet plan
        enhanced_df = self.enhance_bet_plan(bet_plan_data, advisory_data)
        
        # Save and display results
        enhanced_path = self.save_enhanced_plan(enhanced_df, bet_plan_path)
        self.print_enhancement_report(enhanced_df)
        
        print(f"\n💡 NEXT STEPS:")
        print("-" * 30)
        print("   1. Review enhanced_bet_plan_YYYYMMDD.xlsx")
        print("   2. Compare recommended vs original stakes") 
        print("   3. Use intelligent_daily_runner.py for automated workflow")
        print("   4. Paper test recommendations before live betting")
        
        return True

def main():
    enhancer = BetPlanEnhancer()
    success = enhancer.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
