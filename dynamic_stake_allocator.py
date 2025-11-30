# dynamic_stake_allocator.py - UPDATED
# dynamic_stake_allocator.py - REALITY-DRIVEN STAKE ALLOCATION
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths
import quant_data_core

class DynamicStakeAllocator:
    def __init__(self):
        self.base_dir = quant_paths.get_project_root()
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def load_base_bet_plan(self):
        """Load the latest bet plan to get base stakes"""
        latest_bet_plan = quant_paths.find_latest_bet_plan_master()
        target_date = quant_paths.parse_date_from_filename(latest_bet_plan.stem) if latest_bet_plan else None
        
        if not latest_bet_plan:
            print("❌ No bet plan files found")
            return None, None
        
        try:
            bets_df = pd.read_excel(latest_bet_plan, sheet_name='bets')
            print(f"✅ Loaded base bet plan: {latest_bet_plan.name}")
            return bets_df, target_date
        except Exception as e:
            print(f"❌ Error loading bet plan: {e}")
            return None, None
    
    def calculate_base_stakes(self, bets_df):
        """Calculate base stakes from bet plan"""
        base_slot_stakes = {}
        
        for slot in self.slots:
            slot_bets = bets_df[bets_df['slot'] == slot]
            if not slot_bets.empty:
                total_stake = slot_bets['stake'].sum()
                base_slot_stakes[slot] = total_stake
            else:
                base_slot_stakes[slot] = 0
        
        base_daily_stake = sum(base_slot_stakes.values())
        return base_slot_stakes, base_daily_stake
    
    def load_reality_performance(self):
        """Load reality performance data from quant_reality_pnl.json"""
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.json"
        
        if not pnl_file.exists():
            print("❌ No quant_reality_pnl.json found")
            return None
        
        try:
            with open(pnl_file, 'r') as f:
                performance_data = json.load(f)
            print("✅ Loaded reality performance data")
            return performance_data
        except Exception as e:
            print(f"❌ Error loading performance data: {e}")
            return None
    
    def calculate_slot_rois(self, performance_data):
        """Calculate ROI for each slot from performance data"""
        slot_rois = {}
        
        if 'by_slot' in performance_data:
            for slot_data in performance_data['by_slot']:
                slot = slot_data['slot']
                roi_pct = slot_data.get('roi_pct', 0)
                slot_rois[slot] = roi_pct
        
        # Ensure all slots have ROI data
        for slot in self.slots:
            if slot not in slot_rois:
                slot_rois[slot] = 0
        
        return slot_rois
    
    def apply_reality_overlay(self, base_slot_stakes, performance_data):
        """Apply reality-based stake adjustments"""
        print("🎯 Applying reality-based stake overlay...")
        
        # Get overall ROI
        overall_roi = performance_data.get('overall', {}).get('overall_roi', 0)
        slot_rois = self.calculate_slot_rois(performance_data)
        
        final_slot_stakes = {}
        
        # Define slot ROI multiplier policy
        for slot, base_stake in base_slot_stakes.items():
            slot_roi = slot_rois.get(slot, 0)
            
            # Clamp ROI between -50 and +100
            clamped_roi = max(-50, min(100, slot_roi))
            
            # Slot-specific multiplier based on ROI
            if clamped_roi > 40:
                slot_mult = 1.5
            elif clamped_roi > 20:
                slot_mult = 1.3
            elif clamped_roi > 5:
                slot_mult = 1.1
            elif clamped_roi < -20:
                slot_mult = 0.7
            elif clamped_roi < -5:
                slot_mult = 0.9
            else:
                slot_mult = 1.0
            
            # Global multiplier based on overall ROI
            if overall_roi > 30:
                global_mult = 1.1
            elif overall_roi < -10:
                global_mult = 0.9
            else:
                global_mult = 1.0
            
            # Calculate final stake
            final_stake = base_stake * slot_mult * global_mult
            
            # Round to nearest 5 rupees, minimum 10 if base was >= 10
            if base_stake >= 10:
                final_stake = max(10, round(final_stake / 5) * 5)
            else:
                final_stake = round(final_stake)
            
            final_slot_stakes[slot] = final_stake
        
        return final_slot_stakes, overall_roi, slot_rois
    
    def generate_stake_plan(self, base_slot_stakes, final_slot_stakes, target_date, overall_roi, slot_rois):
        """Generate the dynamic stake plan JSON"""
        total_daily_stake = sum(final_slot_stakes.values())
        
        stake_plan = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.strftime("%Y-%m-%d") if target_date else "UNKNOWN",
            "base_slot_stakes": base_slot_stakes,
            "slot_stakes": final_slot_stakes,
            "total_daily_stake": total_daily_stake,
            "overall_roi": overall_roi,
            "slot_rois": slot_rois,
            "logic_version": "v1_reality_simple",
            "central_pnl_source": "quant_reality_pnl.json"
        }
        
        return stake_plan
    
    def save_stake_plan(self, stake_plan):
        """Save dynamic stake plan to JSON"""
        output_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stake_plan, f, indent=2)
        
        print(f"💾 Dynamic stake plan saved: {output_file}")
        return output_file
    
    def print_console_summary(self, stake_plan):
        """Print console summary"""
        print("\n" + "="*60)
        print("💰 DYNAMIC STAKE ALLOCATOR – REALITY LINKED")
        print("="*60)
        
        base_stakes = stake_plan['base_slot_stakes']
        final_stakes = stake_plan['slot_stakes']
        overall_roi = stake_plan['overall_roi']
        
        base_total = sum(base_stakes.values())
        final_total = stake_plan['total_daily_stake']
        
        print(f"📅 Target Date: {stake_plan['target_date']}")
        print(f"📊 Base Stakes: {', '.join(f'{slot}=₹{amt}' for slot, amt in base_stakes.items())} (Total=₹{base_total})")
        print(f"🎯 Final Stakes: {', '.join(f'{slot}=₹{amt}' for slot, amt in final_stakes.items())} (Total=₹{final_total})")
        print(f"📈 Overall ROI (window): {overall_roi:+.1f}%")
        
        # Show slot ROI breakdown
        print(f"\n🎯 Slot ROI Breakdown:")
        for slot, roi in stake_plan['slot_rois'].items():
            base = base_stakes[slot]
            final = final_stakes[slot]
            change_pct = ((final - base) / base * 100) if base > 0 else 0
            trend = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "🟡"
            print(f"   {trend} {slot}: ROI={roi:+.1f}% → Stake: ₹{base}→₹{final} ({change_pct:+.1f}%)")
        
        print("="*60)
    
    def run_allocation(self):
        """Run complete dynamic stake allocation"""
        print("🚀 DYNAMIC STAKE ALLOCATOR - REALITY-DRIVEN")
        print("="*50)
        
        # Step 1: Load base bet plan
        bets_df, target_date = self.load_base_bet_plan()
        if bets_df is None:
            return False
        
        # Step 2: Calculate base stakes
        base_slot_stakes, base_daily_stake = self.calculate_base_stakes(bets_df)
        
        # Step 3: Load reality performance
        performance_data = self.load_reality_performance()
        if performance_data is None:
            return False
        
        # Step 4: Apply reality overlay
        final_slot_stakes, overall_roi, slot_rois = self.apply_reality_overlay(base_slot_stakes, performance_data)
        
        # Step 5: Generate stake plan
        stake_plan = self.generate_stake_plan(base_slot_stakes, final_slot_stakes, target_date, overall_roi, slot_rois)
        
        # Step 6: Save stake plan
        self.save_stake_plan(stake_plan)
        
        # Step 7: Print summary
        self.print_console_summary(stake_plan)
        
        print("✅ Dynamic stake allocation completed!")
        return True

def main():
    allocator = DynamicStakeAllocator()
    success = allocator.run_allocation()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())