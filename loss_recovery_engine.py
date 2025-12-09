# loss_recovery_engine.py - UPDATED
# loss_recovery_engine.py - REALITY-DRIVEN RISK ZONE MANAGEMENT
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths

class LossRecoveryEngine:
    def __init__(self):
        self.base_dir = quant_paths.get_project_root()
        
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
    
    def calculate_recent_performance(self, performance_data, days=7):
        """Calculate recent performance metrics"""
        try:
            # Try to use quant_pnl_summary if available
            try:
                import quant_pnl_summary
                current_perf = quant_pnl_summary.get_current_performance_summary()
                
                rolling_roi = current_perf.get('overall', {}).get('overall_roi', 0)
                recent_7d_roi = current_perf.get('recent_7d', {}).get('roi', 0)
                days_processed = current_perf.get('overall', {}).get('days_processed', 0)
                
                print("✅ Used quant_pnl_summary for performance data")
                return rolling_roi, recent_7d_roi, days_processed
                
            except ImportError:
                # Fallback: calculate from raw data
                print("⚠️  quant_pnl_summary not available, using direct calculation")
                
                overall_data = performance_data.get('overall', {})
                rolling_roi = overall_data.get('overall_roi', 0)
                days_processed = overall_data.get('days_processed', 0)
                
                # Simple 7-day ROI calculation (placeholder - would need daily data)
                recent_7d_roi = rolling_roi * 0.8  # Approximate
                
                return rolling_roi, recent_7d_roi, days_processed
                
        except Exception as e:
            print(f"⚠️  Error calculating recent performance: {e}")
            return 0, 0, 0
    
    def determine_risk_zone(self, rolling_roi, recent_7d_roi):
        """Determine risk zone and mode based on performance"""
        print("🎯 Determining risk zone from performance...")
        
        # Zone determination logic
        if rolling_roi > 20 and recent_7d_roi > 10:
            zone = "GREEN"
            risk_mode = "AGGRESSIVE_BASE"
        elif rolling_roi > -5:
            zone = "AMBER" 
            risk_mode = "BASE"
        elif rolling_roi > -20:
            zone = "ORANGE"
            risk_mode = "DEFENSIVE"
        else:
            zone = "RED"
            risk_mode = "CAP_PROTECT"
        
        # Check for drawdown flag (simplified)
        drawdown_flag = (recent_7d_roi < -10) or (rolling_roi < -15)
        
        return zone, risk_mode, drawdown_flag
    
    def generate_recovery_plan(self, zone, risk_mode, rolling_roi, recent_7d_roi, days_processed, drawdown_flag):
        """Generate loss recovery plan JSON"""
        recovery_plan = {
            "timestamp": datetime.now().isoformat(),
            "zone": zone,
            "current_zone": zone,
            "recommended_risk_mode": risk_mode,
            "rolling_roi": round(rolling_roi, 2),
            "last_7d_roi": round(recent_7d_roi, 2),
            "days_processed": days_processed,
            "drawdown_flag": drawdown_flag,
            "logic_version": "v1_simple_zone"
        }
        
        return recovery_plan
    
    def save_recovery_plan(self, recovery_plan):
        """Save loss recovery plan to JSON"""
        output_file = quant_paths.get_performance_logs_dir() / "loss_recovery_plan.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(recovery_plan, f, indent=2)
        
        print(f"💾 Loss recovery plan saved: {output_file}")
        return output_file
    
    def print_console_summary(self, recovery_plan):
        """Print console summary"""
        print("\n" + "="*50)
        print("🛡️  LOSS RECOVERY ENGINE – REALITY LINKED")
        print("="*50)
        
        zone = recovery_plan['zone']
        risk_mode = recovery_plan['recommended_risk_mode']
        rolling_roi = recovery_plan['rolling_roi']
        last_7d_roi = recovery_plan['last_7d_roi']
        drawdown = recovery_plan['drawdown_flag']
        
        zone_emoji = {
            "GREEN": "🟢",
            "AMBER": "🟡", 
            "ORANGE": "🟠",
            "RED": "🔴"
        }.get(zone, "⚪")
        
        print(f"{zone_emoji} Zone: {zone}, Risk Mode: {risk_mode}")
        print(f"📊 Rolling ROI: {rolling_roi:+.1f}%, 7d ROI: {last_7d_roi:+.1f}%")
        print(f"📅 Days Processed: {recovery_plan['days_processed']}")
        print(f"⚠️  Drawdown Flag: {'ACTIVE' if drawdown else 'Inactive'}")
        
        # Risk guidance
        risk_guidance = {
            "AGGRESSIVE_BASE": "Full betting with pattern boosts",
            "BASE": "Standard betting strategy",
            "DEFENSIVE": "Reduced stakes, focus on high-confidence picks", 
            "CAP_PROTECT": "Minimum stakes, capital protection mode"
        }.get(risk_mode, "Standard strategy")
        
        print(f"💡 Guidance: {risk_guidance}")
        print("="*50)
    
    def run_recovery_analysis(self):
        """Run complete loss recovery analysis"""
        print("🚀 LOSS RECOVERY ENGINE - REALITY-DRIVEN")
        print("="*50)
        
        # Step 1: Load reality performance
        performance_data = self.load_reality_performance()
        if performance_data is None:
            return False
        
        # Step 2: Calculate recent performance
        rolling_roi, recent_7d_roi, days_processed = self.calculate_recent_performance(performance_data)
        
        # Step 3: Determine risk zone
        zone, risk_mode, drawdown_flag = self.determine_risk_zone(rolling_roi, recent_7d_roi)
        
        # Step 4: Generate recovery plan
        recovery_plan = self.generate_recovery_plan(zone, risk_mode, rolling_roi, recent_7d_roi, days_processed, drawdown_flag)
        
        # Step 5: Save recovery plan
        self.save_recovery_plan(recovery_plan)
        
        # Step 6: Print summary
        self.print_console_summary(recovery_plan)
        
        print("✅ Loss recovery analysis completed!")
        return True

def main():
    recovery_engine = LossRecoveryEngine()
    success = recovery_engine.run_recovery_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())