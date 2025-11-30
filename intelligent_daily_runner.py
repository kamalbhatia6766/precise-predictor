# intelligent_daily_runner.py
import subprocess
import sys
from pathlib import Path
import time
import json

class IntelligentDailyRunner:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        
    def run_analytics_phase(self):
        """Run all analytics scripts"""
        print("📊 PHASE 1: Running Smart Analytics...")
        print("-" * 40)
        
        analytics_scripts = [
            "dynamic_stake_allocator.py",
            "pattern_intelligence_engine.py", 
            "adaptive_fusion_2.0.py",
            "money_manager.py", 
            "real_time_performance_dashboard.py"
        ]
        
        for script in analytics_scripts:
            script_path = self.base_dir / script
            if not script_path.exists():
                print(f"⚠️ Script not found: {script}")
                continue
                
            print(f"   Running {script}...")
            result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"   ✅ {script} completed successfully")
            else:
                print(f"   ⚠️ {script} had issues (but continuing)")
                print(f"      Error: {result.stderr[:100]}...")
        
        return True
    
    def run_prediction_phase(self, source, target):
        """Run prediction phase"""
        print(f"\n🎯 PHASE 2: Generating {source.upper()} Predictions...")
        print("-" * 50)
        
        cmd = [
            sys.executable, "precise_daily_runner.py", 
            "--mode", "predict", 
            "--source", source, 
            "--target", target
        ]
        
        print(f"   Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Prediction completed successfully")
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')[-5:]
            for line in lines:
                print(f"      {line}")
            return True
        else:
            print(f"   ❌ Prediction failed")
            print(f"      Error: {result.stderr[:200]}...")
            return False
    
    def run_enhancement_phase(self):
        """Run bet plan enhancement"""
        print(f"\n💰 PHASE 3: Enhancing Bet Plan with Smart Stakes...")
        print("-" * 55)
        
        cmd = [sys.executable, "bet_plan_enhancer.py"]
        
        print(f"   Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Bet plan enhancement completed successfully")
            # Show key enhancement summary
            lines = result.stdout.strip().split('\n')
            enhancement_lines = [line for line in lines if '₹' in line or '→' in line or 'STAKE_' in line]
            for line in enhancement_lines[-10:]:
                print(f"      {line}")
            return True
        else:
            print(f"   ❌ Enhancement failed")
            print(f"      Error: {result.stderr[:200]}...")
            return False

    def _get_confidence_level(self, score):
        """Map confidence score to level (same logic as confidence_scorer.py)"""
        if score >= 80:
            return "VERY_HIGH"
        elif score >= 65:
            return "HIGH" 
        elif score >= 50:
            return "MEDIUM"
        elif score >= 35:
            return "LOW"
        else:
            return "VERY_LOW"

    def run_confidence_phase(self):
        """Run confidence scoring and conviction filtering"""
        print(f"\n🎯 PHASE 4: Running Confidence & Conviction Analysis...")
        print("-" * 55)
        
        # Run confidence scorer
        print("   Running confidence_scorer.py...")
        result1 = subprocess.run([sys.executable, "confidence_scorer.py"], capture_output=True, text=True)
        
        if result1.returncode == 0:
            print("   ✅ Confidence scoring completed successfully")
        else:
            print(f"   ⚠️ Confidence scoring had issues (but continuing)")
            print(f"      Error: {result1.stderr[:100]}...")
        
        # Run high conviction filter
        print("   Running high_conviction_filter.py...")
        result2 = subprocess.run([sys.executable, "high_conviction_filter.py"], capture_output=True, text=True)
        
        if result2.returncode == 0:
            print("   ✅ High conviction filtering completed successfully")
        else:
            print(f"   ⚠️ Conviction filtering had issues (but continuing)")
            print(f"      Error: {result2.stderr[:100]}...")
        
        # Display compact conviction summary
        print("\n   📊 CONFIDENCE & CONVICTION SUMMARY:")
        print("   " + "-" * 40)
        
        # Try to read confidence data
        confidence_file = self.base_dir / "logs" / "performance" / "prediction_confidence.json"
        if confidence_file.exists():
            try:
                with open(confidence_file, 'r') as f:
                    confidence_data = json.load(f)
                
                # Correct JSON structure: data["confidence_scores"][slot]
                confidence_scores = confidence_data.get("confidence_scores", {})
                
                slots = ['FRBD', 'GZBD', 'GALI', 'DSWR']
                for slot in slots:
                    if slot in confidence_scores:
                        slot_data = confidence_scores[slot]
                        score = slot_data.get('confidence_score', 0)
                        level = self._get_confidence_level(score)
                        print(f"   {slot}: {score:.0f}/100 ({level})")
                    else:
                        print(f"   {slot}: No data")
            except Exception as e:
                print(f"   ⚠️ Could not read confidence data: {e}")
        else:
            print("   ⚠️ Confidence data not available")
        
        # Try to find latest high conviction bet plan
        bet_plan_dir = self.base_dir / "predictions" / "bet_engine"
        high_conviction_files = list(bet_plan_dir.glob("high_conviction_bet_plan_*.xlsx"))
        if high_conviction_files:
            latest_file = max(high_conviction_files, key=lambda x: x.stat().st_mtime)
            print(f"   💾 High conviction plan: {latest_file.name}")
        
        return True  # Continue even if confidence phase has issues
    
    def run_evaluation_phase(self):
        """Run evaluation phase"""
        print(f"\n📈 PHASE 5: Running Evaluation & P&L Update...")
        print("-" * 45)
        
        cmd = [sys.executable, "precise_daily_runner.py", "--mode", "evaluate"]
        
        print(f"   Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ Evaluation completed successfully")
            # Show P&L summary
            lines = result.stdout.strip().split('\n')
            pnl_lines = [line for line in lines if '₹' in line or 'profit' in line.lower() or 'CUMULATIVE' in line]
            for line in pnl_lines[-8:]:
                print(f"      {line}")
            return True
        else:
            print(f"   ⚠️ Evaluation had issues (but workflow continues)")
            return True  # Continue anyway

    def run_strategy_phase(self):
        """Run strategy recommendation analysis"""
        print(f"\n🧠 PHASE 6: Running Strategy Recommendation Engine...")
        print("-" * 55)
        
        # First run prediction validator to ensure we have fresh data
        print("   Running prediction_validator.py...")
        result1 = subprocess.run(
            [sys.executable, "prediction_validator.py", "--days", "10"], 
            capture_output=True, text=True
        )
        
        if result1.returncode == 0:
            print("   ✅ Prediction validation completed successfully")
        else:
            print(f"   ⚠️ Prediction validation had issues (but continuing)")
            print(f"      Error: {result1.stderr[:100]}...")
        
        # Run strategy recommendation engine
        print("   Running strategy_recommendation_engine.py...")
        result2 = subprocess.run(
            [sys.executable, "strategy_recommendation_engine.py", "--days", "10"], 
            capture_output=True, text=True
        )
        
        if result2.returncode == 0:
            print("   ✅ Strategy recommendation completed successfully")
        else:
            print(f"   ⚠️ Strategy recommendation had issues (but continuing)")
            print(f"      Error: {result2.stderr[:100]}...")
        
        # Display compact strategy recommendation
        print("\n   🧠 STRATEGY RECO SUMMARY (last 10 days)")
        print("   " + "-" * 40)
        
        # Try to read strategy recommendation data
        strategy_file = self.base_dir / "logs" / "performance" / "strategy_recommendation.json"
        if strategy_file.exists():
            try:
                with open(strategy_file, 'r') as f:
                    strategy_data = json.load(f)
                
                recommended = strategy_data.get('recommended_strategy', 'NONE')
                confidence = strategy_data.get('confidence_level', 'LOW')
                risk_mode = strategy_data.get('risk_mode', 'SUPER_DEFENSIVE')
                
                if recommended != 'NONE':
                    strategies = strategy_data.get('strategies', {})
                    best_stats = strategies.get(recommended, {})
                    
                    stake = best_stats.get('total_stake', 0)
                    profit = best_stats.get('total_profit', 0)
                    roi = best_stats.get('roi_percent', 0)
                    
                    print(f"   Recommended : {recommended}")
                    print(f"   Confidence  : {confidence}")
                    print(f"   Risk mode   : {risk_mode}")
                    print(f"   Stats       : stake=₹{stake:,.0f}, profit=₹{profit:+,.0f}, ROI={roi:+.1f}%")
                else:
                    print("   ⚠️ No strategy recommendation available")
                    
            except Exception as e:
                print(f"   ⚠️ Could not read strategy data: {e}")
        else:
            print("   ⚠️ Strategy recommendation data not available")
        
        return True  # Continue even if strategy phase has issues
    
    def run_complete_workflow(self, source='fusion', target='tomorrow'):
        """Run complete intelligent workflow"""
        print("\n" + "="*70)
        print("🚀 INTELLIGENT DAILY RUNNER - COMPLETE WORKFLOW")
        print("="*70)
        print(f"   Source: {source.upper()}")
        print(f"   Target: {target.upper()}")
        print(f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        phases = [
            ("Smart Analytics", self.run_analytics_phase),
            ("Prediction Generation", lambda: self.run_prediction_phase(source, target)),
            ("Bet Plan Enhancement", self.run_enhancement_phase),
            ("Confidence & Conviction", self.run_confidence_phase),
            ("Evaluation & P&L", self.run_evaluation_phase),
            ("Strategy Recommendation", self.run_strategy_phase)  # NEW PHASE
        ]
        
        start_time = time.time()
        all_success = True
        
        for phase_name, phase_func in phases:
            phase_start = time.time()
            
            try:
                success = phase_func()
                phase_time = time.time() - phase_start
                
                if success:
                    print(f"   ✅ {phase_name}: SUCCESS ({phase_time:.1f}s)")
                else:
                    print(f"   ❌ {phase_name}: FAILED ({phase_time:.1f}s)")
                    all_success = False
                    # Decide whether to continue on failure
                    if phase_name == "Prediction Generation":
                        print("   🛑 Stopping workflow - Prediction is critical")
                        break
                    else:
                        print("   ⚠️ Continuing workflow despite failure")
                        
            except Exception as e:
                print(f"   💥 {phase_name}: CRASHED - {e}")
                all_success = False
                print("   ⚠️ Continuing workflow despite crash")
            
            print()
        
        total_time = time.time() - start_time
        
        print("="*70)
        if all_success:
            print("🎉 INTELLIGENT WORKFLOW COMPLETED SUCCESSFULLY!")
        else:
            print("⚠️ INTELLIGENT WORKFLOW COMPLETED WITH SOME ISSUES")
        
        print(f"⏱️  Total Time: {total_time:.1f} seconds")
        print("\n💡 NEXT ACTIONS:")
        print("   1. Check enhanced_bet_plan_YYYYMMDD.xlsx for stake recommendations")
        print("   2. Review pattern insights and confidence levels") 
        print("   3. Paper test recommendations before live implementation")
        print("   4. Monitor P&L performance in bet_pnl_history.xlsx")
        print("   5. Check strategy_recommendation.json for meta-strategy advice")
        
        return all_success

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Intelligent Daily Runner - Complete Workflow')
    parser.add_argument('--source', default='fusion', choices=['fusion', 'scr9'], 
                       help='Prediction source (fusion or scr9)')
    parser.add_argument('--target', default='tomorrow', 
                       help='Target date (today, tomorrow, or specific date)')
    
    args = parser.parse_args()
    
    runner = IntelligentDailyRunner()
    success = runner.run_complete_workflow(args.source, args.target)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())