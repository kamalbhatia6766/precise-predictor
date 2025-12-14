# reality_check_engine.py
"""
REALITY CHECK ENGINE - Strategy Performance vs Actual Results Analyzer

PURPOSE:
Compare BASE vs DYNAMIC vs FINAL strategies against actual historical results
to validate meta-strategy recommendations.

USAGE:
py -3.12 reality_check_engine.py --days 10
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys


MIN_BASE_ROI_FOR_OVERLAY = 0.10  # Require at least +10% ROI for overlays
HIGH_CONFIDENCE_LEVELS = {"MEDIUM", "HIGH"}


def is_overlay_strategy(strategy_name: str) -> bool:
    """Identify overlay strategies that sit on top of BASE."""
    if not strategy_name:
        return False

    strategy_name = strategy_name.upper()
    overlay_strategies = {
        "STRAT_S40_BOOST",
        # Future overlays can be added here
    }
    return strategy_name in overlay_strategies

class RealityCheckEngine:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.results = {
            'base': {'stake': 0, 'return': 0, 'profit': 0},
            'dynamic': {'stake': 0, 'return': 0, 'profit': 0},
            'final': {'stake': 0, 'return': 0, 'profit': 0}
        }
        self.daily_data = []
        self.heatmap_data = []
        
    def load_pnl_history(self, days=10):
        """Load P&L history from Excel file"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        
        if not pnl_file.exists():
            print("❌ P&L history file not found")
            return None
            
        try:
            df = pd.read_excel(pnl_file, sheet_name='day_level')
            # Get last N days
            df = df.tail(days)
            return df
        except Exception as e:
            print(f"❌ Error loading P&L history: {e}")
            return None
    
    def load_dynamic_stakes(self):
        """Load dynamic stake plan"""
        stake_file = self.base_dir / "logs" / "performance" / "dynamic_stake_plan.json"
        
        if not stake_file.exists():
            print("⚠️ Dynamic stake plan not found")
            return None
            
        try:
            with open(stake_file, 'r') as f:
                data = json.load(f)
            
            stakes = data.get('stakes', {})
            dynamic_total_stake = sum(stakes.values())
            
            print(f"✅ Loaded dynamic stakes: {stakes}")
            return dynamic_total_stake
            
        except Exception as e:
            print(f"⚠️ Error loading dynamic stakes: {e}")
            return None
    
    def load_strategy_recommendation(self):
        """Load strategy recommendation data"""
        strategy_file = self.base_dir / "logs" / "performance" / "strategy_recommendation.json"
        
        if not strategy_file.exists():
            print("⚠️ Strategy recommendation not found")
            return {}

        try:
            with open(strategy_file, 'r') as f:
                data = json.load(f)

            recommended_strategy = data.get('recommended_strategy') or data.get('strategy') or 'NONE'
            confidence_level = data.get('confidence_level') or data.get('confidence') or 'LOW'
            risk_mode = data.get('risk_mode') or data.get('risk') or 'DEFENSIVE'

            metrics = {}
            try:
                metrics = data.get('metrics', {}) or {}
            except Exception:
                metrics = {}

            return {
                'recommended_strategy': recommended_strategy,
                'confidence_level': confidence_level,
                'risk_mode': risk_mode,
                'window_days': data.get('window_days', 10),
                'strategies': data.get('strategies', {}),
                'reason': data.get('reason'),
                'metrics': metrics
            }

        except Exception as e:
            print(f"⚠️ Error loading strategy recommendation: {e}")
            return {}
    
    def get_final_plan_stake(self, date_str):
        """Get final stake for a specific date from final bet plan"""
        # Convert date to YYYYMMDD format
        try:
            target_date = date_str.replace('-', '') if '-' in date_str else date_str
            final_file = self.base_dir / "predictions" / "bet_engine" / f"final_bet_plan_{target_date}.xlsx"
            
            if not final_file.exists():
                return None
                
            df = pd.read_excel(final_file, sheet_name='final_slot_plan')
            
            # Look for TOTAL row
            total_row = df[df['date'] == 'TOTAL']
            if not total_row.empty:
                return float(total_row['final_slot_stake'].iloc[0])
            
            # Otherwise sum non-empty slots
            slot_rows = df[df['slot'].isin(['FRBD', 'GZBD', 'GALI', 'DSWR'])]
            if not slot_rows.empty:
                return float(slot_rows['final_slot_stake'].sum())
                
            return None
            
        except Exception as e:
            print(f"⚠️ Error reading final plan for {date_str}: {e}")
            return None
    
    def calculate_strategy_performance(self, pnl_df, dynamic_total_stake, strategy_recommendation, days):
        """Calculate performance for all three strategies"""
        for day_index, day_row in pnl_df.iterrows():
            try:
                # Extract base data
                base_stake = float(day_row.get('stake_total', 0))
                base_return = float(day_row.get('return_total', 0))
                base_profit = float(day_row.get('profit_total', 0))
                date_str = str(day_row.get('date', ''))
                
                if base_stake == 0:
                    continue
                
                base_roi = (base_profit / base_stake * 100) if base_stake > 0 else 0
                
                # Strategy 1: BASE
                self.results['base']['stake'] += base_stake
                self.results['base']['return'] += base_return
                self.results['base']['profit'] += base_profit
                
                # Strategy 2: DYNAMIC
                dynamic_stake = 0
                dynamic_return = 0
                dynamic_profit = 0
                dynamic_roi = 0
                
                if dynamic_total_stake and dynamic_total_stake > 0:
                    dynamic_stake = dynamic_total_stake
                    scale_factor = dynamic_total_stake / base_stake
                    dynamic_return = base_return * scale_factor
                    dynamic_profit = base_profit * scale_factor
                    dynamic_roi = (dynamic_profit / dynamic_total_stake * 100) if dynamic_total_stake > 0 else 0
                    
                    self.results['dynamic']['stake'] += dynamic_stake
                    self.results['dynamic']['return'] += dynamic_return
                    self.results['dynamic']['profit'] += dynamic_profit
                
                # Strategy 3: FINAL
                final_stake = 0
                final_return = 0
                final_profit = 0
                final_roi = 0
                final_plan_available = False
                
                final_total_stake = self.get_final_plan_stake(date_str)
                if final_total_stake and final_total_stake > 0:
                    final_stake = final_total_stake
                    scale_factor = final_total_stake / base_stake
                    final_return = base_return * scale_factor
                    final_profit = base_profit * scale_factor
                    final_roi = (final_profit / final_total_stake * 100) if final_total_stake > 0 else 0
                    final_plan_available = True
                    
                    self.results['final']['stake'] += final_stake
                    self.results['final']['return'] += final_return
                    self.results['final']['profit'] += final_profit
                
                # Store daily data
                daily_record = {
                    'date': date_str,
                    'base_stake': base_stake,
                    'base_return': base_return,
                    'base_profit': base_profit,
                    'base_roi': base_roi,
                    'dynamic_stake': dynamic_stake,
                    'dynamic_return': dynamic_return,
                    'dynamic_profit': dynamic_profit,
                    'dynamic_roi': dynamic_roi,
                    'final_stake': final_stake,
                    'final_return': final_return,
                    'final_profit': final_profit,
                    'final_roi': final_roi,
                    'final_plan_available': final_plan_available,
                    'recommended_strategy': strategy_recommendation.get('recommended_strategy', 'NONE'),
                    'meta_confidence_level': strategy_recommendation.get('confidence_level', 'LOW'),
                    'risk_mode': strategy_recommendation.get('risk_mode', 'DEFENSIVE')
                }
                self.daily_data.append(daily_record)
                
                # Store heatmap data
                for strategy in ['BASE', 'DYNAMIC', 'FINAL']:
                    if strategy == 'BASE' or (strategy == 'DYNAMIC' and dynamic_total_stake) or (strategy == 'FINAL' and final_plan_available):
                        self.heatmap_data.append({
                            'date': date_str,
                            'strategy': strategy,
                            'stake': daily_record[f'{strategy.lower()}_stake'],
                            'profit': daily_record[f'{strategy.lower()}_profit'],
                            'roi_percent': daily_record[f'{strategy.lower()}_roi']
                        })
                
            except Exception as e:
                print(f"⚠️ Error processing day {day_row.get('date', 'unknown')}: {e}")
                continue

    def compute_meta_alignment(self, strategy_recommendation, best_by_roi, base_roi_ratio, overall_roi_ratio):
        """Determine whether the meta-strategy aligns with reality (overlay-aware)."""
        strategy_name = (strategy_recommendation.get('recommended_strategy', 'NONE') if strategy_recommendation else 'NONE') or 'NONE'
        strategy_name_upper = strategy_name.upper()
        best_name = (best_by_roi.get('strategy', 'NONE') if best_by_roi else 'NONE') or 'NONE'
        best_name_upper = best_name.upper()

        base_roi_value = float(base_roi_ratio or 0.0)
        overlay = is_overlay_strategy(strategy_name_upper)

        mismatch_flag = False
        meta_status = "UNKNOWN"
        meta_message = ""

        if not strategy_name or strategy_name_upper == 'NONE':
            mismatch_flag = False
            meta_status = "NO_STRATEGY"
            meta_message = "No strategy recommendation available; using BASE only."
        elif not overlay:
            mismatch_flag = strategy_name_upper != best_name_upper if best_name_upper != 'NONE' else False
            meta_status = "MISMATCH" if mismatch_flag else "MATCH"
            meta_message = (
                "Recommended strategy differs from best-by-ROI reality."
                if mismatch_flag
                else "Meta-strategy aligned with best-by-ROI reality."
            )
        else:
            if base_roi_value < MIN_BASE_ROI_FOR_OVERLAY:
                mismatch_flag = True
                meta_status = "MISMATCH"
                meta_message = (
                    "Overlay strategy active while BASE ROI is below safe threshold "
                    f"({base_roi_value:.2f} < {MIN_BASE_ROI_FOR_OVERLAY:.2f})."
                )
            else:
                mismatch_flag = False
                meta_status = "OVERLAY_OK"
                meta_message = "Overlay on BASE allowed: positive BASE ROI with pattern-backed overlay."

        return {
            'meta_status': meta_status,
            'meta_message': meta_message,
            'meta_mismatch': mismatch_flag,
            'strategy_name': strategy_name,
            'best_name': best_name,
            'overlay': overlay,
            'base_roi_ratio': base_roi_ratio,
            'overall_roi_ratio': overall_roi_ratio
        }

    def generate_reports(self, days, strategy_recommendation):
        """Generate Excel and JSON reports"""
        # Excel Report
        excel_file = self.base_dir / "logs" / "performance" / "reality_check_report.xlsx"
        excel_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Sheet 1: day_level
        df_day_level = pd.DataFrame(self.daily_data)
        
        # Sheet 2: summary
        summary_data = []
        for strategy in ['base', 'dynamic', 'final']:
            data = self.results[strategy]
            stake = data['stake']
            profit = data['profit']
            roi = (profit / stake * 100) if stake > 0 else 0
            
            summary_data.append({
                'strategy': strategy.upper(),
                'total_stake': stake,
                'total_return': data['return'],
                'total_profit': profit,
                'roi_percent': roi
            })
        
        # Find best strategies
        valid_strategies = [s for s in summary_data if s['total_stake'] > 0]
        if valid_strategies:
            best_by_roi = max(valid_strategies, key=lambda x: x['roi_percent'])
            best_by_profit = max(valid_strategies, key=lambda x: x['total_profit'])
        else:
            best_by_roi = best_by_profit = {'strategy': 'NONE', 'roi_percent': 0, 'total_profit': 0}

        base_total_stake = self.results['base']['stake']
        base_total_profit = self.results['base']['profit']
        base_roi_ratio = (base_total_profit / base_total_stake) if base_total_stake > 0 else 0
        overall_roi_ratio = base_roi_ratio
        try:
            metrics = strategy_recommendation.get('metrics', {}) if strategy_recommendation else {}
            if metrics and metrics.get('overall_roi') is not None:
                overall_roi_ratio = float(metrics.get('overall_roi'))
        except Exception:
            overall_roi_ratio = base_roi_ratio

        meta_alignment = self.compute_meta_alignment(strategy_recommendation, best_by_roi, base_roi_ratio, overall_roi_ratio)

        df_summary = pd.DataFrame([{
            'window_days': days,
            'base_total_stake': self.results['base']['stake'],
            'base_total_return': self.results['base']['return'],
            'base_total_profit': self.results['base']['profit'],
            'base_roi': (self.results['base']['profit'] / self.results['base']['stake'] * 100) if self.results['base']['stake'] > 0 else 0,
            'dynamic_total_stake': self.results['dynamic']['stake'],
            'dynamic_total_return': self.results['dynamic']['return'],
            'dynamic_total_profit': self.results['dynamic']['profit'],
            'dynamic_roi': (self.results['dynamic']['profit'] / self.results['dynamic']['stake'] * 100) if self.results['dynamic']['stake'] > 0 else 0,
            'final_total_stake': self.results['final']['stake'],
            'final_total_return': self.results['final']['return'],
            'final_total_profit': self.results['final']['profit'],
            'final_roi': (self.results['final']['profit'] / self.results['final']['stake'] * 100) if self.results['final']['stake'] > 0 else 0,
            'best_strategy_by_roi': best_by_roi['strategy'],
            'best_strategy_by_profit': best_by_profit['strategy'],
            'meta_recommended_strategy': strategy_recommendation.get('recommended_strategy', 'NONE'),
            'meta_confidence_level': strategy_recommendation.get('confidence_level', 'LOW'),
            'meta_risk_mode': strategy_recommendation.get('risk_mode', 'DEFENSIVE'),
            'meta_alignment': meta_alignment['meta_status'],
            'meta_mismatch': meta_alignment['meta_mismatch'],
            'meta_strategy': meta_alignment.get('strategy_name'),
            'meta_confidence': strategy_recommendation.get('confidence_level', 'LOW') if strategy_recommendation else 'LOW',
            'meta_risk_mode': strategy_recommendation.get('risk_mode', 'DEFENSIVE') if strategy_recommendation else 'DEFENSIVE',
            'base_roi_ratio': base_roi_ratio,
            'overall_roi_ratio': overall_roi_ratio
        }])
        
        # Sheet 3: heatmap_data
        df_heatmap = pd.DataFrame(self.heatmap_data)
        
        # Save Excel
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_day_level.to_excel(writer, sheet_name='day_level', index=False)
            df_summary.to_excel(writer, sheet_name='summary', index=False)
            df_heatmap.to_excel(writer, sheet_name='heatmap_data', index=False)
        
        # JSON Summary
        json_file = self.base_dir / "logs" / "performance" / "reality_check_summary.json"
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "window_days": days,
            "strategy_totals": {
                "BASE": {
                    "total_stake": self.results['base']['stake'],
                    "total_return": self.results['base']['return'],
                    "total_profit": self.results['base']['profit'],
                    "roi_percent": (self.results['base']['profit'] / self.results['base']['stake'] * 100) if self.results['base']['stake'] > 0 else 0
                },
                "DYNAMIC": {
                    "total_stake": self.results['dynamic']['stake'],
                    "total_return": self.results['dynamic']['return'],
                    "total_profit": self.results['dynamic']['profit'],
                    "roi_percent": (self.results['dynamic']['profit'] / self.results['dynamic']['stake'] * 100) if self.results['dynamic']['stake'] > 0 else 0
                },
                "FINAL": {
                    "total_stake": self.results['final']['stake'],
                    "total_return": self.results['final']['return'],
                    "total_profit": self.results['final']['profit'],
                    "roi_percent": (self.results['final']['profit'] / self.results['final']['stake'] * 100) if self.results['final']['stake'] > 0 else 0
                }
            },
            "best_strategy_by_roi": best_by_roi['strategy'],
            "best_strategy_by_profit": best_by_profit['strategy'],
            "meta_recommended_strategy": strategy_recommendation.get('recommended_strategy', 'NONE'),
            "meta_confidence_level": strategy_recommendation.get('confidence_level', 'LOW'),
            "meta_risk_mode": strategy_recommendation.get('risk_mode', 'DEFENSIVE'),
            "meta_alignment": meta_alignment['meta_status'],
            "meta_mismatch": meta_alignment['meta_mismatch'],
            "meta_strategy": meta_alignment.get('strategy_name'),
            "meta_confidence": strategy_recommendation.get('confidence_level', 'LOW') if strategy_recommendation else 'LOW',
            "meta_risk_mode": strategy_recommendation.get('risk_mode', 'DEFENSIVE') if strategy_recommendation else 'DEFENSIVE',
            "base_roi_ratio": base_roi_ratio,
            "overall_roi_ratio": overall_roi_ratio
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        return excel_file, json_file, best_by_roi, best_by_profit
    
    def display_console_summary(self, days, strategy_recommendation, best_by_roi, best_by_profit):
        """Display formatted console summary"""
        print("\n" + "="*60)
        print("📊 REALITY CHECK - Strategy Performance vs Actual Results")
        print("="*60)
        print(f"Last {days} days:")
        print("-" * 40)
        
        # Display strategy comparison
        for strategy in ['base', 'dynamic', 'final']:
            data = self.results[strategy]
            stake = data['stake']
            profit = data['profit']
            roi = (profit / stake * 100) if stake > 0 else 0
            
            if stake > 0:  # Only show strategies with data
                print(f"{strategy.upper():8}: stake=₹{stake:,.0f}, profit=₹{profit:+,.0f}, ROI={roi:+.1f}%")
        
        print()
        print("🎯 STRATEGY RECOMMENDATION:")
        print(f"  Recommended: {strategy_recommendation.get('recommended_strategy', 'NONE')}")
        print(f"  Confidence : {strategy_recommendation.get('confidence_level', 'LOW')}")
        print(f"  Risk Mode  : {strategy_recommendation.get('risk_mode', 'DEFENSIVE')}")
        
        print()
        print("📈 REALITY CHECK BEST:")
        print(f"  By ROI    : {best_by_roi['strategy']} (ROI: {best_by_roi['roi_percent']:+.1f}%)")
        print(f"  By Profit : {best_by_profit['strategy']} (Profit: ₹{best_by_profit['total_profit']:+,.0f})")

        metrics = strategy_recommendation.get('metrics', {}) if strategy_recommendation else {}
        base_stake_total = self.results['base']['stake']
        base_profit_total = self.results['base']['profit']
        base_roi_ratio = (base_profit_total / base_stake_total) if base_stake_total > 0 else 0

        overall_roi_ratio = base_roi_ratio
        try:
            if metrics and metrics.get('overall_roi') is not None:
                overall_roi_ratio = float(metrics.get('overall_roi'))
        except Exception:
            overall_roi_ratio = base_roi_ratio

        alignment = self.compute_meta_alignment(strategy_recommendation, best_by_roi, base_roi_ratio, overall_roi_ratio)

        base_roi_percent = base_roi_ratio * 100
        overall_roi_percent = overall_roi_ratio * 100

        strategy_name = strategy_recommendation.get('recommended_strategy', 'NONE') if strategy_recommendation else 'NONE'
        confidence_level = strategy_recommendation.get('confidence_level', 'LOW') if strategy_recommendation else 'LOW'
        risk_mode = strategy_recommendation.get('risk_mode', 'DEFENSIVE') if strategy_recommendation else 'DEFENSIVE'

        strategy_display = (strategy_name or 'NONE').upper()
        confidence_display = (confidence_level or 'N/A')
        risk_display = (risk_mode or 'N/A')
        reality_best_display = (best_by_roi.get('strategy', 'NONE') if best_by_roi else 'NONE')

        print("\nMeta vs Reality:")
        print(f"  Meta strategy   : {strategy_display} ({confidence_display}, {risk_display})")
        print(f"  Reality best ROI: {reality_best_display}")

        if alignment['meta_status'] == "OVERLAY_OK":
            print(f"\n✅ OVERLAY OK – {strategy_display} is treated as an overlay on BASE with positive ROI.")
            print(f"   - BASE ROI     : {base_roi_percent:+.1f}%")
            print(f"   - Overall ROI  : {overall_roi_percent:+.1f}%")
        elif alignment['meta_status'] == "MATCH":
            print("\n✅ MATCH – Meta-strategy aligned with reality.")
        elif alignment['meta_status'] == "NO_STRATEGY":
            print(f"\nℹ️  {alignment['meta_message']}")
        else:
            reason = alignment['meta_message'] or "Meta-strategy differs from reality"
            print("\n⚠️  MISMATCH – Meta-strategy differs from reality")
            print(f"   - Reason: {reason}")
    
    def run_reality_check(self, days=10):
        """Run complete reality check analysis"""
        print("🔍 REALITY CHECK ENGINE - Strategy Performance Analyzer")
        print("=" * 60)
        print(f"Analyzing last {days} days of performance data...")

        # Step 1: Load P&L history
        pnl_df = self.load_pnl_history(days)
        if pnl_df is None or pnl_df.empty:
            print("⚠️ No P&L data available for analysis; skipping reality check outputs")
            return True
        
        # Step 2: Load dynamic stakes
        dynamic_total_stake = self.load_dynamic_stakes()
        
        # Step 3: Load strategy recommendation
        strategy_recommendation = self.load_strategy_recommendation()
        
        # Step 4: Calculate strategy performance
        self.calculate_strategy_performance(pnl_df, dynamic_total_stake, strategy_recommendation, days)
        
        # Step 5: Generate reports
        excel_file, json_file, best_by_roi, best_by_profit = self.generate_reports(days, strategy_recommendation)
        
        # Step 6: Display console summary
        self.display_console_summary(days, strategy_recommendation, best_by_roi, best_by_profit)
        
        print()
        print("💾 OUTPUT FILES:")
        print(f"  Excel: {excel_file}")
        print(f"  JSON : {json_file}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Reality Check Engine - Strategy Performance vs Actual Results')
    parser.add_argument('--days', type=int, default=10, help='Number of days to analyze (default: 10)')

    args = parser.parse_args()

    engine = RealityCheckEngine()
    success = engine.run_reality_check(args.days)

    return 0 if success else 1


def run_reality_check(days: int = 10) -> bool:
    """Run the reality check analysis for the specified window."""
    engine = RealityCheckEngine()
    return engine.run_reality_check(days)

if __name__ == "__main__":
    exit(main())
