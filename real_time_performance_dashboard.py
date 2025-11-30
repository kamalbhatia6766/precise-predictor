# real_time_performance_dashboard.py
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

class RealTimeDashboard:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        
    def load_all_performance_data(self):
        """Load all performance data for dashboard"""
        data = {}
        
        # Load all relevant files
        files_to_load = {
            'pnl': 'bet_pnl_history.xlsx',
            'golden_days': 'golden_days_analysis.xlsx', 
            'backtest': 'auto_backtest_results.xlsx',
            'stake_plan': 'dynamic_stake_plan.json',
            'fusion_plan': 'adaptive_fusion_plan.json',
            'pattern_intel': 'pattern_intelligence_report.xlsx',
            'money_plan': 'money_management_plan.json'
        }
        
        performance_dir = self.base_dir / "logs" / "performance"
        
        for key, filename in files_to_load.items():
            file_path = performance_dir / filename
            try:
                if file_path.suffix == '.xlsx' and file_path.exists():
                    if key == 'pnl':
                        data[key] = pd.read_excel(file_path, sheet_name='day_level')
                    else:
                        data[key] = pd.read_excel(file_path)
                elif file_path.suffix == '.json' and file_path.exists():
                    with open(file_path, 'r') as f:
                        data[key] = json.load(f)
            except Exception as e:
                print(f"⚠️ Could not load {filename}: {e}")
                
        return data
    
    def generate_dashboard_metrics(self, data):
        """Generate comprehensive dashboard metrics"""
        metrics = {}
        
        # Basic P&L metrics
        if 'pnl' in data:
            pnl_df = data['pnl']
            metrics['total_profit'] = float(pnl_df['profit_total'].sum())
            metrics['total_stake'] = float(pnl_df['stake_total'].sum())
            metrics['total_return'] = float(pnl_df['return_total'].sum())
            metrics['overall_roi'] = (metrics['total_profit'] / metrics['total_stake'] * 100) if metrics['total_stake'] > 0 else 0
            
            # Recent performance (last 7 days)
            recent_days = pnl_df.tail(7)
            metrics['recent_profit'] = float(recent_days['profit_total'].sum())
            metrics['recent_roi'] = (metrics['recent_profit'] / recent_days['stake_total'].sum() * 100) if recent_days['stake_total'].sum() > 0 else 0
        
        # Slot performance
        if 'pnl' in data:
            slot_performance = {}
            for slot in ['FRBD', 'GZBD', 'GALI', 'DSWR']:
                profit_col = f'profit_{slot.lower()}'
                if profit_col in data['pnl'].columns:
                    slot_profit = float(data['pnl'][profit_col].sum())
                    slot_performance[slot] = {
                        'total_profit': slot_profit,
                        'performance_tier': 'HIGH' if slot_profit > 1000 else 'MEDIUM' if slot_profit > 0 else 'LOW'
                    }
            metrics['slot_performance'] = slot_performance
        
        # Source performance
        if 'backtest' in data:
            backtest_df = data['backtest']
            source_performance = {}
            for source in ['fusion', 'scr9']:
                source_data = backtest_df[backtest_df['source'] == source]
                if len(source_data) > 0:
                    source_profit = float(source_data['profit_total'].sum())
                    source_performance[source] = {
                        'total_profit': source_profit,
                        'roi': (source_profit / source_data['stake_total'].sum() * 100) if source_data['stake_total'].sum() > 0 else 0
                    }
            metrics['source_performance'] = source_performance
        
        # Risk metrics
        if 'money_plan' in data:
            metrics['risk_metrics'] = data['money_plan'].get('bankroll_rules', {})
        
        return metrics
    
    def generate_dashboard_recommendations(self, metrics):
        """Generate dashboard recommendations"""
        recommendations = []
        
        # Profit-based recommendations
        if metrics.get('recent_roi', 0) > 100:
            recommendations.append({
                'type': 'PERFORMANCE',
                'priority': 'HIGH',
                'message': f"🚀 Excellent recent performance! ROI: {metrics['recent_roi']:.1f}%",
                'action': 'Consider increasing stakes gradually'
            })
        elif metrics.get('recent_roi', 0) < 0:
            recommendations.append({
                'type': 'PERFORMANCE', 
                'priority': 'HIGH',
                'message': f"⚠️ Recent performance negative. ROI: {metrics['recent_roi']:.1f}%",
                'action': 'Review strategies and reduce stakes'
            })
        
        # Slot-based recommendations
        if 'slot_performance' in metrics:
            high_perf_slots = [s for s, data in metrics['slot_performance'].items() if data['performance_tier'] == 'HIGH']
            low_perf_slots = [s for s, data in metrics['slot_performance'].items() if data['performance_tier'] == 'LOW']
            
            if high_perf_slots:
                recommendations.append({
                    'type': 'SLOT_OPTIMIZATION',
                    'priority': 'MEDIUM',
                    'message': f"🎯 High performing slots: {', '.join(high_perf_slots)}",
                    'action': 'Focus resources on these slots'
                })
            
            if low_perf_slots:
                recommendations.append({
                    'type': 'SLOT_OPTIMIZATION',
                    'priority': 'MEDIUM', 
                    'message': f"📉 Underperforming slots: {', '.join(low_perf_slots)}",
                    'action': 'Reduce exposure or review strategy'
                })
        
        return recommendations
    
    def save_dashboard_report(self, metrics, recommendations):
        """Save dashboard report"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dashboard data
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'recommendations': recommendations,
            'system_status': 'OPTIMAL' if metrics.get('recent_roi', 0) > 50 else 'GOOD' if metrics.get('recent_roi', 0) > 0 else 'NEEDS_ATTENTION'
        }
        
        # Save JSON
        json_file = output_dir / "real_time_dashboard.json"
        with open(json_file, 'w') as f:
            json.dump(dashboard_data, f, indent=2, default=str)
            
        # Save Excel
        excel_file = output_dir / "performance_dashboard.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Metrics sheet
            metrics_flat = []
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            for k, v in subvalue.items():
                                metrics_flat.append({'category': key, 'subcategory': subkey, 'metric': k, 'value': v})
                        else:
                            metrics_flat.append({'category': key, 'subcategory': '', 'metric': subkey, 'value': subvalue})
                else:
                    metrics_flat.append({'category': 'general', 'subcategory': '', 'metric': key, 'value': value})
            
            pd.DataFrame(metrics_flat).to_excel(writer, sheet_name='metrics', index=False)
            
            # Recommendations sheet
            if recommendations:
                pd.DataFrame(recommendations).to_excel(writer, sheet_name='recommendations', index=False)
        
        print(f"💾 Real-time dashboard saved to: {json_file}")
        print(f"💾 Excel report saved to: {excel_file}")
        
        return dashboard_data
    
    def print_console_dashboard(self, metrics, recommendations, dashboard_data):
        """Print console dashboard"""
        print("\n" + "="*100)
        print("📊 REAL-TIME PERFORMANCE DASHBOARD")
        print("="*100)
        
        print(f"\n💰 OVERALL PERFORMANCE:")
        print("-" * 50)
        print(f"   Total Profit:    ₹{metrics.get('total_profit', 0):+,.0f}")
        print(f"   Total ROI:       {metrics.get('overall_roi', 0):.1f}%")
        print(f"   Recent Profit:   ₹{metrics.get('recent_profit', 0):+,.0f} (7 days)")
        print(f"   Recent ROI:      {metrics.get('recent_roi', 0):.1f}%")
        print(f"   System Status:   {dashboard_data['system_status']}")
        
        print(f"\n🎯 SLOT PERFORMANCE:")
        print("-" * 35)
        if 'slot_performance' in metrics:
            for slot, data in metrics['slot_performance'].items():
                profit = data['total_profit']
                tier = data['performance_tier']
                print(f"   {slot}: ₹{profit:+,.0f} | {tier}")
        
        print(f"\n🔧 SOURCE PERFORMANCE:")
        print("-" * 35)
        if 'source_performance' in metrics:
            for source, data in metrics['source_performance'].items():
                profit = data['total_profit']
                roi = data['roi']
                print(f"   {source.upper():6}: ₹{profit:+,.0f} | ROI: {roi:.1f}%")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 45)
        for rec in recommendations[:5]:  # Show top 5
            priority_icon = "🚀" if rec['priority'] == 'HIGH' else "✅" if rec['priority'] == 'MEDIUM' else "ℹ️"
            print(f"   {priority_icon} {rec['message']}")
            print(f"      → {rec['action']}")
    
    def run(self):
        """Main execution"""
        print("📊 REAL-TIME PERFORMANCE DASHBOARD - Generating comprehensive report...")
        
        # Load all data
        data = self.load_all_performance_data()
        
        if not data:
            print("❌ No performance data found")
            return False
            
        # Generate metrics and recommendations
        metrics = self.generate_dashboard_metrics(data)
        recommendations = self.generate_dashboard_recommendations(metrics)
        dashboard_data = self.save_dashboard_report(metrics, recommendations)
        
        # Output dashboard
        self.print_console_dashboard(metrics, recommendations, dashboard_data)
        
        return True

def main():
    dashboard = RealTimeDashboard()
    success = dashboard.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())