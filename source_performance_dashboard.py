# source_performance_dashboard.py - COMPLETE CODE
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys

class SourcePerformanceDashboard:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def load_pnl_data(self):
        """Load P&L data from bet_pnl_history.xlsx"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        
        if not pnl_file.exists():
            print("❌ P&L file not found:", pnl_file)
            return None
            
        try:
            day_df = pd.read_excel(pnl_file, sheet_name='day_level')
            day_df['date'] = pd.to_datetime(day_df['date']).dt.date
            return day_df.sort_values('date')
        except Exception as e:
            print(f"❌ Error reading P&L file: {e}")
            return None
            
    def load_prediction_log(self):
        """Load prediction log data"""
        log_file = self.base_dir / "logs" / "predictions" / "daily_prediction_log.xlsx"
        
        if not log_file.exists():
            print("❌ Prediction log not found:", log_file)
            return None
            
        try:
            log_df = pd.read_excel(log_file)
            log_df['target_date'] = pd.to_datetime(log_df['target_date']).dt.date
            
            # Handle missing source column (older entries)
            if 'source' not in log_df.columns:
                log_df['source'] = 'unknown'
            else:
                log_df['source'] = log_df['source'].fillna('unknown')
                
            return log_df
        except Exception as e:
            print(f"❌ Error reading prediction log: {e}")
            return None
            
    def generate_daily_summary(self, pnl_df, log_df):
        """Generate daily performance summary"""
        daily_data = []
        
        # Calculate cumulative profit
        pnl_df = pnl_df.copy()
        pnl_df['cumulative_profit'] = pnl_df['profit_total'].cumsum()
        
        # Get last 10 days
        recent_days = pnl_df.tail(10)
        
        for _, day_row in recent_days.iterrows():
            date = day_row['date']
            
            # Find prediction log entries for this date
            date_logs = log_df[log_df['target_date'] == date] if log_df is not None else pd.DataFrame()
            
            # Count sources
            fusion_runs = len(date_logs[date_logs['source'] == 'fusion'])
            scr9_runs = len(date_logs[date_logs['source'] == 'scr9'])
            unknown_runs = len(date_logs[date_logs['source'] == 'unknown'])
            
            # Get unique sources used
            sources_used = []
            if fusion_runs > 0:
                sources_used.append('fusion')
            if scr9_runs > 0:
                sources_used.append('scr9')
            if unknown_runs > 0:
                sources_used.append('unknown')
                
            sources_str = ','.join(sources_used) if sources_used else '-'
            
            daily_data.append({
                'date': date,
                'profit': day_row['profit_total'],
                'cumulative_profit': day_row['cumulative_profit'],
                'sources_used': sources_str,
                'fusion_runs': fusion_runs,
                'scr9_runs': scr9_runs
            })
            
        return daily_data
        
    def generate_source_summary(self, log_df):
        """Generate per-source usage summary"""
        if log_df is None or log_df.empty:
            return []
            
        source_data = []
        sources = log_df['source'].unique()
        
        for source in sources:
            source_logs = log_df[log_df['source'] == source]
            
            distinct_days = source_logs['target_date'].nunique()
            total_runs = len(source_logs)
            total_stake = source_logs['total_stake'].sum()
            avg_stake = total_stake / distinct_days if distinct_days > 0 else 0
            last_used = source_logs['target_date'].max()
            
            source_data.append({
                'source': source,
                'distinct_days': distinct_days,
                'total_runs': total_runs,
                'total_stake': total_stake,
                'avg_stake_per_day': avg_stake,
                'last_used': last_used
            })
            
        return source_data
        
    def print_console_summary(self, daily_data, source_data):
        """Print formatted console summary"""
        print("\n" + "="*80)
        print("SOURCE PERFORMANCE DASHBOARD")
        print("="*80)
        
        # Daily performance table
        print("\nLAST 10 DAYS PERFORMANCE:")
        print("-" * 75)
        print(f"{'Date':12} {'Profit':>8} {'CumProfit':>10} {'Sources':>12} {'Fusion':>8} {'Scr9':>8}")
        print("-" * 75)
        
        for day in daily_data:
            profit_str = f"₹{day['profit']:+.0f}"
            cum_str = f"₹{day['cumulative_profit']:.0f}"
            print(f"{day['date']}  {profit_str:>8}  {cum_str:>10}  {day['sources_used']:>12}  {day['fusion_runs']:>6}  {day['scr9_runs']:>6}")
            
        # Source summary table
        print("\nPER-SOURCE USAGE SUMMARY:")
        print("-" * 85)
        print(f"{'Source':8} {'Days':>6} {'Runs':>6} {'TotalStake':>12} {'AvgStake':>10} {'LastUsed':>12}")
        print("-" * 85)
        
        for source in source_data:
            stake_str = f"₹{source['total_stake']:.0f}"
            avg_str = f"₹{source['avg_stake_per_day']:.1f}"
            print(f"{source['source']:8}  {source['distinct_days']:>4}  {source['total_runs']:>4}  {stake_str:>12}  {avg_str:>10}  {source['last_used']}")
            
    def save_excel_summary(self, daily_data, source_data):
        """Save summary to Excel file"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / "source_performance_summary.xlsx"
        
        # Convert to DataFrames
        daily_df = pd.DataFrame(daily_data)
        source_df = pd.DataFrame(source_data)
        
        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            daily_df.to_excel(writer, sheet_name='by_day', index=False)
            source_df.to_excel(writer, sheet_name='by_source', index=False)
            
        print(f"\n💾 Dashboard saved to: {output_file}")
        
    def run(self):
        """Main execution function"""
        print("📊 SOURCE PERFORMANCE DASHBOARD")
        print("Loading data...")
        
        # Load data
        pnl_df = self.load_pnl_data()
        log_df = self.load_prediction_log()
        
        if pnl_df is None:
            print("❌ Cannot proceed without P&L data")
            return False
            
        # Generate summaries
        daily_data = self.generate_daily_summary(pnl_df, log_df)
        source_data = self.generate_source_summary(log_df)
        
        # Print console output
        self.print_console_summary(daily_data, source_data)
        
        # Save Excel file
        self.save_excel_summary(daily_data, source_data)
        
        return True

def main():
    dashboard = SourcePerformanceDashboard()
    success = dashboard.run()
    
    if success:
        print("\n✅ Dashboard completed successfully!")
        return 0
    else:
        print("\n❌ Dashboard failed!")
        return 1

if __name__ == "__main__":
    exit(main())