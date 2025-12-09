# quant_system_auditor.py
"""
QUANT SYSTEM AUDITOR - System Audit & Consistency Guard

PURPOSE:
Check file consistency, date alignment, and pipeline completeness across
the entire quant lab system for the last N days.

USAGE:
py -3.12 quant_system_auditor.py --days 30
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import sys
from collections import defaultdict

class QuantSystemAuditor:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.audit_dates = []
        self.date_level_data = []
        self.file_level_data = []
        self.summary_data = {}
        
    def discover_all_dates(self, days=30):
        """Discover all dates from various sources"""
        all_dates = set()
        
        # 1. From P&L history
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if pnl_file.exists():
            try:
                df = pd.read_excel(pnl_file, sheet_name='day_level')
                date_col = self._find_column_case_insensitive(df, 'date')
                if date_col:
                    for date_val in df[date_col].dropna():
                        normalized = self._normalize_date(date_val)
                        if normalized:
                            all_dates.add(normalized)
            except Exception as e:
                print(f"⚠️ Error reading P&L history: {e}")
        
        # 2. From prediction log
        pred_file = self.base_dir / "logs" / "predictions" / "daily_prediction_log.xlsx"
        if pred_file.exists():
            try:
                df = pd.read_excel(pred_file)
                date_col = self._find_column_case_insensitive(df, 'date')
                if date_col:
                    for date_val in df[date_col].dropna():
                        normalized = self._normalize_date(date_val)
                        if normalized:
                            all_dates.add(normalized)
            except Exception as e:
                print(f"⚠️ Error reading prediction log: {e}")
        
        # 3. From bet plan files
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        if bet_engine_dir.exists():
            file_patterns = [
                "bet_plan_master_*.xlsx",
                "enhanced_bet_plan_*.xlsx", 
                "high_conviction_bet_plan_*.xlsx",
                "final_bet_plan_*.xlsx"
            ]
            
            for pattern in file_patterns:
                files = list(bet_engine_dir.glob(pattern))
                for file in files:
                    date_str = file.stem.split('_')[-1]
                    if date_str.isdigit() and len(date_str) == 8:
                        normalized = self._normalize_date(date_str)
                        if normalized:
                            all_dates.add(normalized)
        
        # Sort dates and take the most recent N
        sorted_dates = sorted(all_dates, reverse=True)
        self.audit_dates = sorted_dates[:days]
        
        print(f"📅 Found {len(all_dates)} unique dates, auditing last {len(self.audit_dates)} dates")
        return True
    
    def _find_column_case_insensitive(self, df, column_name):
        """Find column name case-insensitively"""
        column_name_lower = column_name.lower()
        for col in df.columns:
            if col.lower() == column_name_lower:
                return col
        return None
    
    def _normalize_date(self, date_val):
        """Normalize date to YYYY-MM-DD format"""
        if isinstance(date_val, str):
            # Handle YYYYMMDD format
            if len(date_val) == 8 and date_val.isdigit():
                return f"{date_val[:4]}-{date_val[4:6]}-{date_val[6:8]}"
            # Handle other string formats
            try:
                dt = datetime.strptime(date_val.split()[0], '%Y-%m-%d')
                return dt.strftime('%Y-%m-%d')
            except:
                try:
                    dt = datetime.strptime(date_val.split()[0], '%d-%m-%Y')
                    return dt.strftime('%Y-%m-%d')
                except:
                    return None
        elif isinstance(date_val, datetime):
            return date_val.strftime('%Y-%m-%d')
        elif pd.notna(date_val):
            try:
                return pd.to_datetime(date_val).strftime('%Y-%m-%d')
            except:
                return None
        return None
    
    def _date_to_filename_format(self, date_str):
        """Convert YYYY-MM-DD to YYYYMMDD for filename matching"""
        return date_str.replace('-', '')
    
    def check_prediction_log(self, date_str):
        """Check if date exists in prediction log"""
        pred_file = self.base_dir / "logs" / "predictions" / "daily_prediction_log.xlsx"
        if not pred_file.exists():
            return False
        
        try:
            df = pd.read_excel(pred_file)
            date_col = self._find_column_case_insensitive(df, 'date')
            if not date_col:
                return False
            
            for date_val in df[date_col].dropna():
                if self._normalize_date(date_val) == date_str:
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Error checking prediction log for {date_str}: {e}")
            return False
    
    def check_pnl_history(self, date_str):
        """Check if date exists in P&L history"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if not pnl_file.exists():
            return False
        
        try:
            df = pd.read_excel(pnl_file, sheet_name='day_level')
            date_col = self._find_column_case_insensitive(df, 'date')
            if not date_col:
                return False
            
            for date_val in df[date_col].dropna():
                if self._normalize_date(date_val) == date_str:
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Error checking P&L history for {date_str}: {e}")
            return False
    
    def check_plan_files(self, date_str):
        """Check existence of various plan files for a date"""
        filename_date = self._date_to_filename_format(date_str)
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        
        plans = {
            'master': bet_engine_dir / f"bet_plan_master_{filename_date}.xlsx",
            'enhanced': bet_engine_dir / f"enhanced_bet_plan_{filename_date}.xlsx",
            'conviction': bet_engine_dir / f"high_conviction_bet_plan_{filename_date}.xlsx",
            'final': bet_engine_dir / f"final_bet_plan_{filename_date}.xlsx"
        }
        
        return {plan_type: file_path.exists() for plan_type, file_path in plans.items()}
    
    def check_final_plan_date_mismatch(self, date_str):
        """Check if final plan internal dates match filename date"""
        filename_date = self._date_to_filename_format(date_str)
        final_file = self.base_dir / "predictions" / "bet_engine" / f"final_bet_plan_{filename_date}.xlsx"
        
        if not final_file.exists():
            return False
        
        try:
            df = pd.read_excel(final_file, sheet_name='final_slot_plan')
            date_col = self._find_column_case_insensitive(df, 'date')
            if not date_col:
                return True  # Mismatch because no date column
            
            # Check non-TOTAL rows
            for date_val in df[date_col].dropna():
                if str(date_val).strip().upper() == 'TOTAL':
                    continue
                normalized = self._normalize_date(date_val)
                if normalized and normalized != date_str:
                    return True  # Mismatch found
            
            return False  # No mismatch
        except Exception as e:
            print(f"⚠️ Error checking final plan dates for {date_str}: {e}")
            return True  # Treat errors as mismatches
    
    def generate_date_level_audit(self):
        """Generate per-date audit data"""
        for date_str in self.audit_dates:
            # Check all components
            has_prediction_log = self.check_prediction_log(date_str)
            has_pnl_row = self.check_pnl_history(date_str)
            plan_files = self.check_plan_files(date_str)
            final_plan_mismatch = self.check_final_plan_date_mismatch(date_str)
            
            # Build issues list
            issues = []
            if not has_prediction_log:
                issues.append("missing prediction_log")
            if not has_pnl_row:
                issues.append("missing PnL_row")
            if not plan_files['master']:
                issues.append("missing master_plan")
            if not plan_files['enhanced']:
                issues.append("missing enhanced_plan")
            if not plan_files['conviction']:
                issues.append("missing conviction_plan")
            if not plan_files['final']:
                issues.append("missing final_plan")
            if final_plan_mismatch:
                issues.append("final_plan_date_mismatch")
            
            issues_text = "OK" if not issues else ", ".join(issues)
            
            self.date_level_data.append({
                'date': date_str,
                'has_prediction_log_row': has_prediction_log,
                'has_master_plan': plan_files['master'],
                'has_enhanced_plan': plan_files['enhanced'],
                'has_conviction_plan': plan_files['conviction'],
                'has_final_plan': plan_files['final'],
                'has_pnl_row': has_pnl_row,
                'final_plan_date_mismatch': final_plan_mismatch,
                'issues': issues_text
            })
    
    def generate_file_level_audit(self):
        """Generate per-file audit data"""
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        
        if not bet_engine_dir.exists():
            return
        
        file_patterns = {
            'MASTER': "bet_plan_master_*.xlsx",
            'ENHANCED': "enhanced_bet_plan_*.xlsx",
            'CONVICTION': "high_conviction_bet_plan_*.xlsx", 
            'FINAL': "final_bet_plan_*.xlsx"
        }
        
        for file_type, pattern in file_patterns.items():
            files = list(bet_engine_dir.glob(pattern))
            for file in files:
                # Parse date from filename
                date_str = file.stem.split('_')[-1]
                parsed_date = self._normalize_date(date_str) if date_str.isdigit() and len(date_str) == 8 else ""
                
                # For FINAL files, check internal date
                internal_primary_date = ""
                date_mismatch = False
                notes = ""
                
                if file_type == 'FINAL' and file.exists():
                    try:
                        df = pd.read_excel(file, sheet_name='final_slot_plan')
                        date_col = self._find_column_case_insensitive(df, 'date')
                        if date_col:
                            # Find most common non-TOTAL date
                            date_counts = defaultdict(int)
                            for date_val in df[date_col].dropna():
                                if str(date_val).strip().upper() == 'TOTAL':
                                    continue
                                normalized = self._normalize_date(date_val)
                                if normalized:
                                    date_counts[normalized] += 1
                            
                            if date_counts:
                                internal_primary_date = max(date_counts.items(), key=lambda x: x[1])[0]
                                date_mismatch = (internal_primary_date != parsed_date) if parsed_date else True
                    except Exception as e:
                        notes = f"Error reading file: {e}"
                
                self.file_level_data.append({
                    'file_name': file.name,
                    'file_type': file_type,
                    'parsed_date': parsed_date,
                    'exists': True,
                    'internal_primary_date': internal_primary_date,
                    'date_mismatch': date_mismatch,
                    'notes': notes
                })
    
    def generate_summary(self, days):
        """Generate summary statistics"""
        df_date = pd.DataFrame(self.date_level_data)
        
        if df_date.empty:
            self.summary_data = {
                'window_days': days,
                'total_audited_dates': 0,
                'full_ok_days': 0,
                'partial_days': 0,
                'no_pnl_days': 0,
                'missing_final_plan_days': 0,
                'any_final_date_mismatch': False
            }
            return
        
        full_ok_days = len(df_date[df_date['issues'] == 'OK'])
        partial_days = len(df_date[(df_date['issues'] != 'OK') & df_date['has_pnl_row']])
        no_pnl_days = len(df_date[~df_date['has_pnl_row']])
        missing_final_plan_days = len(df_date[~df_date['has_final_plan'] & df_date['has_pnl_row']])
        any_final_date_mismatch = df_date['final_plan_date_mismatch'].any()
        
        self.summary_data = {
            'window_days': days,
            'total_audited_dates': len(df_date),
            'full_ok_days': full_ok_days,
            'partial_days': partial_days,
            'no_pnl_days': no_pnl_days,
            'missing_final_plan_days': missing_final_plan_days,
            'any_final_date_mismatch': any_final_date_mismatch
        }
    
    def save_audit_report(self):
        """Save audit report to Excel"""
        excel_file = self.base_dir / "logs" / "performance" / "system_audit_report.xlsx"
        excel_file.parent.mkdir(parents=True, exist_ok=True)
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Sheet 1: date_level
            df_date = pd.DataFrame(self.date_level_data)
            if not df_date.empty:
                df_date.to_excel(writer, sheet_name='date_level', index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name='date_level', index=False)
            
            # Sheet 2: file_level
            df_file = pd.DataFrame(self.file_level_data)
            if not df_file.empty:
                df_file.to_excel(writer, sheet_name='file_level', index=False)
            else:
                pd.DataFrame().to_excel(writer, sheet_name='file_level', index=False)
            
            # Sheet 3: summary
            df_summary = pd.DataFrame([self.summary_data])
            df_summary.to_excel(writer, sheet_name='summary', index=False)
        
        return excel_file
    
    def display_console_summary(self, days):
        """Display formatted console summary"""
        print("\n" + "="*60)
        print("🧪 QUANT SYSTEM AUDIT")
        print("="*60)
        print(f"Window: last {days} days")
        print()
        
        print("Date-level summary:")
        print(f"  Total audited dates : {self.summary_data['total_audited_dates']}")
        print(f"  Fully OK days       : {self.summary_data['full_ok_days']}")
        print(f"  Partial days        : {self.summary_data['partial_days']}")
        print(f"  No PnL days         : {self.summary_data['no_pnl_days']}")
        print(f"  Missing final plans : {self.summary_data['missing_final_plan_days']}")
        print(f"  Any date mismatch   : {'YES' if self.summary_data['any_final_date_mismatch'] else 'NO'}")
        
        # Show recent issues (last 5 problematic dates)
        problematic_dates = [d for d in self.date_level_data if d['issues'] != 'OK']
        recent_issues = problematic_dates[:5]
        
        if recent_issues:
            print("\nRecent issues:")
            for issue in recent_issues:
                print(f"  {issue['date']}: {issue['issues']}")
        else:
            print("\n✅ No issues found in recent dates!")
    
    def run_audit(self, days=30):
        """Run complete system audit"""
        print("🔍 QUANT SYSTEM AUDITOR - Consistency & Completeness Check")
        print("=" * 60)
        
        # Step 1: Discover all dates
        if not self.discover_all_dates(days):
            return False
        
        # Step 2: Generate date-level audit
        self.generate_date_level_audit()
        
        # Step 3: Generate file-level audit  
        self.generate_file_level_audit()
        
        # Step 4: Generate summary
        self.generate_summary(days)
        
        # Step 5: Save report
        excel_file = self.save_audit_report()
        
        # Step 6: Display console summary
        self.display_console_summary(days)
        
        print(f"\n💾 Audit report saved to: {excel_file}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Quant System Auditor - Consistency & Completeness Check')
    parser.add_argument('--days', type=int, default=30, help='Number of days to audit (default: 30)')
    
    args = parser.parse_args()
    
    auditor = QuantSystemAuditor()
    success = auditor.run_audit(args.days)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())