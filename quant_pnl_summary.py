"""
quant_pnl_summary.py - P&L Summary and Analysis Utilities
Optional module for clean separation of P&L calculation logic.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import quant_paths


def to_native(value):
    """Convert numpy/pandas scalar types to native Python"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value

class QuantPnLSummary:
    """P&L summary and analysis utilities"""
    
    def __init__(self):
        self.unit = 10  # ₹10 base unit
        
    def load_master_pnl_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Load master P&L data from quant_reality_pnl.xlsx
        
        Returns:
            Tuple of (slot_pnl_df, layer_pnl_df, summary_data)
        """
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.xlsx"
        
        if not pnl_file.exists():
            raise FileNotFoundError(f"Master P&L file not found: {pnl_file}")
        
        try:
            slot_pnl_df = pd.read_excel(pnl_file, sheet_name='daily_slot_pnl')
            layer_pnl_df = pd.read_excel(pnl_file, sheet_name='daily_layer_pnl')
            
            # Load summary data from JSON if available
            json_file = pnl_file.with_suffix('.json')
            if json_file.exists():
                import json
                with open(json_file, 'r') as f:
                    summary_data = json.load(f)
            else:
                summary_data = {}
            
            return slot_pnl_df, layer_pnl_df, summary_data

        except Exception as e:
            raise ValueError(f"Error loading master P&L data: {e}")

    def _load_quant_reality_json(self) -> Dict:
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.json"
        if not pnl_file.exists():
            return {}
        try:
            with open(pnl_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _build_daily_pnl(self, quant_data: Dict) -> pd.DataFrame:
        entries = quant_data.get("daily") or quant_data.get("records") or []
        rows = []
        for entry in entries:
            date_val = pd.to_datetime(entry.get("date") or entry.get("DATE"), errors="coerce")
            if pd.isna(date_val):
                continue
            stake = float(entry.get("total_stake", entry.get("stake", 0)) or 0)
            ret = float(entry.get("total_return", entry.get("return", 0)) or 0)
            pnl_val = entry.get("pnl")
            pnl = float(pnl_val) if pnl_val is not None else ret - stake
            rows.append({"DATE": date_val.normalize(), "STAKE": stake, "PNL": pnl})
        if not rows:
            return pd.DataFrame(columns=["DATE", "STAKE", "PNL"])
        df = pd.DataFrame(rows)
        return df.groupby("DATE")[["STAKE", "PNL"]].sum().reset_index().sort_values("DATE")

    def _calculate_winning_streak(self, daily_df: pd.DataFrame) -> int:
        if daily_df is None or daily_df.empty:
            return 0
        streak = 0
        for pnl in reversed(daily_df.sort_values("DATE")["PNL"].tolist()):
            if pnl > 0:
                streak += 1
            else:
                break
        return streak
    
    def get_current_performance_summary(self) -> Dict:
        """Get current performance summary for dashboards"""
        try:
            slot_pnl_df, layer_pnl_df, summary_data = self.load_master_pnl_data()
            quant_data = self._load_quant_reality_json()

            daily_df = self._build_daily_pnl(quant_data)
            recent_window = daily_df.tail(7)
            recent_stake = recent_window['STAKE'].sum()
            recent_pnl = recent_window['PNL'].sum()
            recent_roi = (recent_pnl / recent_stake * 100) if recent_stake > 0 else 0
            current_streak = self._calculate_winning_streak(daily_df)

            overall_block = summary_data.get('overall', {}) or quant_data.get('overall', {})
            if overall_block:
                overall_info = overall_block
            else:
                overall_stake = daily_df['STAKE'].sum()
                overall_pnl = daily_df['PNL'].sum()
                overall_roi = (overall_pnl / overall_stake * 100) if overall_stake > 0 else 0
                overall_info = {'overall_roi': overall_roi, 'total_pnl': overall_pnl}
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall': overall_info,
                'recent_7d': {
                    'stake': recent_stake,
                    'return': recent_stake + recent_pnl,
                    'roi': recent_roi,
                    'winning_streak': current_streak
                },
                'best_slot': max(summary_data.get('by_slot', []), key=lambda x: x.get('roi_pct', -100), default={}),
                'worst_slot': min(summary_data.get('by_slot', []), key=lambda x: x.get('roi_pct', 100), default={}),
                'best_layer': max(summary_data.get('by_layer', []), key=lambda x: x.get('roi_pct', -100), default={})
            }
            
        except Exception as e:
            print(f"⚠️  Error generating performance summary: {e}")
            return {}
    
    def generate_strategy_insights(self, summary_data: Dict) -> Dict:
        """Generate strategic insights from P&L data"""
        insights = {
            'timestamp': datetime.now().isoformat(),
            'recommendations': [],
            'warnings': [],
            'opportunities': []
        }
        
        by_slot = summary_data.get('by_slot', [])
        by_layer = summary_data.get('by_layer', [])
        
        # Slot-based insights
        for slot in by_slot:
            roi = slot.get('roi_pct', 0)
            if roi > 50:
                insights['recommendations'].append(
                    f"Consider increasing stake on {slot['slot']} (ROI: {roi:.1f}%)"
                )
            elif roi < -20:
                insights['warnings'].append(
                    f"Review strategy for {slot['slot']} (ROI: {roi:.1f}%)"
                )
        
        # Layer-based insights
        main_layer = next((layer for layer in by_layer if layer['layer_type'] == 'Main'), {})
        andar_layer = next((layer for layer in by_layer if layer['layer_type'] == 'ANDAR'), {})
        bahar_layer = next((layer for layer in by_layer if layer['layer_type'] == 'BAHAR'), {})
        
        if main_layer.get('roi_pct', 0) > andar_layer.get('roi_pct', 0) and main_layer.get('roi_pct', 0) > bahar_layer.get('roi_pct', 0):
            insights['recommendations'].append("Main number strategy is outperforming digit bets")
        
        # Overall performance insight
        overall_roi = summary_data.get('overall', {}).get('overall_roi', 0)
        if overall_roi > 0:
            insights['opportunities'].append(f"System is profitable overall (+{overall_roi:.1f}%)")
        else:
            insights['warnings'].append(f"System is currently unprofitable ({overall_roi:.1f}%)")
        
        return insights
    
    def export_dashboard_data(self) -> bool:
        """Export formatted data for external dashboards"""
        try:
            current_perf = self.get_current_performance_summary()
            insights = self.generate_strategy_insights(current_perf.get('overall', {}))
            
            dashboard_data = {
                'performance': current_perf,
                'insights': insights,
                'export_time': datetime.now().isoformat()
            }

            # Save for external consumption
            dashboard_file = quant_paths.get_performance_logs_dir() / "dashboard_data.json"
            import json
            with open(dashboard_file, 'w') as f:
                # Normalize numpy/pandas scalar types to native Python for safe JSON serialization
                json.dump(dashboard_data, f, indent=2, default=to_native)

            print(f"💾 Dashboard data exported: {dashboard_file}")
            return True

        except Exception as e:
            error_details = str(e)
            failed_type = None
            if isinstance(e, TypeError):
                import re
                match = re.search(r"Object of type (.+?) is not JSON serializable", error_details)
                if match:
                    failed_type = match.group(1)
            if not failed_type:
                failed_type = type(e).__name__
            print(f"❌ Error exporting dashboard data ({failed_type}): {error_details}")
            return False

# Global instance for easy import
_pnl_summary = QuantPnLSummary()

# Module-level functions
def get_current_performance_summary() -> Dict:
    """Get current performance summary"""
    return _pnl_summary.get_current_performance_summary()

def generate_strategy_insights(summary_data: Dict) -> Dict:
    """Generate strategy insights"""
    return _pnl_summary.generate_strategy_insights(summary_data)

def export_dashboard_data() -> bool:
    """Export dashboard data"""
    return _pnl_summary.export_dashboard_data()

if __name__ == "__main__":
    print("📊 QUANT P&L SUMMARY UTILITIES")
    print("=" * 50)
    
    summary = QuantPnLSummary()
    
    try:
        perf_data = summary.get_current_performance_summary()
        print("✅ Current Performance Summary:")
        print(f"   Overall ROI: {perf_data.get('overall', {}).get('overall_roi', 0):.1f}%")
        print(f"   Recent 7d ROI: {perf_data.get('recent_7d', {}).get('roi', 0):.1f}%")
        print(f"   Winning Streak: {perf_data.get('recent_7d', {}).get('winning_streak', 0)} days")
        
        insights = summary.generate_strategy_insights(perf_data.get('overall', {}))
        print(f"\n💡 Strategy Insights:")
        for rec in insights.get('recommendations', [])[:3]:
            print(f"   📋 {rec}")
        
        success = summary.export_dashboard_data()
        if success:
            print("✅ Dashboard data exported successfully!")
            
    except Exception as e:
        print(f"❌ Error: {e}")