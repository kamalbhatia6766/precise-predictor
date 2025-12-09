# confidence_scorer.py
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

from quant_core.config_core import ROI_SLUMP_THRESHOLD
from quant_core.pnl_core import classify_slot_regime, load_pnl_snapshot

class ConfidenceScorer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.slot_perf = self._load_slot_perf_data()
        self.topn_profile = self._load_topn_roi_profile()
        self.pnl_snapshot = load_pnl_snapshot()

    def _load_slot_perf_data(self):
        perf_file = self.base_dir / "logs" / "performance" / "quant_reality_pnl.json"
        if not perf_file.exists():
            return {}
        try:
            data = json.loads(perf_file.read_text())
        except Exception:
            return {}
        slot_map = {}
        for entry in data.get("by_slot", []):
            slot = str(entry.get("slot", "")).upper()
            slot_map[slot] = {
                "pnl": entry.get("pnl"),
                "roi": entry.get("roi"),
            }
        return slot_map

    def _load_topn_roi_profile(self):
        path = self.base_dir / "logs" / "performance" / "topn_roi_profile.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _coerce_roi_by_n(self, roi_profile):
        raw_map = roi_profile.get("roi_by_N") if isinstance(roi_profile, dict) else {}
        roi_by_n = {}
        for key, val in (raw_map or {}).items():
            try:
                roi_by_n[int(key)] = val
            except Exception:
                continue
        for n in range(1, 6):
            key = f"Top{n}"
            if n not in roi_by_n and key in roi_profile:
                roi_by_n[n] = roi_profile.get(key)
        return roi_by_n
        
    def load_performance_data(self):
        """Load all performance data for confidence scoring"""
        data = {}
        perf_dir = self.base_dir / "logs" / "performance"
        
        # Load P&L history
        pnl_file = perf_dir / "bet_pnl_history.xlsx"
        if pnl_file.exists():
            data['pnl'] = pd.read_excel(pnl_file, sheet_name='day_level')
            data['pnl']['date'] = pd.to_datetime(data['pnl']['date']).dt.date
        
        # Load prediction log
        pred_log_file = self.base_dir / "logs" / "predictions" / "daily_prediction_log.xlsx"
        if pred_log_file.exists():
            data['pred_log'] = pd.read_excel(pred_log_file)
        
        # Load advisory data
        advisory_files = {
            'patterns': 'pattern_intelligence_summary.json',
            'fusion': 'adaptive_fusion_plan.json',
            'weights': 'smart_fusion_weights.json',
            'stakes': 'dynamic_stake_plan.json'
        }
        
        for key, filename in advisory_files.items():
            file_path = perf_dir / filename
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data[key] = json.load(f)
        
        return data
    
    def calculate_slot_confidence(self, slot, data):
        """Calculate confidence score (0-100) for a slot"""
        confidence_factors = {}

        roi_profile = self.topn_profile.get("per_slot_roi", {}).get(slot, {}) if self.topn_profile else {}
        roi_by_n = self._coerce_roi_by_n(roi_profile)
        slot_perf = self.slot_perf.get(slot, {})
        slot_roi = slot_perf.get("roi") if slot_perf else None
        slot_pnl = slot_perf.get("pnl") if slot_perf else None
        if slot_roi is None and roi_profile:
            slot_roi = roi_profile.get("Top5")
        
        # Factor 1: Recent P&L Performance (40% weight)
        recent_profit_score = self.calculate_recent_profit_score(slot, data)
        confidence_factors['recent_profit'] = recent_profit_score
        
        # Factor 2: Pattern Alignment (25% weight)
        pattern_score = self.calculate_pattern_score(slot, data)
        confidence_factors['pattern_alignment'] = pattern_score
        
        # Factor 3: Source Performance (20% weight)
        source_score = self.calculate_source_score(slot, data)
        confidence_factors['source_performance'] = source_score
        
        # Factor 4: Stake Recommendation (15% weight)
        stake_score = self.calculate_stake_score(slot, data)
        confidence_factors['stake_recommendation'] = stake_score
        
        # Calculate weighted confidence score
        weights = {
            'recent_profit': 0.40,
            'pattern_alignment': 0.25,
            'source_performance': 0.20,
            'stake_recommendation': 0.15
        }

        weighted_score = 0
        for factor, score in confidence_factors.items():
            if factor not in weights:
                continue
            weighted_score += score * weights[factor]

        # ROI/P&L guardrails
        if slot_roi is not None and slot_pnl is not None:
            if slot_roi <= -50 and slot_pnl < 0:
                weighted_score = min(weighted_score, 40)
            elif -50 < slot_roi < 50:
                weighted_score = min(weighted_score, 60)
            elif slot_roi >= 50 and slot_pnl > 0:
                weighted_score = max(weighted_score, 70)

        if roi_profile and all((roi_profile.get(k) == -100.0 for k in ("Top1", "Top5", "Top10") if roi_profile.get(k) is not None)):
            weighted_score = min(weighted_score, 55)

        cool_off_active = self._should_cool_off(slot, roi_by_n, slot_roi, slot_pnl)
        if cool_off_active:
            weighted_score = min(weighted_score, 40)
        confidence_factors["cool_off"] = cool_off_active

        return min(100, max(0, weighted_score)), confidence_factors

    def _should_cool_off(self, slot, roi_by_n, slot_roi, slot_pnl) -> bool:
        """
        Apply a hard cool-off when recent ROI is deeply negative for all Top1..Top5
        picks and the slot is already in a SLUMP regime.
        """

        if not roi_by_n:
            return False

        top_1_to_5 = [roi_by_n.get(n) for n in range(1, 6)]
        if any(val is None for val in top_1_to_5):
            return False

        if not all(val <= -99 for val in top_1_to_5):
            return False

        regime = classify_slot_regime(self.pnl_snapshot, slot)
        if regime != "SLUMP":
            return False

        if slot_roi is not None and slot_roi > ROI_SLUMP_THRESHOLD:
            return False

        if slot_pnl is not None and slot_pnl >= 0:
            return False

        return True
    
    def calculate_recent_profit_score(self, slot, data):
        """Calculate score based on recent profit performance"""
        if 'pnl' not in data:
            return 50
            
        pnl_df = data['pnl']
        profit_col = f'profit_{slot.lower()}'
        
        if profit_col not in pnl_df.columns:
            return 50
        
        # Last 7 days performance
        recent_data = pnl_df.tail(7)
        slot_profits = recent_data[profit_col]
        
        if len(slot_profits) == 0:
            return 50
        
        # Calculate score based on profit trend
        total_profit = slot_profits.sum()
        winning_days = len(slot_profits[slot_profits > 0])
        total_days = len(slot_profits)
        win_rate = (winning_days / total_days) * 100 if total_days > 0 else 0
        
        # Score calculation
        profit_score = min(100, max(0, 50 + (total_profit / 10)))
        win_rate_score = win_rate
        
        return (profit_score * 0.6 + win_rate_score * 0.4)
    
    def calculate_pattern_score(self, slot, data):
        """Calculate score based on pattern alignment"""
        if 'patterns' not in data:
            return 50
            
        patterns_data = data['patterns']
        
        # Check if slot has strong pattern alignment
        slot_pattern_score = 50
        
        # Look for slot-specific pattern performance
        if 'top_recommendations' in patterns_data:
            recommendations = patterns_data['top_recommendations']
            if recommendations:
                # If there are strong pattern recommendations, increase score
                slot_pattern_score = min(100, 60 + (len(recommendations) * 10))
        
        return slot_pattern_score
    
    def calculate_source_score(self, slot, data):
        """Calculate score based on source performance"""
        if 'fusion' not in data or 'slot_optimization' not in data['fusion']:
            return 50
            
        slot_optimization = data['fusion']['slot_optimization']
        slot_data = slot_optimization.get(slot, {})
        
        performance_tier = slot_data.get('performance_tier', 'UNKNOWN')
        preferred_source = slot_data.get('preferred_source', 'fusion')
        
        # Score based on performance tier
        tier_scores = {
            'HIGH': 85,
            'MEDIUM': 65,
            'LOW': 35,
            'UNKNOWN': 50
        }
        
        return tier_scores.get(performance_tier, 50)
    
    def calculate_stake_score(self, slot, data):
        """Calculate score based on stake recommendations"""
        if 'stakes' not in data or 'stakes' not in data['stakes']:
            return 50
            
        stakes = data['stakes']['stakes']
        slot_stake = stakes.get(slot, 55)
        
        # Higher stake = higher confidence
        if slot_stake >= 80:
            return 85
        elif slot_stake >= 60:
            return 70
        elif slot_stake <= 30:
            return 35
        elif slot_stake <= 45:
            return 55
        else:
            return 65
    
    def generate_confidence_reasons(self, slot, confidence_score, factors):
        """Generate human-readable reasons for confidence score"""
        reasons = []
        
        # Recent profit reasons
        profit_score = factors['recent_profit']
        if profit_score >= 70:
            reasons.append("strong_recent_profit")
        elif profit_score <= 40:
            reasons.append("weak_recent_performance")
        
        # Pattern reasons
        pattern_score = factors['pattern_alignment']
        if pattern_score >= 70:
            reasons.append("high_pattern_alignment")
        elif pattern_score <= 40:
            reasons.append("low_pattern_support")
        
        # Source reasons
        source_score = factors['source_performance']
        if source_score >= 75:
            reasons.append("optimal_source_performance")
        elif source_score <= 45:
            reasons.append("suboptimal_source")
        
        # Stake reasons
        stake_score = factors['stake_recommendation']
        if stake_score >= 75:
            reasons.append("high_stake_confidence")
        elif stake_score <= 45:
            reasons.append("low_stake_confidence")
        
        # Overall confidence level
        if confidence_score >= 80:
            reasons.append("very_high_confidence")
        elif confidence_score >= 65:
            reasons.append("high_confidence")
        elif confidence_score <= 45:
            reasons.append("low_confidence")
        else:
            reasons.append("medium_confidence")

        if factors.get("cool_off"):
            reasons.append("cool_off_active")

        return reasons
    
    def save_confidence_data(self, confidence_data):
        """Save confidence scores and report"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON data
        json_output = {
            'timestamp': datetime.now().isoformat(),
            'confidence_scores': confidence_data
        }
        
        json_file = output_dir / "prediction_confidence.json"
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2)
        
        # Save Excel report
        excel_data = []
        for slot, data in confidence_data.items():
            excel_data.append({
                'slot': slot,
                'confidence_score': data['confidence_score'],
                'recent_profit_score': data['factors']['recent_profit'],
                'pattern_score': data['factors']['pattern_alignment'],
                'source_score': data['factors']['source_performance'],
                'stake_score': data['factors']['stake_recommendation'],
                'reasons': ', '.join(data['reasons']),
                'confidence_level': self.get_confidence_level(
                    data['confidence_score'], data['factors'].get('cool_off', False)
                )
            })
        
        excel_file = output_dir / "prediction_confidence_report.xlsx"
        pd.DataFrame(excel_data).to_excel(excel_file, index=False)
        
        print(f"💾 Confidence data saved to: {json_file}")
        print(f"💾 Confidence report saved to: {excel_file}")
        
        return json_file, excel_file
    
    def get_confidence_level(self, score, cool_off: bool = False):
        """Convert numeric score to confidence level"""
        if cool_off:
            return "COOL_OFF"
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
    
    def print_confidence_report(self, confidence_data):
        """Print confidence report to console"""
        print("\n" + "="*80)
        print("🎯 PREDICTION CONFIDENCE SCORER - SLOT CONFIDENCE ANALYSIS")
        print("="*80)
        
        print(f"\n📊 CONFIDENCE SCORES (0-100):")
        print("-" * 50)
        
        for slot, data in confidence_data.items():
            score = data['confidence_score']
            cool_off = data['factors'].get('cool_off', False)
            level = self.get_confidence_level(score, cool_off)
            level_icon = (
                "🧊" if level == "COOL_OFF" else "🚀" if level == "VERY_HIGH" else "✅" if level == "HIGH" else "⚡" if level == "MEDIUM" else "⚠️" if level == "LOW" else "🔴"
            )
            
            print(f"   {slot}: {level_icon} {score:3.0f}/100 ({level:10})")
            
            # Show key reasons
            main_reasons = data['reasons'][:3]
            print(f"        Reasons: {', '.join(main_reasons)}")
        
        print(f"\n💡 CONFIDENCE BREAKDOWN:")
        print("-" * 40)
        high_conf_slots = [s for s, d in confidence_data.items() if d['confidence_score'] >= 65]
        low_conf_slots = [s for s, d in confidence_data.items() if d['confidence_score'] <= 45]
        
        if high_conf_slots:
            print(f"   🎯 High Confidence Slots: {', '.join(high_conf_slots)}")
        if low_conf_slots:
            print(f"   ⚠️  Low Confidence Slots: {', '.join(low_conf_slots)}")
    
    def run(self):
        """Main execution"""
        print("🎯 PREDICTION CONFIDENCE SCORER - Calculating slot confidence...")
        
        # Load performance data
        data = self.load_performance_data()
        
        if not data:
            print("❌ No performance data found")
            return False
        
        # Calculate confidence for each slot
        confidence_data = {}
        for slot in self.slots:
            confidence_score, factors = self.calculate_slot_confidence(slot, data)
            reasons = self.generate_confidence_reasons(slot, confidence_score, factors)
            
            confidence_data[slot] = {
                'confidence_score': confidence_score,
                'factors': factors,
                'reasons': reasons
            }
        
        # Save and display results
        self.save_confidence_data(confidence_data)
        self.print_confidence_report(confidence_data)
        
        return True

def main():
    scorer = ConfidenceScorer()
    success = scorer.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())