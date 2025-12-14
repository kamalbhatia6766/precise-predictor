# strategy_recommendation_engine.py - UPDATED
# strategy_recommendation_engine.py - REALITY-DRIVEN STRATEGY RECOMMENDATION
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings

warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_paths


class StrategyRecommendationEngine:
    MIN_S40_HIT_RATE = 0.30
    MIN_S40_SAMPLE = 30
    MIN_BASE_ROI = 0.10
    S40_OUTPERFORM_MARGIN = 0.05

    def __init__(self):
        self.base_dir = quant_paths.get_project_root()
        self.strategy_families = ["S40", "PACK_164950", "PACK_2DIGIT", "PACK_3DIGIT", "PACK_4DIGIT", "PACK_5DIGIT", "PACK_6DIGIT"]
        
    def load_pattern_intelligence(self):
        """Load pattern intelligence data"""
        pattern_file = quant_paths.get_performance_logs_dir() / "pattern_intelligence.json"
        
        if not pattern_file.exists():
            print("❌ No pattern_intelligence.json found")
            return None
        
        try:
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
            print("✅ Loaded pattern intelligence data")
            return pattern_data
        except Exception as e:
            print(f"❌ Error loading pattern intelligence: {e}")
            return None
    
    def load_reality_performance(self):
        """Load reality performance data"""
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.json"

        if not pnl_file.exists():
            print("⚠️  No quant_reality_pnl.json found")
            return None

        try:
            with open(pnl_file, 'r') as f:
                performance_data = json.load(f)
            print("✅ Loaded reality performance data")
            return performance_data
        except Exception as e:
            print(f"⚠️  Error loading performance data: {e}")
            return None

    def load_reality_check_summary(self):
        """Load summary from reality_check_engine if available"""
        summary_file = quant_paths.get_performance_logs_dir() / "reality_check_summary.json"

        if not summary_file.exists():
            print("⚠️  No reality_check_summary.json found")
            return None

        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            print("✅ Loaded reality check summary")
            return summary_data
        except Exception as e:
            print(f"⚠️  Error loading reality check summary: {e}")
            return None

    def _normalize_percentage(self, value):
        """Convert percentage-like numbers to 0-1 floats."""
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None

        if value > 1:
            return value / 100.0
        return value

    def extract_s40_stats(self, pattern_data):
        """Extract S40 stats from pattern intelligence."""
        pattern_stats = pattern_data.get("pattern_stats", {}) if isinstance(pattern_data, dict) else {}
        s40_stats = pattern_stats.get("S40", {}) if isinstance(pattern_stats, dict) else {}

        hit_rate_raw = s40_stats.get("hit_rate")
        s40_hit_rate = self._normalize_percentage(hit_rate_raw)
        s40_hits = s40_stats.get("hits")

        return s40_hit_rate, s40_hits

    def extract_reality_metrics(self, performance_data, reality_summary):
        """Extract overall ROI and base ROI from available reality data."""
        overall_roi = None
        base_roi = None

        if isinstance(performance_data, dict):
            overall_section = performance_data.get("overall", {})
            overall_roi = self._normalize_percentage(overall_section.get("overall_roi"))

        if isinstance(reality_summary, dict):
            base_stats = reality_summary.get("strategy_totals", {}).get("BASE", {})
            base_roi = self._normalize_percentage(base_stats.get("roi_percent"))

        if base_roi is None:
            base_roi = overall_roi

        return base_roi, overall_roi
    
    def find_top_performing_family(self, pattern_data):
        """Find the top performing pattern family"""
        print("🔍 Analyzing pattern family performance...")

        pattern_stats = pattern_data.get('pattern_stats', {}) if isinstance(pattern_data, dict) else {}

        top_family = None
        top_hit_rate = -1

        for family in self.strategy_families:
            family_stats = pattern_stats.get(family, {}) if isinstance(pattern_stats, dict) else {}
            hit_rate = family_stats.get('hit_rate', 0)

            if hit_rate > top_hit_rate:
                top_family = family
                top_hit_rate = hit_rate

        # Fallback if no data
        if top_family is None:
            top_family = "S40"
            top_hit_rate = 0

        return top_family, top_hit_rate
    
    def map_family_to_strategy(self, top_family, top_hit_rate):
        """Map top performing family to strategy recommendation"""
        print("🎯 Mapping family performance to strategy...")

        if top_family == "S40":
            recommended_strategy = "STRAT_S40_BOOST"
        elif top_family == "PACK_164950":
            recommended_strategy = "STRAT_164950_CORE"
        elif top_family in ["PACK_3DIGIT", "PACK_4DIGIT"]:
            recommended_strategy = "STRAT_MID_PACK_FOCUS"
        else:
            recommended_strategy = "STRAT_BALANCED_CORE"

        return recommended_strategy

    def build_recommendation_payload(
        self,
        strategy,
        confidence,
        risk_mode,
        reason,
        top_family,
        top_hit_rate,
        metrics,
        note=None
    ):
        payload = {
            "timestamp": datetime.now().isoformat(),
            "recommended_strategy": strategy,
            "strategy": strategy,
            "confidence_level": confidence,
            "confidence": confidence,
            "risk_mode": risk_mode,
            "top_family": top_family,
            "top_family_hit_rate": round(top_hit_rate, 2) if top_hit_rate is not None else None,
            "source_files": ["pattern_intelligence.json", "quant_reality_pnl.json", "reality_check_summary.json"],
            "logic_version": "v2_reality_gated",
            "reason": reason,
            "metrics": metrics,
        }

        if note:
            payload["note"] = note

        return payload
    
    def save_strategy_recommendation(self, strategy_recommendation):
        """Save strategy recommendation to JSON"""
        output_file = quant_paths.get_performance_logs_dir() / "strategy_recommendation.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(strategy_recommendation, f, indent=2)

        print(f"💾 Strategy recommendation saved: {output_file}")
        return output_file
    
    def print_console_summary(self, strategy_recommendation):
        """Print console summary"""
        print("\n" + "="*60)
        print("🎯 STRATEGY RECOMMENDATION ENGINE – REALITY LINKED")
        print("="*60)
        
        strategy = strategy_recommendation.get('recommended_strategy')
        top_family = strategy_recommendation.get('top_family')
        hit_rate = strategy_recommendation.get('top_family_hit_rate') or 0

        confidence = strategy_recommendation.get('confidence_level', strategy_recommendation.get('confidence', 'LOW'))
        risk_mode = strategy_recommendation.get('risk_mode', 'DEFENSIVE')
        reason = strategy_recommendation.get('reason', 'No reason provided')
        metrics = strategy_recommendation.get('metrics', {})
        
        # Strategy descriptions
        strategy_descriptions = {
            "STRAT_S40_BOOST": "Focus on S40 numbers with boosted stakes",
            "STRAT_164950_CORE": "Prioritize 164950 family numbers", 
            "STRAT_MID_PACK_FOCUS": "Emphasize 3-4 digit pack coverage",
            "STRAT_BALANCED_CORE": "Balanced approach across all patterns"
        }
        
        description = strategy_descriptions.get(strategy, "Standard balanced strategy")
        
        print(f"📋 Recommended Strategy: {strategy}")
        print(f"   {description}")
        print(f"   Confidence : {confidence}")
        print(f"   Risk Mode  : {risk_mode}")
        print(f"🎯 Top Performing Family: {top_family} (Hit Rate: {hit_rate:.1f}%)")
        print(f"🔍 Reason: {reason}")

        print("📊 Reality:")
        base_roi = metrics.get("base_roi")
        overall_roi = metrics.get("overall_roi")
        if base_roi is not None:
            print(f"   BASE ROI    : {base_roi*100:.1f}%")
        if overall_roi is not None:
            print(f"   Overall ROI : {overall_roi*100:.1f}%")

        print("📊 S40:")
        s40_hit_rate = metrics.get("s40_hit_rate")
        s40_hits = metrics.get("s40_hits")
        if s40_hit_rate is not None:
            print(f"   Hit Rate    : {s40_hit_rate*100:.1f}%")
        if s40_hits is not None:
            print(f"   Hits        : {s40_hits}")

        # Additional context
        print(f"\n💡 Implementation:")
        if strategy == "STRAT_S40_BOOST":
            print("   - Apply 20% stake boost to S40 numbers")
            print("   - Focus pattern intelligence on S40 performance")
        elif strategy == "STRAT_164950_CORE":
            print("   - Prioritize 164950 family in number selection")
            print("   - Adjust pack weights for 014569 digits")
        elif strategy == "STRAT_MID_PACK_FOCUS":
            print("   - Boost 3-4 digit pack coverage")
            print("   - Adjust dynamic stake allocation for mid-packs")
        else:
            print("   - Maintain balanced stake allocation")
            print("   - Use standard pattern weights")
        
        print("="*60)

    def run_strategy_analysis(self):
        """Run complete strategy recommendation analysis"""
        print("🚀 STRATEGY RECOMMENDATION ENGINE - REALITY-DRIVEN")
        print("="*50)

        # Step 1: Load pattern intelligence
        pattern_data = self.load_pattern_intelligence()
        if pattern_data is None:
            print("⚠️ Pattern intelligence missing; emitting safe default recommendation")
            strategy_recommendation = self.build_recommendation_payload(
                strategy="STRAT_S40_BOOST",
                confidence="LOW",
                risk_mode="DEFENSIVE",
                reason="Pattern data unavailable; defaulting to legacy S40 boost",
                top_family="S40",
                top_hit_rate=0,
                metrics={},
                note="Pattern-only recommendation; reality data unavailable",
            )
            self.save_strategy_recommendation(strategy_recommendation)
            self.print_console_summary(strategy_recommendation)
            return True

        # Step 2: Load reality performance
        performance_data = self.load_reality_performance()
        reality_summary = self.load_reality_check_summary()

        if performance_data is None or reality_summary is None:
            print("⚠️  Reality data missing - falling back to pattern-only recommendation")
            top_family, top_hit_rate = self.find_top_performing_family(pattern_data)
            strategy_recommendation = self.build_recommendation_payload(
                strategy="STRAT_S40_BOOST",
                confidence="LOW",
                risk_mode="DEFENSIVE",
                reason="Pattern-only recommendation; reality data unavailable",
                top_family=top_family,
                top_hit_rate=top_hit_rate,
                metrics={},
                note="Pattern-only recommendation; reality data unavailable",
            )
            self.save_strategy_recommendation(strategy_recommendation)
            self.print_console_summary(strategy_recommendation)
            return True

        # Step 3: Extract stats
        top_family, top_hit_rate = self.find_top_performing_family(pattern_data)
        s40_hit_rate, s40_hits = self.extract_s40_stats(pattern_data)
        base_roi, overall_roi = self.extract_reality_metrics(performance_data, reality_summary)

        metrics = {
            "s40_hit_rate": s40_hit_rate,
            "s40_hits": s40_hits,
            "base_roi": base_roi,
            "overall_roi": overall_roi,
        }

        s40_ok = (
            s40_hit_rate is not None
            and s40_hit_rate >= self.MIN_S40_HIT_RATE
            and (s40_hits or 0) >= self.MIN_S40_SAMPLE
        )

        base_ok = base_roi is not None and base_roi >= self.MIN_BASE_ROI

        if not base_ok:
            strategy = "STRAT_BASE"
            confidence = "LOW"
            risk_mode = "DEFENSIVE"
            reason = "Base ROI not strong; avoid aggressive overlays"
        elif s40_ok:
            strategy = "STRAT_S40_BOOST"
            confidence = "HIGH" if (s40_hit_rate >= 0.5 and (s40_hits or 0) >= self.MIN_S40_SAMPLE * 2) else "MEDIUM"
            risk_mode = "DEFENSIVE"
            reason = "S40 pattern strong with sufficient sample and positive base ROI"
        else:
            strategy = "STRAT_BASE"
            confidence = "MEDIUM"
            risk_mode = "DEFENSIVE"
            reason = "BASE performing well, S40 conditions not met"

        strategy_recommendation = self.build_recommendation_payload(
            strategy=strategy,
            confidence=confidence,
            risk_mode=risk_mode,
            reason=reason,
            top_family=top_family,
            top_hit_rate=top_hit_rate,
            metrics=metrics,
        )

        self.save_strategy_recommendation(strategy_recommendation)
        self.print_console_summary(strategy_recommendation)

        print("✅ Strategy recommendation completed!")
        return True

def main():
    strategy_engine = StrategyRecommendationEngine()
    success = strategy_engine.run_strategy_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
