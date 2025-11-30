# pattern_intelligence_enhanced.py - UPDATED WITH GOLDEN INTEGRATION
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class PatternIntelligenceEnhanced:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.scripts = [f"SCR{i}" for i in range(1, 10)]
        
    def load_hit_memory(self):
        """Load enhanced hit memory with time-shift tracking"""
        memory_file = self.base_dir / "logs" / "performance" / "script_hit_memory.xlsx"
        
        if not memory_file.exists():
            print("❌ No script_hit_memory.xlsx found")
            return None
        
        try:
            df = pd.read_excel(memory_file)
            print(f"✅ Loaded {len(df)} hit records")
            return df
        except Exception as e:
            print(f"❌ Error loading hit memory: {e}")
            return None

    def load_golden_insights(self):
        """Load golden block insights for pattern enhancement"""
        golden_file = self.base_dir / "logs" / "performance" / "golden_block_insights.json"
        
        if golden_file.exists():
            try:
                with open(golden_file, 'r') as f:
                    insights = json.load(f)
                print("✅ Loaded golden block insights")
                return insights
            except Exception as e:
                print(f"⚠️  Error loading golden insights: {e}")
        
        return None

    def analyze_cross_slot_patterns(self, hit_memory_df):
        """Enhanced cross-slot pattern analysis"""
        cross_patterns = {}
        
        # Filter for cross-slot hits
        cross_hits = hit_memory_df[
            hit_memory_df['hit_family'].isin(['CROSS_SAME_DAY', 'CROSS_NEXT_DAY', 'CROSS_PREV_DAY'])
        ]
        
        # Count cross-slot pairs
        for _, hit in cross_hits.iterrows():
            pair_key = f"{hit['predicted_slot']}→{hit['real_slot']}"
            if pair_key not in cross_patterns:
                cross_patterns[pair_key] = {
                    'hits': 0,
                    'total_rank': 0,
                    'scripts': set(),
                    'time_shifts': defaultdict(int)
                }
            
            cross_patterns[pair_key]['hits'] += 1
            cross_patterns[pair_key]['total_rank'] += hit['rank']
            cross_patterns[pair_key]['scripts'].add(hit['script'])
            cross_patterns[pair_key]['time_shifts'][hit['time_shift']] += 1
        
        # Calculate averages and format
        enhanced_patterns = {}
        for pair, data in cross_patterns.items():
            avg_rank = data['total_rank'] / data['hits'] if data['hits'] > 0 else 0
            enhanced_patterns[pair] = {
                'hits': data['hits'],
                'avg_rank': round(avg_rank, 1),
                'script_count': len(data['scripts']),
                'time_shift_distribution': dict(data['time_shifts']),
                'strength_score': round(data['hits'] * (1 / max(1, avg_rank)), 2)
            }
        
        return dict(sorted(enhanced_patterns.items(), key=lambda x: x[1]['strength_score'], reverse=True))

    def analyze_script_effectiveness(self, hit_memory_df):
        """Enhanced script effectiveness analysis"""
        script_stats = {}
        
        for script in self.scripts:
            script_hits = hit_memory_df[hit_memory_df['script'] == script]
            
            if not script_hits.empty:
                total_hits = len(script_hits)
                direct_hits = len(script_hits[script_hits['hit_family'] == 'DIRECT'])
                cross_hits = len(script_hits[script_hits['hit_family'].isin(['CROSS_SAME_DAY', 'CROSS_NEXT_DAY', 'CROSS_PREV_DAY'])])
                
                # Calculate effectiveness scores
                direct_effectiveness = (direct_hits / total_hits * 100) if total_hits > 0 else 0
                cross_effectiveness = (cross_hits / total_hits * 100) if total_hits > 0 else 0
                overall_effectiveness = (total_hits / len(script_hits['date'].unique()) * 100) if len(script_hits['date'].unique()) > 0 else 0
                
                # Calculate average rank (lower is better)
                avg_rank = script_hits['rank'].mean() if not script_hits.empty else 0
                
                script_stats[script] = {
                    'total_hits': total_hits,
                    'direct_hits': direct_hits,
                    'cross_hits': cross_hits,
                    'direct_effectiveness': round(direct_effectiveness, 1),
                    'cross_effectiveness': round(cross_effectiveness, 1),
                    'overall_effectiveness': round(overall_effectiveness, 1),
                    'avg_rank': round(avg_rank, 1),
                    'performance_score': round(overall_effectiveness / max(1, avg_rank), 2)
                }
        
        return dict(sorted(script_stats.items(), key=lambda x: x[1]['performance_score'], reverse=True))

    def analyze_time_patterns(self, hit_memory_df):
        """Enhanced time pattern analysis"""
        time_patterns = {}
        
        # Convert dates to proper format
        hit_memory_df['date'] = pd.to_datetime(hit_memory_df['date']).dt.date
        hit_memory_df['day_of_week'] = pd.to_datetime(hit_memory_df['date']).dt.day_name()
        
        # Analyze by slot
        for slot in self.slots:
            slot_hits = hit_memory_df[hit_memory_df['real_slot'] == slot]
            if not slot_hits.empty:
                # Day of week analysis
                day_counts = slot_hits['day_of_week'].value_counts()
                best_day = day_counts.index[0] if not day_counts.empty else "Unknown"
                best_day_hits = day_counts.iloc[0] if not day_counts.empty else 0
                
                # Time shift analysis
                time_shift_counts = slot_hits['time_shift'].value_counts()
                
                time_patterns[slot] = {
                    'best_day': best_day,
                    'best_day_hits': int(best_day_hits),
                    'day_distribution': day_counts.to_dict(),
                    'time_shift_distribution': time_shift_counts.to_dict(),
                    'total_hits': len(slot_hits),
                    'hit_rate': round(len(slot_hits) / len(slot_hits['date'].unique()) * 100, 1) if len(slot_hits['date'].unique()) > 0 else 0
                }
        
        return time_patterns

    def generate_adaptive_pattern_packs(self, golden_insights):
        """Generate adaptive pattern packs combining base and golden patterns"""
        adaptive_packs = {
            "timestamp": datetime.now().isoformat(),
            "source": "PatternIntelligenceEnhanced + GoldenBlockAnalyzer",
            "tens_core_base": [9, 7, 4],  # From existing pattern analysis
            "ones_core_base": [4, 2, 8],   # From existing pattern analysis
            "tens_core_golden": [],
            "ones_core_golden": [],
            "hero_numbers": [],
            "cross_slot_pairs_top": [],
            "boost_scripts": [],
            "time_boost_slots": {},
            "notes": "Auto-generated adaptive pattern packs"
        }
        
        # Integrate golden insights if available
        if golden_insights:
            # Add golden digits
            adaptive_packs["tens_core_golden"] = golden_insights.get('digits', {}).get('tens', [])[:3]
            adaptive_packs["ones_core_golden"] = golden_insights.get('digits', {}).get('ones', [])[:3]
            
            # Add hero numbers
            adaptive_packs["hero_numbers"] = golden_insights.get('digits', {}).get('hero_numbers', [])[:5]
            
            # Add top cross-slot patterns
            cross_pairs = golden_insights.get('cross_slot_pairs', {})
            adaptive_packs["cross_slot_pairs_top"] = list(cross_pairs.keys())[:5] if cross_pairs else []
            
            # Add boost scripts
            scripts = golden_insights.get('scripts', {})
            if scripts:
                top_scripts = sorted(scripts.items(), key=lambda x: x[1].get('golden_score', 0), reverse=True)[:3]
                adaptive_packs["boost_scripts"] = [script for script, _ in top_scripts]
            
            # Add time boosts
            day_patterns = golden_insights.get('day_of_week', {})
            for slot, pattern in day_patterns.items():
                adaptive_packs["time_boost_slots"][slot] = {
                    'best_day': pattern.get('best_day'),
                    'boost_factor': 1.1  # 10% boost on best days
                }
        
        return adaptive_packs

    def generate_enhanced_config(self, cross_patterns, script_stats, time_patterns, adaptive_packs):
        """Generate enhanced pattern intelligence configuration"""
        config = {
            "timestamp": datetime.now().isoformat(),
            "s40_enabled": True,
            "digit_packs_enabled": True,
            "memory_bonus_enabled": True,
            "cross_slot_boost": {},
            "script_weights": {},
            "time_awareness": {},
            "digit_preferences": {},
            "adaptive_packs_integrated": True,
            "pattern_weights": {
                "s40_bonus": 0.20,
                "digit_pack_bonus": 0.05,
                "max_pattern_bonus": 0.50,
                "cross_slot_bonus": 0.15,
                "golden_digit_boost": 0.08,
                "hero_number_boost": 0.10,
                "time_awareness_boost": 0.05
            }
        }
        
        # Add cross-slot boosts
        for pattern, data in list(cross_patterns.items())[:10]:  # Top 10 patterns
            predicted_slot, real_slot = pattern.split('→')
            boost_key = f"{predicted_slot}_{real_slot}"
            config["cross_slot_boost"][boost_key] = min(0.3, data['strength_score'] * 0.05)
        
        # Add script weights
        for script, stats in list(script_stats.items())[:5]:  # Top 5 scripts
            config["script_weights"][script] = min(2.0, 1.0 + (stats['performance_score'] * 0.1))
        
        # Add time awareness
        for slot, patterns in time_patterns.items():
            config["time_awareness"][slot] = {
                "preferred_day": patterns['best_day'],
                "boost_factor": 1.1,
                "hit_rate": patterns['hit_rate']
            }
        
        # Add adaptive packs reference
        config["adaptive_packs"] = adaptive_packs
        
        return config

    def run_enhanced_analysis(self):
        """Run complete enhanced pattern analysis"""
        print("🧠 PATTERN INTELLIGENCE ENHANCED - Analyzing Cross-Slot Patterns...")
        
        # Step 1: Load data
        hit_memory_df = self.load_hit_memory()
        if hit_memory_df is None:
            return False
            
        golden_insights = self.load_golden_insights()
        
        # Step 2: Run enhanced analyses
        print("📊 Analyzing cross-slot patterns...")
        cross_patterns = self.analyze_cross_slot_patterns(hit_memory_df)
        
        print("📊 Analyzing script effectiveness...")
        script_stats = self.analyze_script_effectiveness(hit_memory_df)
        
        print("📊 Analyzing time patterns...")
        time_patterns = self.analyze_time_patterns(hit_memory_df)
        
        print("📊 Generating adaptive pattern packs...")
        adaptive_packs = self.generate_adaptive_pattern_packs(golden_insights)
        
        print("📊 Generating enhanced configuration...")
        enhanced_config = self.generate_enhanced_config(cross_patterns, script_stats, time_patterns, adaptive_packs)
        
        # Step 3: Save outputs
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save enhanced patterns
        enhanced_file = output_dir / "enhanced_pattern_analysis.json"
        with open(enhanced_file, 'w') as f:
            json.dump({
                "cross_patterns": cross_patterns,
                "script_stats": script_stats, 
                "time_patterns": time_patterns,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Save adaptive packs
        adaptive_file = output_dir / "adaptive_pattern_packs.json"
        with open(adaptive_file, 'w') as f:
            json.dump(adaptive_packs, f, indent=2)
        
        # Save enhanced config
        config_file = output_dir / "pattern_intelligence_config.json"
        with open(config_file, 'w') as f:
            json.dump(enhanced_config, f, indent=2)
        
        print(f"💾 Enhanced patterns saved: {enhanced_file}")
        print(f"💾 Pattern config saved: {config_file}")
        print(f"💾 Adaptive packs saved: {adaptive_file}")
        
        # Step 4: Print enhanced summary
        self.print_enhanced_summary(cross_patterns, script_stats, time_patterns, adaptive_packs)
        
        return True

    def print_enhanced_summary(self, cross_patterns, script_stats, time_patterns, adaptive_packs):
        """Print enhanced analysis summary"""
        print("\n" + "=" * 80)
        print("🧠 PATTERN INTELLIGENCE ENHANCED - Cross-Slot Pattern Analysis")
        print("=" * 80)
        
        print(f"\n📊 HIT ANALYSIS SUMMARY:")
        total_hits = sum(stats['total_hits'] for stats in script_stats.values())
        print(f"   Total hits in memory: {total_hits}")
        print(f"   Scripts analyzed: {len(script_stats)}")
        
        print(f"\n🎯 TOP CROSS-SLOT PATTERNS:")
        for pattern, data in list(cross_patterns.items())[:5]:
            print(f"   {pattern}: {data['hits']} hits (avg rank: {data['avg_rank']})")
        
        print(f"\n🏆 BEST PERFORMING SCRIPTS:")
        for script, stats in list(script_stats.items())[:3]:
            print(f"   {script}: {stats['overall_effectiveness']}% effectiveness ({stats['direct_effectiveness']}% direct, {stats['cross_effectiveness']}% cross)")
        
        print(f"\n⏰ TIME PATTERNS:")
        for slot, patterns in time_patterns.items():
            print(f"   {slot}: Best on {patterns['best_day']} ({patterns['best_day_hits']} hits)")
        
        print(f"\n💡 ADAPTIVE PATTERN RECOMMENDATIONS:")
        if adaptive_packs.get('hero_numbers'):
            print(f"   1. Focus on hero numbers: {', '.join(map(str, adaptive_packs['hero_numbers'][:3]))}")
        if adaptive_packs.get('cross_slot_pairs_top'):
            print(f"   2. Monitor cross-slot: {adaptive_packs['cross_slot_pairs_top'][0]}")
        if adaptive_packs.get('boost_scripts'):
            print(f"   3. Boost scripts: {', '.join(adaptive_packs['boost_scripts'][:2])}")
        
        print(f"\n✅ Enhanced pattern analysis completed!")

def main():
    analyzer = PatternIntelligenceEnhanced()
    success = analyzer.run_enhanced_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())