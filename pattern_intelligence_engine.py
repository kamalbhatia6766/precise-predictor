# pattern_intelligence_engine.py - UPDATED WITH CENTRAL PACK REGISTRY & DEFENSIVE LOGIC
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
from utils_2digit import is_valid_2d_number, to_2d_str
import warnings
warnings.filterwarnings('ignore')

# 🆕 Import central pack registry
import pattern_packs

class PatternIntelligenceEngine:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
        # 🆕 Use central pack registry instead of local definitions
        self.S40 = pattern_packs.S40
        self.PACK_164950_FAMILY = pattern_packs.PACK_164950_FAMILY
        
        # Keep sequence families (these are different from digit packs)
        self.PACK_3_FAMILIES = pattern_packs.PACK_3_FAMILIES
        self.PACK_4_FAMILIES = pattern_packs.PACK_4_FAMILIES
        
        # 🆕 Define core families for analysis (compatible with strategy engine)
        self.CORE_FAMILIES = [
            "S40", 
            "PACK_164950",
            "PACK_00_19", 
            "PACK_20_39", 
            "PACK_40_59", 
            "PACK_60_79", 
            "PACK_80_99"
        ]

    def load_hit_memory(self):
        """Load hit memory data - DEFENSIVE VERSION"""
        memory_file = self.base_dir / "logs" / "performance" / "script_hit_memory.xlsx"
        
        if not memory_file.exists():
            print("❌ No script_hit_memory.xlsx found")
            return None
        
        try:
            df = pd.read_excel(memory_file)
            print(f"✅ Loaded {len(df)} hit records")
            
            # 🆕 DEFENSIVE: Check required columns
            required_columns = ['real_number']
            for col in required_columns:
                if col not in df.columns:
                    print(f"❌ Required column '{col}' missing in hit memory")
                    return None
                    
            return df
        except Exception as e:
            print(f"❌ Error loading hit memory: {e}")
            return None

    def safe_family_check(self, number, family):
        """🆕 Safe family membership check with error handling"""
        try:
            if family == "S40":
                return pattern_packs.is_s40(number)
            elif family == "PACK_164950":
                return pattern_packs.is_164950_family(number)
            else:
                # For coarse packs and digit packs
                tags = pattern_packs.get_digit_pack_tags(number)
                return family in tags
        except Exception as e:
            print(f"⚠️  Error checking family {family} for number {number}: {e}")
            return False

    def analyze_pattern_performance(self, hit_memory_df):
        """Analyze pattern performance using central pack registry - DEFENSIVE VERSION"""
        pattern_stats = {}
        
        # 🆕 Initialize stats for core families
        for family in self.CORE_FAMILIES:
            pattern_stats[family] = {
                'hits': 0,
                'opportunities': 0,
                'hit_rate': 0.0
            }
        
        # 🆕 DEFENSIVE: Check if we have valid data
        if hit_memory_df is None or hit_memory_df.empty:
            print("⚠️  No hit memory data to analyze")
            return pattern_stats
        
        # Analyze hits by pattern family
        total_hits = len(hit_memory_df)
        
        for _, hit in hit_memory_df.iterrows():
            number = hit['real_number']
            
            # 🆕 DEFENSIVE: Skip invalid numbers
            if pd.isna(number):
                continue
                
            try:
                number = int(float(number))  # Ensure integer
            except (ValueError, TypeError):
                continue
            
            # 🆕 Check all core families using safe method
            for family in self.CORE_FAMILIES:
                if self.safe_family_check(number, family):
                    pattern_stats[family]['hits'] += 1
        
        # Calculate hit rates
        for family in pattern_stats:
            if total_hits > 0:
                pattern_stats[family]['hit_rate'] = (pattern_stats[family]['hits'] / total_hits) * 100
            pattern_stats[family]['opportunities'] = total_hits
        
        return pattern_stats

    def generate_pattern_weights(self, pattern_stats):
        """Generate pattern weights based on performance - DEFENSIVE VERSION"""
        weights = {}
        
        for family, stats in pattern_stats.items():
            hit_rate = stats['hit_rate']
            
            # 🆕 DEFENSIVE: Handle zero or very low hit rates
            if hit_rate > 10:
                weights[family] = 1.5
            elif hit_rate > 5:
                weights[family] = 1.2
            elif hit_rate > 2:
                weights[family] = 1.0
            elif hit_rate > 0:
                weights[family] = 0.8
            else:
                weights[family] = 0.5  # Very conservative for zero hit rate
        
        return weights

    def save_pattern_intelligence(self, pattern_stats, pattern_weights):
        """Save pattern intelligence data"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 🆕 Include central pack universe stats
        try:
            pack_stats = pattern_packs.get_pack_universe_stats()
        except Exception as e:
            print(f"⚠️  Error getting pack universe stats: {e}")
            pack_stats = {"error": "Could not load pack stats"}
        
        # Save pattern stats
        stats_file = output_dir / "pattern_intelligence.json"
        output_data = {
            'pattern_stats': pattern_stats,
            'pattern_weights': pattern_weights,
            'timestamp': datetime.now().isoformat(),
            'pack_universe_stats': pack_stats,
            'core_families_analyzed': self.CORE_FAMILIES,
            'total_hits_processed': sum(stats['hits'] for stats in pattern_stats.values())
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Pattern intelligence saved: {stats_file}")
            return stats_file
        except Exception as e:
            print(f"❌ Error saving pattern intelligence: {e}")
            return None

    def run_analysis(self):
        """Run complete pattern intelligence analysis"""
        print("🧠 PATTERN INTELLIGENCE ENGINE - Central Pack Registry")
        print("=" * 60)
        
        # 🆕 Print central pack usage
        try:
            stats = pattern_packs.get_pack_universe_stats()
            print(f"🧮 Using central pack registry: {stats['total_digit_packs']} digit packs")
            print(f"   Special packs: S40={len(self.S40)}, 164950={36} numbers")  # 6x6=36 for 164950
        except Exception as e:
            print(f"⚠️  Could not load pack stats: {e}")
        
        # Load hit memory
        hit_memory_df = self.load_hit_memory()
        if hit_memory_df is None:
            print("❌ Cannot proceed without hit memory data")
            return False
        
        # Analyze pattern performance
        print("📊 Analyzing pattern performance...")
        pattern_stats = self.analyze_pattern_performance(hit_memory_df)
        
        # Generate pattern weights
        print("⚖️ Generating pattern weights...")
        pattern_weights = self.generate_pattern_weights(pattern_stats)
        
        # Save results
        output_file = self.save_pattern_intelligence(pattern_stats, pattern_weights)
        
        if output_file:
            # Print summary
            self.print_analysis_summary(pattern_stats, pattern_weights)
            print(f"✅ Pattern intelligence analysis completed!")
            return True
        else:
            print("❌ Pattern intelligence analysis failed!")
            return False

    def print_analysis_summary(self, pattern_stats, pattern_weights):
        """Print analysis summary"""
        print("\n" + "="*60)
        print("🎯 PATTERN INTELLIGENCE SUMMARY")
        print("="*60)
        
        total_hits = sum(stats['hits'] for stats in pattern_stats.values())
        print(f"📊 Total hits analyzed: {total_hits}")
        
        print(f"\n📈 PATTERN PERFORMANCE (Top 8):")
        # Sort by hit rate, filter families with at least 1 hit
        active_families = [(fam, stats) for fam, stats in pattern_stats.items() if stats['hits'] > 0]
        sorted_stats = sorted(active_families, key=lambda x: x[1]['hit_rate'], reverse=True)[:8]
        
        if not sorted_stats:
            print("   No pattern hits found in the data")
            return
            
        for family, stats in sorted_stats:
            hit_rate = stats['hit_rate']
            trend = "🟢" if hit_rate > 5 else "🟡" if hit_rate > 2 else "🔴"
            weight = pattern_weights.get(family, 1.0)
            hits = stats['hits']
            print(f"   {trend} {family:12}: {hit_rate:5.1f}% ({hits:2} hits) weight: {weight:.1f}x")
        
        print(f"\n💡 RECOMMENDATIONS:")
        if sorted_stats:
            best_family = sorted_stats[0][0]
            best_rate = sorted_stats[0][1]['hit_rate']
            print(f"   Focus on: {best_family} ({best_rate:.1f}% hit rate)")
            
            # Find worst performing active family
            worst_family = sorted_stats[-1][0]
            worst_rate = sorted_stats[-1][1]['hit_rate']
            if worst_rate < 2.0:
                print(f"   Review: {worst_family} ({worst_rate:.1f}% hit rate)")
        
        # 🆕 Show S40 performance specifically (important for current strategy)
        s40_stats = pattern_stats.get('S40', {'hit_rate': 0, 'hits': 0})
        print(f"\n⭐ S40 PERFORMANCE: {s40_stats['hit_rate']:.1f}% ({s40_stats['hits']} hits)")

def main():
    engine = PatternIntelligenceEngine()
    success = engine.run_analysis()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
