# golden_block_analyzer.py - NEW FILE FOR PHASE 3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')


def normalize_hit_type(df: pd.DataFrame):
    """
    Ensure we have a single, clean HIT_TYPE column as an upper-case string Series.
    Returns (df, hit_type_col_name or None).
    """
    if df is None or df.empty:
        return df, None

    # Drop duplicate columns by name
    df = df.loc[:, ~df.columns.duplicated()]

    # Normalise column names to upper-case, stripped
    df.columns = [str(c).strip().upper() for c in df.columns]

    hit_type_col = None
    # Prefer HIT_TYPE if present
    if "HIT_TYPE" in df.columns:
        hit_type_col = "HIT_TYPE"
    elif "HIT_KIND" in df.columns:
        hit_type_col = "HIT_KIND"
    elif "HITCATEGORY" in df.columns:
        hit_type_col = "HITCATEGORY"
    elif "HIT_FAMILY" in df.columns:
        hit_type_col = "HIT_FAMILY"

    if hit_type_col is None:
        # No hit-type info; caller must handle None
        return df, None

    # Extract column and reduce to Series if it is accidentally a DataFrame
    col_obj = df[hit_type_col]
    if isinstance(col_obj, pd.DataFrame):
        # Take first underlying column if duplicated
        first_col = col_obj.columns[0]
        col_series = col_obj[first_col]
    else:
        col_series = col_obj

    # Force to clean upper-case strings
    col_series = (
        col_series
        .astype(str)
        .str.strip()
        .str.upper()
    )
    # Push back into df
    df[hit_type_col] = col_series

    return df, hit_type_col

class GoldenBlockAnalyzer:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.scripts = [f"SCR{i}" for i in range(1, 10)]
        
    def load_pnl_history(self):
        """Load P&L history for golden days analysis"""
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        
        if not pnl_file.exists():
            print("❌ P&L history file not found")
            return None
            
        try:
            df = pd.read_excel(pnl_file, sheet_name='day_level')
            df['date'] = pd.to_datetime(df['date']).dt.date
            
            # Filter to last 30 days for golden analysis
            unique_dates = sorted(df['date'].unique())
            if len(unique_dates) > 30:
                cutoff_date = unique_dates[-30]
                df = df[df['date'] >= cutoff_date]
                print(f"📊 Using last 30 days for golden analysis ({len(unique_dates)} total days available)")
            else:
                print(f"📊 Using all {len(unique_dates)} available days for golden analysis")
            
            return df.sort_values('date')
        except Exception as e:
            print(f"❌ Error loading P&L history: {e}")
            return None

    def load_hit_memory(self):
        """Load enhanced hit memory with time-shift tracking"""
        memory_file = self.base_dir / "logs" / "performance" / "script_hit_memory.xlsx"
        
        if not memory_file.exists():
            print("❌ No script_hit_memory.xlsx found")
            return None

        try:
            df = pd.read_excel(memory_file)
            print(f"📊 Loaded enhanced hit memory: {len(df)} records")

            # Convert date columns
            columns_lower = {str(col).strip().lower(): col for col in df.columns}
            if 'date' in columns_lower:
                date_col = columns_lower['date']
                df[date_col] = pd.to_datetime(df[date_col]).dt.date
            if 'prediction_date' in columns_lower:
                pred_col = columns_lower['prediction_date']
                df[pred_col] = pd.to_datetime(df[pred_col]).dt.date
            
            return df
        except Exception as e:
            print(f"❌ Error loading hit memory: {e}")
            return None

    def identify_golden_days(self, pnl_df, top_n=5):
        """Identify top profit days (golden days)"""
        if pnl_df.empty or 'profit_total' not in pnl_df.columns:
            print("❌ No profit data available")
            return []

        positive_df = pnl_df[pnl_df['profit_total'] > 0]
        if positive_df.empty:
            print("❌ No profitable days found to label as golden")
            return []

        # Get top profit days
        top_days = positive_df.nlargest(top_n, 'profit_total')[['date', 'profit_total']]
        
        golden_days = []
        for _, row in top_days.iterrows():
            golden_days.append({
                'date': row['date'].strftime('%Y-%m-%d'),
                'profit': float(row['profit_total'])
            })
        
        print(f"🎯 Identified {len(golden_days)} golden days:")
        for day in golden_days:
            print(f"   {day['date']}: Profit ₹{day['profit']:+,.0f}")

        return golden_days

    @staticmethod
    def _classify_direct_and_cross(golden_hits: pd.DataFrame, hit_type_col: str | None):
        """
        Returns (direct_mask, cross_mask) for the subset golden_hits.
        Both are boolean Series aligned with golden_hits.index.
        """
        if hit_type_col is None or hit_type_col not in golden_hits.columns:
            all_false = pd.Series(False, index=golden_hits.index)
            all_true = pd.Series(True, index=golden_hits.index)
            return all_false, all_true

        col_obj = golden_hits[hit_type_col]
        if isinstance(col_obj, pd.DataFrame):
            col_obj = col_obj.iloc[:, 0]

        ht = (
            col_obj
            .astype(str)
            .str.strip()
            .str.upper()
        )

        direct_labels = {"DIRECT", "SAME_DAY"}
        cross_mask = ht.str.startswith("CROSS", na=False)
        direct_mask = ht.isin(direct_labels)

        return direct_mask, cross_mask

    def analyze_script_contributions(self, hit_memory_df, golden_days, hit_type_col: str | None):
        """Analyze script contributions on golden days"""
        script_analysis = {}

        golden_dates = [day['date'] for day in golden_days]

        for script in self.scripts:
            script_hits = hit_memory_df[hit_memory_df['SCRIPT'] == script]

            # Filter for golden days
            golden_hits = script_hits[script_hits['DATE'].astype(str).isin(golden_dates)]

            if not golden_hits.empty:
                direct_mask, cross_mask = self._classify_direct_and_cross(golden_hits, hit_type_col)
                direct_hits = int(direct_mask.sum())
                cross_hits = int(cross_mask.sum())
                total_hits = len(golden_hits)

                # Calculate golden score (weighted by hit type and rank)
                golden_score = 0
                for _, hit in golden_hits.iterrows():
                    rank_weight = 1.0 / hit['RANK']
                    if hit_type_col and hit_type_col in hit:
                        hit_label = str(hit[hit_type_col]).strip().upper()
                    else:
                        hit_label = "CROSS"
                    hit_weight = 1.0 if hit_label in {"DIRECT", "SAME_DAY"} else 0.7
                    golden_score += rank_weight * hit_weight

                script_analysis[script] = {
                    'direct_hits': direct_hits,
                    'cross_hits': cross_hits,
                    'total_hits': total_hits,
                    'golden_score': round(golden_score, 2),
                    'effectiveness': round((total_hits / len(golden_days)) * 100, 1) if golden_days else 0
                }
            else:
                script_analysis[script] = {
                    'direct_hits': 0,
                    'cross_hits': 0,
                    'total_hits': 0,
                    'golden_score': 0,
                    'effectiveness': 0
                }

        return script_analysis

    def analyze_cross_slot_patterns(self, hit_memory_df, golden_days, hit_type_col: str | None):
        """Analyze cross-slot patterns on golden days"""
        cross_patterns = {}

        # Filter for cross-slot hits on golden days
        golden_dates = [day['date'] for day in golden_days]
        golden_hits = hit_memory_df[hit_memory_df['DATE'].astype(str).isin(golden_dates)]
        direct_mask, cross_mask = self._classify_direct_and_cross(golden_hits, hit_type_col)
        cross_hits = golden_hits[cross_mask]

        # Count cross-slot pairs
        for _, hit in cross_hits.iterrows():
            pair_key = f"{hit['PREDICTED_SLOT']}→{hit['REAL_SLOT']}"
            if pair_key not in cross_patterns:
                cross_patterns[pair_key] = {
                    'hits': 0,
                    'total_rank': 0,
                    'days': set()
                }
            
            cross_patterns[pair_key]['hits'] += 1
            cross_patterns[pair_key]['total_rank'] += hit['RANK']
            cross_patterns[pair_key]['days'].add(hit['DATE'])
        
        # Calculate averages and format results
        formatted_patterns = {}
        for pair, data in cross_patterns.items():
            avg_rank = data['total_rank'] / data['hits'] if data['hits'] > 0 else 0
            formatted_patterns[pair] = {
                'hits': data['hits'],
                'avg_rank': round(avg_rank, 1),
                'unique_days': len(data['days'])
            }
        
        # Sort by hits descending
        return dict(sorted(formatted_patterns.items(), key=lambda x: x[1]['hits'], reverse=True))

    def analyze_digit_patterns(self, hit_memory_df, golden_days):
        """Analyze digit patterns on golden days"""
        golden_dates = [day['date'] for day in golden_days]
        golden_hits = hit_memory_df[hit_memory_df['DATE'].astype(str).isin(golden_dates)]

        # Extract all numbers from golden hits
        all_numbers = golden_hits['REAL_NUMBER'].tolist()
        
        # Analyze tens and ones digits
        tens_digits = [num // 10 for num in all_numbers]
        ones_digits = [num % 10 for num in all_numbers]
        
        # Get most common digits
        tens_counter = Counter(tens_digits)
        ones_counter = Counter(ones_digits)
        
        top_tens = [digit for digit, _ in tens_counter.most_common(5)]
        top_ones = [digit for digit, _ in ones_counter.most_common(5)]
        
        # Identify hero numbers (appear multiple times)
        number_counter = Counter(all_numbers)
        hero_numbers = [num for num, count in number_counter.most_common(10) if count >= 2]
        
        return {
            'tens': top_tens,
            'ones': top_ones,
            'hero_numbers': hero_numbers,
            'total_unique_numbers': len(set(all_numbers)),
            'most_frequent_number': number_counter.most_common(1)[0] if number_counter else None
        }

    def analyze_time_patterns(self, hit_memory_df, golden_days):
        """Analyze time/day patterns on golden days"""
        time_patterns = {}

        golden_dates = [day['date'] for day in golden_days]
        golden_hits = hit_memory_df[hit_memory_df['DATE'].astype(str).isin(golden_dates)]
        
        # Convert dates to day of week
        golden_hits['day_of_week'] = pd.to_datetime(golden_hits['DATE']).dt.day_name()
        
        # Analyze by slot
        for slot in self.slots:
            slot_hits = golden_hits[golden_hits['REAL_SLOT'] == slot]
            if not slot_hits.empty:
                day_counts = slot_hits['day_of_week'].value_counts()
                best_day = day_counts.index[0] if not day_counts.empty else "Unknown"
                best_day_hits = day_counts.iloc[0] if not day_counts.empty else 0
                
                time_patterns[slot] = {
                    'best_day': best_day,
                    'hits': int(best_day_hits),
                    'all_days': day_counts.to_dict()
                }
        
        return time_patterns

    def generate_golden_insights(self):
        """Generate complete golden block insights"""
        print("🧠 GOLDEN BLOCK ANALYZER - Extracting Profit Patterns")
        print("=" * 60)
        
        # Step 1: Load data
        pnl_df = self.load_pnl_history()
        if pnl_df is None:
            return False
            
        hit_memory_df = self.load_hit_memory()
        if hit_memory_df is None:
            return False

        hit_memory_df, hit_type_col = normalize_hit_type(hit_memory_df)
        if hit_type_col is None:
            print("⚠️ HIT_TYPE column missing in hit memory; golden script analysis will ignore direct/cross split.")

        # Step 2: Identify golden days
        golden_days = self.identify_golden_days(pnl_df)
        if not golden_days:
            print("❌ No golden days identified")
            return False

        # Step 3: Analyze various aspects
        print("\n📊 Analyzing script contributions...")
        script_analysis = self.analyze_script_contributions(hit_memory_df, golden_days, hit_type_col)

        print("📊 Analyzing cross-slot patterns...")
        cross_patterns = self.analyze_cross_slot_patterns(hit_memory_df, golden_days, hit_type_col)
        
        print("📊 Analyzing digit patterns...")
        digit_analysis = self.analyze_digit_patterns(hit_memory_df, golden_days)
        
        print("📊 Analyzing time patterns...")
        time_patterns = self.analyze_time_patterns(hit_memory_df, golden_days)
        
        # Step 4: Compile insights
        insights = {
            "timestamp": datetime.now().isoformat(),
            "window_days": 30,
            "golden_days": golden_days,
            "scripts": script_analysis,
            "cross_slot_pairs": cross_patterns,
            "digits": digit_analysis,
            "day_of_week": time_patterns,
            "summary": {
                "total_golden_days": len(golden_days),
                "total_profits": sum(day['profit'] for day in golden_days),
                "avg_profit_per_golden_day": sum(day['profit'] for day in golden_days) / len(golden_days),
                "top_script": max(
                    {k: v for k, v in script_analysis.items() if v.get('golden_score', 0) > 0}.items(),
                    key=lambda x: x[1]['golden_score'],
                    default=("NONE", {})
                )[0],
                "top_cross_pattern": list(cross_patterns.keys())[0] if cross_patterns else "NONE"
            }
        }
        
        # Step 5: Save insights
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = output_dir / "golden_block_insights.json"
        with open(json_file, 'w') as f:
            json.dump(insights, f, indent=2)
        
        # Save Excel report
        excel_file = output_dir / "golden_block_insights.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Golden days sheet
            pd.DataFrame(golden_days).to_excel(writer, sheet_name='golden_days', index=False)
            
            # Scripts sheet
            scripts_df = pd.DataFrame.from_dict(script_analysis, orient='index')
            scripts_df.reset_index(inplace=True)
            scripts_df.rename(columns={'index': 'script'}, inplace=True)
            scripts_df.to_excel(writer, sheet_name='scripts', index=False)
            
            # Cross patterns sheet
            cross_df = pd.DataFrame.from_dict(cross_patterns, orient='index')
            cross_df.reset_index(inplace=True)
            cross_df.rename(columns={'index': 'pattern'}, inplace=True)
            cross_df.to_excel(writer, sheet_name='cross_patterns', index=False)
            
            # Digits sheet
            digits_data = {
                'tens_digits': digit_analysis['tens'],
                'ones_digits': digit_analysis['ones'],
                'hero_numbers': digit_analysis['hero_numbers'] + [''] * (10 - len(digit_analysis['hero_numbers']))
            }
            pd.DataFrame(dict([(k, pd.Series(v)) for k, v in digits_data.items()])).to_excel(writer, sheet_name='digits', index=False)
        
        print(f"\n💾 Golden insights saved:")
        print(f"   JSON: {json_file}")
        print(f"   Excel: {excel_file}")
        
        # Step 6: Print summary
        self.print_insights_summary(insights)
        
        return True

    def print_insights_summary(self, insights):
        """Print formatted insights summary"""
        print("\n" + "=" * 60)
        print("🎯 GOLDEN BLOCK INSIGHTS SUMMARY")
        print("=" * 60)
        
        print(f"📊 Golden Days: {len(insights['golden_days'])} days")
        print(f"💰 Total Profit: ₹{insights['summary']['total_profits']:+,.0f}")
        print(f"📈 Average Profit: ₹{insights['summary']['avg_profit_per_golden_day']:+,.0f}")
        
        top_script = insights['summary']['top_script']
        if top_script != "NONE" and top_script in insights['scripts']:
            print(f"\n🏆 Top Script: {top_script}")
            print(f"   Golden Score: {insights['scripts'][top_script]['golden_score']}")
            print(f"   Direct Hits: {insights['scripts'][top_script]['direct_hits']}")
            print(f"   Cross Hits: {insights['scripts'][top_script]['cross_hits']}")
        
        if insights['cross_slot_pairs']:
            top_pattern = insights['summary']['top_cross_pattern']
            print(f"\n🔗 Top Cross Pattern: {top_pattern}")
            print(f"   Hits: {insights['cross_slot_pairs'][top_pattern]['hits']}")
            print(f"   Avg Rank: {insights['cross_slot_pairs'][top_pattern]['avg_rank']}")
        
        if insights['digits']['hero_numbers']:
            print(f"\n🔢 Hero Numbers: {', '.join(map(str, insights['digits']['hero_numbers'][:5]))}")
        
        print(f"\n⏰ Best Days by Slot:")
        for slot, pattern in insights['day_of_week'].items():
            print(f"   {slot}: {pattern['best_day']} ({pattern['hits']} hits)")

def main():
    analyzer = GoldenBlockAnalyzer()
    success = analyzer.generate_golden_insights()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
