# pattern_packs_lab.py
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

class PatternPacksLab:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        
    def load_pattern_data(self):
        """Load all pattern-related data"""
        data = {}
        perf_dir = self.base_dir / "logs" / "performance"
        
        # Load golden days analysis
        golden_file = perf_dir / "golden_days_analysis.xlsx"
        if golden_file.exists():
            try:
                data['golden_days'] = pd.read_excel(golden_file, sheet_name='slot_hits')
                print(f"✅ Loaded {len(data['golden_days'])} golden day hits")
            except Exception as e:
                print(f"⚠️ Error loading golden days: {e}")
        
        # Load pattern intelligence
        pattern_file = perf_dir / "pattern_intelligence_summary.json"
        if pattern_file.exists():
            with open(pattern_file, 'r') as f:
                data['pattern_intel'] = json.load(f)
        
        # Load real results
        results_file = self.base_dir / "number_prediction_learn.xlsx"
        if results_file.exists():
            # This would need custom loading logic based on your file structure
            print("✅ Real results file available")
            data['results_file'] = results_file
        
        return data
    
    def analyze_golden_patterns(self, data):
        """Analyze patterns from golden days"""
        if 'golden_days' not in data:
            return []
            
        hits_df = data['golden_days']
        pattern_suggestions = []
        
        # Analyze number ranges
        range_patterns = self.analyze_number_ranges(hits_df)
        pattern_suggestions.extend(range_patterns)
        
        # Analyze digit patterns
        digit_patterns = self.analyze_digit_patterns(hits_df)
        pattern_suggestions.extend(digit_patterns)
        
        # Analyze S40 patterns
        s40_patterns = self.analyze_s40_patterns(hits_df)
        pattern_suggestions.extend(s40_patterns)
        
        return pattern_suggestions
    
    def analyze_number_ranges(self, hits_df):
        """Analyze winning number ranges"""
        ranges = [
            (0, 9, "SINGLE_DIGITS"),
            (10, 39, "LOW_RANGE"),
            (40, 69, "MID_RANGE"), 
            (70, 99, "HIGH_RANGE")
        ]
        
        suggestions = []
        total_hits = len(hits_df)
        
        for range_start, range_end, range_name in ranges:
            range_hits = hits_df[(hits_df['hit_number'] >= range_start) & 
                                (hits_df['hit_number'] <= range_end)]
            hit_count = len(range_hits)
            hit_percentage = (hit_count / total_hits) * 100 if total_hits > 0 else 0
            
            if hit_percentage > 20:  # Significant pattern threshold
                suggestions.append({
                    'pattern_type': 'NUMBER_RANGE',
                    'pattern_name': f"GOLDEN_{range_name}",
                    'numbers': f"{range_start}-{range_end}",
                    'hit_count': hit_count,
                    'hit_percentage': hit_percentage,
                    'confidence': 'HIGH' if hit_percentage > 30 else 'MEDIUM',
                    'suggestion': f"Focus on {range_name} numbers",
                    'implementation': f"SUGGESTED_RANGE_{range_name.upper()} = list(range({range_start}, {range_end+1}))"
                })
        
        return suggestions
    
    def analyze_digit_patterns(self, hits_df):
        """Analyze digit-based patterns"""
        suggestions = []
        total_hits = len(hits_df)
        
        # Tens digit analysis
        for tens in range(10):
            tens_hits = hits_df[hits_df['tens_digit'] == tens]
            hit_count = len(tens_hits)
            hit_percentage = (hit_count / total_hits) * 100 if total_hits > 0 else 0
            
            if hit_percentage > 15:  # Significant tens pattern
                suggestions.append({
                    'pattern_type': 'TENS_DIGIT',
                    'pattern_name': f"STRONG_TENS_{tens}",
                    'numbers': f"Tens digit {tens}",
                    'hit_count': hit_count,
                    'hit_percentage': hit_percentage,
                    'confidence': 'HIGH' if hit_percentage > 20 else 'MEDIUM',
                    'suggestion': f"Prioritize numbers with tens digit {tens}",
                    'implementation': f"SUGGESTED_TENS_{tens} = [n for n in range(100) if n // 10 == {tens}]"
                })
        
        # Ones digit analysis
        for ones in range(10):
            ones_hits = hits_df[hits_df['ones_digit'] == ones]
            hit_count = len(ones_hits)
            hit_percentage = (hit_count / total_hits) * 100 if total_hits > 0 else 0
            
            if hit_percentage > 15:  # Significant ones pattern
                suggestions.append({
                    'pattern_type': 'ONES_DIGIT',
                    'pattern_name': f"STRONG_ONES_{ones}",
                    'numbers': f"Ones digit {ones}",
                    'hit_count': hit_count,
                    'hit_percentage': hit_percentage,
                    'confidence': 'HIGH' if hit_percentage > 20 else 'MEDIUM',
                    'suggestion': f"Prioritize numbers with ones digit {ones}",
                    'implementation': f"SUGGESTED_ONES_{ones} = [n for n in range(100) if n % 10 == {ones}]"
                })
        
        return suggestions
    
    def analyze_s40_patterns(self, hits_df):
        """Analyze S40-related patterns"""
        suggestions = []
        
        # S40 numbers (70-99, 0-9)
        s40_hits = hits_df[hits_df['s40_flag'] == True]
        s40_count = len(s40_hits)
        total_hits = len(hits_df)
        s40_percentage = (s40_count / total_hits) * 100 if total_hits > 0 else 0
        
        if s40_percentage > 25:
            suggestions.append({
                'pattern_type': 'S40_NUMBERS',
                'pattern_name': "GOLDEN_S40",
                'numbers': "70-99, 0-9",
                'hit_count': s40_count,
                'hit_percentage': s40_percentage,
                'confidence': 'HIGH' if s40_percentage > 30 else 'MEDIUM',
                'suggestion': "Increase S40 number weight in predictions",
                'implementation': "SUGGESTED_S40_GOLDEN = list(range(70, 100)) + list(range(0, 10))"
            })
        
        return suggestions
    
    def generate_pattern_packs_suggestions(self, pattern_suggestions):
        """Generate Python code for suggested pattern packs"""
        if not pattern_suggestions:
            return "# No significant pattern suggestions found"
        
        code_lines = [
            '# AUTO-SUGGESTED PATTERN PACKS - GENERATED BY PATTERN_PACKS_LAB.PY',
            '# DO NOT EDIT THIS FILE DIRECTLY - REVIEW AND MERGE INTO pattern_packs.py MANUALLY',
            '# Generated on: ' + datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '',
            '# Golden Days Based Pattern Suggestions',
            ''
        ]
        
        # Group by pattern type
        range_patterns = [p for p in pattern_suggestions if p['pattern_type'] == 'NUMBER_RANGE']
        tens_patterns = [p for p in pattern_suggestions if p['pattern_type'] == 'TENS_DIGIT']
        ones_patterns = [p for p in pattern_suggestions if p['pattern_type'] == 'ONES_DIGIT']
        s40_patterns = [p for p in pattern_suggestions if p['pattern_type'] == 'S40_NUMBERS']
        
        # Add range patterns
        if range_patterns:
            code_lines.append('# Number Range Patterns')
            for pattern in range_patterns:
                code_lines.append(f"# {pattern['pattern_name']}: {pattern['hit_percentage']:.1f}% hit rate")
                code_lines.append(pattern['implementation'])
                code_lines.append('')
        
        # Add tens patterns
        if tens_patterns:
            code_lines.append('# Tens Digit Patterns')
            for pattern in tens_patterns:
                code_lines.append(f"# {pattern['pattern_name']}: {pattern['hit_percentage']:.1f}% hit rate")
                code_lines.append(pattern['implementation'])
                code_lines.append('')
        
        # Add ones patterns
        if ones_patterns:
            code_lines.append('# Ones Digit Patterns')
            for pattern in ones_patterns:
                code_lines.append(f"# {pattern['pattern_name']}: {pattern['hit_percentage']:.1f}% hit rate")
                code_lines.append(pattern['implementation'])
                code_lines.append('')
        
        # Add S40 patterns
        if s40_patterns:
            code_lines.append('# S40 Number Patterns')
            for pattern in s40_patterns:
                code_lines.append(f"# {pattern['pattern_name']}: {pattern['hit_percentage']:.1f}% hit rate")
                code_lines.append(pattern['implementation'])
                code_lines.append('')
        
        return '\n'.join(code_lines)
    
    def save_pattern_lab_results(self, pattern_suggestions, python_code):
        """Save pattern lab results"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save Excel report
        if pattern_suggestions:
            excel_file = output_dir / "pattern_packs_lab_report.xlsx"
            df = pd.DataFrame(pattern_suggestions)
            df.to_excel(excel_file, index=False)
            print(f"💾 Pattern lab report saved to: {excel_file}")
        
        # Save Python suggestions
        python_file = output_dir / "pattern_packs_suggested.py"
        with open(python_file, 'w') as f:
            f.write(python_code)
        print(f"💾 Pattern packs suggestions saved to: {python_file}")
        
        return excel_file, python_file
    
    def print_pattern_lab_report(self, pattern_suggestions):
        """Print pattern lab report to console"""
        print("\n" + "="*80)
        print("🔍 PATTERN PACKS LAB - AUTO PATTERN DISCOVERY")
        print("="*80)
        
        if not pattern_suggestions:
            print("\n❌ No significant patterns discovered")
            return
        
        print(f"\n📊 PATTERNS DISCOVERED: {len(pattern_suggestions)}")
        print("-" * 50)
        
        # Group by confidence
        high_conf = [p for p in pattern_suggestions if p['confidence'] == 'HIGH']
        medium_conf = [p for p in pattern_suggestions if p['confidence'] == 'MEDIUM']
        
        if high_conf:
            print(f"\n🎯 HIGH CONFIDENCE PATTERNS:")
            print("-" * 40)
            for pattern in high_conf:
                print(f"   {pattern['pattern_name']:20} : {pattern['hit_percentage']:5.1f}% hit rate ({pattern['hit_count']} hits)")
                print(f"      Suggestion: {pattern['suggestion']}")
        
        if medium_conf:
            print(f"\n✅ MEDIUM CONFIDENCE PATTERNS:")
            print("-" * 45)
            for pattern in medium_conf:
                print(f"   {pattern['pattern_name']:20} : {pattern['hit_percentage']:5.1f}% hit rate ({pattern['hit_count']} hits)")
    
    def run(self):
        """Main execution"""
        print("🔍 PATTERN PACKS LAB - Discovering new patterns from Golden Days...")
        
        # Load pattern data
        data = self.load_pattern_data()
        
        if not data:
            print("❌ No pattern data found")
            return False
        
        # Analyze patterns
        pattern_suggestions = self.analyze_golden_patterns(data)
        
        # Generate Python code
        python_code = self.generate_pattern_packs_suggestions(pattern_suggestions)
        
        # Save results
        self.save_pattern_lab_results(pattern_suggestions, python_code)
        self.print_pattern_lab_report(pattern_suggestions)
        
        print(f"\n💡 NEXT STEPS:")
        print("-" * 30)
        print("   1. Review pattern_packs_lab_report.xlsx")
        print("   2. Check pattern_packs_suggested.py for code suggestions")
        print("   3. Manually merge high-confidence patterns into pattern_packs.py")
        print("   4. Test new patterns in predictions")
        
        return True

def main():
    lab = PatternPacksLab()
    success = lab.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())