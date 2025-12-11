# pattern_packs_automerge.py
"""
PATTERN PACKS AUTOMERGE - Safe Pattern Overlay Generator

PURPOSE:
Convert pattern_packs_suggested.py SUGGESTED_* lists into a safe overlay file
that works with pattern_packs.py without modifying the base file.

USAGE:
py -3.12 pattern_packs_automerge.py
"""

import importlib.util
import sys
from pathlib import Path
import shutil

class PatternPacksAutomerge:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.suggested_patterns = {}
        
    def load_suggested_patterns(self):
        """Dynamically import pattern_packs_suggested.py"""
        suggested_file = self.base_dir / "logs" / "performance" / "pattern_packs_suggested.py"

        if not suggested_file.exists():
            print("Pattern packs not available – skipping automerge")
            return True
            
        try:
            # Dynamic import
            spec = importlib.util.spec_from_file_location("pattern_packs_suggested", suggested_file)
            suggested_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(suggested_module)
            
            # Extract all SUGGESTED_* variables
            for attr_name in dir(suggested_module):
                if attr_name.startswith('SUGGESTED_'):
                    self.suggested_patterns[attr_name] = getattr(suggested_module, attr_name)
            
            print(f"✅ Loaded {len(self.suggested_patterns)} suggested patterns")
            return True
            
        except Exception as e:
            print(f"❌ Error loading suggested patterns: {e}")
            return False
    
    def generate_overlay_file(self):
        """Generate pattern_packs_auto.py overlay file"""
        overlay_content = '''"""
PATTERN_PACKS_AUTO.PY - Auto-generated overlay from pattern_packs_suggested.py

⚠️  DO NOT EDIT MANUALLY - REGENERATE VIA pattern_packs_automerge.py
⚠️  This is an OVERLAY file - use with pattern_packs.py for enhanced patterns

USAGE:
# In your prediction scripts, you can optionally use:
# from pattern_packs_auto import *

Golden Days lab discovered these patterns with high hit rates.
"""

from pattern_packs import *  # Import all existing packs

# =============================================================================
# GOLDEN DAYS OVERLAY PATTERNS - AUTO GENERATED
# =============================================================================

'''
        
        # Add pattern definitions
        overlay_groups = {
            'range': [],
            'tens': [], 
            'ones': []
        }
        
        for pattern_name, pattern_list in self.suggested_patterns.items():
            # Determine pattern type and create clean name
            if 'RANGE' in pattern_name:
                clean_name = pattern_name.replace('SUGGESTED_', 'GOLDEN_')
                overlay_groups['range'].append((clean_name, pattern_list))
            elif 'TENS' in pattern_name:
                clean_name = pattern_name.replace('SUGGESTED_', 'GOLDEN_TENS_')
                overlay_groups['tens'].append((clean_name, pattern_list))
            elif 'ONES' in pattern_name:
                clean_name = pattern_name.replace('SUGGESTED_', 'GOLDEN_ONES_')
                overlay_groups['ones'].append((clean_name, pattern_list))
            else:
                clean_name = pattern_name.replace('SUGGESTED_', 'GOLDEN_')
                overlay_content += f"# {clean_name} - Custom pattern\\n"
                overlay_content += f"{clean_name} = {pattern_list}\\n\\n"
        
        # Add range patterns
        if overlay_groups['range']:
            overlay_content += "# Golden Days range-based overlays\\n"
            for clean_name, pattern_list in overlay_groups['range']:
                overlay_content += f"{clean_name} = {pattern_list}\\n"
            overlay_content += "\\n"
        
        # Add tens patterns  
        if overlay_groups['tens']:
            overlay_content += "# Golden Days tens digit overlays\\n"
            for clean_name, pattern_list in overlay_groups['tens']:
                overlay_content += f"{clean_name} = {pattern_list}\\n"
            overlay_content += "\\n"
            
        # Add ones patterns
        if overlay_groups['ones']:
            overlay_content += "# Golden Days ones digit overlays\\n" 
            for clean_name, pattern_list in overlay_groups['ones']:
                overlay_content += f"{clean_name} = {pattern_list}\\n"
            overlay_content += "\\n"
        
        # Add meta information
        overlay_content += '''# Meta information about Golden Days overlays
GOLDEN_META_INFO = {
    "comment": "Overlay packs generated from Golden Days lab analysis",
    "ranges": [''' + ', '.join([f'"{name}"' for name, _ in overlay_groups['range']]) + '''],
    "tens": [''' + ', '.join([f'"{name}"' for name, _ in overlay_groups['tens']]) + '''],
    "ones": [''' + ', '.join([f'"{name}"' for name, _ in overlay_groups['ones']]) + '''],
    "generated_on": "''' + str(Path(__file__).resolve().name) + '''"
}

# Combined overlay sets for easy access
GOLDEN_OVERLAY_RANGES = []
GOLDEN_OVERLAY_TENS = []
GOLDEN_OVERLAY_ONES = []

'''
        
        # Add combined sets
        for clean_name, pattern_list in overlay_groups['range']:
            overlay_content += f"GOLDEN_OVERLAY_RANGES.extend({clean_name})\\n"
        
        for clean_name, pattern_list in overlay_groups['tens']:
            overlay_content += f"GOLDEN_OVERLAY_TENS.extend({clean_name})\\n"
            
        for clean_name, pattern_list in overlay_groups['ones']:
            overlay_content += f"GOLDEN_OVERLAY_ONES.extend({clean_name})\\n"
        
        overlay_content += '''
# Remove duplicates from combined sets
GOLDEN_OVERLAY_RANGES = list(set(GOLDEN_OVERLAY_RANGES))
GOLDEN_OVERLAY_TENS = list(set(GOLDEN_OVERLAY_TENS))  
GOLDEN_OVERLAY_ONES = list(set(GOLDEN_OVERLAY_ONES))

print("✅ Golden Days overlay patterns loaded successfully")
'''
        
        # Write overlay file
        overlay_file = self.base_dir / "pattern_packs_auto.py"
        with open(overlay_file, 'w') as f:
            f.write(overlay_content)
        
        return overlay_file, overlay_groups
    
    def display_summary(self, overlay_groups):
        """Display summary of generated overlays"""
        print("\\n🔍 PATTERN PACKS AUTOMERGE - OVERLAY SUMMARY")
        print("=" * 50)
        
        total_patterns = 0
        
        for pattern_type in ['range', 'tens', 'ones']:
            patterns = overlay_groups[pattern_type]
            if patterns:
                print(f"\\n📊 {pattern_type.upper()} PATTERNS ({len(patterns)}):")
                for clean_name, pattern_list in patterns:
                    sample = pattern_list[:3] if len(pattern_list) > 3 else pattern_list
                    print(f"   {clean_name}: {len(pattern_list)} numbers, sample: {sample}")
                    total_patterns += len(pattern_list)
        
        print(f"\\n📈 TOTAL: {total_patterns} overlay patterns generated")
        print("💾 Overlay file: pattern_packs_auto.py")
        print("\\n💡 USAGE INSTRUCTIONS:")
        print("   1. In prediction scripts: from pattern_packs_auto import *")
        print("   2. Use GOLDEN_* patterns alongside existing patterns")
        print("   3. Test with small stakes first")
        print("   4. Monitor performance in pattern_packs_lab_report.xlsx")
    
    def run_automerge(self):
        """Run complete automerge process"""
        print("🔄 PATTERN PACKS AUTOMERGE - Generating Safe Overlay")
        print("=" * 60)
        
        # Load suggested patterns
        if not self.load_suggested_patterns():
            return False
        
        # Generate overlay file
        overlay_file, overlay_groups = self.generate_overlay_file()
        
        # Display summary
        self.display_summary(overlay_groups)
        
        return True

def main():
    automerger = PatternPacksAutomerge()
    success = automerger.run_automerge()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())