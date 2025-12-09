# pattern_packs_exporter.py - UPDATED WITH ADAPTIVE PACKS
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class PatternPacksExporter:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        
    def load_pattern_packs(self):
        """Load pattern packs data"""
        pattern_file = self.base_dir / "logs" / "performance" / "pattern_packs.json"
        
        if not pattern_file.exists():
            print("❌ Pattern packs file not found")
            return None
            
        try:
            with open(pattern_file, 'r') as f:
                pattern_data = json.load(f)
            return pattern_data
        except Exception as e:
            print(f"❌ Error loading pattern packs: {e}")
            return None

    def load_adaptive_packs(self):
        """✅ PHASE 3: Load adaptive pattern packs"""
        adaptive_file = self.base_dir / "logs" / "performance" / "adaptive_pattern_packs.json"
        
        if adaptive_file.exists():
            try:
                with open(adaptive_file, 'r') as f:
                    adaptive_data = json.load(f)
                return adaptive_data
            except Exception as e:
                print(f"⚠️  Error loading adaptive packs: {e}")
        
        return None

    def export_pattern_packs(self):
        """Export pattern packs to Excel with adaptive packs integration"""
        print("🎯 PATTERN PACKS EXPORTER - Auto-Generating Pattern Packs")
        print("=" * 60)
        
        # Load pattern packs
        pattern_data = self.load_pattern_packs()
        if pattern_data is None:
            return False
        
        # ✅ PHASE 3: Load adaptive packs
        adaptive_data = self.load_adaptive_packs()
        
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON (existing functionality)
        with open(output_dir / "pattern_packs.json", 'w') as f:
            json.dump(pattern_data, f, indent=2)
        
        # Create Excel with multiple sheets
        excel_file = output_dir / "pattern_packs.xlsx"
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Base pattern packs sheet
            base_data = []
            for slot in self.slots:
                if slot in pattern_data:
                    slot_data = pattern_data[slot]
                    base_data.append({
                        'Slot': slot,
                        'Tens Core': ', '.join(map(str, slot_data.get('tens_core', []))),
                        'Ones Core': ', '.join(map(str, slot_data.get('ones_core', []))),
                        'S40 Numbers': ', '.join(map(str, slot_data.get('s40_numbers', []))),
                        'Hit Rate': f"{slot_data.get('hit_rate', 0):.1f}%"
                    })
            
            pd.DataFrame(base_data).to_excel(writer, sheet_name='base_pattern_packs', index=False)
            
            # ✅ PHASE 3: Adaptive packs sheet
            if adaptive_data:
                adaptive_sheets = []
                
                # Core digits comparison
                core_data = {
                    'Type': ['Base Tens', 'Base Ones', 'Golden Tens', 'Golden Ones'],
                    'Digits': [
                        ', '.join(map(str, adaptive_data.get('tens_core_base', []))),
                        ', '.join(map(str, adaptive_data.get('ones_core_base', []))),
                        ', '.join(map(str, adaptive_data.get('tens_core_golden', []))),
                        ', '.join(map(str, adaptive_data.get('ones_core_golden', [])))
                    ]
                }
                pd.DataFrame(core_data).to_excel(writer, sheet_name='adaptive_cores', index=False)
                
                # Hero numbers
                hero_numbers = adaptive_data.get('hero_numbers', [])
                pd.DataFrame({'Hero Numbers': hero_numbers}).to_excel(writer, sheet_name='hero_numbers', index=False)
                
                # Cross-slot patterns
                cross_patterns = adaptive_data.get('cross_slot_pairs_top', [])
                pd.DataFrame({'Top Cross Patterns': cross_patterns}).to_excel(writer, sheet_name='cross_patterns', index=False)
                
                # Boost scripts
                boost_scripts = adaptive_data.get('boost_scripts', [])
                pd.DataFrame({'Boost Scripts': boost_scripts}).to_excel(writer, sheet_name='boost_scripts', index=False)
                
                print("   ✅ Adaptive packs integrated into Excel export")
            
            # Summary sheet
            summary_data = {
                'Export Time': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Total Slots': [len(self.slots)],
                'Adaptive Packs Integrated': ['Yes' if adaptive_data else 'No'],
                'Hero Numbers Count': [len(adaptive_data.get('hero_numbers', [])) if adaptive_data else 0],
                'Golden Days Used': [len(adaptive_data.get('tens_core_golden', [])) if adaptive_data else 0]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='summary', index=False)
        
        print(f"✅ Pattern packs saved to: {output_dir / 'pattern_packs.json'}")
        print(f"✅ Pattern packs Excel saved to: {excel_file}")
        
        if adaptive_data:
            print(f"✅ Adaptive pattern packs integrated successfully")
            print(f"   Hero numbers: {len(adaptive_data.get('hero_numbers', []))}")
            print(f"   Golden digits: {len(adaptive_data.get('tens_core_golden', []))} tens, {len(adaptive_data.get('ones_core_golden', []))} ones")
        
        print("✅ PATTERN PACKS EXPORT COMPLETED SUCCESSFULLY")
        return True

def main():
    exporter = PatternPacksExporter()
    success = exporter.export_pattern_packs()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())