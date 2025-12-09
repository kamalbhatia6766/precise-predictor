import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import glob
import os
import re
from collections import Counter, defaultdict
import argparse

class DeepSeekFusion:
    """
    FUSION BRAIN - Combines predictions from all SCR scripts (1-11)
    Uses frequency + rank consensus to generate ultimate predictions
    """
    
    def __init__(self):
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.base_dir = Path(__file__).resolve().parent
        self.setup_directories()
        
    def setup_directories(self):
        """Create fusion output directory"""
        fusion_dir = self.base_dir / "predictions" / "fusion"
        fusion_dir.mkdir(parents=True, exist_ok=True)
    
    def get_opposite(self, n):
        """Get opposite number (23 → 32)"""
        if n < 10:
            return n * 10
        else:
            return (n % 10) * 10 + (n // 10)
    
    def find_script_predictions(self, script_pattern, target_date):
        """Find and parse predictions from a script pattern"""
        predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        
        try:
            # Search in script-specific directories
            search_patterns = [
                f"predictions/{script_pattern}/*.xlsx",
                f"predictions/{script_pattern}/*.txt", 
                f"predictions/{script_pattern}/*.csv",
                f"outputs/predictions/{script_pattern}/*.xlsx"
            ]
            
            for pattern in search_patterns:
                for file_path in glob.glob(str(self.base_dir / pattern)):
                    if self.is_recent_file(file_path, target_date):
                        script_preds = self.parse_prediction_file(file_path, target_date)
                        if script_preds:
                            for slot in self.slots:
                                predictions[slot].extend(script_preds.get(slot, []))
        
        except Exception as e:
            print(f"⚠️  Error processing {script_pattern}: {e}")
            
        return predictions
    
    def is_recent_file(self, file_path, target_date):
        """Check if file is relevant to target date"""
        try:
            # Extract date from filename
            filename = Path(file_path).stem.lower()
            date_patterns = [
                target_date.strftime("%Y%m%d"),
                target_date.strftime("%Y-%m-%d"),
                (target_date - timedelta(days=1)).strftime("%Y%m%d"),  # Yesterday's predictions for today
            ]
            
            return any(pattern in filename for pattern in date_patterns)
        except:
            return True  # If we can't determine, include it
    
    def parse_prediction_file(self, file_path, target_date):
        """Parse prediction file and extract numbers"""
        predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.xlsx':
                return self.parse_excel_file(file_path, target_date)
            elif file_ext in ['.txt', '.csv']:
                return self.parse_text_file(file_path)
            else:
                return self.guess_parse_file(file_path)
                
        except Exception as e:
            print(f"⚠️  Could not parse {file_path}: {e}")
            
        return predictions
    
    def parse_excel_file(self, file_path, target_date):
        """Parse Excel prediction files"""
        predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        
        try:
            # Try to read Excel file
            xl = pd.ExcelFile(file_path)
            
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df.columns = [str(col).strip().upper() for col in df.columns]
                
                # Look for date column and filter by target date
                date_columns = [col for col in df.columns if 'DATE' in col]
                if date_columns:
                    date_col = date_columns[0]
                    # Convert date column to comparable format
                    try:
                        df[date_col] = pd.to_datetime(df[date_col]).dt.date
                        df = df[df[date_col] == target_date]
                    except:
                        pass  # Use all rows if date parsing fails
                
                # Extract numbers from slot columns
                for slot in self.slots:
                    slot_columns = [col for col in df.columns if slot in col and 'OPP' not in col]
                    for col in slot_columns:
                        numbers = self.extract_numbers_from_column(df[col])
                        predictions[slot].extend(numbers)
                        
        except Exception as e:
            print(f"⚠️  Excel parsing failed for {file_path}: {e}")
            
        return predictions
    
    def parse_text_file(self, file_path):
        """Parse text prediction files"""
        predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Look for slot patterns like "FRBD: 12, 34, 56" or "FRBD (Top-15): 12, 34, 56"
            for slot in self.slots:
                slot_patterns = [
                    f"{slot}\\s*[:\\-]\\s*([\\d,\\s]+)",  # FRBD: 12, 34, 56
                    f"{slot}\\s*\\([^)]+\\)\\s*[:\\-]\\s*([\\d,\\s]+)",  # FRBD (Top-15): 12, 34, 56
                ]
                
                for pattern in slot_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for match in matches:
                        numbers = self.extract_numbers_from_string(match)
                        predictions[slot].extend(numbers)
                        
        except Exception as e:
            print(f"⚠️  Text parsing failed for {file_path}: {e}")
            
        return predictions
    
    def guess_parse_file(self, file_path):
        """Fallback parsing for unknown file types"""
        predictions = {"FRBD": [], "GZBD": [], "GALI": [], "DSWR": []}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Extract all 2-digit numbers
            all_numbers = re.findall(r'\b\d{2}\b', content)
            numbers_list = [int(num) for num in all_numbers if 0 <= int(num) <= 99]
            
            # Distribute numbers evenly across slots (fallback)
            chunk_size = max(1, len(numbers_list) // 4)
            for i, slot in enumerate(self.slots):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < 3 else len(numbers_list)
                predictions[slot] = numbers_list[start_idx:end_idx]
                
        except Exception as e:
            print(f"⚠️  Fallback parsing failed for {file_path}: {e}")
            
        return predictions
    
    def extract_numbers_from_column(self, column):
        """Extract numbers from pandas column"""
        numbers = []
        for value in column:
            if pd.notna(value):
                numbers.extend(self.extract_numbers_from_string(str(value)))
        return numbers
    
    def extract_numbers_from_string(self, text):
        """Extract 2-digit numbers from string"""
        numbers = []
        number_strings = re.findall(r'\b\d{2}\b', str(text))
        for num_str in number_strings:
            try:
                num = int(num_str)
                if 0 <= num <= 99:
                    numbers.append(num)
            except ValueError:
                continue
        return numbers
    
    def collect_all_script_predictions(self, target_date):
        """Collect predictions from all SCR scripts (1-11)"""
        print(f"🎯 Collecting predictions for {target_date}...")
        
        script_patterns = [
            "deepseek_scr1", "deepseek_scr2", "deepseek_scr3", "deepseek_scr4", "deepseek_scr5",
            "deepseek_scr6", "deepseek_scr7", "deepseek_scr8", "deepseek_scr9", 
            "deepseek_scr10", "deepseek_scr11"
        ]
        
        all_predictions = {}
        
        for script_pattern in script_patterns:
            print(f"   🔍 Scanning {script_pattern}...")
            script_preds = self.find_script_predictions(script_pattern, target_date)
            
            # Filter out empty predictions
            if any(script_preds.values()):
                all_predictions[script_pattern] = script_preds
                slot_counts = {slot: len(nums) for slot, nums in script_preds.items()}
                print(f"     ✅ Found: {slot_counts}")
            else:
                print(f"     ⚠️  No predictions found")
        
        print(f"📊 Collected predictions from {len(all_predictions)} scripts")
        return all_predictions
    
    def build_fusion_ranking(self, all_predictions):
        """Build fusion ranking using frequency + rank consensus"""
        fusion_results = {}
        
        for slot in self.slots:
            print(f"\n🎲 Building fusion for {slot}...")
            
            # Collect all numbers and their occurrences
            number_data = defaultdict(lambda: {'scripts': [], 'ranks': [], 'freq': 0})
            
            for script_name, predictions in all_predictions.items():
                slot_numbers = predictions.get(slot, [])
                for rank, number in enumerate(slot_numbers, 1):
                    number_data[number]['scripts'].append(script_name)
                    number_data[number]['ranks'].append(rank)
                    number_data[number]['freq'] += 1
            
            if not number_data:
                print(f"   ⚠️  No predictions found for {slot}")
                fusion_results[slot] = []
                continue
            
            # Prepare candidates with scores
            candidates = []
            for number, data in number_data.items():
                best_rank = min(data['ranks'])
                avg_rank = sum(data['ranks']) / len(data['ranks'])
                candidates.append({
                    'number': number,
                    'frequency': data['freq'],
                    'best_rank': best_rank,
                    'avg_rank': avg_rank,
                    'scripts': data['scripts']
                })
            
            # Sort by frequency (desc), then best_rank (asc), then avg_rank (asc)
            candidates.sort(key=lambda x: (-x['frequency'], x['best_rank'], x['avg_rank']))
            
            # Take top 15
            top_numbers = [candidate['number'] for candidate in candidates[:15]]
            
            print(f"   ✅ Top 5: {top_numbers[:5]}")
            print(f"   📊 Frequency range: {candidates[0]['frequency']}-{candidates[-1]['frequency']} scripts")
            
            fusion_results[slot] = top_numbers
        
        return fusion_results
    
    def generate_fusion_predictions(self, target_date):
        """Generate fusion predictions for target date"""
        # Collect all script predictions
        all_predictions = self.collect_all_script_predictions(target_date)
        
        if not all_predictions:
            print("❌ No predictions found from any script!")
            return None
        
        # Build fusion ranking
        fusion_ranking = self.build_fusion_ranking(all_predictions)
        
        # Prepare output data
        output_data = {
            'date': [target_date],
            'type': ['TOMORROW']  # Default type
        }
        
        for slot in self.slots:
            numbers = fusion_ranking.get(slot, [])
            output_data[slot] = [', '.join(f"{n:02d}" for n in numbers)]
            
            # Generate opposites for top 3 numbers
            top_3 = numbers[:3]
            opposites = [self.get_opposite(n) for n in top_3]
            output_data[f'{slot}_OPP'] = [', '.join(f"{n:02d}" for n in opposites)]
        
        return output_data
    
    def save_fusion_predictions(self, output_data, target_date):
        """Save fusion predictions to Excel file"""
        if not output_data:
            return None
        
        output_dir = self.base_dir / "predictions" / "fusion"
        filename = f"fusion_predictions_{target_date.strftime('%Y%m%d')}.xlsx"
        output_path = output_dir / filename
        
        try:
            df = pd.DataFrame(output_data)
            df.to_excel(output_path, index=False, sheet_name='fusion')
            
            print(f"💾 Fusion predictions saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving fusion predictions: {e}")
            return None
    
    def run(self, target_date):
        """Main execution"""
        print("=" * 70)
        print("🎯 DEEPSEEK FUSION BRAIN - Phase C1")
        print("   Combining SCR1-11 predictions using frequency + rank consensus")
        print("=" * 70)
        
        print(f"📅 Target Date: {target_date}")
        print(f"🕒 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Generate fusion predictions
        output_data = self.generate_fusion_predictions(target_date)
        
        if not output_data:
            print("❌ Fusion failed - no output generated")
            return False
        
        # Save predictions
        output_path = self.save_fusion_predictions(output_data, target_date)
        
        if not output_path:
            return False
        
        # Print summary
        print("\n" + "=" * 70)
        print("📊 FUSION PREDICTIONS SUMMARY")
        print("=" * 70)
        
        for slot in self.slots:
            numbers = output_data.get(slot, [''])[0]
            opposites = output_data.get(f'{slot}_OPP', [''])[0]
            print(f"\n{slot}:")
            print(f"  Numbers : {numbers}")
            print(f"  Opposites: {opposites}")
        
        print(f"\n✅ Fusion completed successfully!")
        print(f"📁 File: {output_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='DeepSeek Fusion Brain - Combine SCR1-11 predictions')
    parser.add_argument('--date', required=True, help='Target date in YYYY-MM-DD format')
    
    args = parser.parse_args()
    
    try:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    except ValueError:
        print("❌ Invalid date format. Use YYYY-MM-DD")
        return 1
    
    fusion = DeepSeekFusion()
    
    try:
        success = fusion.run(target_date)
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Fusion error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())