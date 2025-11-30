# precise_bet_engine.py - ULTRA v5 ROCKET MODE - CLEAR DATES + BREAKDOWN
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import re
from collections import Counter, defaultdict
import warnings
import argparse
import json
warnings.filterwarnings('ignore')

try:
    import pattern_packs
    PATTERN_PACKS_AVAILABLE = True
except ImportError:
    print("⚠️  pattern_packs.py not found - pattern bonuses disabled")
    PATTERN_PACKS_AVAILABLE = False

class PreciseBetEngine:
    def __init__(self):
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.base_unit = 10
        
        # ULTRA v5 constants
        self.N_DAYS = 30
        self.W_DIRECT = 0.15
        self.W_CROSS = 0.05
        self.W_S40_HIT = 0.05
        self.S40_BONUS = 0.20
        self.DIGIT_PACK_BONUS = 0.05
        self.MAX_PATTERN_BONUS = 0.5
        
        self.GOLDEN_DIGIT_BOOST = 0.08
        self.HERO_NUMBER_BOOST = 0.10
        self.TIME_AWARENESS_BOOST = 0.05
        self.MAX_QUANTUM_BOOST = 0.25
        
        self.NEAR_MISS_BOOST = 0.06
        self.MIRROR_BOOST = 0.04
        self.DIGITAL_ROOT_BOOST = 0.03
        
        self.EV_GAP = 0.03
        self.HARD_CAP = 12
        self.SOFT_CAP = 15
        self.MIN_BINS = 3
        self.MAX_PER_BIN = 6
        
        self.pattern_config = self.load_enhanced_pattern_intelligence()
        self.adaptive_packs = self.load_adaptive_pattern_packs()
        self.golden_insights = self.load_golden_insights()

    # ✅ ROCKET MODE: CRYSTAL CLEAR DATE MAPPING
    def print_rocket_summary(self, bets_df, summary_df, target_date, source_file, target_mode):
        """🚀 ULTRA CLEAR SUMMARY - NO CONFUSION"""
        system_today = datetime.now().date()
        
        print("\n" + "="*80)
        print("🎯 PRECISE BET ENGINE - ULTRA v5 ROCKET MODE")
        print("="*80)
        
        # ✅ CRYSTAL CLEAR DATE MAPPING
        print("📅 DATE MAPPING (CRYSTAL CLEAR):")
        print(f"   🖥️  System Date: {system_today}")
        print(f"   🎯 Target Date: {target_date}") 
        print(f"   📊 Mode: {target_mode.upper()}")
        print(f"   📁 Source: {source_file.name}")
        
        print(f"💰 Base Unit: ₹{self.base_unit}")
        
        print("\n📊 SLOT BREAKDOWN (MAIN + ANDAR/BAHAR):")
        print("-" * 80)
        
        grand_total = 0
        for slot in self.slots:
            slot_bets = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'Main')]
            slot_andars = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'ANDAR')]
            slot_bahars = bets_df[(bets_df['slot'] == slot) & (bets_df['layer_type'] == 'BAHAR')]
            
            if not slot_bets.empty:
                # Main numbers with tiers
                main_numbers = []
                main_total = 0
                for _, bet in slot_bets.iterrows():
                    number = bet['number_or_digit']
                    tier = bet['tier']
                    stake = bet['stake']
                    main_numbers.append(f"{number}({tier} ₹{stake})")
                    main_total += stake
                
                # ANDAR/BAHAR digits
                andar_digit = slot_andars['number_or_digit'].iloc[0] if not slot_andars.empty else "None"
                bahar_digit = slot_bahars['number_or_digit'].iloc[0] if not slot_bahars.empty else "None"
                andar_stake = slot_andars['stake'].iloc[0] if not slot_andars.empty else 0
                bahar_stake = slot_bahars['stake'].iloc[0] if not slot_bahars.empty else 0
                
                slot_total = main_total + andar_stake + bahar_stake
                grand_total += slot_total
                
                print(f"   {slot}:")
                print(f"     🔢 Main: {', '.join(main_numbers)}")
                print(f"     📊 ANDAR: {andar_digit}(₹{andar_stake}), BAHAR: {bahar_digit}(₹{bahar_stake})")
                print(f"     💰 Total: ₹{slot_total}")
                print()
        
        print(f"💵 GRAND TOTAL: ₹{grand_total}")
        print(f"🚀 ULTRA v5 QUANTUM SELF-LEARNING: ACTIVE")

    # ✅ ALL WORKING METHODS FROM YOUR CURRENT VERSION
    def find_latest_predictions_file(self, source='scr9'):
        if source == 'fusion':
            predictions_dir = Path(__file__).resolve().parent / "predictions" / "fusion"
            if not predictions_dir.exists():
                raise FileNotFoundError(f"Fusion directory not found: {predictions_dir}")
            fusion_files = list(predictions_dir.glob("fusion_predictions_*.xlsx"))
            if not fusion_files:
                raise FileNotFoundError("No fusion_predictions_*.xlsx files found")
            return max(fusion_files, key=lambda x: x.stat().st_mtime)
        else:
            scr9_dir = Path(__file__).resolve().parent / "predictions" / "deepseek_scr9"
            if not scr9_dir.exists():
                raise FileNotFoundError(f"SCR9 directory not found: {scr9_dir}")
            ultimate_files = list(scr9_dir.glob("ultimate_predictions_*.xlsx"))
            if not ultimate_files:
                raise FileNotFoundError("No ultimate_predictions_*.xlsx files found")
            return max(ultimate_files, key=lambda x: x.stat().st_mtime)

    def load_ultimate_predictions(self, file_path):
        try:
            df = pd.read_excel(file_path)
            df.columns = [str(col).strip().lower() for col in df.columns]
            print(f"📁 Loaded: {file_path.name}")
            return df
        except Exception as e:
            raise ValueError(f"Error loading predictions: {e}")

    def select_target_data(self, df, target_mode):
        df.columns = [str(col).strip().lower() for col in df.columns]
        
        if 'type' not in df.columns:
            print("⚠️  No 'type' column found - using first row")
            return df.iloc[[0]]
        
        print(f"🎯 Target mode: {target_mode}")
        
        if target_mode == 'today':
            target_rows = df[df['type'].str.upper() == 'TODAY_EMPTY']
            if not target_rows.empty:
                print("   Using TODAY_EMPTY rows")
                return target_rows
            else:
                print("⚠️  No TODAY_EMPTY rows found - falling back to TOMORROW")
                target_mode = 'tomorrow'
        
        if target_mode == 'tomorrow':
            target_rows = df[df['type'].str.upper() == 'TOMORROW']
            if not target_rows.empty:
                print("   Using TOMORROW rows")
                return target_rows
            else:
                print("⚠️  No TOMORROW rows found - using first available row")
                return df.iloc[[0]]
        
        if target_mode == 'auto':
            today_rows = df[df['type'].str.upper() == 'TODAY_EMPTY']
            if not today_rows.empty:
                print("   Auto-selected TODAY_EMPTY rows")
                return today_rows
            else:
                tomorrow_rows = df[df['type'].str.upper() == 'TOMORROW']
                if not tomorrow_rows.empty:
                    print("   Auto-selected TOMORROW rows (TODAY_EMPTY not available)")
                    return tomorrow_rows
                else:
                    print("⚠️  No TODAY_EMPTY or TOMORROW rows found - using first available row")
                    return df.iloc[[0]]
        
        return df.iloc[[0]]

    def convert_wide_to_long_format(self, df, target_rows):
        slot_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if (any(slot.lower() in col_lower for slot in self.slots) and 
                '_opp' not in col_lower and
                col_lower not in ['date', 'type']):
                slot_columns.append(col)
        
        if not slot_columns:
            raise ValueError("No slot columns found in the data")
        
        print(f"🔍 Found slot columns: {slot_columns}")
        
        long_data = []
        for _, row in target_rows.iterrows():
            date_val = row.get('date', '')
            for slot_col in slot_columns:
                numbers_str = row[slot_col]
                if pd.notna(numbers_str):
                    numbers = self.parse_numbers(str(numbers_str))
                    if numbers:
                        slot_name = None
                        for slot in self.slots:
                            if slot.lower() in slot_col.lower():
                                slot_name = slot
                                break
                        if slot_name:
                            long_data.append({
                                'date': date_val,
                                'slot': slot_name,
                                'numbers': numbers
                            })
        
        long_df = pd.DataFrame(long_data)
        print(f"🔄 Converted to long format: {len(long_df)} rows")
        return long_df

    def parse_numbers(self, numbers_str):
        if pd.isna(numbers_str):
            return []
        numbers = []
        for num_str in str(numbers_str).split(','):
            num_str = num_str.strip()
            if num_str.isdigit():
                try:
                    numbers.append(int(num_str))
                except ValueError:
                    continue
        return numbers

    def get_target_date(self, long_df):
        today = datetime.now().date()
        dates = []
        for date_val in long_df['date'].unique():
            try:
                if isinstance(date_val, datetime):
                    parsed_date = date_val.date()
                elif isinstance(date_val, str):
                    parsed_date = pd.to_datetime(date_val).date()
                else:
                    continue
                dates.append(parsed_date)
            except:
                continue
        
        if not dates:
            raise ValueError("No valid dates found in predictions")
        
        future_dates = [d for d in dates if d >= today]
        if not future_dates:
            print("⚠️  No future dates found - using latest date")
            target_date = max(dates)
        else:
            target_date = min(future_dates)
        
        return target_date

    def load_script_hit_memory(self, target_date):
        memory_file = Path(__file__).resolve().parent / "logs" / "performance" / "script_hit_memory.xlsx"
        if not memory_file.exists():
            print("⚠️  No script_hit_memory.xlsx found - using pure SCR9 ranks")
            return None
        try:
            df = pd.read_excel(memory_file)
            df['date'] = pd.to_datetime(df['date']).dt.date
            cutoff_date = target_date - timedelta(days=self.N_DAYS)
            filtered_df = df[df['date'] >= cutoff_date]
            print(f"📊 Loaded script hit memory: {len(filtered_df)} records")
            return filtered_df
        except Exception as e:
            print(f"⚠️  Error loading script_hit_memory: {e}")
            return None

    def build_history_table(self, memory_df):
        history = defaultdict(lambda: {'direct_hits': 0, 'cross_hits': 0, 's40_hits': 0, 'digit_tags': set()})
        if memory_df is None:
            return history
        for _, row in memory_df.iterrows():
            slot = row['real_slot']
            number = row['real_number']
            hit_type = row['hit_type']
            is_s40 = row.get('is_s40', False)
            digit_tags = str(row.get('digit_pack_tags', '')).split(',')
            key = (slot, number)
            if hit_type == 'DIRECT':
                history[key]['direct_hits'] += 1
            elif hit_type == 'CROSS_SLOT':
                history[key]['cross_hits'] += 1
            if is_s40:
                history[key]['s40_hits'] += 1
            for tag in digit_tags:
                if tag.strip():
                    history[key]['digit_tags'].add(tag.strip())
        return history

    def assign_tiers(self, shortlist):
        if not shortlist:
            return {}
        tiers = {}
        n = len(shortlist)
        tier_sizes = {'A': n // 3, 'B': n // 3, 'C': n - 2 * (n // 3)}
        idx = 0
        for tier, size in tier_sizes.items():
            for i in range(size):
                if idx < len(shortlist):
                    tiers[shortlist[idx]['number']] = tier
                    idx += 1
        return tiers

    def get_andar_bahar(self, shortlist):
        if not shortlist:
            return None, None
        tens_digits = [int(str(num['number']).zfill(2)[0]) for num in shortlist]
        ones_digits = [int(str(num['number']).zfill(2)[1]) for num in shortlist]
        tens_counter = Counter(tens_digits)
        ones_counter = Counter(ones_digits)
        def break_tie(counter, digit_list):
            max_count = max(counter.values())
            candidates = [d for d, count in counter.items() if count == max_count]
            if len(candidates) == 1:
                return candidates[0]
            positions = {}
            for candidate in candidates:
                for pos, item in enumerate(shortlist):
                    num_str = str(item['number']).zfill(2)
                    if (digit_list is tens_digits and int(num_str[0]) == candidate) or (digit_list is ones_digits and int(num_str[1]) == candidate):
                        positions[candidate] = pos
                        break
            return min(positions.keys(), key=lambda x: positions[x])
        andar_digit = break_tie(tens_counter, tens_digits)
        bahar_digit = break_tie(ones_counter, ones_digits)
        return andar_digit, bahar_digit

    def load_enhanced_pattern_intelligence(self):
        config = {
            "s40_enabled": True,
            "digit_packs_enabled": True,
            "memory_bonus_enabled": True,
            "pattern_weights": {
                "s40_bonus": self.S40_BONUS,
                "digit_pack_bonus": self.DIGIT_PACK_BONUS,
                "max_pattern_bonus": self.MAX_PATTERN_BONUS,
            },
        }
        try:
            pattern_file = Path(__file__).resolve().parent / "logs" / "performance" / "pattern_intelligence_config.json"
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    external_config = json.load(f)
                config.update(external_config)
                print("✅ Loaded enhanced pattern intelligence configuration")
        except Exception as e:
            print(f"⚠️  Error loading enhanced pattern config: {e}")
        return config

    def load_adaptive_pattern_packs(self):
        try:
            adaptive_file = Path(__file__).resolve().parent / "logs" / "performance" / "adaptive_pattern_packs.json"
            if adaptive_file.exists():
                with open(adaptive_file, 'r') as f:
                    adaptive_data = json.load(f)
                print("✅ Loaded adaptive pattern packs for quantum boost")
                return adaptive_data
        except Exception as e:
            print(f"⚠️  Error loading adaptive pattern packs: {e}")
        return {}

    def load_golden_insights(self):
        try:
            golden_file = Path(__file__).resolve().parent / "logs" / "performance" / "golden_block_insights.json"
            if golden_file.exists():
                with open(golden_file, 'r') as f:
                    insights = json.load(f)
                print("✅ Loaded golden block insights")
                return insights
        except Exception as e:
            print(f"⚠️  Error loading golden insights: {e}")
        return {}

    def calculate_enhanced_scores(self, numbers_list, slot, history, target_date):
        if not numbers_list:
            return []
        scored_numbers = []
        for rank, number in enumerate(numbers_list, 1):
            base_score = 1.0 / rank
            digit_tags = []
            history_key = (slot, number)
            
            quantum_boosted_score, quantum_debug = self.apply_pattern_boost(slot, number, base_score, target_date)
            pattern_bonus = 0.0
            
            if PATTERN_PACKS_AVAILABLE and self.pattern_config.get('s40_enabled', True):
                if pattern_packs.is_s40(number):
                    s40_bonus = self.pattern_config.get('pattern_weights', {}).get('s40_bonus', self.S40_BONUS)
                    pattern_bonus += s40_bonus
                    
                if self.pattern_config.get('digit_packs_enabled', True):
                    digit_tags = pattern_packs.get_digit_pack_tags(number)
                    # ✅ UPDATED: Family-level bonuses instead of per-pack
                    families = self._get_family_categories(digit_tags)
                    for fam in families:
                        fam_bonus = self.DIGIT_PACK_BONUS
                        pattern_bonus += fam_bonus
            
            memory_bonus = 0.0
            if self.pattern_config.get('memory_bonus_enabled', True) and history_key in history:
                h = history[history_key]
                memory_bonus = (
                    self.W_DIRECT * h['direct_hits'] +
                    self.W_CROSS * h['cross_hits'] +
                    self.W_S40_HIT * h['s40_hits']
                )
                memory_bonus = memory_bonus / max(1, self.N_DAYS / 10)
            
            for other_slot in self.slots:
                if other_slot != slot:
                    cross_boost = self.get_cross_slot_boost(other_slot, slot, number)
                    pattern_bonus += cross_boost
            
            time_boost = self.get_time_awareness_boost(slot, target_date)
            if time_boost > 1.0:
                pattern_bonus += (time_boost - 1.0) * 0.5
            
            digit_boost = self.get_digit_preference_boost(slot, number)
            pattern_bonus += digit_boost
            
            max_pattern = self.pattern_config.get('pattern_weights', {}).get('max_pattern_bonus', self.MAX_PATTERN_BONUS)
            pattern_bonus = min(pattern_bonus, max_pattern)
            
            final_score = quantum_boosted_score + pattern_bonus + memory_bonus
            
            direct_hits = history[history_key]['direct_hits'] if history_key in history else 0
            cross_hits = history[history_key]['cross_hits'] if history_key in history else 0
            s40_hits = history[history_key]['s40_hits'] if history_key in history else 0
            
            scored_numbers.append({
                'number': number,
                'rank': rank,
                'base_score': base_score,
                'pattern_bonus': pattern_bonus,
                'memory_bonus': memory_bonus,
                'final_score': final_score,
                'is_s40': PATTERN_PACKS_AVAILABLE and pattern_packs.is_s40(number),
                'digit_pack_tags': ','.join(digit_tags),
                'direct_hits_30d': direct_hits,
                'cross_hits_30d': cross_hits,
                's40_hits_30d': s40_hits,
                'quantum_boosted_score': quantum_boosted_score,
                'quantum_boost_components': quantum_debug,
            })
        return scored_numbers

    def _get_family_categories(self, digit_tags):
        """Convert fine-grained pack tags to family categories"""
        families = set()
        for tag in digit_tags:
            if tag.startswith("pack2_"):
                families.add("PACK_2DIGIT")
            elif tag.startswith("pack3_"):
                families.add("PACK_3DIGIT")
            elif tag.startswith("pack4_"):
                families.add("PACK_4DIGIT")
            elif tag.startswith("pack5_"):
                families.add("PACK_5DIGIT")
            elif tag.startswith("pack6_"):
                families.add("PACK_6DIGIT")
            elif tag == "PACK_164950":
                families.add("PACK_164950")
            elif tag == "S40":
                families.add("S40")
        return families

    def apply_pattern_boost(self, slot, number, base_score, target_date):
        debug_components = {
            'base_score': base_score,
            'digit_match_boost': 0.0,
            'hero_number_boost': 0.0,
            'time_boost': 0.0,
            'cross_slot_boost': 0.0,
            'near_miss_boost': 0.0,
            'mirror_boost': 0.0,
            'digital_root_boost': 0.0,
            'total_quantum_boost': 0.0
        }
        quantum_boost = 0.0
        tens_digit = number // 10
        ones_digit = number % 10
        
        if self.adaptive_packs.get('tens_core_base') and tens_digit in self.adaptive_packs['tens_core_base']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST * 0.6
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST * 0.6
        if self.adaptive_packs.get('ones_core_base') and ones_digit in self.adaptive_packs['ones_core_base']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST * 0.6
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST * 0.6
        if self.adaptive_packs.get('tens_core_golden') and tens_digit in self.adaptive_packs['tens_core_golden']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST
        if self.adaptive_packs.get('ones_core_golden') and ones_digit in self.adaptive_packs['ones_core_golden']:
            quantum_boost += self.GOLDEN_DIGIT_BOOST
            debug_components['digit_match_boost'] += self.GOLDEN_DIGIT_BOOST
        
        if self.adaptive_packs.get('hero_numbers') and number in self.adaptive_packs['hero_numbers']:
            quantum_boost += self.HERO_NUMBER_BOOST
            debug_components['hero_number_boost'] += self.HERO_NUMBER_BOOST
        
        current_day = target_date.strftime('%A')
        time_boost_slots = self.adaptive_packs.get('time_boost_slots', {})
        if slot in time_boost_slots and time_boost_slots[slot].get('best_day') == current_day:
            quantum_boost += self.TIME_AWARENESS_BOOST
            debug_components['time_boost'] += self.TIME_AWARENESS_BOOST
        
        cross_pairs = self.adaptive_packs.get('cross_slot_pairs_top', [])
        for pair in cross_pairs[:2]:
            if f"→{slot}" in pair:
                quantum_boost += 0.03
                debug_components['cross_slot_boost'] += 0.03
        
        if hasattr(self, 'real_numbers_history'):
            near_miss_boost = self.get_near_miss_boost(number, self.real_numbers_history)
            quantum_boost += near_miss_boost
            debug_components['near_miss_boost'] += near_miss_boost
        
        mirror_boost = self.get_mirror_boost(number)
        quantum_boost += mirror_boost
        debug_components['mirror_boost'] += mirror_boost
        
        digital_root_boost = self.get_digital_root_boost(number)
        quantum_boost += digital_root_boost
        debug_components['digital_root_boost'] += digital_root_boost
        
        quantum_boost = min(quantum_boost, self.MAX_QUANTUM_BOOST)
        debug_components['total_quantum_boost'] = quantum_boost
        boosted_score = base_score * (1.0 + quantum_boost)
        return boosted_score, debug_components

    def get_near_miss_boost(self, number, real_numbers_history):
        boost = 0.0
        for real_num in real_numbers_history:
            if abs(number - real_num) == 1:
                boost += self.NEAR_MISS_BOOST
        return min(boost, self.NEAR_MISS_BOOST * 3)

    def get_mirror_boost(self, number):
        if number < 10:
            return 0.0
        tens = number // 10
        ones = number % 10
        mirror_num = ones * 10 + tens
        if mirror_num == number:
            return 0.0
        return self.MIRROR_BOOST

    def get_digital_root_boost(self, number):
        digital_root = number
        while digital_root > 9:
            digital_root = sum(int(d) for d in str(digital_root))
        common_roots = [3, 6, 9]
        if digital_root in common_roots:
            return self.DIGITAL_ROOT_BOOST
        return 0.0

    def get_cross_slot_boost(self, real_slot, predicted_slot, number):
        boost_key = f"{real_slot}_{predicted_slot}"
        cross_boost = self.pattern_config.get('cross_slot_boost', {}).get(boost_key, 0.0)
        if hasattr(self, 'history'):
            history_key = (real_slot, number)
            if history_key in self.history:
                cross_hits = self.history[history_key]['cross_hits']
                cross_boost += min(0.2, cross_hits * 0.05)
        return cross_boost

    def get_time_awareness_boost(self, slot, target_date):
        time_awareness = self.pattern_config.get('time_awareness', {})
        slot_config = time_awareness.get(slot, {})
        current_day = target_date.strftime('%A')
        preferred_day = slot_config.get('preferred_day')
        if current_day == preferred_day:
            return slot_config.get('boost_factor', 1.0)
        return 1.0

    def get_digit_preference_boost(self, slot, number):
        digit_preferences = self.pattern_config.get('digit_preferences', {})
        slot_digits = digit_preferences.get(slot, {})
        tens_digit = number // 10
        ones_digit = number % 10
        boost = 0.0
        if tens_digit in slot_digits.get('common_tens', []):
            boost += 0.05
        if ones_digit in slot_digits.get('common_ones', []):
            boost += 0.05
        return boost

    def build_dynamic_shortlist(self, scored_numbers, ev_gap=0.03, hard_cap=12, soft_cap=15):
        if not scored_numbers:
            return [], []
        sorted_numbers = sorted(scored_numbers, key=lambda x: (-x['final_score'], x['rank']))
        if not sorted_numbers:
            return [], []
        top_score = sorted_numbers[0]['final_score']
        shortlist = sorted_numbers[:3]
        for item in sorted_numbers[3:]:
            if item['final_score'] >= top_score * (1 - ev_gap) and len(shortlist) < hard_cap:
                shortlist.append(item)
        if len(sorted_numbers) >= 10:
            top_10_scores = [x['final_score'] for x in sorted_numbers[:10]]
            score_range = max(top_10_scores) - min(top_10_scores)
            if score_range <= 2 * ev_gap * top_score:
                for item in sorted_numbers[len(shortlist):]:
                    if item['final_score'] >= top_score * (1 - ev_gap) and len(shortlist) < soft_cap:
                        shortlist.append(item)
        shortlist = self._apply_diversity_guards(shortlist, sorted_numbers)
        shortlist = self._apply_mirror_hedge(shortlist, sorted_numbers, top_score, ev_gap)
        shortlist = sorted(shortlist, key=lambda x: (-x['final_score'], x['rank']))
        shortlisted_numbers = [item['number'] for item in shortlist]
        for item in scored_numbers:
            item['shortlisted'] = item['number'] in shortlisted_numbers
        return shortlist, scored_numbers

    def _apply_diversity_guards(self, shortlist, all_sorted_numbers):
        if not shortlist:
            return shortlist
        bins = [item['number'] // 10 for item in shortlist]
        bin_counts = Counter(bins)
        unique_bins = set(bins)
        if len(unique_bins) < self.MIN_BINS:
            missing_bins = set(range(10)) - unique_bins
            for bin_num in missing_bins:
                if len(unique_bins) >= self.MIN_BINS:
                    break
                for candidate in all_sorted_numbers:
                    if candidate['number'] // 10 == bin_num and candidate not in shortlist:
                        most_common_bin = bin_counts.most_common(1)[0][0]
                        worst_in_bin = None
                        for item in shortlist:
                            if item['number'] // 10 == most_common_bin:
                                if worst_in_bin is None or item['final_score'] < worst_in_bin['final_score']:
                                    worst_in_bin = item
                        if worst_in_bin and worst_in_bin['final_score'] < candidate['final_score']:
                            shortlist.remove(worst_in_bin)
                            shortlist.append(candidate)
                            bin_counts[most_common_bin] -= 1
                            bin_counts[bin_num] = bin_counts.get(bin_num, 0) + 1
                            unique_bins.add(bin_num)
                        break
        for bin_num, count in list(bin_counts.items()):
            if count > self.MAX_PER_BIN:
                bin_items = [item for item in shortlist if item['number'] // 10 == bin_num]
                bin_items_sorted = sorted(bin_items, key=lambda x: x['final_score'])
                while len(bin_items) > self.MAX_PER_BIN and bin_items_sorted:
                    worst = bin_items_sorted.pop(0)
                    shortlist.remove(worst)
                    bin_items.remove(worst)
        return shortlist

    def _apply_mirror_hedge(self, shortlist, all_sorted_numbers, top_score, ev_gap):
        if len(shortlist) < 2:
            return shortlist
        top_numbers = shortlist[:2]
        for top_item in top_numbers:
            number = top_item['number']
            mirror_num = int(str(number).zfill(2)[::-1])
            if mirror_num == number or any(item['number'] == mirror_num for item in shortlist):
                continue
            mirror_item = None
            for candidate in all_sorted_numbers:
                if candidate['number'] == mirror_num:
                    mirror_item = candidate
                    break
            if mirror_item and mirror_item['final_score'] >= top_score * (1 - 2 * ev_gap):
                if len(shortlist) < self.HARD_CAP:
                    shortlist.append(mirror_item)
                else:
                    worst_item = min(shortlist, key=lambda x: x['final_score'])
                    if mirror_item['final_score'] > worst_item['final_score']:
                        shortlist.remove(worst_item)
                        shortlist.append(mirror_item)
        return shortlist

    # ✅ FIXED: Near-miss bug resolved - proper date comparison
    def load_real_numbers_history(self, days=30, target_date=None):
        try:
            real_file = Path(__file__).resolve().parent / "number prediction learn.xlsx"
            real_df = self.load_real_results(real_file)
            if real_df.empty:
                print("   No real results data found")
                return []
            
            # ✅ FIX: Ensure date column is datetime64[ns]
            real_df['date'] = pd.to_datetime(real_df['date'])
            
            if target_date is None:
                target_date = real_df['date'].max()
            else:
                # ✅ FIX: Convert target_date to same type as real_df['date']
                target_date = pd.to_datetime(target_date)
            
            start_date = target_date - timedelta(days=days)
            
            # ✅ FIX: Now both are same type - comparison will work
            filtered_df = real_df[(real_df['date'] >= start_date) & (real_df['date'] <= target_date)]
            real_numbers = filtered_df['number'].tolist()
            print(f"   Loaded {len(real_numbers)} real numbers from last {days} calendar days")
            return real_numbers
        except Exception as e:
            print(f"⚠️  Error loading real numbers history: {e}")
            return []

    def load_real_results(self, file_path):
        try:
            df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=None)
            all_data = []
            current_year = 2025
            current_month = 1
            for idx, row in df_raw.iterrows():
                if pd.notna(row[0]) and '2025' in str(row[0]):
                    try:
                        date_str = str(row[0]).strip()
                        if '2025' in date_str:
                            date_parts = date_str.split('-')
                            if len(date_parts) >= 2:
                                current_month = int(date_parts[1])
                    except:
                        current_month += 1
                    continue
                if pd.notna(row[0]) and str(row[0]).strip().isdigit():
                    day_num = int(row[0])
                    for slot_idx, slot_name in enumerate(self.slots, 1):
                        if (pd.notna(row[slot_idx]) and 
                            str(row[slot_idx]).strip() not in ['XX', ''] and
                            str(row[slot_idx]).strip().isdigit()):
                            try:
                                date_obj = datetime(current_year, current_month, day_num)
                                number = int(str(row[slot_idx]).strip())
                                all_data.append({
                                    'date': date_obj,
                                    'slot': slot_name,
                                    'number': number
                                })
                            except:
                                continue
            real_df = pd.DataFrame(all_data)
            return real_df
        except Exception as e:
            print(f"⚠️  Error loading real results: {e}")
            return pd.DataFrame()

    # ✅ NEW METHOD: Intraday scoring with family multipliers
    def calculate_enhanced_scores_intraday(self, numbers_list, slot, history, target_date, family_multipliers):
        """
        Intraday scoring with pattern family multipliers
        PRESERVES ALL EXISTING v1 LOGIC + adds family multipliers
        """
        if not numbers_list:
            return []
            
        scored_numbers = []
        for rank, number in enumerate(numbers_list, 1):
            base_score = 1.0 / rank
            digit_tags = []
            history_key = (slot, number)
            
            # ✅ PRESERVE EXACT v1 QUANTUM BOOST LOGIC
            quantum_boosted_score, quantum_debug = self.apply_pattern_boost(slot, number, base_score, target_date)
            pattern_bonus = 0.0
            
            # ✅ ENHANCED: Apply family multipliers to pattern bonuses
            if PATTERN_PACKS_AVAILABLE and self.pattern_config.get('s40_enabled', True):
                if pattern_packs.is_s40(number):
                    s40_bonus = self.pattern_config.get('pattern_weights', {}).get('s40_bonus', self.S40_BONUS)
                    # Apply S40 family multiplier
                    if 'S40' in family_multipliers:
                        s40_bonus *= family_multipliers['S40']
                    pattern_bonus += s40_bonus
                    
                if self.pattern_config.get('digit_packs_enabled', True):
                    digit_tags = pattern_packs.get_digit_pack_tags(number)
                    # ✅ UPDATED: Family-level bonuses with multipliers
                    families = self._get_family_categories(digit_tags)
                    for fam in families:
                        fam_bonus = self.DIGIT_PACK_BONUS
                        if fam in family_multipliers:
                            fam_bonus *= family_multipliers[fam]
                        pattern_bonus += fam_bonus
            
            # ✅ PRESERVE EXACT v1 MEMORY BONUS LOGIC
            memory_bonus = 0.0
            if self.pattern_config.get('memory_bonus_enabled', True) and history_key in history:
                h = history[history_key]
                memory_bonus = (
                    self.W_DIRECT * h['direct_hits'] +
                    self.W_CROSS * h['cross_hits'] +
                    self.W_S40_HIT * h['s40_hits']
                )
                memory_bonus = memory_bonus / max(1, self.N_DAYS / 10)
            
            # ✅ PRESERVE EXACT v1 CROSS-SLOT BOOST LOGIC
            for other_slot in self.slots:
                if other_slot != slot:
                    cross_boost = self.get_cross_slot_boost(other_slot, slot, number)
                    pattern_bonus += cross_boost
            
            # ✅ PRESERVE EXACT v1 TIME BOOST LOGIC
            time_boost = self.get_time_awareness_boost(slot, target_date)
            if time_boost > 1.0:
                pattern_bonus += (time_boost - 1.0) * 0.5
            
            # ✅ PRESERVE EXACT v1 DIGIT PREFERENCE LOGIC
            digit_boost = self.get_digit_preference_boost(slot, number)
            pattern_bonus += digit_boost
            
            max_pattern = self.pattern_config.get('pattern_weights', {}).get('max_pattern_bonus', self.MAX_PATTERN_BONUS)
            pattern_bonus = min(pattern_bonus, max_pattern)
            
            # ✅ FINAL SCORE COMBINATION (PRESERVED FROM v1)
            final_score = quantum_boosted_score + pattern_bonus + memory_bonus
            
            direct_hits = history[history_key]['direct_hits'] if history_key in history else 0
            cross_hits = history[history_key]['cross_hits'] if history_key in history else 0
            s40_hits = history[history_key]['s40_hits'] if history_key in history else 0
            
            scored_numbers.append({
                'number': number,
                'rank': rank,
                'base_score': base_score,
                'pattern_bonus': pattern_bonus,
                'memory_bonus': memory_bonus,
                'final_score': final_score,
                'is_s40': PATTERN_PACKS_AVAILABLE and pattern_packs.is_s40(number),
                'digit_pack_tags': ','.join(digit_tags),
                'direct_hits_30d': direct_hits,
                'cross_hits_30d': cross_hits,
                's40_hits_30d': s40_hits,
                'quantum_boosted_score': quantum_boosted_score,
                'quantum_boost_components': quantum_debug,
            })
            
        return scored_numbers

    # ✅ MODIFIED METHOD: Added intraday support parameters with safe defaults
    def generate_enhanced_bet_plan(self, df, target_rows, target_date, history, 
                                   slot_filter=None, family_multipliers=None, mode="normal"):
        """
        Enhanced with intraday recalculation support
        PRESERVES EXACT v1 BEHAVIOR WHEN NO NEW PARAMETERS PROVIDED
        """
        
        # ✅ PRESERVE EXACT v1 BEHAVIOR - NO CHANGES TO EXISTING LOGIC
        bets_data = []
        summary_data = []
        diagnostic_data = []
        quantum_debug_data = []
        ultra_debug_data = []
        explainability_data = []
        
        self.history = history
        print("🔍 Loading real numbers history for near-miss learning...")
        self.real_numbers_history = self.load_real_numbers_history(30, target_date)
        print(f"   Loaded {len(self.real_numbers_history)} real numbers for near-miss analysis")
        
        long_df = self.convert_wide_to_long_format(df, target_rows)
        target_df = long_df.copy()
        target_df['date_parsed'] = target_df['date'].apply(
            lambda x: pd.to_datetime(x).date() if isinstance(x, str) else x.date() if isinstance(x, datetime) else x
        )
        target_df = target_df[target_df['date_parsed'] == target_date]
        
        if target_df.empty:
            raise ValueError(f"No predictions found for target date {target_date}")
        
        # ✅ INTRADAY SUPPORT: Filter slots if provided (preserves v1 behavior when None)
        slots_to_process = slot_filter if slot_filter is not None else self.slots
        
        for slot in slots_to_process:
            if slot not in self.slots:
                print(f"⚠️  Skipping invalid slot: {slot}")
                continue
                
            slot_data = target_df[target_df['slot'] == slot]
            if slot_data.empty:
                print(f"⚠️  No data found for slot {slot} on {target_date}")
                continue
            
            numbers_list = slot_data['numbers'].iloc[0]
            
            # ✅ INTRADAY SUPPORT: Apply family multipliers if provided
            if mode == "intraday" and family_multipliers:
                print(f"🎯 Processing {slot} with intraday learning...")
                scored_numbers = self.calculate_enhanced_scores_intraday(numbers_list, slot, history, target_date, family_multipliers)
            else:
                # ✅ EXACT v1 BEHAVIOR PRESERVED
                print(f"🎯 Processing {slot}: {len(numbers_list)} numbers")
                scored_numbers = self.calculate_enhanced_scores(numbers_list, slot, history, target_date)
            
            if not scored_numbers:
                print(f"   Empty scored list - using fallback")
                continue
            
            shortlist, all_scored = self.build_dynamic_shortlist(scored_numbers)
            tiers = self.assign_tiers(shortlist)
            
            for item in all_scored:
                diagnostic_data.append({
                    'date': target_date,
                    'slot': slot,
                    'number': item['number'],
                    'base_rank': item['rank'],
                    'base_score': item['base_score'],
                    'is_s40': item['is_s40'],
                    'digit_pack_tags': item['digit_pack_tags'],
                    'direct_hits_30d': item['direct_hits_30d'],
                    'cross_hits_30d': item['cross_hits_30d'],
                    's40_hits_30d': item['s40_hits_30d'],
                    'pattern_bonus': item['pattern_bonus'],
                    'memory_bonus': item['memory_bonus'],
                    'final_score': item['final_score'],
                    'shortlisted': item['shortlisted'],
                    'quantum_boosted_score': item.get('quantum_boosted_score', 0),
                })
                
                quantum_components = item.get('quantum_boost_components', {})
                quantum_debug_data.append({
                    'date': target_date,
                    'slot': slot,
                    'number': item['number'],
                    'base_score': quantum_components.get('base_score', 0),
                    'digit_match_boost': quantum_components.get('digit_match_boost', 0),
                    'hero_number_boost': quantum_components.get('hero_number_boost', 0),
                    'time_boost': quantum_components.get('time_boost', 0),
                    'cross_slot_boost': quantum_components.get('cross_slot_boost', 0),
                    'near_miss_boost': quantum_components.get('near_miss_boost', 0),
                    'mirror_boost': quantum_components.get('mirror_boost', 0),
                    'digital_root_boost': quantum_components.get('digital_root_boost', 0),
                    'total_quantum_boost': quantum_components.get('total_quantum_boost', 0),
                    'final_quantum_score': item.get('quantum_boosted_score', 0)
                })
            
            if shortlist:
                shortlist_numbers = [item['number'] for item in shortlist]
                print(f"   Shortlist ({len(shortlist)}): {', '.join(str(n) for n in shortlist_numbers)}")
            else:
                print(f"   Empty shortlist - using fallback")
                shortlist = scored_numbers[:3]
                tiers = self.assign_tiers(shortlist)
            
            andar_digit, bahar_digit = self.get_andar_bahar(shortlist)
            main_stake_total = 0
            main_max_return = 0
            
            for item in shortlist:
                number = item['number']
                tier = tiers.get(number, 'C')
                if tier == 'A':
                    stake = 2 * self.base_unit
                elif tier == 'B':
                    stake = 1 * self.base_unit
                else:
                    stake = 0.5 * self.base_unit
                
                potential_return = stake * 90
                main_stake_total += stake
                main_max_return += potential_return
                
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'Main',
                    'number_or_digit': f"{number:02d}",
                    'tier': tier,
                    'stake': stake,
                    'potential_return': potential_return,
                    'source_rank': item['rank'],
                    'notes': f"ULTRA v5 scoring: final_score={item['final_score']:.3f}",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            if andar_digit is not None:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'ANDAR',
                    'number_or_digit': str(andar_digit),
                    'tier': 'NA',
                    'stake': self.base_unit,
                    'potential_return': 90,
                    'source_rank': '',
                    'notes': f"most frequent tens digit",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            if bahar_digit is not None:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': 'BAHAR',
                    'number_or_digit': str(bahar_digit),
                    'tier': 'NA',
                    'stake': self.base_unit,
                    'potential_return': 90,
                    'source_rank': '',
                    'notes': f"most frequent ones digit",
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            for layer_type in ['S36', 'PackCore', 'PackBooster']:
                bets_data.append({
                    'date': target_date,
                    'slot': slot,
                    'layer_type': layer_type,
                    'number_or_digit': '',
                    'tier': 'NA',
                    'stake': '',
                    'potential_return': '',
                    'source_rank': '',
                    'notes': 'ULTRA v5 implementation',
                    'actual_result': '',
                    'hit_flag_main': '',
                    'hit_flag_andar': '',
                    'hit_flag_bahar': '',
                    'net_pnl': ''
                })
            
            andar_stake = self.base_unit if andar_digit is not None else 0
            bahar_stake = self.base_unit if bahar_digit is not None else 0
            total_stake = main_stake_total + andar_stake + bahar_stake
            max_total_return = main_max_return + (90 if andar_digit else 0) + (90 if bahar_digit else 0)
            
            summary_data.append({
                'date': target_date,
                'slot': slot,
                'main_count': len(shortlist),
                'main_stake': main_stake_total,
                'main_max_return': main_max_return,
                'andar_stake': andar_stake,
                'bahar_stake': bahar_stake,
                'total_stake': total_stake,
                'max_total_return': max_total_return,
                'enhanced_patterns': True,
                'ultra_mode': True,
                'dynamic_top_n': True
            })
            
            print(f"   Main: {len(shortlist)} numbers, ₹{main_stake_total} stake")
            print(f"   ANDAR: {andar_digit}, BAHAR: {bahar_digit}")
            print(f"   Total stake: ₹{total_stake}, Max return: ₹{max_total_return}")
        
        bets_df = pd.DataFrame(bets_data)
        summary_df = pd.DataFrame(summary_data)
        diagnostic_df = pd.DataFrame(diagnostic_data)
        quantum_debug_df = pd.DataFrame(quantum_debug_data)
        ultra_debug_df = pd.DataFrame(ultra_debug_data)
        explainability_df = pd.DataFrame(explainability_data)
        
        return bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df

    def save_bet_plan(self, bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df, target_date):
        output_dir = Path(__file__).resolve().parent / "predictions" / "bet_engine"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"bet_plan_master_{target_date.strftime('%Y%m%d')}.xlsx"
        file_path = output_dir / filename
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            bets_df.to_excel(writer, sheet_name='bets', index=False)
            summary_df.to_excel(writer, sheet_name='summary', index=False)
            diagnostic_df.to_excel(writer, sheet_name='diagnostic_scores', index=False)
            quantum_debug_df.to_excel(writer, sheet_name='quantum_debug', index=False)
            ultra_debug_df.to_excel(writer, sheet_name='ultra_debug', index=False)
            explainability_df.to_excel(writer, sheet_name='explainability', index=False)
        
        explain_json = output_dir / f"bet_engine_explainability_{target_date.strftime('%Y%m%d')}.json"
        explain_summary = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.strftime('%Y-%m-%d'),
            "total_numbers_analyzed": len(explainability_df),
            "dynamic_top_n_used": True,
            "avg_final_score": explainability_df['final_score'].mean() if not explainability_df.empty else 0,
        }
        
        with open(explain_json, 'w') as f:
            json.dump(explain_summary, f, indent=2)
        
        return file_path

def main():
    parser = argparse.ArgumentParser(description='Precise Bet Engine v5 Rocket - Ultra clear output')
    parser.add_argument('--target', choices=['today', 'tomorrow', 'auto'], default='tomorrow')
    parser.add_argument('--source', choices=['scr9', 'fusion'], default='scr9')
    
    args = parser.parse_args()
    
    try:
        engine = PreciseBetEngine()
        
        print(f"🔍 Locating latest {args.source.upper()} predictions...")
        latest_file = engine.find_latest_predictions_file(args.source)
        df = engine.load_ultimate_predictions(latest_file)
        
        target_rows = engine.select_target_data(df, args.target)
        long_df = engine.convert_wide_to_long_format(df, target_rows)
        target_date = engine.get_target_date(long_df)
        
        # ✅ ROCKET MODE: Clear date mapping
        system_today = datetime.now().date()
        print(f"\n📅 DATE MAPPING (CRYSTAL CLEAR):")
        print(f"   🖥️  System Date: {system_today}")
        print(f"   🎯 Target Date: {target_date}") 
        print(f"   📊 Mode: {args.target.upper()}")
        
        print("🧠 Loading script hit memory...")
        memory_df = engine.load_script_hit_memory(target_date)
        history = engine.build_history_table(memory_df)
        
        print("🎲 Generating ULTRA bet plan...")
        bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df = engine.generate_enhanced_bet_plan(df, target_rows, target_date, history)
        
        if bets_df.empty:
            print("❌ No bets generated")
            return 1
        
        output_path = engine.save_bet_plan(bets_df, summary_df, diagnostic_df, quantum_debug_df, ultra_debug_df, explainability_df, target_date)
        print(f"💾 Bet plan saved: {output_path}")
        
        # ✅ ROCKET MODE: Ultra clear summary
        engine.print_rocket_summary(bets_df, summary_df, target_date, latest_file, args.target)
        
        print(f"✅ ULTRA v5 ROCKET MODE completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())