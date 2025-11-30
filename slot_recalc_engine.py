# slot_recalc_engine.py - Enhanced with central pack registry + better logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
from precise_bet_engine import PreciseBetEngine

# 🆕 Import central pack registry
import pattern_packs


def _print_central_pack_registry(prefix: str = ""):
    """Log a safe, best-effort summary of the central pack registry."""
    try:
        stats = pattern_packs.get_pack_universe_stats()
        # This summary is aligned with the current pattern_packs.get_pack_universe_stats() schema
        total_packs = (
            stats.get("total_digit_packs")
            or stats.get("total_packs_2_to_6")
            or stats.get("total_packs")
            or 0
        )

        s40_count = len(getattr(pattern_packs, "S40", []))
        if not s40_count:
            s40_count = stats.get("s40_count") or 0

        if hasattr(pattern_packs, "PACK_164950_NUMBERS"):
            pack_164950_count = len(pattern_packs.PACK_164950_NUMBERS)
        elif hasattr(pattern_packs, "PACK_164950_FAMILY"):
            pack_164950_count = len(pattern_packs.PACK_164950_FAMILY)
        else:
            pack_164950_count = stats.get("pack_164950_count") or 0

        print(
            f"{prefix}🧮 Using central pack registry: {total_packs} digit packs, "
            f"S40={s40_count}, 164950={pack_164950_count}"
        )
    except Exception:
        print(
            f"{prefix}🧮 Using central pack registry (stats summary unavailable, but registry is active)"
        )

class SlotRecalcEngine:
    def __init__(self, verbose=False):
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.base_unit = 10
        self.verbose = verbose
        # 🆕 Print central pack usage sanity line        
        # Load configuration
        self.config = self.load_recalc_config()
        self.precise_engine = PreciseBetEngine()
        
    def load_recalc_config(self):
        config_file = Path(__file__).resolve().parent / "slot_recalc_config.json"
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print("✅ Loaded slot recalculation configuration")
            return config
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            return {
                "hit_family_multiplier": 1.2,
                "unhit_family_multiplier": 0.8,
                "max_stake_multiplier": 2.0,
                "min_stake_multiplier": 0.5
            }

    def load_bet_plan(self, target_date):
        bets_dir = Path(__file__).resolve().parent / "predictions" / "bet_engine"
        date_str = target_date.strftime("%Y%m%d")
        file_pattern = f"bet_plan_master_{date_str}.xlsx"
        
        bet_files = list(bets_dir.glob(file_pattern))
        if not bet_files:
            raise FileNotFoundError(f"No bet plan found for {target_date}")
        
        bet_file = bet_files[0]
        print(f"   📁 Bet plan file: {bet_file.name}")
        
        try:
            bets_df = pd.read_excel(bet_file, sheet_name='bets')
            return bets_df
        except Exception as e:
            raise ValueError(f"Error loading bet plan: {e}")

    def load_real_results_for_date(self, target_date):
        try:
            real_df = self.precise_engine.load_real_results(
                Path(__file__).resolve().parent / "number prediction learn.xlsx"
            )
            if real_df.empty:
                return pd.DataFrame()
            
            # Convert dates for comparison
            real_df['date'] = pd.to_datetime(real_df['date']).dt.date
            target_date = target_date.date() if isinstance(target_date, datetime) else target_date
            
            date_results = real_df[real_df['date'] == target_date]
            return date_results
        except Exception as e:
            print(f"⚠️  Error loading real results: {e}")
            return pd.DataFrame()

    def get_completed_open_slots(self, target_date):
        real_results = self.load_real_results_for_date(target_date)
        completed_slots = real_results['slot'].unique().tolist() if not real_results.empty else []
        open_slots = [slot for slot in self.slots if slot not in completed_slots]
        
        # Store date range for logging
        if not real_results.empty:
            self.real_results_date_range = (
                real_results['date'].min(),
                real_results['date'].max()
            )
        else:
            self.real_results_date_range = None
            
        return completed_slots, open_slots

    def analyze_family_performance(self, target_date):
        try:
            # Load pattern intelligence data
            pattern_file = Path(__file__).resolve().parent / "logs" / "performance" / "pattern_intelligence_summary.json"
            if pattern_file.exists():
                with open(pattern_file, 'r') as f:
                    pattern_data = json.load(f)
                return pattern_data
            return {}
        except Exception as e:
            print(f"⚠️  Error loading family performance: {e}")
            return {}

    def calculate_stake_multiplier(self, number, slot, family_performance):
        multiplier = 1.0
        
        # 🆕 Use central pattern_packs instead of local definitions
        try:
            # Get number's pattern families using central registry
            digit_tags = pattern_packs.get_digit_pack_tags(number)
            families = self._get_family_categories(digit_tags)
            
            if pattern_packs.is_s40(number):
                families.add("S40")
                
            # Apply multipliers based on family performance
            for family in families:
                if family in family_performance:
                    perf = family_performance[family]
                    hit_rate = perf.get('hit_rate', 0)
                    if hit_rate > 0.6:  # High performing family
                        multiplier *= self.config.get('hit_family_multiplier', 1.2)
                    elif hit_rate < 0.3:  # Low performing family  
                        multiplier *= self.config.get('unhit_family_multiplier', 0.8)
                        
            # Apply bounds
            max_mult = self.config.get('max_stake_multiplier', 2.0)
            min_mult = self.config.get('min_stake_multiplier', 0.5)
            multiplier = max(min(multiplier, max_mult), min_mult)
            
            return multiplier, families
            
        except ImportError:
            print("⚠️  pattern_packs not available, using fallback")
            return 1.0, set()

    def _get_family_categories(self, digit_tags):
        """🆕 Convert fine-grained pack tags to family categories using central registry"""
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

    def adjust_stakes(self, bets_df, open_slots, family_performance):
        adjusted_bets = []
        
        for _, bet in bets_df.iterrows():
            slot = bet['slot']
            layer_type = bet['layer_type']
            number_str = str(bet['number_or_digit'])  # ✅ FIX: Convert to string first
            
            # Only adjust MAIN bets in open slots
            if layer_type == 'Main' and slot in open_slots and number_str.replace('.', '').isdigit():
                number = int(float(number_str))  # ✅ FIX: Handle both int and float
                old_stake = float(bet['stake'])
                
                multiplier, families = self.calculate_stake_multiplier(number, slot, family_performance)
                new_stake = old_stake * multiplier
                
                # Round to nearest 0.5
                new_stake = round(new_stake * 2) / 2
                
                # Verbose logging
                if self.verbose:
                    family_str = ", ".join(sorted(families)) if families else "None"
                    print(f"   {slot} {number:02d} → old ₹{old_stake:.0f} → new ₹{new_stake:.0f} (×{multiplier:.2f}) [families: {family_str}]")
                
                # Update bet
                adjusted_bet = bet.copy()
                adjusted_bet['stake'] = new_stake
                adjusted_bet['potential_return'] = new_stake * 90
                adjusted_bet['notes'] = f"INTRADAY adjusted: {multiplier:.2f}x"
                adjusted_bets.append(adjusted_bet)
            else:
                # Keep original bet for completed slots and non-Main layers
                adjusted_bets.append(bet)
                
        return pd.DataFrame(adjusted_bets)

    def display_intraday_plan(self, adjusted_df, open_slots):
        print(f"\n🎯 INTRADAY BET PLAN - {len(open_slots)} SLOTS OPEN")
        print("=" * 60)
        
        for slot in self.slots:
            slot_bets = adjusted_df[adjusted_df['slot'] == slot]
            main_bets = slot_bets[slot_bets['layer_type'] == 'Main']
            andar_bets = slot_bets[slot_bets['layer_type'] == 'ANDAR']
            bahar_bets = slot_bets[slot_bets['layer_type'] == 'BAHAR']
            
            status = "🟢 OPEN" if slot in open_slots else "🔴 COMPLETED"
            print(f"\n{slot} {status}:")
            
            # Main numbers - format properly
            if not main_bets.empty:
                print("  🔢 Main numbers:")
                for _, bet in main_bets.iterrows():
                    number_str = str(bet['number_or_digit'])
                    # ✅ FIX: Handle both string and float numbers
                    try:
                        number = int(float(number_str))
                        formatted_number = f"{number:02d}"
                    except:
                        formatted_number = number_str
                    
                    tier = bet['tier']
                    stake = bet['stake']
                    # Clean formatting - remove .0 for whole numbers
                    stake_str = f"₹{stake:.0f}" if stake == int(stake) else f"₹{stake:.1f}"
                    print(f"     {formatted_number} (tier {tier}) stake {stake_str}")
            
            # ANDAR/BAHAR - format properly
            andar_digit = andar_bets['number_or_digit'].iloc[0] if not andar_bets.empty else "None"
            bahar_digit = bahar_bets['number_or_digit'].iloc[0] if not bahar_bets.empty else "None"
            
            andar_stake = andar_bets['stake'].iloc[0] if not andar_bets.empty else 0
            bahar_stake = bahar_bets['stake'].iloc[0] if not bahar_bets.empty else 0
            
            # Clean stake formatting
            andar_stake_str = f"₹{andar_stake:.0f}" if andar_stake == int(andar_stake) else f"₹{andar_stake:.1f}"
            bahar_stake_str = f"₹{bahar_stake:.0f}" if bahar_stake == int(bahar_stake) else f"₹{bahar_stake:.1f}"
            
            print(f"  📊 ANDAR: {andar_digit} {andar_stake_str}, BAHAR: {bahar_digit} {bahar_stake_str}")
            
            # Total stake for slot
            total_stake = slot_bets['stake'].sum()
            total_stake_str = f"₹{total_stake:.0f}" if total_stake == int(total_stake) else f"₹{total_stake:.1f}"
            print(f"  💰 Total stake this slot: {total_stake_str}")

    def run_recalculation(self, target_date):
        # ✅ TASK B: Clear date mapping
        system_date = datetime.now().date()
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
        
        print(f"\n📅 DATE MAPPING (INTRADAY):")
        print(f"   🖥️  System Date: {system_date}")
        print(f"   🎯 Target Date (bet plan & intraday): {target_date_obj}")
        print(f"   📊 Mode: INTRADAY RECALC")
        
        # 🆕 Print central pack usage
        _print_central_pack_registry(prefix="   ")
        
        # Load bet plan
        print(f"\n🔍 Loading bet plan for {target_date}...")
        bets_df = self.load_bet_plan(target_date_obj)
        
        # Get completed/open slots
        completed_slots, open_slots = self.get_completed_open_slots(target_date_obj)
        
        # ✅ TASK B: Print real results range if available
        if hasattr(self, 'real_results_date_range') and self.real_results_date_range:
            min_date, max_date = self.real_results_date_range
            print(f"   📈 Real results range in learn file: {min_date} to {max_date}")
        
        print(f"\n📊 SLOT STATUS:")
        print(f"   Completed slots : {', '.join(completed_slots) if completed_slots else 'None'}")
        print(f"   Open slots      : {', '.join(open_slots) if open_slots else 'None'}")
        
        # Check if recalculation is needed
        if not open_slots:
            print(f"\n✅ All slots completed for today. No recalculation needed.")
            print(f"ℹ️  Reason: For the target date above, all 4 slots already have real results in 'number prediction learn.xlsx'.")
            print(f"   Intraday engine only reweights stakes when some slots are still open.")
            return
        
        # Load family performance
        print(f"\n📈 Analyzing family performance...")
        family_performance = self.analyze_family_performance(target_date_obj)
        
        # Adjust stakes
        print(f"🔄 Adjusting stakes for open slots: {', '.join(open_slots)}")
        adjusted_df = self.adjust_stakes(bets_df, open_slots, family_performance)
        
        # Display plan
        self.display_intraday_plan(adjusted_df, open_slots)
        
        # Save adjusted plan
        self.save_adjusted_plan(adjusted_df, target_date_obj)
        
        print(f"\n✅ INTRADAY RECALCULATION COMPLETED!")

    def save_adjusted_plan(self, adjusted_df, target_date):
        output_dir = Path(__file__).resolve().parent / "predictions" / "bet_engine"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"bet_plan_intraday_{target_date.strftime('%Y%m%d')}.xlsx"
        file_path = output_dir / filename
        
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            adjusted_df.to_excel(writer, sheet_name='bets', index=False)
        
        print(f"💾 Intraday bet plan saved: {file_path}")

def main():
    parser = argparse.ArgumentParser(description='Slot Recalculation Engine - Intraday Learning')
    parser.add_argument('--date', required=True, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--verbose', action='store_true', help='Print per-number stake change details')
    
    args = parser.parse_args()
    
    try:
        print(f"🔁 SLOT RECALC ENGINE – {args.date}")
        print("=" * 50)
        
        engine = SlotRecalcEngine(verbose=args.verbose)
        engine.run_recalculation(args.date)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 🆕 Print central pack usage when run directly
    _print_central_pack_registry()
    exit(main())
