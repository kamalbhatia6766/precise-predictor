# dynamic_stake_allocator.py - UPDATED
# dynamic_stake_allocator.py - REALITY-DRIVEN STAKE ALLOCATION
import argparse
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# 🆕 Import central helpers
import quant_data_core
import quant_paths


def fmt_rupees(value: float) -> str:
    """Format rupee amounts with clean rounding for console output."""
    try:
        amt = float(value)
    except (TypeError, ValueError):
        return "₹0"

    if abs(amt - round(amt)) < 0.01:
        return f"₹{int(round(amt))}"
    return f"₹{amt:.2f}"

class DynamicStakeAllocator:
    def __init__(self, force_refresh: bool | None = None):
        self.base_dir = quant_paths.get_project_root()
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        env_force = os.getenv("FORCE_DYNAMIC_STAKE_REFRESH", "0")
        self.force_refresh = bool(int(env_force)) if force_refresh is None else bool(force_refresh)
        self.meta_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan_meta.json"
        
    def load_base_bet_plan(self):
        """Load the latest bet plan to get base stakes (always from master file)."""
        latest_bet_plan = quant_paths.find_latest_bet_plan_master()
        target_date = quant_paths.parse_date_from_filename(latest_bet_plan.stem) if latest_bet_plan else None

        if target_date:
            canonical_master = quant_paths.get_bet_plan_master_path(target_date.strftime("%Y-%m-%d"))
            if canonical_master.exists():
                latest_bet_plan = canonical_master

        if not latest_bet_plan:
            print("❌ No bet plan files found")
            return None, None, None, None

        try:
            bets_df = pd.read_excel(latest_bet_plan, sheet_name='bets')
            summary_df = None
            try:
                summary_df = pd.read_excel(latest_bet_plan, sheet_name='summary')
            except Exception:
                summary_df = None
            print(f"✅ Loaded base bet plan: {latest_bet_plan.name}")
            return bets_df, summary_df, target_date, latest_bet_plan
        except Exception as e:
            print(f"❌ Error loading bet plan: {e}")
            return None, None, None, None

    def _load_strategy_context(self):
        """Load strategy recommendation context when available for signature building."""
        strategy_file = quant_paths.get_performance_logs_dir() / "strategy_recommendation.json"
        if not strategy_file.exists():
            return {}
        try:
            with open(strategy_file, "r") as f:
                return json.load(f) or {}
        except Exception:
            return {}

    def _load_meta_store(self):
        if not self.meta_file.exists():
            return {}
        try:
            with open(self.meta_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_meta_store(self, meta):
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.meta_file, "w") as f:
            json.dump(meta, f, indent=2)

    def _compute_config_signature(self, performance_data, base_slot_stakes):
        strategy_context = self._load_strategy_context()
        overall_block = performance_data.get("overall", {}) if performance_data else {}
        date_range = overall_block.get("date_range", {}) if isinstance(overall_block, dict) else {}
        slot_rois = self.calculate_slot_rois(performance_data) if performance_data else {}

        payload = {
            "base_slot_stakes": {k: round(float(v or 0), 2) for k, v in base_slot_stakes.items()},
            "overall_roi": round(float(overall_block.get("overall_roi", 0) or 0), 4),
            "date_window": {
                "start": date_range.get("start"),
                "end": date_range.get("end"),
            },
            "slot_rois": {k: round(float(v or 0), 3) for k, v in slot_rois.items()},
            "strategy": strategy_context.get("recommended_strategy") or strategy_context.get("strategy"),
            "risk_mode": strategy_context.get("risk_mode"),
            "base_roi": strategy_context.get("metrics", {}).get("base_roi"),
        }

        signature_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(signature_str.encode()).hexdigest()[:16]
    
    def calculate_base_stakes(self, bets_df):
        """Calculate base stakes from bet plan"""
        base_slot_stakes = {}
        
        for slot in self.slots:
            slot_bets = bets_df[bets_df['slot'] == slot]
            if not slot_bets.empty:
                total_stake = slot_bets['stake'].sum()
                base_slot_stakes[slot] = total_stake
            else:
                base_slot_stakes[slot] = 0
        
        base_daily_stake = sum(base_slot_stakes.values())
        return base_slot_stakes, base_daily_stake

    def _scale_bet_plan(self, bets_df, summary_df, final_slot_stakes, base_slot_stakes):
        """Apply slot-wise scaling to bets and summary dataframes"""
        if bets_df is None:
            return None, None

        scaled_bets = bets_df.copy()
        scaled_summary = summary_df.copy() if summary_df is not None else None

        for slot in self.slots:
            base_total = float(base_slot_stakes.get(slot, 0) or 0)
            final_total = float(final_slot_stakes.get(slot, base_total) or 0)

            if base_total <= 0:
                continue

            factor = final_total / base_total if base_total else 1.0
            if factor == 1.0:
                continue

            slot_mask = scaled_bets['slot'] == slot
            stake_numeric = pd.to_numeric(scaled_bets.loc[slot_mask, 'stake'], errors='coerce')
            numeric_mask = stake_numeric.notna()
            scaled_values = (stake_numeric[numeric_mask] * factor).round(1)
            scaled_bets.loc[slot_mask, 'stake'] = scaled_values.combine_first(scaled_bets.loc[slot_mask, 'stake'])

            if 'potential_return' in scaled_bets.columns:
                scaled_returns = pd.to_numeric(scaled_bets.loc[slot_mask, 'stake'], errors='coerce') * 90
                scaled_bets.loc[slot_mask, 'potential_return'] = scaled_returns.round(1).combine_first(
                    scaled_bets.loc[slot_mask, 'potential_return']
                )

            if scaled_summary is not None and 'slot' in scaled_summary.columns:
                summary_mask = scaled_summary['slot'] == slot
                slot_rows = scaled_bets[slot_mask]
                main_total = pd.to_numeric(
                    slot_rows[slot_rows['layer_type'] == 'Main']['stake'], errors='coerce'
                ).sum()
                andar_total = pd.to_numeric(
                    slot_rows[slot_rows['layer_type'] == 'ANDAR']['stake'], errors='coerce'
                ).sum()
                bahar_total = pd.to_numeric(
                    slot_rows[slot_rows['layer_type'] == 'BAHAR']['stake'], errors='coerce'
                ).sum()
                max_total_return = pd.to_numeric(slot_rows['potential_return'], errors='coerce').sum()

                scaled_summary.loc[summary_mask, 'main_stake'] = main_total
                scaled_summary.loc[summary_mask, 'andar_stake'] = andar_total
                scaled_summary.loc[summary_mask, 'bahar_stake'] = bahar_total
                scaled_summary.loc[summary_mask, 'total_stake'] = main_total + andar_total + bahar_total
                scaled_summary.loc[summary_mask, 'max_total_return'] = max_total_return

        return scaled_bets, scaled_summary

    def save_final_bet_plan(
        self,
        bets_df,
        summary_df,
        target_date,
        total_final_stake,
        base_slot_stakes,
        final_slot_stakes,
    ):
        if bets_df is None or target_date is None:
            return None

        date_str = target_date.strftime("%Y-%m-%d")
        date_str_clean = target_date.strftime("%Y%m%d")
        output_path = quant_paths.get_final_bet_plan_path(date_str)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        final_slot_plan_df = self._prepare_final_slot_plan(
            date_str,
            date_str_clean,
            bets_df,
            summary_df,
            base_slot_stakes,
            final_slot_stakes,
        )

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            final_slot_plan_df.to_excel(writer, sheet_name='final_slot_plan', index=False)
            bets_df.to_excel(writer, sheet_name='bets', index=False)
            if summary_df is not None:
                summary_df.to_excel(writer, sheet_name='summary', index=False)

        print(f"💾 Final bet plan saved: {output_path} (Total stake={fmt_rupees(total_final_stake)})")
        return output_path

    def _load_template_final_plan(self, target_date_clean):
        bet_dir = quant_paths.get_bet_engine_dir()
        latest_file = None
        latest_date = ""

        for file in bet_dir.glob("final_bet_plan_*.xlsx"):
            date_str = file.stem.split('_')[-1]
            if date_str.isdigit() and len(date_str) == 8 and date_str != target_date_clean:
                if date_str > latest_date:
                    latest_date = date_str
                    latest_file = file

        if not latest_file:
            return None, None

        try:
            df = pd.read_excel(latest_file, sheet_name='final_slot_plan')
            return df, latest_file
        except Exception:
            try:
                df = pd.read_excel(latest_file, sheet_name=0)
                return df, latest_file
            except Exception:
                return None, latest_file

    def _prepare_final_slot_plan(
        self,
        date_str,
        date_str_clean,
        bets_df,
        summary_df,
        base_slot_stakes,
        final_slot_stakes,
    ):
        template_df, template_file = self._load_template_final_plan(date_str_clean)

        if template_df is not None and not template_df.empty:
            print(f"ℹ️ Using template final plan structure from {template_file.name}")
            columns = list(template_df.columns)
        else:
            columns = [
                'date', 'slot', 'meta_strategy', 'numeric_strategy', 'meta_confidence_level',
                'risk_mode', 'zone', 'base_slot_stake', 'dynamic_slot_stake',
                'conviction_slot_stake', 'final_slot_stake'
            ]

        template_lookup = {}
        if template_df is not None and 'slot' in template_df.columns:
            template_lookup = {
                str(row['slot']): row.to_dict()
                for _, row in template_df.iterrows()
                if pd.notna(row.get('slot', ''))
            }

        rows = []
        for slot in self.slots:
            base_row = {col: '' for col in columns}
            if slot in template_lookup:
                for col in columns:
                    base_row[col] = template_lookup[slot].get(col, base_row[col])

            if 'slot' in columns:
                base_row['slot'] = slot
            if 'date' in columns:
                base_row['date'] = date_str
            if 'base_slot_stake' in columns:
                base_row['base_slot_stake'] = float(base_slot_stakes.get(slot, 0))
            if 'dynamic_slot_stake' in columns:
                base_row['dynamic_slot_stake'] = float(final_slot_stakes.get(slot, base_slot_stakes.get(slot, 0)))
            if 'conviction_slot_stake' in columns:
                base_row['conviction_slot_stake'] = float(base_row.get('conviction_slot_stake', 0) or base_slot_stakes.get(slot, 0))
            if 'final_slot_stake' in columns:
                base_row['final_slot_stake'] = float(final_slot_stakes.get(slot, 0))

            rows.append(base_row)

        total_row = {col: '' for col in columns}
        if 'date' in columns:
            total_row['date'] = 'TOTAL'
        if 'slot' in columns:
            total_row['slot'] = ''
        if 'base_slot_stake' in columns:
            total_row['base_slot_stake'] = sum(base_slot_stakes.values())
        if 'dynamic_slot_stake' in columns:
            total_row['dynamic_slot_stake'] = sum(final_slot_stakes.values())
        if 'conviction_slot_stake' in columns:
            total_row['conviction_slot_stake'] = sum(
                float(base_row.get('conviction_slot_stake', 0) or 0) for base_row in rows
            )
        if 'final_slot_stake' in columns:
            total_row['final_slot_stake'] = sum(final_slot_stakes.values())

        rows.append(total_row)

        # Fill defaults for meta columns if missing
        for row in rows:
            row.setdefault('meta_strategy', 'DYNAMIC_STAKE_PLAN')
            row.setdefault('numeric_strategy', 'DYNAMIC')
            row.setdefault('meta_confidence_level', 'AUTO')
            row.setdefault('risk_mode', 'ADAPTIVE')
            row.setdefault('zone', 'UNKNOWN')

        return pd.DataFrame(rows, columns=columns)
    
    def load_reality_performance(self):
        """Load reality performance data from quant_reality_pnl.json"""
        pnl_file = quant_paths.get_performance_logs_dir() / "quant_reality_pnl.json"
        
        if not pnl_file.exists():
            print("❌ No quant_reality_pnl.json found")
            return None
        
        try:
            with open(pnl_file, 'r') as f:
                performance_data = json.load(f)
            print("✅ Loaded reality performance data")
            return performance_data
        except Exception as e:
            print(f"❌ Error loading performance data: {e}")
            return None
    
    def calculate_slot_rois(self, performance_data):
        """Calculate ROI for each slot from performance data"""
        slot_rois = {}
        
        if 'by_slot' in performance_data:
            for slot_data in performance_data['by_slot']:
                slot = slot_data['slot']
                roi_pct = slot_data.get('roi_pct', 0)
                slot_rois[slot] = roi_pct
        
        # Ensure all slots have ROI data
        for slot in self.slots:
            if slot not in slot_rois:
                slot_rois[slot] = 0
        
        return slot_rois
    
    def apply_reality_overlay(self, base_slot_stakes, performance_data):
        """Apply reality-based stake adjustments"""
        print("🎯 Applying reality-based stake overlay...")
        
        # Get overall ROI
        overall_roi = performance_data.get('overall', {}).get('overall_roi', 0)
        slot_rois = self.calculate_slot_rois(performance_data)
        
        final_slot_stakes = {}
        
        # Define slot ROI multiplier policy
        for slot, base_stake in base_slot_stakes.items():
            slot_roi = slot_rois.get(slot, 0)
            
            # Clamp ROI between -50 and +100
            clamped_roi = max(-50, min(100, slot_roi))
            
            # Slot-specific multiplier based on ROI
            if clamped_roi > 40:
                slot_mult = 1.5
            elif clamped_roi > 20:
                slot_mult = 1.3
            elif clamped_roi > 5:
                slot_mult = 1.1
            elif clamped_roi < -20:
                slot_mult = 0.7
            elif clamped_roi < -5:
                slot_mult = 0.9
            else:
                slot_mult = 1.0
            
            # Global multiplier based on overall ROI
            if overall_roi > 30:
                global_mult = 1.1
            elif overall_roi < -10:
                global_mult = 0.9
            else:
                global_mult = 1.0
            
            # Calculate final stake
            final_stake = base_stake * slot_mult * global_mult
            
            # Round to nearest 5 rupees, minimum 10 if base was >= 10
            if base_stake >= 10:
                final_stake = max(10, round(final_stake / 5) * 5)
            else:
                final_stake = round(final_stake)
            
            final_slot_stakes[slot] = final_stake
        
        return final_slot_stakes, overall_roi, slot_rois

    def _detect_stake_column(self, df):
        for cand in ['final_slot_stake', 'stake', 'final_stake', 'amount']:
            if cand in df.columns:
                return cand
        return None

    def _read_final_plan_total(self, file_path):
        try:
            df = pd.read_excel(file_path, sheet_name='final_slot_plan')
            source = 'final_slot_plan'
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name=0)
                source = 'first_sheet'
                print(f"⚠️ final_slot_plan sheet not found in {file_path.name}, using first sheet for totals")
            except Exception:
                return None

        stake_col = self._detect_stake_column(df)
        if not stake_col:
            return None

        if 'date' in df.columns:
            total_row = df[df['date'] == 'TOTAL']
            if not total_row.empty:
                return float(pd.to_numeric(total_row[stake_col], errors='coerce').iloc[0])

        if 'slot' in df.columns:
            return float(pd.to_numeric(df[df['slot'].isin(self.slots)][stake_col], errors='coerce').sum())

        return float(pd.to_numeric(df[stake_col], errors='coerce').sum())

    def _load_existing_final_slot_stakes(self, file_path):
        """Load slot-wise final stakes from an existing final bet plan (idempotence helper)."""
        try:
            df = pd.read_excel(file_path, sheet_name='final_slot_plan')
        except Exception:
            try:
                df = pd.read_excel(file_path, sheet_name=0)
            except Exception:
                return None

        stake_col = self._detect_stake_column(df)
        if not stake_col:
            return None

        slot_stakes = {}
        for slot in self.slots:
            try:
                slot_value = float(
                    pd.to_numeric(
                        df[(df.get('slot') == slot) | (df.get('slot') == str(slot))][stake_col],
                        errors='coerce'
                    ).sum()
                )
                slot_stakes[slot] = slot_value
            except Exception:
                slot_stakes[slot] = 0.0

        if not any(slot_stakes.values()):
            return None

        return slot_stakes

    def _persist_meta_entry(
        self,
        meta_store,
        date_key,
        base_total,
        final_total,
        config_signature,
    ):
        meta_store[date_key] = {
            "target_date": date_key,
            "base_total_stake": round(float(base_total or 0), 2),
            "final_total_stake": round(float(final_total or 0), 2),
            "config_signature": config_signature,
            "stake_state": "DYNAMIC_LOCKED",
            "timestamp": datetime.now().isoformat(),
        }
        self._save_meta_store(meta_store)
    
    def generate_stake_plan(
        self,
        base_slot_stakes,
        final_slot_stakes,
        target_date,
        overall_roi,
        slot_rois,
        config_signature=None,
    ):
        """Generate the dynamic stake plan JSON"""
        total_daily_stake = sum(final_slot_stakes.values())

        stake_plan = {
            "timestamp": datetime.now().isoformat(),
            "target_date": target_date.strftime("%Y-%m-%d") if target_date else "UNKNOWN",
            "base_slot_stakes": base_slot_stakes,
            "slot_stakes": final_slot_stakes,
            "total_daily_stake": total_daily_stake,
            "overall_roi": overall_roi,
            "slot_rois": slot_rois,
            "logic_version": "v1_reality_simple",
            "central_pnl_source": "quant_reality_pnl.json",
            "config_signature": config_signature,
            "stake_state": "DYNAMIC_LOCKED",
        }
        
        return stake_plan
    
    def save_stake_plan(self, stake_plan):
        """Save dynamic stake plan to JSON"""
        output_file = quant_paths.get_performance_logs_dir() / "dynamic_stake_plan.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(stake_plan, f, indent=2)
        
        print(f"💾 Dynamic stake plan saved: {output_file}")
        return output_file
    
    def print_console_summary(self, stake_plan):
        """Print console summary"""
        print("\n" + "="*60)
        print("💰 DYNAMIC STAKE ALLOCATOR – REALITY LINKED")
        print("="*60)

        base_stakes = stake_plan['base_slot_stakes']
        final_stakes = stake_plan['slot_stakes']
        overall_roi = stake_plan['overall_roi']

        base_total = sum(base_stakes.values())
        final_total = stake_plan['total_daily_stake']

        print(f"📅 Target Date: {stake_plan['target_date']}")
        print(
            f"📊 Base Stakes: {', '.join(f'{slot}={fmt_rupees(amt)}' for slot, amt in base_stakes.items())} "
            f"(Total={fmt_rupees(base_total)})"
        )
        print(
            f"🎯 Final Stakes: {', '.join(f'{slot}={fmt_rupees(amt)}' for slot, amt in final_stakes.items())} "
            f"(Total={fmt_rupees(final_total)})"
        )
        print(f"📈 Overall ROI (window): {overall_roi:+.1f}%")
        
        # Show slot ROI breakdown
        print(f"\n🎯 Slot ROI Breakdown:")
        for slot, roi in stake_plan['slot_rois'].items():
            base = base_stakes[slot]
            final = final_stakes[slot]
            change_pct = ((final - base) / base * 100) if base > 0 else 0
            if roi > 0:
                trend = "🟢"
            elif roi <= -20:
                trend = "🔴"
            else:
                trend = "🟡"
            print(
                f"   {trend} {slot}: ROI={roi:+.1f}% → Stake: {fmt_rupees(base)}→{fmt_rupees(final)} "
                f"({change_pct:+.1f}%)"
            )
        
        print("="*60)
    
    def run_allocation(self):
        """Run complete dynamic stake allocation"""
        print("🚀 DYNAMIC STAKE ALLOCATOR - REALITY-DRIVEN")
        print("="*50)
        
        # Step 1: Load base bet plan
        bets_df, summary_df, target_date, _ = self.load_base_bet_plan()
        if bets_df is None:
            return False

        # Step 2: Calculate base stakes
        base_slot_stakes, base_daily_stake = self.calculate_base_stakes(bets_df)

        date_key = target_date.strftime("%Y-%m-%d") if target_date else "UNKNOWN"
        meta_store = self._load_meta_store()
        existing_meta = meta_store.get(date_key)

        existing_final_total = None
        existing_final_slot_stakes = None
        candidate_final = None
        if target_date is not None:
            candidate_final = quant_paths.get_final_bet_plan_path(target_date.strftime("%Y-%m-%d"))
            if candidate_final.exists():
                existing_final_total = self._read_final_plan_total(candidate_final)
                existing_final_slot_stakes = self._load_existing_final_slot_stakes(candidate_final)
                if existing_final_total is not None:
                    print(
                        f"ℹ️ Existing final plan detected ({candidate_final.name}): "
                        f"total stake={fmt_rupees(existing_final_total)}"
                    )

        # Step 3: Load reality performance
        performance_data = self.load_reality_performance()
        if performance_data is None:
            return False

        # Step 4: Apply reality overlay
        final_slot_stakes, overall_roi, slot_rois = self.apply_reality_overlay(base_slot_stakes, performance_data)

        config_signature = self._compute_config_signature(performance_data, base_slot_stakes)

        # Step 5: Apply overlay to bet plan and save final plan with idempotence
        desired_total = sum(final_slot_stakes.values())
        already_aligned = False
        write_final_plan = True
        locked_noop = False

        if existing_meta and candidate_final and candidate_final.exists():
            if not self.force_refresh and existing_meta.get("config_signature") == config_signature:
                locked_noop = True
                print(
                    f"ℹ️ Dynamic stakes already locked for {date_key} (config unchanged) – idempotent no-op."
                )
            elif not self.force_refresh and existing_meta.get("config_signature") != config_signature:
                locked_noop = True
                print(
                    "⚠️ Config signature changed but stakes are locked; skipping refresh. "
                    "Use --force-refresh-stakes or FORCE_DYNAMIC_STAKE_REFRESH=1 to recompute."
                )
            elif self.force_refresh:
                print(
                    f"🔄 Force refresh enabled for {date_key} – recomputing dynamic stakes despite lock."
                )

        if locked_noop:
            final_slot_stakes = existing_final_slot_stakes or final_slot_stakes
            desired_total = sum(final_slot_stakes.values()) if final_slot_stakes else desired_total
            write_final_plan = False
            already_aligned = True
        else:
            if existing_final_total is not None and desired_total > 0:
                total_diff = abs(desired_total - existing_final_total)
                if total_diff < 0.1 and existing_final_slot_stakes:
                    already_aligned = True
                    write_final_plan = False
                    final_slot_stakes = existing_final_slot_stakes
                    desired_total = sum(final_slot_stakes.values())
                    print(
                        "ℹ️ Base bet plan already aligned; reusing existing final stakes (idempotent). "
                        f"Total ≈ {fmt_rupees(existing_final_total)}"
                    )
                elif total_diff >= 0.1 and self.force_refresh:
                    print(
                        f"⚠️ Final bet plan updated with forced refresh: previous total={fmt_rupees(existing_final_total)}, "
                        f"new total={fmt_rupees(desired_total)}"
                    )
                elif total_diff >= 0.1:
                    print(
                        f"⚠️ Final bet plan updated: previous total={fmt_rupees(existing_final_total)}, "
                        f"new total={fmt_rupees(desired_total)} (config/ROI change)"
                    )

            if not already_aligned and desired_total > 0 and abs(base_daily_stake - desired_total) / max(desired_total, 1) <= 0.01:
                already_aligned = True
                print(
                    "ℹ️ Base bet plan already aligned with dynamic final stakes "
                    f"(total ≈ {fmt_rupees(base_daily_stake)}). No additional scaling applied."
                )

        stake_plan = self.generate_stake_plan(
            base_slot_stakes,
            final_slot_stakes,
            target_date,
            overall_roi,
            slot_rois,
            config_signature=config_signature,
        )

        if already_aligned:
            scaled_bets, scaled_summary = bets_df, summary_df
        else:
            scaled_bets, scaled_summary = self._scale_bet_plan(bets_df, summary_df, final_slot_stakes, base_slot_stakes)

        # Step 6: Persist stake plan (after any idempotent overrides)
        self.save_stake_plan(stake_plan)

        if write_final_plan:
            self.save_final_bet_plan(
                scaled_bets,
                scaled_summary,
                target_date,
                desired_total,
                base_slot_stakes,
                final_slot_stakes,
            )
            self._persist_meta_entry(meta_store, date_key, base_daily_stake, desired_total, config_signature)
        elif not existing_meta:
            # No meta exists yet but we are keeping the current plan; still persist metadata for lock-in
            self._persist_meta_entry(meta_store, date_key, base_daily_stake, desired_total, config_signature)
        elif candidate_final:
            print(f"ℹ️ Skipped rewriting existing final plan ({candidate_final.name}) to keep stakes stable.")

        # Step 8: Print summary
        self.print_console_summary(stake_plan)

        print("✅ Dynamic stake allocation completed!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Dynamic Stake Allocator – reality linked")
    parser.add_argument(
        "--force-refresh-stakes",
        action="store_true",
        help="Recompute and overwrite dynamic stakes even when a lock exists",
    )
    args = parser.parse_args()

    allocator = DynamicStakeAllocator(force_refresh=args.force_refresh_stakes)
    success = allocator.run_allocation()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
