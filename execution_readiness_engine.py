# execution_readiness_engine.py
"""
EXECUTION READINESS ENGINE - Live Execution Go/No-Go Decision Maker

PURPOSE:
Analyze system health, performance history, and confidence levels to determine
optimal execution mode and stake sizing for live betting.

USAGE:
py -3.12 execution_readiness_engine.py --days 10
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import argparse
import sys

class ExecutionReadinessEngine:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.environment_score = 0
        self.decision_data = {}
        self.score_components = {}
        
    def load_reality_check_summary(self):
        """Load reality check summary data"""
        reality_file = self.base_dir / "logs" / "performance" / "reality_check_summary.json"
        
        if not reality_file.exists():
            print("⚠️ Reality check summary not found")
            return {
                'base_roi': 0,
                'best_strategy_by_roi': 'NONE',
                'window_days': 0,
                'strategy_totals': {
                    'BASE': {'roi_percent': 0, 'total_profit': 0},
                    'DYNAMIC': {'roi_percent': 0, 'total_profit': 0},
                    'FINAL': {'roi_percent': 0, 'total_profit': 0}
                }
            }
            
        try:
            with open(reality_file, 'r') as f:
                data = json.load(f)
            
            base_roi = data.get('strategy_totals', {}).get('BASE', {}).get('roi_percent', 0)
            best_strategy = data.get('best_strategy_by_roi', 'NONE')
            window_days = data.get('window_days', 0)
            
            print(f"✅ Loaded reality check: BASE ROI={base_roi:.1f}%, best={best_strategy}, window={window_days} days")
            
            return {
                'base_roi': base_roi,
                'best_strategy_by_roi': best_strategy,
                'window_days': window_days,
                'strategy_totals': data.get('strategy_totals', {})
            }
            
        except Exception as e:
            print(f"⚠️ Error loading reality check: {e}")
            return {
                'base_roi': 0,
                'best_strategy_by_roi': 'NONE',
                'window_days': 0,
                'strategy_totals': {}
            }
    
    def load_strategy_recommendation(self):
        """Load strategy recommendation data"""
        strategy_file = self.base_dir / "logs" / "performance" / "strategy_recommendation.json"
        
        if not strategy_file.exists():
            print("⚠️ Strategy recommendation not found")
            return {
                'recommended_strategy': 'NONE',
                'confidence_level': 'LOW',
                'risk_mode': 'DEFENSIVE',
                'window_days': 0,
                'reason': 'Strategy recommendation missing'
            }
            
        try:
            with open(strategy_file, 'r') as f:
                data = json.load(f)
            
            recommended = data.get('recommended_strategy', 'NONE')
            confidence = data.get('confidence_level', 'LOW')
            risk_mode = data.get('risk_mode', 'DEFENSIVE')
            window_days = data.get('window_days', 0)
            
            print(f"✅ Loaded strategy recommendation: {recommended} ({confidence}, {risk_mode})")
            
            return {
                'recommended_strategy': recommended,
                'confidence_level': confidence,
                'risk_mode': risk_mode,
                'window_days': window_days,
                'reason': data.get('reason', 'No reason provided')
            }
            
        except Exception as e:
            print(f"⚠️ Error loading strategy recommendation: {e}")
            return {
                'recommended_strategy': 'NONE',
                'confidence_level': 'LOW',
                'risk_mode': 'DEFENSIVE',
                'window_days': 0,
                'reason': f'Error: {e}'
            }
    
    def load_prediction_confidence(self):
        """Load prediction confidence data"""
        confidence_file = self.base_dir / "logs" / "performance" / "prediction_confidence.json"
        
        if not confidence_file.exists():
            print("⚠️ Prediction confidence not found")
            return {
                'avg_confidence_score': 0,
                'count_high_slots': 0,
                'slot_scores': {}
            }
            
        try:
            with open(confidence_file, 'r') as f:
                data = json.load(f)
            
            confidence_scores = data.get('confidence_scores', {})
            slot_scores = {}
            
            for slot, slot_data in confidence_scores.items():
                if isinstance(slot_data, dict):
                    score = slot_data.get('confidence_score', 0)
                    slot_scores[slot] = score
            
            if slot_scores:
                avg_score = sum(slot_scores.values()) / len(slot_scores)
                count_high = sum(1 for score in slot_scores.values() if score >= 65)
            else:
                avg_score = 0
                count_high = 0
            
            print(f"✅ Loaded prediction confidence: avg={avg_score:.1f}, high_slots={count_high}")
            
            return {
                'avg_confidence_score': avg_score,
                'count_high_slots': count_high,
                'slot_scores': slot_scores
            }
            
        except Exception as e:
            print(f"⚠️ Error loading prediction confidence: {e}")
            return {
                'avg_confidence_score': 0,
                'count_high_slots': 0,
                'slot_scores': {}
            }
    
    def load_system_audit_summary(self):
        """Load system audit summary data"""
        audit_file = self.base_dir / "logs" / "performance" / "system_audit_report.xlsx"
        
        if not audit_file.exists():
            print("⚠️ System audit report not found")
            return {
                'total_audited_dates': 0,
                'full_ok_days': 0,
                'partial_days': 0,
                'no_pnl_days': 0,
                'missing_final_plan_days': 0,
                'any_final_date_mismatch': False
            }
            
        try:
            df = pd.read_excel(audit_file, sheet_name='summary')
            
            if df.empty:
                return {
                    'total_audited_dates': 0,
                    'full_ok_days': 0,
                    'partial_days': 0,
                    'no_pnl_days': 0,
                    'missing_final_plan_days': 0,
                    'any_final_date_mismatch': False
                }
            
            row = df.iloc[0]
            
            audit_data = {
                'total_audited_dates': int(row.get('total_audited_dates', 0)),
                'full_ok_days': int(row.get('full_ok_days', 0)),
                'partial_days': int(row.get('partial_days', 0)),
                'no_pnl_days': int(row.get('no_pnl_days', 0)),
                'missing_final_plan_days': int(row.get('missing_final_plan_days', 0)),
                'any_final_date_mismatch': bool(row.get('any_final_date_mismatch', False))
            }
            
            print(f"✅ Loaded system audit: {audit_data['full_ok_days']} OK days, {audit_data['missing_final_plan_days']} missing final plans")
            
            return audit_data
            
        except Exception as e:
            print(f"⚠️ Error loading system audit: {e}")
            return {
                'total_audited_dates': 0,
                'full_ok_days': 0,
                'partial_days': 0,
                'no_pnl_days': 0,
                'missing_final_plan_days': 0,
                'any_final_date_mismatch': False
            }
    
    def discover_latest_final_plan(self):
        """Discover and load latest final bet plan"""
        bet_engine_dir = self.base_dir / "predictions" / "bet_engine"
        
        if not bet_engine_dir.exists():
            print("⚠️ Bet engine directory not found")
            return 0
        
        final_files = list(bet_engine_dir.glob("final_bet_plan_*.xlsx"))
        if not final_files:
            print("⚠️ No final bet plan files found")
            return 0
        
        # Find latest file by date in filename
        latest_file = None
        latest_date = ""
        
        for file in final_files:
            date_str = file.stem.split('_')[-1]
            if date_str.isdigit() and len(date_str) == 8:
                if date_str > latest_date:
                    latest_date = date_str
                    latest_file = file
        
        if not latest_file:
            print("⚠️ Could not find valid final bet plan")
            return 0

        try:
            df = pd.read_excel(latest_file, sheet_name='final_slot_plan')
            source_sheet = 'final_slot_plan'
        except Exception as e_first:
            print(f"⚠️ final_slot_plan sheet not found in {latest_file.name}, falling back to first sheet ({e_first})")
            try:
                df = pd.read_excel(latest_file, sheet_name=0)
                source_sheet = 'first_sheet'
            except Exception as e_second:
                print(f"⚠️ Could not read any valid sheet from final bet plan: {e_second}")
                return 0

        stake_col = None
        for cand in ['final_slot_stake', 'stake', 'final_stake', 'amount']:
            if cand in df.columns:
                stake_col = cand
                break

        if stake_col is None:
            print("⚠️ No recognizable stake column in final bet plan")
            return 0

        if 'date' in df.columns:
            total_row = df[df['date'] == 'TOTAL']
            if not total_row.empty:
                total_stake = float(pd.to_numeric(total_row[stake_col], errors='coerce').iloc[0])
            else:
                slot_rows = df[df['slot'].isin(['FRBD', 'GZBD', 'GALI', 'DSWR'])] if 'slot' in df.columns else df
                total_stake = float(pd.to_numeric(slot_rows[stake_col], errors='coerce').sum())
        elif 'slot' in df.columns:
            slot_rows = df[df['slot'].isin(['FRBD', 'GZBD', 'GALI', 'DSWR'])]
            total_stake = float(pd.to_numeric(slot_rows[stake_col], errors='coerce').sum())
        else:
            total_stake = float(pd.to_numeric(df[stake_col], errors='coerce').sum())

        print(f"✅ Loaded final bet plan ({source_sheet}): {latest_file.name}, total stake=₹{total_stake:.0f}")
        return total_stake
    
    def compute_environment_score(self, reality_summary, strategy_reco, conf_summary, audit_summary):
        """Compute environment score (0-100)"""
        # Extract variables
        base_roi = reality_summary.get('base_roi', 0)
        best_strategy = reality_summary.get('best_strategy_by_roi', 'NONE')
        meta_strategy = strategy_reco.get('recommended_strategy', 'NONE')
        avg_conf = conf_summary.get('avg_confidence_score', 0)
        count_high = conf_summary.get('count_high_slots', 0)
        
        total_audited = audit_summary.get('total_audited_dates', 0)
        full_ok = audit_summary.get('full_ok_days', 0)
        missing_final = audit_summary.get('missing_final_plan_days', 0)
        no_pnl = audit_summary.get('no_pnl_days', 0)
        date_mismatch = audit_summary.get('any_final_date_mismatch', False)
        
        # 1. Baseline from BASE ROI
        if base_roi <= 0:
            baseline = 20
        elif base_roi < 100:
            baseline = 40
        elif base_roi < 300:
            baseline = 60
        elif base_roi < 600:
            baseline = 75
        else:
            baseline = 80
        
        # 2. Meta vs reality alignment
        if best_strategy == meta_strategy and base_roi > 0:
            bonus_meta = 5
        else:
            bonus_meta = 0
        
        # 3. Confidence contribution
        if avg_conf >= 75 and count_high >= 2:
            bonus_conf = 10
        elif avg_conf >= 60:
            bonus_conf = 5
        elif avg_conf < 40:
            bonus_conf = -5
        else:
            bonus_conf = 0
        
        # 4. System audit contribution
        audit_adjust = 0
        
        if date_mismatch:
            audit_adjust -= 10
        
        if total_audited > 0:
            ratio_missing_final = missing_final / total_audited
            if ratio_missing_final >= 0.7:
                audit_adjust -= 5
            elif ratio_missing_final >= 0.3:
                audit_adjust -= 2
        
        if no_pnl > 0:
            audit_adjust -= 5
        
        if full_ok >= 5:
            audit_adjust += 3
        
        # 5. Total environment score
        env_score = baseline + bonus_meta + bonus_conf + audit_adjust
        env_score = max(0, min(100, env_score))  # Clamp to [0, 100]
        
        # Store components for reporting
        self.score_components = {
            "baseline_from_roi": baseline,
            "bonus_meta": bonus_meta,
            "bonus_confidence": bonus_conf,
            "audit_adjust": audit_adjust,
            "base_roi": base_roi,
            "avg_confidence_score": avg_conf,
            "count_high_slots": count_high,
            "total_audited_dates": total_audited,
            "missing_final_plan_days": missing_final,
            "full_ok_days": full_ok,
            "no_pnl_days": no_pnl
        }
        
        return env_score
    
    def build_decision(self, env_score, base_final_total_stake, reality_summary, strategy_reco):
        """Build execution decision based on environment score"""
        base_roi = reality_summary.get('base_roi', 0)
        best_strategy = reality_summary.get('best_strategy_by_roi', 'NONE')
        meta_strategy = strategy_reco.get('recommended_strategy', 'NONE')
        window_days_reality = reality_summary.get('window_days', 0)
        window_days_strategy = strategy_reco.get('window_days', 0)
        
        notes = []
        
        # Add ROI note
        notes.append(f"BASE ROI over last {window_days_reality} days: {base_roi:.1f}%")
        
        # Add strategy alignment note
        notes.append(f"Meta recommended strategy: {meta_strategy}")
        notes.append(f"Reality best-by-ROI: {best_strategy}")
        
        # Add audit note
        full_ok = self.score_components.get('full_ok_days', 0)
        missing_final = self.score_components.get('missing_final_plan_days', 0)
        notes.append(f"Audit: {full_ok} full OK days, {missing_final} days missing final plan")
        
        # Decision logic
        if base_roi <= 0:
            if env_score >= 40 and any(stats.get('roi_percent', 0) > 0 for stats in reality_summary.get('strategy_totals', {}).values()):
                mode = "PAPER_ONLY"
                multiplier = 0.0
                notes.append("ROI <= 0% but system shows some positive performance - paper trading only")
            else:
                mode = "SKIP_TODAY"
                multiplier = 0.0
                notes.append("Negative ROI across all strategies - skipping today")
        else:
            if env_score >= 80:
                mode = "GO_LIVE_FULL"
                multiplier = 1.0
                notes.append("High confidence environment - full stake recommended")
            elif 60 <= env_score < 80:
                mode = "GO_LIVE_LIGHT"
                multiplier = 0.5
                notes.append("Medium confidence environment - reduced stake recommended")
            elif 40 <= env_score < 60:
                mode = "PAPER_ONLY"
                multiplier = 0.0
                notes.append("Low confidence environment - paper trading only")
            else:
                mode = "SKIP_TODAY"
                multiplier = 0.0
                notes.append("Very low confidence environment - skipping today")
        
        recommended_stake = round(base_final_total_stake * multiplier, 0)
        
        return {
            "mode": mode,
            "stake_multiplier": multiplier,
            "recommended_real_total_stake": recommended_stake,
            "base_final_total_stake": base_final_total_stake,
            "window_days_reality": window_days_reality,
            "window_days_strategy": window_days_strategy,
            "meta_strategy": meta_strategy,
            "base_roi": base_roi,
            "reality_best_strategy_by_roi": best_strategy,
            "meta_confidence_level": strategy_reco.get('confidence_level', 'LOW'),
            "meta_risk_mode": strategy_reco.get('risk_mode', 'DEFENSIVE'),
            "notes": notes
        }
    
    def save_reports(self):
        """Save JSON and Excel reports"""
        # JSON output
        json_file = self.base_dir / "logs" / "performance" / "execution_readiness_summary.json"
        json_file.parent.mkdir(parents=True, exist_ok=True)
        
        json_data = {
            "timestamp": datetime.now().isoformat(),
            "environment_score": self.environment_score,
            "mode": self.decision_data.get('mode', 'SKIP_TODAY'),
            "stake_multiplier": self.decision_data.get('stake_multiplier', 0.0),
            "recommended_real_total_stake": self.decision_data.get('recommended_real_total_stake', 0.0),
            "base_final_total_stake": self.decision_data.get('base_final_total_stake', 0.0),
            "window_days_reality": self.decision_data.get('window_days_reality', 0),
            "window_days_strategy": self.decision_data.get('window_days_strategy', 0),
            "base_roi": self.decision_data.get('base_roi', 0.0),
            "meta_strategy": self.decision_data.get('meta_strategy', 'NONE'),
            "reality_best_strategy_by_roi": self.decision_data.get('reality_best_strategy_by_roi', 'NONE'),
            "meta_confidence_level": self.decision_data.get('meta_confidence_level', 'LOW'),
            "meta_risk_mode": self.decision_data.get('meta_risk_mode', 'DEFENSIVE'),
            "score_components": self.score_components,
            "notes": self.decision_data.get('notes', []),
            "advisory_only": True
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Excel output
        excel_file = self.base_dir / "logs" / "performance" / "execution_readiness_report.xlsx"
        
        # Sheet 1: summary
        summary_data = [{
            'environment_score': self.environment_score,
            'mode': self.decision_data.get('mode', 'SKIP_TODAY'),
            'stake_multiplier': self.decision_data.get('stake_multiplier', 0.0),
            'recommended_real_total_stake': self.decision_data.get('recommended_real_total_stake', 0.0),
            'base_final_total_stake': self.decision_data.get('base_final_total_stake', 0.0),
            'window_days_reality': self.decision_data.get('window_days_reality', 0),
            'window_days_strategy': self.decision_data.get('window_days_strategy', 0),
            'base_roi': self.decision_data.get('base_roi', 0.0),
            'meta_strategy': self.decision_data.get('meta_strategy', 'NONE'),
            'reality_best_strategy_by_roi': self.decision_data.get('reality_best_strategy_by_roi', 'NONE'),
            'meta_confidence_level': self.decision_data.get('meta_confidence_level', 'LOW'),
            'meta_risk_mode': self.decision_data.get('meta_risk_mode', 'DEFENSIVE'),
            'timestamp': datetime.now().isoformat()
        }]
        
        df_summary = pd.DataFrame(summary_data)
        
        # Sheet 2: score_components
        df_components = pd.DataFrame([self.score_components])
        
        # Sheet 3: notes
        notes_data = [{'note': note} for note in self.decision_data.get('notes', [])]
        df_notes = pd.DataFrame(notes_data)
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='summary', index=False)
            df_components.to_excel(writer, sheet_name='score_components', index=False)
            df_notes.to_excel(writer, sheet_name='notes', index=False)
        
        return json_file, excel_file
    
    def display_console_summary(self, json_file, excel_file):
        """Display formatted console summary"""
        print("\n" + "="*60)
        print("🎛️  EXECUTION READINESS – GO / NO-GO")
        print("="*60)
        
        env_score = self.environment_score
        mode = self.decision_data.get('mode', 'SKIP_TODAY')
        multiplier = self.decision_data.get('stake_multiplier', 0.0)
        base_stake = self.decision_data.get('base_final_total_stake', 0)
        recommended_stake = self.decision_data.get('recommended_real_total_stake', 0)

        print(f"Environment score : {env_score} / 100")
        print(f"Mode              : {mode}")
        print(f"Stake multiplier  : {multiplier:.1f}x")
        print(f"Final plan stake  : ₹{base_stake:.0f} → Recommended real stake: ₹{recommended_stake:.0f}")
        print("Note: Multiplier is advisory. Core staking engine may adjust stakes based on master strategy.")
        print()
        
        window_reality = self.decision_data.get('window_days_reality', 0)
        window_strategy = self.decision_data.get('window_days_strategy', 0)
        base_roi = self.decision_data.get('base_roi', 0)
        
        print(f"Reality window    : {window_reality} days")
        print(f"Strategy window   : {window_strategy} days")
        print(f"BASE ROI          : {base_roi:+.1f}%")
        print()
        
        meta_strategy = self.decision_data.get('meta_strategy', 'NONE')
        meta_confidence = self.decision_data.get('meta_confidence_level', 'LOW')
        meta_risk = self.decision_data.get('meta_risk_mode', 'DEFENSIVE')
        reality_best = self.decision_data.get('reality_best_strategy_by_roi', 'NONE')
        
        print("Meta vs Reality:")
        print(f"  Meta strategy   : {meta_strategy} ({meta_confidence}, {meta_risk})")
        print(f"  Reality best ROI: {reality_best}")
        print()
        
        print("Key notes:")
        notes = self.decision_data.get('notes', [])
        for note in notes[:4]:  # Show first 4 notes
            print(f"  - {note}")
        
        print()
        print("💾 Output files:")
        print(f"  JSON : {json_file}")
        print(f"  Excel: {excel_file}")
    
    def run_engine(self, days=10):
        """Run complete execution readiness analysis"""
        print("🎛️  EXECUTION READINESS ENGINE - Live Execution Decision Maker")
        print("=" * 60)
        
        # Step 1: Load all input data
        reality_summary = self.load_reality_check_summary()
        strategy_reco = self.load_strategy_recommendation()
        conf_summary = self.load_prediction_confidence()
        audit_summary = self.load_system_audit_summary()
        base_final_stake = self.discover_latest_final_plan()
        
        # Step 2: Compute environment score
        self.environment_score = self.compute_environment_score(
            reality_summary, strategy_reco, conf_summary, audit_summary
        )
        
        # Step 3: Build execution decision
        self.decision_data = self.build_decision(
            self.environment_score, base_final_stake, reality_summary, strategy_reco
        )
        
        # Step 4: Save reports
        json_file, excel_file = self.save_reports()
        
        # Step 5: Display console summary
        self.display_console_summary(json_file, excel_file)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Execution Readiness Engine - Live Execution Decision Maker')
    parser.add_argument('--days', type=int, default=10, help='Number of days used for interpretation (informational only)')
    
    args = parser.parse_args()
    
    engine = ExecutionReadinessEngine()
    success = engine.run_engine(args.days)
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())