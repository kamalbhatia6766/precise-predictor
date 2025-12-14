# money_manager.py - UPDATED WITH RISK AUTOPILOT
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

class MoneyManager:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.slots = ["FRBD", "GZBD", "GALI", "DSWR"]
        self.reference_bankroll = self.load_reference_bankroll()

    def load_reference_bankroll(self):
        """Load reference bankroll from latest daily_meta_config, fallback to default constant."""
        default_bankroll = 11550
        config_dir = self.base_dir / "logs" / "performance"
        candidates = sorted(config_dir.glob("daily_meta_config*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                ref_val = data.get("current_bankroll") or data.get("reference_bankroll")
                if ref_val is not None:
                    return float(ref_val)
            except Exception:
                continue
        return default_bankroll

    def load_quant_pnl(self):
        """Load quant_reality_pnl.json for realized P&L linkage."""
        pnl_path = self.base_dir / "logs" / "performance" / "quant_reality_pnl.json"
        if not pnl_path.exists():
            return None
        try:
            with open(pnl_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
        
    def load_financial_data(self):
        """Load all financial data including risk configuration"""
        data = {}
        
        # Load P&L history
        pnl_file = self.base_dir / "logs" / "performance" / "bet_pnl_history.xlsx"
        if pnl_file.exists():
            data['pnl'] = pd.read_excel(pnl_file, sheet_name='day_level')
            data['pnl']['date'] = pd.to_datetime(data['pnl']['date']).dt.date

        quant_pnl = self.load_quant_pnl()
        if quant_pnl:
            data['quant_pnl'] = quant_pnl
        
        # Load dynamic stake plan
        stake_file = self.base_dir / "logs" / "performance" / "dynamic_stake_plan.json"
        if stake_file.exists():
            with open(stake_file, 'r') as f:
                data['stake_plan'] = json.load(f)
                
        # Load adaptive fusion plan
        fusion_file = self.base_dir / "logs" / "performance" / "adaptive_fusion_plan.json"
        if fusion_file.exists():
            with open(fusion_file, 'r') as f:
                data['fusion_plan'] = json.load(f)
        
        # ✅ PHASE 2: Load strategy recommendation for risk mode
        strategy_file = self.base_dir / "logs" / "performance" / "strategy_recommendation.json"
        if strategy_file.exists():
            with open(strategy_file, 'r') as f:
                data['strategy'] = json.load(f)
        
        # ✅ PHASE 2: Load loss recovery multipliers
        recovery_file = self.base_dir / "logs" / "performance" / "loss_recovery_plan.json"
        if recovery_file.exists():
            with open(recovery_file, 'r') as f:
                data['recovery'] = json.load(f)
                
        return data
    
    def get_risk_multipliers(self, strategy_data):
        """Get stake multipliers based on risk mode"""
        risk_mode = strategy_data.get('risk_mode', 'NORMAL')
        
        multipliers = {
            'AGGRESSIVE': 1.3,
            'NORMAL': 1.0,
            'DEFENSIVE': 0.7,
            'SUPER_DEFENSIVE': 0.5
        }
        
        return multipliers.get(risk_mode, 1.0)
    
    def get_daily_caps(self, strategy_data):
        """Get daily stake caps based on risk mode"""
        risk_mode = strategy_data.get('risk_mode', 'NORMAL')
        
        caps = {
            'AGGRESSIVE': {'total': 500, 'single': 150},
            'NORMAL': {'total': 300, 'single': 100},
            'DEFENSIVE': {'total': 200, 'single': 70},
            'SUPER_DEFENSIVE': {'total': 100, 'single': 50}
        }
        
        return caps.get(risk_mode, {'total': 300, 'single': 100})
    
    def calculate_kelly_criterion(self, data):
        """Calculate Kelly Criterion optimal stakes with risk multipliers"""
        kelly_stakes = {}
        
        if 'pnl' not in data:
            return kelly_stakes
            
        pnl_df = data['pnl']
        
        # ✅ PHASE 2: Apply risk multipliers
        risk_multiplier = 1.0
        if 'strategy' in data:
            risk_multiplier = self.get_risk_multipliers(data['strategy'])
        
        for slot in self.slots:
            profit_col = f'profit_{slot.lower()}'
            if profit_col in pnl_df.columns:
                slot_data = pnl_df[profit_col]
                
                # Calculate win probability and odds
                winning_days = slot_data[slot_data > 0]
                total_days = len(slot_data)
                win_prob = len(winning_days) / total_days if total_days > 0 else 0
                
                if len(winning_days) > 0:
                    avg_win = winning_days.mean()
                    avg_loss = slot_data[slot_data <= 0].mean() if len(slot_data[slot_data <= 0]) > 0 else 0
                    
                    # Kelly formula: f = (p * b - q) / b
                    # where p = win prob, q = loss prob, b = net odds received on win
                    if avg_loss != 0:
                        net_odds = abs(avg_win / avg_loss)  # b in Kelly formula
                        kelly_fraction = (win_prob * net_odds - (1 - win_prob)) / net_odds
                        
                        # Apply conservative Kelly (half Kelly)
                        conservative_kelly = max(0, kelly_fraction * 0.5)
                        
                        # ✅ PHASE 2: Apply risk multiplier
                        conservative_kelly *= risk_multiplier
                        
                        # Convert to stake (assuming base stake of ₹55)
                        optimal_stake = int(55 * conservative_kelly)
                        
                        kelly_stakes[slot] = {
                            'win_probability': float(win_prob),
                            'avg_win': float(avg_win),
                            'avg_loss': float(avg_loss),
                            'kelly_fraction': float(conservative_kelly),
                            'optimal_stake': optimal_stake,
                            'risk_level': 'HIGH' if conservative_kelly > 0.3 else 'MEDIUM' if conservative_kelly > 0.1 else 'LOW',
                            'risk_multiplier_applied': risk_multiplier  # ✅ PHASE 2: Track multiplier
                        }
        
        return kelly_stakes
    
    def calculate_bankroll_management(self, data, kelly_stakes):
        """Calculate bankroll management rules with risk-aware caps"""
        if 'pnl' not in data:
            return {}

        pnl_df = data['pnl']
        total_profit = float(pnl_df['profit_total'].sum())
        total_stake = float(pnl_df['stake_total'].sum())

        # Calculate risk metrics
        daily_returns = pnl_df['profit_total'] / pnl_df['stake_total']
        sharpe_ratio = float(daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0
        max_drawdown = float(daily_returns.min())

        # ✅ PHASE 2: Get risk-aware daily caps
        daily_caps = {'total': 300, 'single': 100}  # defaults
        if 'strategy' in data:
            daily_caps = self.get_daily_caps(data['strategy'])

        reference_bankroll = self.reference_bankroll
        realized_bankroll = reference_bankroll

        quant_block = data.get('quant_pnl', {})
        quant_summary = quant_block.get('summary') or quant_block.get('overall') or {}
        total_pnl = quant_summary.get('total_pnl')
        if total_pnl is None:
            daily_entries = quant_block.get('daily') or []
            try:
                total_pnl = sum((entry.get('total_return', entry.get('return', 0)) or 0) - (entry.get('total_stake', entry.get('stake', 0)) or 0) for entry in daily_entries)
            except Exception:
                total_pnl = None

        if total_pnl is not None:
            realized_bankroll = reference_bankroll + float(total_pnl)

        bankroll_rules = {
            'current_bankroll': realized_bankroll,
            'recommended_daily_risk': min(daily_caps['total'], realized_bankroll * 0.1),  # 10% or cap max
            'max_single_loss': daily_caps['single'],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'risk_adjustment': 'AGGRESSIVE' if sharpe_ratio > 1.0 else 'MODERATE' if sharpe_ratio > 0.5 else 'CONSERVATIVE',
            'daily_caps': daily_caps,  # ✅ PHASE 2: Include caps
            'risk_mode': data.get('strategy', {}).get('risk_mode', 'NORMAL'),  # ✅ PHASE 2: Include risk mode
            'reference_bankroll': reference_bankroll,
            'realized_bankroll': realized_bankroll,
            'realized_pnl': total_pnl if total_pnl is not None else 0,
            'bankroll_mode': 'PNL_LINKED'
        }

        return bankroll_rules
    
    def generate_money_recommendations(self, kelly_stakes, bankroll_rules):
        """Generate money management recommendations with risk context"""
        recommendations = []
        
        # Kelly-based recommendations
        high_kelly_slots = [s for s, data in kelly_stakes.items() if data['risk_level'] == 'HIGH']
        for slot in high_kelly_slots:
            recommendations.append({
                'type': 'STAKE_OPTIMIZATION',
                'priority': 'HIGH',
                'action': f"Apply Kelly stake for {slot}: ₹{kelly_stakes[slot]['optimal_stake']}",
                'reason': f"Win probability: {kelly_stakes[slot]['win_probability']:.1%}, Kelly fraction: {kelly_stakes[slot]['kelly_fraction']:.3f}",
                'impact': 'HIGH'
            })
        
        # Bankroll recommendations with risk mode context
        risk_mode = bankroll_rules.get('risk_mode', 'NORMAL')
        daily_caps = bankroll_rules.get('daily_caps', {'total': 300, 'single': 100})
        
        recommendations.append({
            'type': 'RISK_MANAGEMENT',
            'priority': 'HIGH',
            'action': f"Apply {risk_mode} risk mode: ₹{daily_caps['total']} daily cap, ₹{daily_caps['single']} single max",
            'reason': f"Strategy recommendation: {risk_mode} mode",
            'impact': 'HIGH'
        })
        
        if bankroll_rules['risk_adjustment'] == 'AGGRESSIVE':
            recommendations.append({
                'type': 'RISK_MANAGEMENT',
                'priority': 'MEDIUM',
                'action': "Increase position sizing - System performing well",
                'reason': f"Sharpe ratio: {bankroll_rules['sharpe_ratio']:.2f} indicates good risk-adjusted returns",
                'impact': 'MEDIUM_HIGH'
            })
        elif bankroll_rules['risk_adjustment'] == 'CONSERVATIVE':
            recommendations.append({
                'type': 'RISK_MANAGEMENT', 
                'priority': 'HIGH',
                'action': "Reduce position sizing - High volatility detected",
                'reason': f"Max drawdown: {bankroll_rules['max_drawdown']:.1%}, Sharpe: {bankroll_rules['sharpe_ratio']:.2f}",
                'impact': 'HIGH'
            })
        
        return recommendations
    
    def save_money_management_plan(self, kelly_stakes, bankroll_rules, recommendations):
        """Save money management plan with risk configuration"""
        output_dir = self.base_dir / "logs" / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create money management plan
        money_plan = {
            'timestamp': datetime.now().isoformat(),
            'kelly_stakes': kelly_stakes,
            'bankroll_rules': bankroll_rules,
            'recommendations': recommendations,
            'daily_limits': {
                'max_total_stake': bankroll_rules.get('daily_caps', {}).get('total', 300),
                'max_single_stake': bankroll_rules.get('daily_caps', {}).get('single', 100),
                'max_daily_loss': 200,
                'target_daily_profit': 500
            },
            'risk_configuration': {  # ✅ PHASE 2: Add risk config
                'risk_mode': bankroll_rules.get('risk_mode', 'NORMAL'),
                'risk_multiplier': self.get_risk_multipliers(bankroll_rules),
                'daily_caps': bankroll_rules.get('daily_caps', {'total': 300, 'single': 100})
            }
        }
        
        plan_file = output_dir / "money_management_plan.json"
        with open(plan_file, 'w') as f:
            json.dump(money_plan, f, indent=2, default=str)
            
        # Save Excel report
        excel_file = output_dir / "money_management_report.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            if kelly_stakes:
                kelly_df = pd.DataFrame([{**{'slot': k}, **v} for k, v in kelly_stakes.items()])
                kelly_df.to_excel(writer, sheet_name='kelly_stakes', index=False)
            
            if bankroll_rules:
                bankroll_df = pd.DataFrame([bankroll_rules])
                bankroll_df.to_excel(writer, sheet_name='bankroll_rules', index=False)
            
            if recommendations:
                rec_df = pd.DataFrame(recommendations)
                rec_df.to_excel(writer, sheet_name='recommendations', index=False)
        
        print(f"💾 Money management plan saved to: {plan_file}")
        print(f"💾 Detailed report saved to: {excel_file}")
    
    def print_console_report(self, kelly_stakes, bankroll_rules, recommendations):
        """Print console report with risk summary"""
        print("\n" + "="*80)
        print("💰 MONEY MANAGER - ADVANCED BANKROLL MANAGEMENT")
        print("="*80)
        
        print(f"\n🎯 KELLY CRITERION OPTIMAL STAKES:")
        print("-" * 50)
        for slot, metrics in kelly_stakes.items():
            multiplier_note = f" (x{metrics['risk_multiplier_applied']:.1f})" if 'risk_multiplier_applied' in metrics else ""
            print(f"   {slot}: ₹{metrics['optimal_stake']:3} | Win Prob: {metrics['win_probability']:.1%} | Risk: {metrics['risk_level']}{multiplier_note}")
        
        risk_mode = bankroll_rules.get('risk_mode', 'NORMAL')
        daily_caps = bankroll_rules.get('daily_caps', {'total': 300, 'single': 100})
        
        print(f"\n📊 BANKROLL MANAGEMENT - {risk_mode} MODE:")
        print("-" * 45)
        print(f"   Current Bankroll: ₹{bankroll_rules['current_bankroll']:.0f}")
        print(f"   Reference Bankroll: ₹{bankroll_rules.get('reference_bankroll', bankroll_rules['current_bankroll']):.0f} | Realized Bankroll: ₹{bankroll_rules.get('realized_bankroll', bankroll_rules['current_bankroll']):.0f}")
        print(f"   Bankroll mode: {bankroll_rules.get('bankroll_mode', 'STATIC_REFERENCE')}")
        print(f"   Effective Bankroll (PNL_LINKED): ₹{bankroll_rules.get('realized_bankroll', bankroll_rules['current_bankroll']):.0f}")
        print(f"   Total Realized P&L (window): ₹{bankroll_rules.get('realized_pnl', 0):.0f}")
        print(f"   Recommended Daily Risk: ₹{bankroll_rules['recommended_daily_risk']:.0f}")
        print(f"   Daily Caps: ₹{daily_caps['total']} total, ₹{daily_caps['single']} single")
        print("   Note: Daily caps and stake multipliers are advisory. Core staking engine may choose higher stakes based on strategy and user preferences.")
        print(f"   Sharpe Ratio: {bankroll_rules['sharpe_ratio']:.2f}")
        print(f"   Risk Adjustment: {bankroll_rules['risk_adjustment']}")
        
        print(f"\n💡 MONEY MANAGEMENT RECOMMENDATIONS:")
        print("-" * 55)
        high_priority = [r for r in recommendations if r['priority'] == 'HIGH']
        for rec in high_priority:
            print(f"   🚀 {rec['action']}")
            print(f"      📝 {rec['reason']}")
    
    def run(self):
        """Main execution"""
        print("💰 MONEY MANAGER - Calculating advanced bankroll management...")
        
        # Load data
        data = self.load_financial_data()

        if 'pnl' not in data:
            print("⚠️ No financial data found; emitting neutral money plan")
            risk_mode = data.get('strategy', {}).get('risk_mode', 'DEFENSIVE')
            daily_caps = self.get_daily_caps({'risk_mode': risk_mode})
            bankroll_rules = {
                'current_bankroll': self.reference_bankroll,
                'reference_bankroll': self.reference_bankroll,
                'realized_bankroll': self.reference_bankroll,
                'realized_pnl': 0,
                'recommended_daily_risk': min(daily_caps['total'], self.reference_bankroll * 0.1),
                'max_single_loss': daily_caps['single'],
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'risk_adjustment': 'CONSERVATIVE',
                'daily_caps': daily_caps,
                'risk_mode': risk_mode,
                'bankroll_mode': 'STATIC_REFERENCE',
            }
            kelly_stakes = {}
            recommendations = self.generate_money_recommendations(kelly_stakes, bankroll_rules)
            self.print_console_report(kelly_stakes, bankroll_rules, recommendations)
            self.save_money_management_plan(kelly_stakes, bankroll_rules, recommendations)
            return True
            
        # Calculate money management
        kelly_stakes = self.calculate_kelly_criterion(data)
        bankroll_rules = self.calculate_bankroll_management(data, kelly_stakes)
        recommendations = self.generate_money_recommendations(kelly_stakes, bankroll_rules)
        
        # Output results
        self.print_console_report(kelly_stakes, bankroll_rules, recommendations)
        self.save_money_management_plan(kelly_stakes, bankroll_rules, recommendations)
        
        return True

def main():
    manager = MoneyManager()
    success = manager.run()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
