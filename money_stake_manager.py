from pathlib import Path
from datetime import datetime
import argparse

class MoneyStakeManager:
    """
    MONEY & STAKE MASTER ORCHESTRATOR

    Runs, in order:
      1) LossRecoveryEngine           → loss_recovery_plan.json
      2) StrategyRecommendationEngine → strategy_recommendation.json
      3) DynamicStakeAllocator        → dynamic_stake_plan.json
      4) MoneyManager                 → money_management_plan.json + XLSX
      5) ExecutionReadinessEngine     → execution_readiness_summary.json + XLSX
    """
    def __init__(self, readiness_days: int = 10, quiet: bool = False):
        self.readiness_days = readiness_days
        self.quiet = quiet
        self.base_dir = Path(__file__).resolve().parent
        self.step_status = {}

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg)

    def run(self) -> bool:
        # Import inside run() so original modules remain owners of logic
        from loss_recovery_engine import LossRecoveryEngine
        from strategy_recommendation_engine import StrategyRecommendationEngine
        from dynamic_stake_allocator import DynamicStakeAllocator
        from money_manager import MoneyManager
        from execution_readiness_engine import ExecutionReadinessEngine

        self._log("=" * 70)
        self._log("💰 MONEY & STAKE MANAGER – MASTER ORCHESTRATOR")
        self._log("=" * 70)
        self._log(f"📂 Base dir : {self.base_dir}")
        self._log(f"🕒 Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._log(f"📊 ReadyDays: {self.readiness_days}")
        self._log("=" * 70)

        ok_all = True

        # 1) Loss recovery (zone + risk mode)
        try:
            self._log("1️⃣  LOSS RECOVERY ENGINE…")
            lr = LossRecoveryEngine()
            ok = lr.run_recovery_analysis()
            self.step_status["loss_recovery"] = bool(ok)
            ok_all &= bool(ok)
        except Exception as e:
            self.step_status["loss_recovery"] = False
            self._log(f"   ❌ LossRecoveryEngine error: {e}")
            ok_all = False

        # 2) Strategy recommendation (S40 / 164950 / packs)
        try:
            self._log("2️⃣  STRATEGY RECOMMENDATION ENGINE…")
            se = StrategyRecommendationEngine()
            ok = se.run_strategy_analysis()
            self.step_status["strategy_recommendation"] = bool(ok)
            ok_all &= bool(ok)
        except Exception as e:
            self.step_status["strategy_recommendation"] = False
            self._log(f"   ❌ StrategyRecommendationEngine error: {e}")
            ok_all = False

        # 3) Dynamic stake allocation (slot-wise stakes)
        try:
            self._log("3️⃣  DYNAMIC STAKE ALLOCATOR…")
            da = DynamicStakeAllocator()
            ok = da.run_allocation()
            self.step_status["dynamic_stake_allocator"] = bool(ok)
            ok_all &= bool(ok)
        except Exception as e:
            self.step_status["dynamic_stake_allocator"] = False
            self._log(f"   ❌ DynamicStakeAllocator error: {e}")
            ok_all = False

        # 4) Money manager (bankroll + caps + JSON plan)
        try:
            self._log("4️⃣  MONEY MANAGER…")
            mm = MoneyManager()
            ok = mm.run()
            self.step_status["money_manager"] = bool(ok)
            ok_all &= bool(ok)
        except Exception as e:
            self.step_status["money_manager"] = False
            self._log(f"   ❌ MoneyManager error: {e}")
            ok_all = False

        # 5) Execution readiness (GO / NO-GO view)
        try:
            self._log("5️⃣  EXECUTION READINESS ENGINE…")
            er = ExecutionReadinessEngine()
            ok = er.run_engine(self.readiness_days)
            self.step_status["execution_readiness"] = bool(ok)
            ok_all &= bool(ok)
        except Exception as e:
            self.step_status["execution_readiness"] = False
            self._log(f"   ❌ ExecutionReadinessEngine error: {e}")
            ok_all = False

        # Compact summary
        self._log("\n" + "-" * 70)
        self._log("📌 PHASE-2 STATUS SNAPSHOT")
        for key, label in [
            ("loss_recovery", "LossRecovery"),
            ("strategy_recommendation", "StrategyReco"),
            ("dynamic_stake_allocator", "DynStake"),
            ("money_manager", "MoneyMgr"),
            ("execution_readiness", "ExecReady"),
        ]:
            flag = self.step_status.get(key, False)
            self._log(f"   {'✅' if flag else '❌'} {label}")
        self._log("-" * 70)

        # Clarify ROI terminology without altering any calculations
        self._log("ROI METRICS LEGEND:")
        self._log(" - BASE ROI (model-only): pre-overlay ROI from base bet plans before dynamic stake adjustments.")
        self._log(" - Overall ROI (realized): execution-level ROI that bet_pnl_tracker and roi_summary report.")
        self._log(" - Rolling/7d ROI: short-window snapshots of the realized ROI trajectory.")

        self._log("✅ ALL DONE" if ok_all else "⚠️ Completed with issues")
        return ok_all


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Money & Stake Manager – run all money-phase engines in correct order"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Window (days) for ExecutionReadinessEngine scoring",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal console output",
    )
    args = parser.parse_args()

    mgr = MoneyStakeManager(readiness_days=args.days, quiet=args.quiet)
    ok = mgr.run()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
