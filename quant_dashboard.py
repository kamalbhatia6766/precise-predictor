"""Console dashboard for quick Quant System health checks."""

import json
from datetime import datetime
from pathlib import Path

from auto_backtest_runner import run_auto_backtest

PERFORMANCE_DIR = Path(__file__).resolve().parent / "logs" / "performance"


def _load_json(path: Path, label: str):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ {label} not found – skipping that section.")
    except Exception as exc:
        print(f"⚠️ Could not read {label}: {exc}")
    return None


def _print_section(title: str):
    print(f"\n{title}")
    print("-" * len(title))


def _summarize_auto_backtest():
    summary, df_results = run_auto_backtest()
    if df_results is None or df_results.empty:
        print("⚠️ Auto backtest data unavailable.")
        return

    best_strategy = summary.get("best_strategy") or "N/A"
    roi = summary.get("best_strategy_roi")
    profit = summary.get("best_strategy_profit")
    window = summary.get("window_days") or "?"

    roi_display = f"{roi:+.1f}%" if roi is not None else "N/A"
    profit_display = f"₹{profit:,.0f}" if profit is not None else "N/A"

    print(
        f"Best strategy (last {window} days): {best_strategy} | ROI: {roi_display} | Profit: {profit_display}"
    )

    top_days = summary.get("top_profit_days", [])
    if top_days:
        print("Top profit days:")
        for item in top_days[:3]:
            profit = item.get("profit")
            if isinstance(profit, (int, float)):
                print(f" • {item['date']}: ₹{profit:+,.0f}")
            else:
                display_profit = "N/A" if profit is None else profit
                print(f" • {item['date']}: ₹{display_profit}")


def _summarize_real_time():
    data = _load_json(PERFORMANCE_DIR / "real_time_dashboard.json", "real_time_dashboard.json")
    if not data:
        return

    metrics = data.get("metrics", {})
    overall_roi = metrics.get("overall_roi")
    total_profit = metrics.get("total_profit")
    recent_roi = metrics.get("recent_roi")
    recent_profit = metrics.get("recent_profit")

    if total_profit is not None and overall_roi is not None:
        print(
            f"Overall P&L: ₹{total_profit:,.0f} | ROI: {overall_roi:+.1f}% | "
            f"Recent: ₹{(recent_profit or 0):+,.0f} ({(recent_roi or 0):+.1f}%)"
        )
    else:
        print("⚠️ real_time_dashboard.json missing key metrics.")

    slot_perf = metrics.get("slot_performance", {})
    if slot_perf:
        slot_summaries = [f"{slot}: ₹{info.get('total_profit', 0):+,.0f}" for slot, info in slot_perf.items()]
        print("Slot performance: " + "; ".join(slot_summaries))


def _summarize_reality_pnl():
    data = _load_json(PERFORMANCE_DIR / "quant_reality_pnl.json", "quant_reality_pnl.json")
    if not data:
        return

    by_slot = data.get("by_slot", [])
    if not by_slot:
        return

    best_slot = max(by_slot, key=lambda x: x.get("pnl", 0))
    worst_slot = min(by_slot, key=lambda x: x.get("pnl", 0))

    print(
        "Best slot: {slot} (P&L ₹{pnl:+,.0f}, ROI {roi:+.1f}%)".format(
            slot=best_slot.get("slot"),
            pnl=best_slot.get("pnl", 0),
            roi=best_slot.get("roi_pct", 0),
        )
    )
    print(
        "Worst slot: {slot} (P&L ₹{pnl:+,.0f}, ROI {roi:+.1f}%)".format(
            slot=worst_slot.get("slot"),
            pnl=worst_slot.get("pnl", 0),
            roi=worst_slot.get("roi_pct", 0),
        )
    )


def _summarize_strategy_recommendation():
    data = _load_json(PERFORMANCE_DIR / "strategy_recommendation.json", "strategy_recommendation.json")
    if not data:
        return

    strategy = data.get("recommended_strategy") or data.get("strategy")
    confidence = data.get("confidence") or data.get("confidence_level")
    reason = data.get("reason")

    print(f"Recommended strategy: {strategy} (confidence: {confidence})")
    if reason:
        print(f"Reason: {reason}")


def run_dashboard():
    print("🧭 QUANT DASHBOARD")
    print("=" * 40)

    _print_section("Auto Backtest Snapshot")
    _summarize_auto_backtest()

    _print_section("Real-Time Performance")
    _summarize_real_time()

    _print_section("Reality P&L")
    _summarize_reality_pnl()

    _print_section("Strategy Recommendation")
    _summarize_strategy_recommendation()

    print("\nDashboard generated at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return True


def main() -> int:
    success = run_dashboard()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
