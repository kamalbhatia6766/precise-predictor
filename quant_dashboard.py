"""Central CLI dashboard for performance and P&L views."""

import argparse

from auto_backtest_runner import run_auto_backtest
from prediction_validator import run_prediction_validation
from quant_pnl_summary import run_summary
from reality_check_engine import run_reality_check
from real_time_performance_dashboard import run_real_time_dashboard
from source_performance_dashboard import run_source_dashboard
from system_recap_engine import run_system_recap


FOOTER_TEMPLATE = "[quant_dashboard] Completed: {command}"


def _print_footer(command: str) -> None:
    print(FOOTER_TEMPLATE.format(command=command))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Central dashboard / CLI hub for performance and P&L views",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("summary", help="Run the P&L summary utilities")
    subparsers.add_parser("realtime", help="Run the real-time performance dashboard")
    subparsers.add_parser("sources", help="Run the source performance dashboard")
    subparsers.add_parser("backtest", help="Run the auto backtest runner")

    validate_parser = subparsers.add_parser(
        "validate", help="Run the prediction validator (historical strategy analysis)",
    )
    validate_parser.add_argument(
        "--days", type=int, default=10, help="Number of days to analyze (default: 10)",
    )

    reality_parser = subparsers.add_parser(
        "reality-check", help="Run the reality check engine",
    )
    reality_parser.add_argument(
        "--days", type=int, default=10, help="Number of days to analyze (default: 10)",
    )

    subparsers.add_parser("recap", help="Run the system recap engine")

    args = parser.parse_args()

    if args.command == "summary":
        success = run_summary()
    elif args.command == "realtime":
        success = run_real_time_dashboard()
    elif args.command == "sources":
        success = run_source_dashboard()
    elif args.command == "backtest":
        success = run_auto_backtest()
    elif args.command == "validate":
        success = run_prediction_validation(args.days)
    elif args.command == "reality-check":
        success = run_reality_check(args.days)
    elif args.command == "recap":
        success = run_system_recap()
    else:
        parser.error("Unknown command")
        return 1

    _print_footer(args.command)
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
