"""Analyze near-miss patterns using PreciseBetEngine helpers."""
import argparse

from precise_bet_engine import analyze_near_miss_history


def main():
    parser = argparse.ArgumentParser(description="Near-miss pattern analyzer")
    parser.add_argument("--days", type=int, default=30, help="History window (days)")
    args = parser.parse_args()

    analyze_near_miss_history(days=args.days)


if __name__ == "__main__":
    main()
