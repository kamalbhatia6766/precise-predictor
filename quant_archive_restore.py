"""Restore key prediction and performance files from archive back into live folders."""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List


def parse_args():
    parser = argparse.ArgumentParser(description="Restore archived predictions/logs back to live folders")
    parser.add_argument("--days", type=int, default=120, help="Number of days back to restore from archive")
    parser.add_argument(
        "--date-from",
        type=str,
        help="Restore archives from this date (YYYY-MM-DD) onward; overrides --days if provided",
    )
    parser.add_argument(
        "--what",
        choices=["predictions", "logs", "both"],
        default="both",
        help="Which categories to restore",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview files to restore without copying")
    parser.add_argument("--verbose", action="store_true", help="Print extra detail while restoring")
    return parser.parse_args()


def parse_cutoff_date(args) -> datetime:
    if args.date_from:
        try:
            return datetime.strptime(args.date_from, "%Y-%m-%d")
        except ValueError:
            raise SystemExit("Invalid --date-from format. Use YYYY-MM-DD.")
    return datetime.now() - timedelta(days=args.days)


def iter_archive_days(archive_root: Path, cutoff: datetime, verbose: bool) -> Iterable[Path]:
    if not archive_root.exists():
        return []
    for folder in sorted(archive_root.iterdir()):
        if not folder.is_dir():
            continue
        try:
            archive_date = datetime.strptime(folder.name, "%Y%m%d")
        except ValueError:
            if verbose:
                print(f"⚠️ Skipping non-date archive folder: {folder.name}")
            continue
        if archive_date >= cutoff:
            yield folder


def collect_prediction_files(archive_day: Path) -> List[Path]:
    predictions_dir = archive_day / "predictions"
    if not predictions_dir.exists():
        return []
    return [p for p in predictions_dir.rglob("*.xlsx") if p.is_file()]


def collect_log_files(archive_day: Path) -> List[Path]:
    logs_dir = archive_day / "logs" / "performance"
    if not logs_dir.exists():
        return []
    return [p for p in logs_dir.rglob("*") if p.suffix.lower() in {".xlsx", ".json", ".csv"} and p.is_file()]


def restore_files(files: List[Path], base_dir: Path, dry_run: bool, verbose: bool):
    for src in files:
        try:
            rel_path = src.relative_to(src.parents[2])
        except ValueError:
            # Expect structure archive/YYYYMMDD/<category>/...
            continue
        dest = base_dir / rel_path
        if dest.exists():
            if verbose:
                print(f"SKIP (exists): {rel_path}")
            continue
        if dry_run:
            print(f"RESTORE: {src} -> {rel_path}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        if verbose:
            print(f"RESTORED: {rel_path}")


if __name__ == "__main__":
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    archive_root = base_dir / "archive"
    cutoff_date = parse_cutoff_date(args)

    if args.what in {"predictions", "both"}:
        for archive_day in iter_archive_days(archive_root, cutoff_date, args.verbose):
            files = collect_prediction_files(archive_day)
            restore_files(files, base_dir, args.dry_run, args.verbose)

    if args.what in {"logs", "both"}:
        for archive_day in iter_archive_days(archive_root, cutoff_date, args.verbose):
            files = collect_log_files(archive_day)
            restore_files(files, base_dir, args.dry_run, args.verbose)

    if args.dry_run:
        print("\n💡 DRY RUN complete. No files were copied.")
    else:
        print("\n✅ Restore complete.")
