"""Quant Housekeeping - keeps predictions/log folders tidy.

Features
- Archives or deletes old predictions and logs based on retention window.
- Keeps only a limited number of recent files per script folder to avoid bloat.
- Detects dates from filenames (YYYYMMDD[_HHMMSS]) so files with copied mtimes can still be cleaned.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path


class QuantHousekeeping:
    def __init__(self, config_path=None):
        self.base_dir = Path(__file__).resolve().parent
        self.config = self.load_config(config_path)

        # Whitelist of files that MUST NEVER be deleted
        self.whitelist = [
            "number prediction learn.xlsx",
            "pattern_packs.py",
            "quant_paths.py",
            "quant_data_core.py",
            "bet_pnl_history.xlsx",
            "quant_reality_pnl.json",
        ]

    def load_config(self, config_path):
        """Load housekeeping configuration"""
        default_config = {
            # How long to keep files (by embedded date or file mtime)
            "retention_days_predictions": 30,
            "retention_days_logs": 60,
            # How many recent files to keep per script folder regardless of age
            "retain_latest_per_slot": 5,
            # Run live cleanup by default (still archives, not deletes)
            "dry_run": False,
            # Archive instead of deleting so the process is reversible
            "archive_instead_of_delete": True,
        }

        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:  # pragma: no cover - config parsing safety
                print(f"⚠️ Error loading config: {e}, using defaults")

        return default_config

    def is_protected_file(self, file_path):
        """Check if file is in whitelist"""
        return file_path.name in self.whitelist

    def _infer_file_datetime(self, file_path: Path) -> datetime:
        """Infer file timestamp from filename (YYYYMMDD or YYYYMMDD_HHMMSS) or fall back to mtime."""
        digits = "".join(filter(str.isdigit, file_path.name))
        for fmt in ("%Y%m%d_%H%M%S", "%Y%m%d%H%M%S", "%Y%m%d"):
            try:
                return datetime.strptime(digits[: len(datetime.now().strftime(fmt))], fmt)
            except ValueError:
                continue
        return datetime.fromtimestamp(file_path.stat().st_mtime)

    def _collect_old_in_folder(self, files, cutoff_date):
        """Keep newest N files and flag anything older than cutoff or beyond retain_latest_per_slot."""
        old_files = []
        files_sorted = sorted(files, key=self._infer_file_datetime, reverse=True)
        keep = set(files_sorted[: self.config["retain_latest_per_slot"]])
        for file in files_sorted:
            if file in keep:
                continue
            file_dt = self._infer_file_datetime(file)
            if file_dt < cutoff_date:
                old_files.append(file)
            else:
                # Beyond retain_latest_per_slot: still eligible for cleanup to keep folders light
                old_files.append(file)
        return old_files

    def scan_predictions_dir(self):
        """Scan predictions directory (all script folders) for old files"""
        predictions_dir = self.base_dir / "predictions"
        if not predictions_dir.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=self.config["retention_days_predictions"])
        old_files = []

        # Scan bet_engine files (keep latest N per pattern)
        bet_engine_dir = predictions_dir / "bet_engine"
        if bet_engine_dir.exists():
            pattern_files = {
                "master": list(bet_engine_dir.glob("bet_plan_master_*.xlsx")),
                "final": list(bet_engine_dir.glob("final_bet_plan_*.xlsx")),
                "live": list(bet_engine_dir.glob("live_bet_sheet_*.xlsx")),
            }

            for files in pattern_files.values():
                old_files.extend(self._collect_old_in_folder(files, cutoff_date))

        # Scan individual script folders (deepseek_scr*, scr9, etc.)
        for subdir in predictions_dir.iterdir():
            if not subdir.is_dir() or subdir.name == "bet_engine":
                continue

            files = [p for p in subdir.rglob("*") if p.is_file()]
            if not files:
                continue

            old_files.extend(self._collect_old_in_folder(files, cutoff_date))

        return old_files

    def scan_logs_dir(self):
        """Scan logs directory for old files"""
        logs_dir = self.base_dir / "logs"
        if not logs_dir.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=self.config["retention_days_logs"])
        old_files = []

        # Recursively scan logs directory
        for file_path in logs_dir.rglob("*"):
            if file_path.is_file() and not self.is_protected_file(file_path):
                # Check if file is older than retention period
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    old_files.append(file_path)

        return old_files

    def archive_files(self, files):
        """Archive files instead of deleting"""
        archive_dir = self.base_dir / "archive" / datetime.now().strftime("%Y%m%d")
        archive_dir.mkdir(parents=True, exist_ok=True)

        archived = []
        for file_path in files:
            try:
                # Create relative path in archive
                if "predictions" in str(file_path):
                    rel_path = file_path.relative_to(self.base_dir / "predictions")
                    target_dir = archive_dir / "predictions" / rel_path.parent
                else:
                    rel_path = file_path.relative_to(self.base_dir / "logs")
                    target_dir = archive_dir / "logs" / rel_path.parent

                target_dir.mkdir(parents=True, exist_ok=True)
                target_path = target_dir / file_path.name

                file_path.rename(target_path)
                archived.append(target_path)

            except Exception as e:  # pragma: no cover - best effort logging
                print(f"❌ Error archiving {file_path}: {e}")

        return archived

    def delete_files(self, files):
        """Delete files permanently"""
        deleted = []
        for file_path in files:
            try:
                file_path.unlink()
                deleted.append(file_path)
            except Exception as e:  # pragma: no cover - best effort logging
                print(f"❌ Error deleting {file_path}: {e}")

        return deleted

    def run_housekeeping(self):
        """Run complete housekeeping"""
        print("🧹 QUANT HOUSEKEEPING - FILE CLEANUP")
        print("=" * 50)
        print(f"Mode: {'DRY RUN' if self.config['dry_run'] else 'LIVE'}")
        print(f"Archive: {self.config['archive_instead_of_delete']}")
        print(f"Predictions retention: {self.config['retention_days_predictions']} days")
        print(f"Logs retention: {self.config['retention_days_logs']} days")
        print()

        # Scan for old files
        old_predictions = self.scan_predictions_dir()
        old_logs = self.scan_logs_dir()
        all_old_files = old_predictions + old_logs

        if not all_old_files:
            print("✅ No old files found for cleanup")
            return True

        print(f"📁 Found {len(all_old_files)} old files:")
        for file_path in all_old_files[:10]:  # Show first 10
            file_date = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d")
            print(f"   {file_path.name} ({file_date})")

        if len(all_old_files) > 10:
            print(f"   ... and {len(all_old_files) - 10} more files")

        if self.config["dry_run"]:
            print(
                f"\n💡 DRY RUN: Would {'archive' if self.config['archive_instead_of_delete'] else 'delete'} {len(all_old_files)} files"
            )
            return True

        # Actual cleanup
        print(f"\n🚀 Performing actual cleanup...")
        if self.config["archive_instead_of_delete"]:
            archived = self.archive_files(all_old_files)
            print(f"✅ Archived {len(archived)} files to {self.base_dir / 'archive'}")
        else:
            deleted = self.delete_files(all_old_files)
            print(f"✅ Deleted {len(deleted)} files")

        return True

    def housekeeping_predictions(self):
        """Housekeeping only for predictions directory"""
        old_files = self.scan_predictions_dir()
        return self._process_files(old_files, "predictions")

    def housekeeping_logs(self):
        """Housekeeping only for logs directory"""
        old_files = self.scan_logs_dir()
        return self._process_files(old_files, "logs")

    def _process_files(self, files, category):
        """Process files for specific category"""
        if not files:
            print(f"✅ No old {category} files found")
            return True

        print(f"📁 Found {len(files)} old {category} files")

        if self.config["dry_run"]:
            print(f"💡 DRY RUN: Would process {len(files)} {category} files")
            return True

        if self.config["archive_instead_of_delete"]:
            archived = self.archive_files(files)
            print(f"✅ Archived {len(archived)} {category} files")
        else:
            deleted = self.delete_files(files)
            print(f"✅ Deleted {len(deleted)} {category} files")

        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quant Housekeeping - File Cleanup")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Preview actions without moving/deleting files")
    parser.add_argument("--live", action="store_true", help="Force live mode (overrides dry-run)")
    parser.add_argument("--predictions-only", action="store_true", help="Clean only predictions")
    parser.add_argument("--logs-only", action="store_true", help="Clean only logs")

    args = parser.parse_args()

    housekeeping = QuantHousekeeping(args.config)

    # Override dry-run if flags are provided
    if args.dry_run:
        housekeeping.config["dry_run"] = True
    if args.live:
        housekeeping.config["dry_run"] = False

    if args.predictions_only:
        success = housekeeping.housekeeping_predictions()
    elif args.logs_only:
        success = housekeeping.housekeeping_logs()
    else:
        success = housekeeping.run_housekeeping()

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
