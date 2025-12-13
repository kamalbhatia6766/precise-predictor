"""Restore snapshot-managed files from snapshots folder."""
from __future__ import annotations

import argparse
import fnmatch
import shutil
from pathlib import Path
from typing import Iterable, List

import quant_paths
from snapshot_manager import GLOB_FILES, FIXED_FILES, LATEST_POINTER, SNAPSHOT_ROOT

BASE_DIR = quant_paths.get_project_root()


def _resolve_snapshot(name: str) -> Path:
    if name.lower() == "latest":
        if not LATEST_POINTER.exists():
            raise FileNotFoundError("No latest snapshot pointer found")
        pointer_value = LATEST_POINTER.read_text().strip()
        candidate = Path(pointer_value)
        if not candidate.is_absolute():
            candidate = SNAPSHOT_ROOT / pointer_value
        if not candidate.exists():
            raise FileNotFoundError(f"Pointer targets missing snapshot: {candidate}")
        return candidate

    candidate = Path(name)
    if not candidate.is_absolute():
        candidate = SNAPSHOT_ROOT / candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Snapshot not found: {candidate}")
    return candidate


def _managed_patterns() -> List[Path]:
    patterns: List[Path] = []
    patterns.extend(FIXED_FILES)
    for base_dir, pattern in GLOB_FILES:
        patterns.append(base_dir / pattern)
    return patterns


def _is_managed(rel_path: Path, patterns: Iterable[Path]) -> bool:
    for pattern in patterns:
        if "*" in pattern.name:
            if rel_path.is_relative_to(pattern.parent) and fnmatch.fnmatch(rel_path.name, pattern.name):
                return True
        elif rel_path == pattern:
            return True
    return False


def restore_snapshot(snapshot_name: str) -> Path:
    snapshot_dir = _resolve_snapshot(snapshot_name)
    managed_patterns = _managed_patterns()
    restored: List[str] = []

    for file_path in snapshot_dir.rglob("*"):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(snapshot_dir)
        if not _is_managed(rel_path, managed_patterns):
            continue
        dest = BASE_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        restored.append(str(rel_path))

    print(f"Restored {len(restored)} file(s) from {snapshot_dir}")
    for path in restored:
        print(f" - {path}")
    return snapshot_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore snapshot-managed files")
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot folder name under snapshots/ or 'latest'",
    )
    args = parser.parse_args()

    try:
        restore_snapshot(args.snapshot)
    except Exception as exc:
        print(f"❌ Restore failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
