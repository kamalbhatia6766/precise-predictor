"""Snapshot creation helpers for restoring key run artifacts."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import quant_paths

BASE_DIR = quant_paths.get_project_root()
SNAPSHOT_ROOT = BASE_DIR / "snapshots"
LATEST_POINTER = SNAPSHOT_ROOT / "latest.txt"

FIXED_FILES: List[Path] = [
    Path("data") / "topn_policy.json",
    Path("data") / "slot_health.json",
    Path("data") / "arjun_pick.json",
    Path("logs") / "performance" / "ultimate_performance.csv",
    Path("logs") / "performance" / "script_hit_memory.xlsx",
]

GLOB_FILES: List[Tuple[Path, str]] = [
    (Path("logs") / "performance", "quant_reality_pnl.*"),
    (Path("predictions") / "deepseek_scr9", "ultimate_predictions_*.xlsx"),
    (Path("predictions") / "bet_engine", "bet_plan_master_*.xlsx"),
]


def _copy_file(src: Path, dest: Path) -> bool:
    if not src.exists():
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return True


def _snapshot_files(snapshot_dir: Path) -> List[str]:
    copied: List[str] = []
    for rel_path in FIXED_FILES:
        src = BASE_DIR / rel_path
        dest = snapshot_dir / rel_path
        if _copy_file(src, dest):
            copied.append(str(rel_path))

    for rel_base, pattern in GLOB_FILES:
        src_dir = BASE_DIR / rel_base
        if not src_dir.exists():
            continue
        latest: Optional[Path] = None
        for src in src_dir.glob(pattern):
            if latest is None or src.stat().st_mtime > latest.stat().st_mtime:
                latest = src
        if latest:
            dest = snapshot_dir / rel_base / latest.name
            if _copy_file(latest, dest):
                copied.append(str(rel_base / latest.name))
    return copied


def create_snapshot() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    snapshot_dir = SNAPSHOT_ROOT / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied = _snapshot_files(snapshot_dir)
    if copied:
        LATEST_POINTER.parent.mkdir(parents=True, exist_ok=True)
        LATEST_POINTER.write_text(str(snapshot_dir))
    return snapshot_dir


def iter_managed_files(snapshot_dir: Path) -> Iterable[Path]:
    for rel_path in FIXED_FILES:
        yield snapshot_dir / rel_path
    for rel_base, pattern in GLOB_FILES:
        for src in (snapshot_dir / rel_base).glob(pattern):
            yield src
