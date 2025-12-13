"""Canonical Python entrypoint for running the quant daily brief."""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


RUN_WARNING = (
    "Do not run start.bat with python. Use call start.bat OR py -3.12 start.py"
)


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> int:
    base_dir = _project_root()
    os.chdir(base_dir)

    print(RUN_WARNING)
    anchor = datetime.now(tz=ZoneInfo("Asia/Kolkata")).isoformat()
    env = os.environ.copy()
    env["SCR9_RUN_STARTED_AT"] = anchor

    cmd = [sys.executable, "quant_daily_brief.py", "--mode", "auto"]
    print(f"Launching daily brief via: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print("❌ quant_daily_brief.py failed; see logs above.")
        return result.returncode

    print("✅ quant_daily_brief.py completed successfully.")
    print(RUN_WARNING)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
