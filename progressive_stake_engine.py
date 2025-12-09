"""Wrapper to run the master money & stake manager."""
import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "money_stake_manager.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
