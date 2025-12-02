"""Convenience wrapper around golden_block_analyzer."""
import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "golden_block_analyzer.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
