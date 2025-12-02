"""Wrapper for prediction hit memory analysis."""
import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "prediction_hit_memory.py"
    cmd = [sys.executable, str(target)] + sys.argv[1:]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
