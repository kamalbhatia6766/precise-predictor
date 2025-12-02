"""Convenience wrapper to rebuild the pack registry."""
import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    target = base_dir / "pattern_packs.py"
    subprocess.run([sys.executable, str(target)], check=True)
    print("✅ Master pack registry refreshed (see console output above)")


if __name__ == "__main__":
    main()
