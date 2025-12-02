"""Quick combined review console leveraging existing summaries."""
import subprocess
import sys
from pathlib import Path


def main():
    base_dir = Path(__file__).resolve().parent
    print("📝 Launching quant_daily_brief for snapshot...")
    subprocess.run([sys.executable, str(base_dir / "quant_daily_brief.py")], check=False)
    print("ℹ️ For ROI details see roi_summary.py; for pattern intelligence see pattern_intelligence_engine.py")


if __name__ == "__main__":
    main()
