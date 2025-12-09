"""Surface prediction confidence metrics or fall back to the daily brief."""
import json
import subprocess
import sys
from pathlib import Path


CONF_FILE = Path(__file__).resolve().parent / "logs" / "performance" / "prediction_confidence.json"


def print_confidence_summary(data: dict) -> None:
    slot_scores = data.get("confidence_scores", {})
    if not slot_scores:
        print("⚠️ No confidence scores available; consider running execution_readiness_engine.py")
        return

    scores = {}
    for slot, slot_data in slot_scores.items():
        if isinstance(slot_data, dict):
            scores[slot] = slot_data.get("confidence_score", 0)
    if not scores:
        print("⚠️ Confidence structure present but empty")
        return

    avg_score = sum(scores.values()) / len(scores)
    high_slots = [slot for slot, score in scores.items() if score >= 65]

    print("🎯 Prediction Confidence Snapshot")
    print(f"   Avg Score : {avg_score:.1f}")
    print(f"   High Slots: {', '.join(high_slots) if high_slots else 'None'}")
    for slot, score in sorted(scores.items()):
        print(f"   {slot}: {score:.1f}")


def main():
    if CONF_FILE.exists():
        try:
            with open(CONF_FILE, "r") as f:
                data = json.load(f) or {}
            print_confidence_summary(data)
            return
        except Exception as exc:
            print(f"⚠️ Error reading confidence file: {exc}")

    print("ℹ️ Falling back to quant_daily_brief.py for confidence info")
    base_dir = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(base_dir / "quant_daily_brief.py")], check=True)


if __name__ == "__main__":
    main()
