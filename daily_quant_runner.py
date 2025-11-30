import subprocess, sys

def run(name):
    print(f"=== RUN {name} ===")
    subprocess.run([sys.executable, name], check=True)


def main():
    run("deepseek_scr9.py")
    run("prediction_hit_memory.py")
    run("pattern_intelligence_engine.py")
    run("money_stake_manager.py")
    run("precise_bet_engine.py")
    print("=== DAILY QUANT RUN COMPLETED ===")


if __name__ == "__main__":
    main()
