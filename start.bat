py -3.12 quant_master_controller.py --mode daily_live --speed-mode full
py -3.12 bet_pnl_tracker.py --days 30
py -3.12 quant_pnl_signals.py
py -3.12 precise_daily_runner.py --mode predict --target auto --source scr9 --speed-mode full
py -3.12 quant_daily_brief.py --mode auto
pause