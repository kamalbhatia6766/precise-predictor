@echo off
cd /d "C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor"

py -3.12 prediction_hit_memory.py --mode rebuild --window 90
py -3.12 script_hit_metrics.py
py -3.12 pattern_intelligence_enhanced.py
py -3.12 topn_roi_scanner.py
py -3.12 deepseek_scr9.py --mode tomorrow
py -3.12 precise_bet_engine.py
py -3.12 quant_arjun_mode.py
py -3.12 bet_pnl_tracker.py
py -3.12 quant_pnl_signals.py
py -3.12 quant_daily_brief.py --mode auto

echo.
echo QUANT DAILY AUTO LOOP FINISHED. Press any key to close...
pause >nul
