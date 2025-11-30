@echo off
cd "C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor"

echo === STEP 1: NEXT-DAY PREDICTIONS ===
py -3.12 prediction_engine.py

echo.
echo === STEP 2: INTRADAY RECALC (OPTIONAL) ===
py -3.12 slot_recalc_engine.py --date 2025-11-29 --verbose

echo.
echo === STEP 3: REALITY PnL ===
py -3.12 bet_pnl_tracker.py

echo.
echo === STEP 4: SUMMARY + DASHBOARD ===
py -3.12 quant_pnl_summary.py
py -3.12 auto_backtest_runner.py

echo.
echo === STEP 5: HOUSEKEEPING (DRY RUN) ===
py -3.12 quant_housekeeping.py --dry-run

echo.
echo DONE.
pause
