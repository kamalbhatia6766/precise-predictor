@echo off
chcp 65001 > nul
cd /d "C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor"

echo ========================================
echo   DAILY LIVE PREDICTIONS - CLEAN MODE
echo ========================================
echo.

echo [1/4] Starting Quantum Master Controller...
py -3.12 quant_master_controller.py --mode daily_live --speed-mode full

echo.
echo ========================================
echo   DAILY PIPELINE COMPLETED!
echo ========================================
echo.
echo Check these files for results:
echo    - predictions\bet_engine\live_bet_sheet_*.xlsx
echo    - predictions\bet_engine\bet_plan_master_*.xlsx
echo    - predictions\bet_engine\prediction_explainability_*.xlsx
echo.
echo Next: Open the latest live_bet_sheet file for today's numbers
echo.

pause