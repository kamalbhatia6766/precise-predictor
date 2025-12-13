@echo off
cd /d "C:\Users\kamal\AppData\Local\Programs\Python\Python312\precise predictor"

REM Capture SCR9 run anchor for stale-file detection (ISO 8601 for cross-script parsing)
for /f "usebackq delims=" %%i in (`powershell -NoLogo -NoProfile -Command "Get-Date -Format o"`) do set SCR9_RUN_STARTED_AT=%%i

py -3.12 -m py_compile deepseek_scr9.py
if errorlevel 1 goto :fail

py -3.12 -m py_compile quant_daily_brief.py
if errorlevel 1 goto :fail

py -3.12 prediction_hit_memory.py --mode rebuild --window 90
if errorlevel 1 goto :fail
py -3.12 script_hit_metrics.py
if errorlevel 1 goto :fail
py -3.12 pattern_intelligence_enhanced.py
if errorlevel 1 goto :fail
py -3.12 topn_roi_scanner.py
if errorlevel 1 goto :fail
py -3.12 deepseek_scr9.py
if errorlevel 1 goto :fail
py -3.12 precise_bet_engine.py
if errorlevel 1 goto :fail
py -3.12 quant_arjun_mode.py
if errorlevel 1 goto :fail
py -3.12 bet_pnl_tracker.py
if errorlevel 1 goto :fail
py -3.12 quant_pnl_signals.py
if errorlevel 1 goto :fail
py -3.12 quant_daily_brief.py --mode auto
if errorlevel 1 goto :fail

echo.
echo QUANT DAILY AUTO LOOP FINISHED. Press any key to close...
pause >nul
goto :eof

:fail
echo.
echo STARTUP HALTED - see errors above (SCR9 compile/runtime failure or stale predictions).
exit /b 1
