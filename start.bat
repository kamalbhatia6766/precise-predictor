@echo off
cd /d "%~dp0"

py -3.12 quant_daily_brief.py --mode auto
