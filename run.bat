@echo off
cd /d "%~dp0"
python -m shiny run app.py --reload --launch-browser
