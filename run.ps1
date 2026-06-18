# KIT 流失預測分析工具 — 啟動腳本（自動開啟瀏覽器）
Set-Location $PSScriptRoot
python -m shiny run app.py --reload --launch-browser
