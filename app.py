"""KIT 流失預測分析工具 — Shiny for Python entry point.

Usage (recommended — auto-opens browser):
    python app.py

Or via shiny CLI:
    python -m shiny run app.py --reload --launch-browser
"""
from shiny import App

from server import server
from ui import app_ui

app = App(app_ui, server)

if __name__ == "__main__":
    import threading
    import webbrowser

    # Open browser after uvicorn is ready (1.5 s grace period).
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:8000")).start()
    app.run(reload=False)
