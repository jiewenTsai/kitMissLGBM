"""Shared Shiny UI helpers: KIT header, goal panel, status alert."""
from shiny import ui

from backend.theme import (
    KIT_ACCENT,
    KIT_BODY_BG,
    KIT_BORDER,
    KIT_DANGER,
    KIT_INFO,
    KIT_PRIMARY,
    KIT_PRIMARY_LIGHT,
    KIT_SECONDARY,
    KIT_SUBTITLE_COLOR,
    KIT_TEXT_MUTED,
    KIT_WARNING,
    kit_css,
)


def make_theme() -> ui.Theme:
    """Lumen preset with KIT brand colour overrides (requires shiny[theme])."""
    theme = ui.Theme(preset="lumen")
    theme.add_defaults(
        f"$primary: {KIT_PRIMARY};",
        f"$success: {KIT_PRIMARY};",
        f"$secondary: {KIT_SECONDARY};",
        f"$info: {KIT_INFO};",
        f"$warning: {KIT_WARNING};",
        f"$danger: {KIT_DANGER};",
        f"$body-bg: {KIT_BODY_BG};",
    )
    theme.add_rules(kit_css())
    return theme


def kit_header(title: str, subtitle: str, tagline: str) -> ui.Tag:
    """KIT branded page header (green gradient + accent underline)."""
    return ui.div(
        ui.h1(title),
        ui.p(ui.tags.span(subtitle, class_="kit-subtitle"), f" · {tagline}"),
        class_="kit-header",
    )


def goal_panel(
    objective: str,
    tool: str,
    steps: list[str],
    result: str,
    citation: str | None = None,
) -> ui.Tag:
    """kitMiss-style green alert panel: 【本分頁目標】 / 工具 / 步驟 / 結果 / 引用."""
    step_items = [ui.tags.li(s) for s in steps]
    body: list = [
        ui.tags.strong("【本分頁目標】"),
        f" {objective}",
        ui.tags.hr(style="margin: 0.5rem 0;"),
        ui.tags.strong("工具："), f" {tool}",
        ui.tags.br(),
        ui.tags.strong("步驟："),
        ui.tags.ol(*step_items, style="margin: 0.3rem 0 0 1.2rem; padding: 0;"),
        ui.tags.strong("結果："), f" {result}",
    ]
    if citation:
        body += [
            ui.tags.hr(style="margin: 0.5rem 0;"),
            ui.tags.small(
                ui.tags.strong("引用："),
                ui.tags.br(),
                ui.HTML(citation),
                class_="text-muted",
            ),
        ]
    return ui.div(*body, class_="alert alert-success")


def status_alert(state: str, message: str) -> ui.Tag:
    """Bootstrap alert for running / success / error states."""
    _prefix = {"running": "【進行中】", "success": "【成功】", "error": "【失敗】"}
    _cls    = {
        "running": "alert alert-info",
        "success": "alert alert-success",
        "error":   "alert alert-danger",
    }
    return ui.div(
        ui.tags.strong(_prefix.get(state, "")),
        f" {message}",
        class_=_cls.get(state, "alert alert-secondary"),
        role="alert",
        style="margin-top: 8px;",
    )


def progress_bar_ui(phase: str, done: int, total: int, msg: str) -> ui.Tag | None:
    """Bootstrap progress bar for a running background task.

    phase="flaml"  → indeterminate striped bar (FLAML search, duration unknown)
    phase="oof"    → determinate bar (done/total folds)
    phase="idle"   → returns None (hidden)
    """
    if phase == "idle":
        return None

    if phase == "flaml":
        bar = ui.div(
            ui.div(
                class_=(
                    "progress-bar progress-bar-striped progress-bar-animated"
                    " bg-success"
                ),
                role="progressbar",
                style="width: 100%;",
                aria_valuenow="100",
                aria_valuemin="0",
                aria_valuemax="100",
            ),
            class_="progress",
            style="height: 1rem;",
        )
    else:  # "oof" or "shap"
        pct = (done / total * 100) if total > 0 else 0
        bar = ui.div(
            ui.div(
                f"{done} / {total}",
                class_="progress-bar bg-success",
                role="progressbar",
                style=f"width: max({pct:.1f}%, 3em); min-width: 3em;",
                aria_valuenow=str(done),
                aria_valuemin="0",
                aria_valuemax=str(total),
            ),
            class_="progress",
            style="height: 1.5rem; font-size: 0.8rem;",
        )

    return ui.div(
        ui.tags.p(msg, class_="text-muted mb-1", style="font-size: 0.88rem;"),
        bar,
        style="margin-top: 6px; margin-bottom: 6px;",
    )


def download_hint() -> ui.Tag:
    """Standard kitMiss download footer hint."""
    return ui.tags.p(
        "若下載失敗並顯示「檢查網際網路連線」，通常不是網路問題："
        "請確認訓練／SHAP 已顯示【成功】、終端機的 app 仍在執行、"
        "按 F5 重新整理後再下載；連續下載多檔請改用「全部 ZIP」。",
        class_="text-muted",
        style="font-size:0.85rem; margin-top:0.5rem;",
    )
