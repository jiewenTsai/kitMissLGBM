# KIT brand constants and CSS — no Shiny imports

KIT_PRIMARY = "#2d7535"
KIT_PRIMARY_LIGHT = "#3a8f45"
KIT_ACCENT = "#c6d033"
KIT_SUBTITLE_COLOR = "#e8f0a0"
KIT_BODY_BG = "#f5f5f5"
KIT_TEXT_MUTED = "#424242"
KIT_BORDER = "#e0e0e0"
KIT_INFO = "#2778c4"
KIT_DANGER = "#d9534f"
KIT_SECONDARY = "#616161"
KIT_WARNING = "#c6d033"  # same as accent

TARGET_DEFAULT = "attrition"


def kit_css() -> str:
    """KIT visual identity CSS rules (no Shiny dependency)."""
    return f"""
    body {{ background-color: {KIT_BODY_BG}; }}
    .kit-header {{
        background: linear-gradient(135deg, {KIT_PRIMARY} 0%, {KIT_PRIMARY_LIGHT} 100%);
        color: #fff;
        padding: 1.25rem 1.5rem;
        border-radius: 0.5rem;
        border-bottom: 4px solid {KIT_ACCENT};
        box-shadow: 0 2px 8px rgba(45, 117, 53, 0.2);
        margin-bottom: 1.25rem;
    }}
    .kit-header h1 {{ margin: 0; font-size: 1.75rem; font-weight: 500; }}
    .kit-header p  {{ margin: 0.35rem 0 0; opacity: 0.92; font-size: 0.95rem; }}
    .kit-subtitle  {{ color: {KIT_SUBTITLE_COLOR}; font-weight: 400; }}
    .nav-tabs .nav-link.active {{
        border-bottom: 3px solid {KIT_PRIMARY};
        font-weight: 500;
        color: {KIT_PRIMARY} !important;
    }}
    .nav-tabs .nav-link {{ color: {KIT_TEXT_MUTED}; }}
    .btn-primary, .btn-primary:focus {{
        background-color: {KIT_PRIMARY};
        border-color: {KIT_PRIMARY};
    }}
    .btn-primary:hover, .btn-primary:active {{
        background-color: {KIT_PRIMARY_LIGHT};
        border-color: {KIT_PRIMARY_LIGHT};
    }}
    .table {{ background: #fff; user-select: text; }}
    .shiny-table table {{ user-select: text; }}
    .well, pre {{ background-color: #fff; border: 1px solid {KIT_BORDER}; }}
    .kit-table-scroll {{ overflow-x: auto; max-width: 100%; margin-bottom: 0.5rem; }}
    .kit-table-scroll table {{ font-size: 0.85rem; white-space: nowrap; }}
    """
