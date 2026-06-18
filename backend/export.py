"""Serialise analysis outputs to bytes for Shiny download handlers."""
import io
import zipfile

import joblib
import matplotlib.pyplot as plt
import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8-BOM CSV bytes (Excel-friendly)."""
    return df.to_csv(index=False).encode("utf-8-sig")


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    """Render Figure to PNG bytes, then close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def model_to_pkl_bytes(model) -> bytes:
    """Joblib-pickle a model to bytes."""
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    return buf.read()


def make_shap_zip(
    shap_df: pd.DataFrame,
    sel_df: pd.DataFrame,
    fig_bar: plt.Figure,
    fig_beeswarm: plt.Figure,
    fig_cum: plt.Figure,
) -> bytes:
    """Bundle all SHAP outputs (2 CSVs + 3 PNGs) into a ZIP archive."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("shap_importance_all.csv", df_to_csv_bytes(shap_df))
        zf.writestr("selected_variables.csv", df_to_csv_bytes(sel_df))
        zf.writestr("shap_bar_ci.png", fig_to_png_bytes(fig_bar))
        zf.writestr("shap_beeswarm.png", fig_to_png_bytes(fig_beeswarm))
        zf.writestr("shap_cumulative.png", fig_to_png_bytes(fig_cum))
    buf.seek(0)
    return buf.read()
