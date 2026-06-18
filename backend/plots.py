"""Matplotlib plot factories — each returns a Figure, no Shiny dependency."""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from backend.theme import KIT_PRIMARY, KIT_DANGER, KIT_SECONDARY


def plot_shap_bar_ci(shap_df: pd.DataFrame, top_n: int) -> plt.Figure:
    """Horizontal bar chart of top_n features with 95 % CI error bars."""
    plot_df = shap_df.iloc[:top_n].iloc[::-1].copy()
    fig_h = max(5.0, top_n * 0.32 + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), dpi=100)
    ax.barh(
        plot_df["feature"], plot_df["mean_abs_shap"],
        color=KIT_PRIMARY, alpha=0.85, height=0.6,
    )
    ax.errorbar(
        plot_df["mean_abs_shap"], plot_df["feature"],
        xerr=[
            plot_df["mean_abs_shap"] - plot_df["ci_low"],
            plot_df["ci_high"] - plot_df["mean_abs_shap"],
        ],
        fmt="none", color="black", capsize=3, linewidth=1,
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top-{top_n} Important Variables (OOF SHAP, 95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def plot_shap_beeswarm(
    sv_full: np.ndarray,
    X: pd.DataFrame,
    top_features: list[str],
    top_n: int,
) -> plt.Figure:
    """Beeswarm of full-data SHAP values for top_n features."""
    feature_names = list(X.columns)
    idx_list = [feature_names.index(f) for f in top_features if f in feature_names]
    fig_h = max(6.0, top_n * 0.32 + 1.5)
    plt.figure(figsize=(10, fig_h), dpi=100)
    shap.summary_plot(
        sv_full[:, idx_list], X.iloc[:, idx_list],
        show=False, max_display=top_n,
    )
    plt.title(f"Beeswarm: Top-{top_n} Variables (direction, full-data model)")
    plt.tight_layout()
    return plt.gcf()


def plot_shap_cumulative(
    shap_df: pd.DataFrame,
    top_n: int,
    cum_threshold: float,
) -> plt.Figure:
    """Cumulative SHAP proportion curve with threshold and Top-N markers."""
    n_cum = int((shap_df["cumulative_pct"] <= cum_threshold).sum())
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
    ax.plot(
        range(1, len(shap_df) + 1), shap_df["cumulative_pct"],
        color=KIT_PRIMARY, linewidth=2,
    )
    ax.axhline(
        cum_threshold, color=KIT_DANGER, linestyle="--", alpha=0.7,
        label=f"{cum_threshold:.0%} threshold ({n_cum} vars)",
    )
    ax.axvline(
        top_n, color=KIT_SECONDARY, linestyle=":", alpha=0.7,
        label=f"Top-{top_n}",
    )
    ax.set_xlabel("Number of variables (ranked by SHAP)")
    ax.set_ylabel("Cumulative SHAP proportion")
    ax.set_title("Cumulative SHAP — Variable Selection Aid")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
