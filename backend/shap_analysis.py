"""OOF SHAP importance with 95 % CI and full-data SHAP for beeswarm."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import StratifiedKFold


def _align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """Return X with only the columns the model was actually trained on.

    FLAML may drop all-NaN or constant columns internally, so the stored X
    can have more columns than the model expects.  Aligning prevents the
    'number of features in data … not the same as in training data' error.
    """
    expected: list[str] | None = None

    # LightGBM estimator: booster_ attribute
    if hasattr(model, "booster_"):
        expected = model.booster_.feature_name()
    # Sklearn-compatible feature_names_in_
    elif hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
    # Fallback: feature_name_ (some versions)
    elif hasattr(model, "feature_name_"):
        expected = list(model.feature_name_)

    if expected is None:
        return X

    available = [f for f in expected if f in X.columns]
    return X[available]


def run_oof_shap(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    shap_seeds: int = 10,
    shap_n_folds: int = 5,
    top_n: int = 20,
    cum_threshold: float = 0.80,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> dict:
    """Compute OOF mean |SHAP| with 95 % CI and full-data SHAP values.

    Returns:
        shap_df        – DataFrame ranked by mean_abs_shap (all features)
        sv_full        – full-data SHAP matrix (N × p) for beeswarm
        X              – feature DataFrame (passed through for beeswarm)
        top_n          – echo of parameter
        cum_threshold  – echo of parameter
    """
    # Align X to the exact features the model was trained on (FLAML may have
    # dropped constant / all-NaN columns internally).
    X = _align_features(model, X)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    sw = np.where(y == 1, n_neg / n_pos, 1.0)

    total = shap_seeds * shap_n_folds
    done = 0
    fold_means: list[np.ndarray] = []

    for seed in range(shap_seeds):
        skf = StratifiedKFold(n_splits=shap_n_folds, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X, y):
            m = model.__class__(**model.get_params())
            m.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=sw[tr_idx])
            explainer = shap.TreeExplainer(m)
            sv = explainer.shap_values(X.iloc[val_idx])
            if isinstance(sv, list):
                sv = sv[1]
            fold_means.append(np.abs(sv).mean(axis=0))
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    mat = np.array(fold_means)
    mean_shap = mat.mean(axis=0)
    ci_low = np.percentile(mat, 2.5, axis=0)
    ci_high = np.percentile(mat, 97.5, axis=0)

    shap_df = (
        pd.DataFrame({
            "feature": X.columns,
            "mean_abs_shap": mean_shap,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    shap_df["rank"] = shap_df.index + 1
    shap_df["cumulative_pct"] = (
        shap_df["mean_abs_shap"].cumsum() / shap_df["mean_abs_shap"].sum()
    )

    # Full-data SHAP for beeswarm (uses the training model as-is)
    exp_full = shap.TreeExplainer(model)
    sv_full = exp_full.shap_values(X)
    if isinstance(sv_full, list):
        sv_full = sv_full[1]

    return {
        "shap_df": shap_df,
        "sv_full": sv_full,
        "X": X,
        "top_n": top_n,
        "cum_threshold": cum_threshold,
    }
