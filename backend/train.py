"""FLAML AutoML training and OOF evaluation utilities."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

N_OOF_SEEDS = 10


def _sample_weight(y: np.ndarray) -> np.ndarray:
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    return np.where(y == 1, n_neg / n_pos, 1.0)


def run_flaml_train(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    time_budget: int,
    n_splits: int,
    seed: int,
):
    """Fit FLAML AutoML (LightGBM); return (best_estimator, best_config)."""
    automl = AutoML()
    automl.fit(
        X,
        y,
        task="classification",
        metric="roc_auc",
        estimator_list=["lgbm"],
        eval_method="cv",
        n_splits=n_splits,
        time_budget=time_budget,
        early_stop=True,
        sample_weight=_sample_weight(y),
        seed=seed,
    )
    return automl.model.estimator, automl.best_config


def compute_oof_metrics(
    model,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    n_splits: int,
    n_seeds: int = N_OOF_SEEDS,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """n_seeds × n_splits OOF evaluation; return summary DataFrame.

    progress_cb(done, total) is called after each fold completes.
    """
    sw = _sample_weight(y)
    buckets: dict[str, list[float]] = {
        "AUC": [], "AUPRC": [], "F1": [], "Recall": [], "Brier": []
    }
    total = n_seeds * n_splits
    done = 0

    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X, y):
            m = model.__class__(**model.get_params())
            m.fit(X.iloc[tr_idx], y[tr_idx], sample_weight=sw[tr_idx])
            prob = m.predict_proba(X.iloc[val_idx])[:, 1]
            pred = (prob >= 0.5).astype(int)
            yv = y[val_idx]
            buckets["AUC"].append(roc_auc_score(yv, prob))
            buckets["AUPRC"].append(average_precision_score(yv, prob))
            buckets["F1"].append(f1_score(yv, pred, zero_division=0))
            buckets["Recall"].append(recall_score(yv, pred, zero_division=0))
            buckets["Brier"].append(brier_score_loss(yv, prob))
            done += 1
            if progress_cb is not None:
                progress_cb(done, total)

    baseline_auprc = float(y.mean())
    _NOTES: dict[str, str] = {
        "AUC":    "ROC 曲線下面積；0.5=隨機，>0.7 可接受，>0.8 良好，>0.9 優異",
        "AUPRC":  "",   # filled below with baseline
        "F1":     "精確率與召回率調和平均；>0.5 可接受，越高越好",
        "Recall": "流失樣本捕獲率（敏感性）；高 Recall 可減少漏報，建議 >0.6",
        "Brier":  "機率校準誤差（越低越好）；<0.1 優異，<0.25 尚可，接近流失率=隨機",
    }
    rows = []
    for metric, values in buckets.items():
        note = _NOTES.get(metric, "")
        if metric == "AUPRC":
            note = (
                f"Precision-Recall AUC；baseline={baseline_auprc:.4f}（流失率），"
                f"提升 {np.mean(values) / baseline_auprc:.2f}x；>2x 顯示模型有效"
            )
        rows.append({
            "指標": metric,
            "平均": f"{np.mean(values):.4f}",
            "SD": f"{np.std(values):.4f}",
            "判斷標準 / 備註": note,
        })
    return pd.DataFrame(rows)
