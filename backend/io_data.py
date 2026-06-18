"""Read CSV bytes or a joblib-pickled model from raw bytes."""
import io

import joblib
import numpy as np
import pandas as pd

from backend.theme import TARGET_DEFAULT


def load_csv_bytes(data: bytes, target: str = TARGET_DEFAULT):
    """Parse CSV bytes; return (full_df, feature_df X, label_array y)."""
    df = pd.read_csv(io.BytesIO(data))
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    y = df[target].astype(int).values
    return df, X, y


def resolve_estimator(model):
    """Return the fitted LightGBM estimator from a saved object."""
    if hasattr(model, "booster_") or (
        hasattr(model, "predict_proba") and not hasattr(model, "model")
    ):
        return model

    # FLAML AutoML wrapper (e.g. notebook or older exports)
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "estimator"):
        return inner.estimator

    raise ValueError(
        "無法辨識的模型格式。請上傳本工具下載的 .pkl，"
        "或 Colab 中 automl.model.estimator 儲存的 LightGBM 模型。"
    )


def _reject_invalid_pkl_bytes(data: bytes) -> None:
    """Raise a clear error when the upload is not a real joblib pickle."""
    if len(data) < 128:
        preview = data.decode("utf-8", errors="replace").strip()
        if preview:
            raise ValueError(
                f"上傳的 .pkl 不是有效模型檔（內容：{preview[:120]}）。"
                "若檔案來自本工具下載，請確認訓練已完成後再下載；"
                "或改用 notebook 中 joblib.dump(model, ...) 儲存的檔案。"
            )
        raise ValueError("上傳的 .pkl 檔案過小或為空，請重新選擇模型檔。")


def load_model_bytes(data: bytes):
    """Deserialise a joblib-pickled model from bytes."""
    _reject_invalid_pkl_bytes(data)
    try:
        return resolve_estimator(joblib.load(io.BytesIO(data)))
    except Exception as e:
        raise ValueError(
            f"無法讀取 .pkl 模型：{e}。"
            "請確認檔案為 joblib 儲存的 LightGBM 模型（automl.model.estimator）。"
        ) from e
