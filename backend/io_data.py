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


def load_model_bytes(data: bytes):
    """Deserialise a joblib-pickled model from bytes."""
    return joblib.load(io.BytesIO(data))
