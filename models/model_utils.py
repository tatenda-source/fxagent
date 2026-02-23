import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SEQUENCE_LENGTH, TRAIN_TEST_SPLIT

FEATURE_COLS = [
    "Close", "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_signal", "BB_upper", "BB_lower", "ATR",
]


def prepare_sequences(
    df: pd.DataFrame,
    target_col: str = "Close",
    feature_cols: list = None,
    seq_length: int = SEQUENCE_LENGTH,
) -> tuple:
    """
    Convert DataFrame into LSTM-ready sequences.

    Returns: (X_train, X_test, y_train, y_test, scaler)
    Shapes: X=(samples, seq_length, features), y=(samples,)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)

    target_idx = available.index(target_col)

    X, y = [], []
    for i in range(seq_length, len(scaled)):
        X.append(scaled[i - seq_length: i])
        y.append(scaled[i, target_idx])

    X = np.array(X)
    y = np.array(y)

    split = int(len(X) * TRAIN_TEST_SPLIT)
    return X[:split], X[split:], y[:split], y[split:], scaler


def inverse_scale_close(scaled_value: float, scaler: MinMaxScaler, num_features: int) -> float:
    """Inverse transform a single scaled Close value back to real price."""
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = scaled_value  # Close is index 0 in feature_cols
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, 0])
