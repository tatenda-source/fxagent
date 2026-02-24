import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SEQUENCE_LENGTH, TRAIN_TEST_SPLIT

FEATURE_COLS = [
    # Price
    "Close",
    # Trend (existing)
    "SMA_20", "SMA_50", "SMA_200", "EMA_12", "EMA_26",
    # Momentum (existing)
    "RSI", "MACD", "MACD_signal", "MACD_histogram",
    # Volatility (existing)
    "BB_upper", "BB_middle", "BB_lower", "ATR",
    # Momentum (new)
    "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI", "ROC",
    # Volume (new)
    "OBV", "MFI",
    # Price returns & volatility (new)
    "LOG_RETURN_1", "LOG_RETURN_5", "VOLATILITY_20", "ATR_RATIO",
    # Time features (new)
    "DAY_SIN", "DAY_COS", "MONTH_SIN", "MONTH_COS",
]


def prepare_sequences(
    df: pd.DataFrame,
    target_col: str = "Close",
    feature_cols: list = None,
    seq_length: int = SEQUENCE_LENGTH,
) -> tuple:
    """
    Convert DataFrame into LSTM-ready sequences.

    Scaler is fit ONLY on training data to prevent data leakage.

    Returns: (X_train, X_test, y_train, y_test, scaler)
    Shapes: X=(samples, seq_length, features), y=(samples,)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values

    # Split chronologically FIRST, then scale
    split_row = int(len(data) * TRAIN_TEST_SPLIT)
    train_data = data[:split_row]
    test_data = data[split_row:]

    # Fit scaler on TRAINING DATA ONLY
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    target_idx = available.index(target_col)

    # Build training sequences
    X_train, y_train = [], []
    for i in range(seq_length, len(scaled_train)):
        X_train.append(scaled_train[i - seq_length: i])
        y_train.append(scaled_train[i, target_idx])

    # Build test sequences — bridge with tail of training data
    combined_for_test = np.concatenate([
        scaled_train[-seq_length:],
        scaled_test,
    ], axis=0)

    X_test, y_test = [], []
    for i in range(seq_length, len(combined_for_test)):
        X_test.append(combined_for_test[i - seq_length: i])
        y_test.append(combined_for_test[i, target_idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test, scaler


def inverse_scale_close(scaled_value: float, scaler: MinMaxScaler, num_features: int) -> float:
    """Inverse transform a single scaled Close value back to real price."""
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = scaled_value  # Close is index 0 in feature_cols
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, 0])
