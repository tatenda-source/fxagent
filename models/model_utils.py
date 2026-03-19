import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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

    Now predicts LOG RETURNS (stationary) instead of raw price.
    Scaler is fit ONLY on training data to prevent data leakage.

    Returns: (X_train, X_test, y_train, y_test, scaler, meta)
    Shapes: X=(samples, seq_length, features), y=(samples,)
    meta contains last_close for converting returns back to price.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values
    close_prices = df["Close"].values

    # Compute log return targets: log(close[t+1] / close[t])
    log_returns = np.log(close_prices[1:] / close_prices[:-1])

    # Split chronologically FIRST, then scale
    split_row = int(len(data) * TRAIN_TEST_SPLIT)
    train_data = data[:split_row]
    test_data = data[split_row:]

    # Use RobustScaler — less sensitive to outliers than MinMax
    scaler = RobustScaler()
    scaler.fit(train_data)

    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)

    # Build training sequences — target is log return (next day)
    X_train, y_train = [], []
    for i in range(seq_length, len(scaled_train) - 1):
        X_train.append(scaled_train[i - seq_length: i])
        y_train.append(log_returns[i])  # log return from day i to i+1

    # Build test sequences — bridge with tail of training data
    combined_for_test = np.concatenate([
        scaled_train[-seq_length:],
        scaled_test,
    ], axis=0)

    # Log returns for the test portion
    test_start_idx = split_row
    X_test, y_test = [], []
    for i in range(seq_length, len(combined_for_test) - 1):
        X_test.append(combined_for_test[i - seq_length: i])
        # Absolute index in original log_returns array
        abs_idx = test_start_idx - seq_length + i
        if abs_idx < len(log_returns):
            y_test.append(log_returns[abs_idx])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test[:len(y_test)])
    y_test = np.array(y_test)

    meta = {
        "last_close": float(close_prices[-1]),
        "close_prices": close_prices,
        "log_returns": log_returns,
    }

    return X_train, X_test, y_train, y_test, scaler, meta


def prepare_sequences_expanding_cv(
    df: pd.DataFrame,
    feature_cols: list = None,
    seq_length: int = SEQUENCE_LENGTH,
    n_splits: int = 3,
    min_train_pct: float = 0.5,
) -> list:
    """
    Expanding window cross-validation splits.

    Each fold uses a growing training window with a fixed-size test window.
    Returns list of (X_train, X_test, y_train, y_test, scaler, meta) tuples.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    available = [c for c in feature_cols if c in df.columns]
    data = df[available].values
    close_prices = df["Close"].values
    log_returns = np.log(close_prices[1:] / close_prices[:-1])

    n = len(data)
    min_train = int(n * min_train_pct)
    remaining = n - min_train
    test_size = remaining // (n_splits + 1)

    splits = []
    for fold in range(n_splits):
        train_end = min_train + fold * test_size
        test_end = min(train_end + test_size, n)

        train_data = data[:train_end]
        test_data = data[train_end:test_end]

        scaler = RobustScaler()
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)
        scaled_test = scaler.transform(test_data)

        X_tr, y_tr = [], []
        for i in range(seq_length, len(scaled_train) - 1):
            X_tr.append(scaled_train[i - seq_length: i])
            y_tr.append(log_returns[i])

        combined = np.concatenate([scaled_train[-seq_length:], scaled_test], axis=0)
        X_te, y_te = [], []
        for i in range(seq_length, len(combined) - 1):
            X_te.append(combined[i - seq_length: i])
            abs_idx = train_end - seq_length + i
            if abs_idx < len(log_returns):
                y_te.append(log_returns[abs_idx])

        if len(X_tr) > 10 and len(y_te) > 0:
            splits.append((
                np.array(X_tr), np.array(X_te[:len(y_te)]),
                np.array(y_tr), np.array(y_te),
                scaler,
                {"last_close": float(close_prices[test_end - 1]),
                 "close_prices": close_prices[:test_end],
                 "log_returns": log_returns[:test_end - 1]},
            ))

    return splits


def log_return_to_price(log_return: float, current_price: float) -> float:
    """Convert a predicted log return back to an absolute price."""
    return current_price * np.exp(log_return)


def inverse_scale_close(scaled_value: float, scaler, num_features: int) -> float:
    """Inverse transform a single scaled Close value back to real price.
    Kept for backward compatibility."""
    dummy = np.zeros((1, num_features))
    dummy[0, 0] = scaled_value
    inv = scaler.inverse_transform(dummy)
    return float(inv[0, 0])
