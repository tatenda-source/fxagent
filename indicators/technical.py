import pandas as pd
import ta

from config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD,
)


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a DataFrame with Open/High/Low/Close columns."""
    df = df.copy()

    # SMA
    for period in SMA_PERIODS:
        df[f"SMA_{period}"] = ta.trend.sma_indicator(df["Close"], window=period)

    # EMA
    for period in EMA_PERIODS:
        df[f"EMA_{period}"] = ta.trend.ema_indicator(df["Close"], window=period)

    # RSI
    df["RSI"] = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)

    # MACD
    macd = ta.trend.MACD(df["Close"], window_slow=MACD_SLOW, window_fast=MACD_FAST, window_sign=MACD_SIGNAL)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_histogram"] = macd.macd_diff()

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"], window=BOLLINGER_PERIOD, window_dev=BOLLINGER_STD)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()

    # ATR
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=ATR_PERIOD)

    return df
