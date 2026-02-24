import numpy as np
import pandas as pd
import ta

from config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD,
    STOCH_K_PERIOD, STOCH_D_PERIOD, WILLIAMS_R_PERIOD,
    CCI_PERIOD, ROC_PERIOD, MFI_PERIOD,
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

    # --- Momentum Indicators ---

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        df["High"], df["Low"], df["Close"],
        window=STOCH_K_PERIOD, smooth_window=STOCH_D_PERIOD,
    )
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    # Williams %R
    df["WILLIAMS_R"] = ta.momentum.williams_r(
        df["High"], df["Low"], df["Close"], lbp=WILLIAMS_R_PERIOD,
    )

    # CCI
    df["CCI"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=CCI_PERIOD)

    # Rate of Change
    df["ROC"] = ta.momentum.roc(df["Close"], window=ROC_PERIOD)

    # --- Volume Indicators ---

    if "Volume" in df.columns and df["Volume"].sum() > 0:
        df["OBV"] = ta.volume.on_balance_volume(df["Close"], df["Volume"])
        df["MFI"] = ta.volume.money_flow_index(
            df["High"], df["Low"], df["Close"], df["Volume"], window=MFI_PERIOD,
        )
    else:
        df["OBV"] = 0.0
        df["MFI"] = 50.0

    # --- Price Returns & Volatility ---

    df["LOG_RETURN_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["LOG_RETURN_5"] = np.log(df["Close"] / df["Close"].shift(5))
    df["VOLATILITY_20"] = df["LOG_RETURN_1"].rolling(window=20).std()
    df["ATR_RATIO"] = df["ATR"] / df["ATR"].rolling(window=20).mean()

    # --- Time Features (cyclical encoding) ---

    if hasattr(df.index, "dayofweek"):
        dow = df.index.dayofweek
    else:
        dow = pd.to_datetime(df.index).dayofweek
    df["DAY_SIN"] = np.sin(2 * np.pi * dow / 5)
    df["DAY_COS"] = np.cos(2 * np.pi * dow / 5)

    if hasattr(df.index, "month"):
        month = df.index.month
    else:
        month = pd.to_datetime(df.index).month
    df["MONTH_SIN"] = np.sin(2 * np.pi * month / 12)
    df["MONTH_COS"] = np.cos(2 * np.pi * month / 12)

    return df
