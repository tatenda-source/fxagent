import streamlit as st
import pandas as pd

from config import ALL_PAIRS, PAIR_DISPLAY
from data.storage import Storage
from data.fetcher import ForexFetcher
from indicators.technical import add_all_indicators


def render():
    st.title("Market Overview")

    storage = Storage()
    fetcher = ForexFetcher()

    # Pair summary cards (rows of 6)
    per_row = 6
    for row_start in range(0, len(ALL_PAIRS), per_row):
        row_pairs = ALL_PAIRS[row_start:row_start + per_row]
        cols = st.columns(per_row)
        for i, pair in enumerate(row_pairs):
            with cols[i]:
            df = storage.get_ohlcv(pair)
            if df.empty:
                try:
                    df = fetcher.fetch_historical(pair, period="5d", interval="1d")
                except Exception:
                    pass

            if not df.empty:
                current = df["Close"].iloc[-1]
                prev = df["Close"].iloc[-2] if len(df) > 1 else current
                change = ((current - prev) / prev) * 100
                st.metric(
                    label=PAIR_DISPLAY.get(pair, pair),
                    value=f"{current:.4f}",
                    delta=f"{change:+.2f}%",
                )
            else:
                st.metric(label=PAIR_DISPLAY.get(pair, pair), value="N/A")

    st.divider()

    # Indicator heatmap
    st.subheader("Indicator Signals")
    heatmap_data = []

    for pair in ALL_PAIRS:
        df = storage.get_ohlcv(pair)
        if df.empty or len(df) < 200:
            continue

        df = add_all_indicators(df)
        df.dropna(inplace=True)
        if df.empty:
            continue

        latest = df.iloc[-1]
        row = {"Pair": PAIR_DISPLAY.get(pair, pair)}

        # RSI signal
        if latest["RSI"] < 30:
            row["RSI"] = "Oversold"
        elif latest["RSI"] > 70:
            row["RSI"] = "Overbought"
        else:
            row["RSI"] = "Neutral"

        # MACD signal
        if latest["MACD"] > latest["MACD_signal"]:
            row["MACD"] = "Bullish"
        else:
            row["MACD"] = "Bearish"

        # Bollinger signal
        if latest["Close"] < latest["BB_lower"]:
            row["Bollinger"] = "Below Lower"
        elif latest["Close"] > latest["BB_upper"]:
            row["Bollinger"] = "Above Upper"
        else:
            row["Bollinger"] = "Within Bands"

        # SMA trend
        if latest["Close"] > latest["SMA_50"]:
            row["SMA 50"] = "Above (Bullish)"
        else:
            row["SMA 50"] = "Below (Bearish)"

        heatmap_data.append(row)

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df.set_index("Pair", inplace=True)

        def color_signal(val):
            bullish = ["Oversold", "Bullish", "Below Lower", "Above (Bullish)"]
            bearish = ["Overbought", "Bearish", "Above Upper", "Below (Bearish)"]
            if val in bullish:
                return "background-color: #2d6a4f; color: white"
            elif val in bearish:
                return "background-color: #9d0208; color: white"
            return "background-color: #495057; color: white"

        st.dataframe(
            heatmap_df.style.map(color_signal),
            use_container_width=True,
        )
    else:
        st.info("No data available yet. Run the pipeline first to populate data.")

    st.divider()

    # Recent signals
    st.subheader("Recent Signals")
    signals = storage.get_all_signals(limit=20)
    if not signals.empty:
        display_cols = ["pair", "signal_type", "confidence", "entry_price", "stop_loss", "take_profit", "status", "pnl"]
        available = [c for c in display_cols if c in signals.columns]
        st.dataframe(signals[available], use_container_width=True)
    else:
        st.info("No signals generated yet.")
