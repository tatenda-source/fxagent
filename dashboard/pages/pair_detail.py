import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import ALL_PAIRS, PAIR_DISPLAY
from data.storage import Storage
from indicators.technical import add_all_indicators


def render():
    st.title("Pair Detail")

    pair = st.selectbox(
        "Select Currency Pair",
        ALL_PAIRS,
        format_func=lambda x: PAIR_DISPLAY.get(x, x),
    )

    storage = Storage()
    df = storage.get_ohlcv(pair)

    if df.empty:
        st.warning(f"No data for {PAIR_DISPLAY.get(pair, pair)}. Run the pipeline first.")
        return

    df = add_all_indicators(df)
    df.dropna(inplace=True)

    if df.empty:
        st.warning("Not enough data for indicators.")
        return

    # Timeframe filter
    period = st.select_slider("Show last N days", options=[30, 60, 90, 180, 365, 730], value=180)
    df = df.tail(period)

    # Main chart: Candlestick + overlays
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[
            PAIR_DISPLAY.get(pair, pair),
            "RSI",
            "MACD",
        ],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
        ),
        row=1, col=1,
    )

    # SMA overlays
    for sma_col, color in [("SMA_20", "#ffd166"), ("SMA_50", "#06d6a0"), ("SMA_200", "#ef476f")]:
        if sma_col in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[sma_col], name=sma_col,
                           line=dict(width=1, color=color)),
                row=1, col=1,
            )

    # Bollinger Bands
    if "BB_upper" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper",
                       line=dict(width=1, dash="dash", color="rgba(255,255,255,0.3)")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower",
                       line=dict(width=1, dash="dash", color="rgba(255,255,255,0.3)"),
                       fill="tonexty", fillcolor="rgba(255,255,255,0.05)"),
            row=1, col=1,
        )

    # RSI
    fig.add_trace(
        go.Scatter(x=df.index, y=df["RSI"], name="RSI", line=dict(color="#118ab2")),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#06d6a0")),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df["MACD_signal"], name="Signal", line=dict(color="#ef476f")),
        row=3, col=1,
    )
    colors = ["#2d6a4f" if v >= 0 else "#9d0208" for v in df["MACD_histogram"]]
    fig.add_trace(
        go.Bar(x=df.index, y=df["MACD_histogram"], name="Histogram",
               marker_color=colors),
        row=3, col=1,
    )

    fig.update_layout(
        height=900,
        template="plotly_dark",
        showlegend=True,
        xaxis_rangeslider_visible=False,
    )

    st.plotly_chart(fig, use_container_width=True)

    # ML Prediction box
    st.subheader("ML Prediction")
    predictions = storage.get_predictions(pair=pair, limit=1)
    if not predictions.empty:
        pred = predictions.iloc[0]
        current = df["Close"].iloc[-1]
        predicted = pred["predicted_price"]
        direction = "UP" if predicted > current else "DOWN"
        change = ((predicted - current) / current) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"{current:.5f}")
        col2.metric("Predicted Price", f"{predicted:.5f}", delta=f"{change:+.2f}%")
        col3.metric("Direction", direction)
    else:
        st.info("No predictions available. Run the full pipeline to generate predictions.")

    # Latest indicator values
    st.subheader("Current Indicator Values")
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("RSI", f"{latest['RSI']:.1f}")
    col2.metric("MACD", f"{latest['MACD']:.6f}")
    col3.metric("ATR", f"{latest['ATR']:.6f}")
    col4.metric("SMA 50", f"{latest['SMA_50']:.5f}")
