import streamlit as st
import plotly.graph_objects as go

from config import ALL_PAIRS, PAIR_DISPLAY
from data.storage import Storage
from indicators.technical import add_all_indicators
from backtesting.engine import BacktestEngine


def render():
    st.title("Backtesting")

    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        pair = st.selectbox(
            "Currency Pair",
            ALL_PAIRS,
            format_func=lambda x: PAIR_DISPLAY.get(x, x),
        )
    with col2:
        initial_balance = st.number_input("Initial Balance ($)", value=10000, step=1000)

    col3, col4, col5 = st.columns(3)
    with col3:
        rsi_oversold = st.slider("RSI Oversold", 10, 40, 30)
    with col4:
        rsi_overbought = st.slider("RSI Overbought", 60, 90, 70)
    with col5:
        min_score = st.slider("Min Signal Score", 1.0, 4.0, 2.0, 0.5)

    if st.button("Run Backtest", type="primary"):
        storage = Storage()
        df = storage.get_ohlcv(pair)

        if df.empty or len(df) < 250:
            st.error("Not enough data. Run the pipeline first to fetch historical data.")
            return

        df = add_all_indicators(df)
        df.dropna(inplace=True)

        if len(df) < 50:
            st.error("Not enough data after applying indicators.")
            return

        def strategy(row, prev):
            score = 0
            direction = None

            # RSI
            if row["RSI"] < rsi_oversold:
                score += 1.0
                direction = "BUY"
            elif row["RSI"] > rsi_overbought:
                score += 1.0
                direction = "SELL"

            if direction is None:
                return None

            # MACD confirmation
            if direction == "BUY" and row["MACD"] > row["MACD_signal"]:
                score += 0.8
            elif direction == "SELL" and row["MACD"] < row["MACD_signal"]:
                score += 0.8

            # Bollinger confirmation
            if direction == "BUY" and row["Close"] < row["BB_lower"]:
                score += 0.7
            elif direction == "SELL" and row["Close"] > row["BB_upper"]:
                score += 0.7

            # SMA trend
            if direction == "BUY" and row["Close"] > row["SMA_50"]:
                score += 0.5
            elif direction == "SELL" and row["Close"] < row["SMA_50"]:
                score += 0.5

            if score < min_score:
                return None

            return {"action": direction, "atr": row["ATR"]}

        engine = BacktestEngine(initial_balance=initial_balance)
        with st.spinner("Running backtest..."):
            results = engine.run(df, strategy)

        # Display results
        metrics = results["metrics"]

        st.subheader("Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{metrics['total_return']}%")
        col2.metric("Win Rate", f"{metrics['win_rate']}%")
        col3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']}")
        col4.metric("Max Drawdown", f"{metrics['max_drawdown']}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Trades", metrics["num_trades"])
        col6.metric("Profit Factor", metrics["profit_factor"])
        col7.metric("Avg Win", f"{metrics['avg_win']:.4f}")
        col8.metric("Avg Loss", f"{metrics['avg_loss']:.4f}")

        # Equity curve
        st.subheader("Equity Curve")
        equity = results["equity_curve"]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=equity.index, y=equity.values, name="Equity",
                       fill="tozeroy", fillcolor="rgba(6,214,160,0.2)",
                       line=dict(color="#06d6a0"))
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            yaxis_title="Balance ($)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Trade log
        if results["trades"]:
            st.subheader("Trade Log")
            import pandas as pd
            trades_df = pd.DataFrame(results["trades"])
            st.dataframe(trades_df, use_container_width=True)
