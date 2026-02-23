import json

import streamlit as st
import pandas as pd

from config import PAIR_DISPLAY
from data.storage import Storage


def render():
    st.title("Active Signals")

    storage = Storage()

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        status_filter = st.selectbox("Status", ["All", "OPEN", "TP_HIT", "SL_HIT", "EXPIRED"])
    with col2:
        type_filter = st.selectbox("Signal Type", ["All", "BUY", "SELL"])
    with col3:
        limit = st.number_input("Max Rows", min_value=10, max_value=500, value=50)

    signals = storage.get_all_signals(limit=limit)

    if signals.empty:
        st.info("No signals generated yet. Run the full pipeline to generate trading signals.")
        return

    # Apply filters
    if status_filter != "All":
        signals = signals[signals["status"] == status_filter]
    if type_filter != "All":
        signals = signals[signals["signal_type"] == type_filter]

    # Format display
    signals["pair_display"] = signals["pair"].map(lambda x: PAIR_DISPLAY.get(x, x))

    # Parse reasons JSON
    signals["reasons_text"] = signals["reasons"].apply(
        lambda x: ", ".join(json.loads(x)) if x and x != "null" else ""
    )

    display_cols = [
        "pair_display", "signal_type", "confidence", "entry_price",
        "stop_loss", "take_profit", "position_size", "status", "pnl", "reasons_text", "timestamp",
    ]
    available = [c for c in display_cols if c in signals.columns]

    def color_pnl(val):
        try:
            v = float(val)
            if v > 0:
                return "color: #2d6a4f"
            elif v < 0:
                return "color: #9d0208"
        except (ValueError, TypeError):
            pass
        return ""

    def color_signal(val):
        if val == "BUY":
            return "color: #2d6a4f"
        elif val == "SELL":
            return "color: #9d0208"
        return ""

    styled = signals[available].style.map(color_pnl, subset=["pnl"]).map(color_signal, subset=["signal_type"])
    st.dataframe(styled, use_container_width=True)

    # Summary stats
    st.divider()
    st.subheader("Summary")
    col1, col2, col3, col4 = st.columns(4)

    total = len(signals)
    open_count = len(signals[signals["status"] == "OPEN"])
    total_pnl = signals["pnl"].sum()
    win_rate = (
        len(signals[signals["pnl"] > 0]) / len(signals[signals["status"] != "OPEN"])
        if len(signals[signals["status"] != "OPEN"]) > 0 else 0
    )

    col1.metric("Total Signals", total)
    col2.metric("Open", open_count)
    col3.metric("Total P&L", f"{total_pnl:.4f}")
    col4.metric("Win Rate", f"{win_rate:.0%}")
