import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from config import ALL_PAIRS, PAIR_DISPLAY, REGIME_ADX_TRENDING
from data.storage import Storage
from data.fetcher import ForexFetcher
from risk.regime import MarketRegime


def render():
    st.title("Market Regimes")
    st.caption("Real-time regime detection: is the market trending, ranging, or volatile?")

    storage = Storage()
    fetcher = ForexFetcher()

    # Detect regimes for all pairs
    regimes_data = []
    for pair in ALL_PAIRS:
        df = storage.get_ohlcv(pair)
        if df.empty:
            try:
                df = fetcher.fetch_historical(pair, period="1y", interval="1d")
            except Exception:
                continue
        if df.empty or len(df) < 50:
            continue

        regime = MarketRegime.detect(df)
        regime["pair"] = pair
        regime["pair_display"] = PAIR_DISPLAY.get(pair, pair)
        regimes_data.append(regime)

    if not regimes_data:
        st.warning("No data available. Run the pipeline first.")
        return

    # --- Regime Summary Cards ---
    trending = [r for r in regimes_data if r["regime"] == MarketRegime.TRENDING]
    ranging = [r for r in regimes_data if r["regime"] == MarketRegime.RANGING]
    volatile = [r for r in regimes_data if r["regime"] == MarketRegime.VOLATILE]

    col1, col2, col3 = st.columns(3)
    col1.metric("Trending", len(trending))
    col2.metric("Ranging", len(ranging))
    col3.metric("Volatile", len(volatile))

    st.divider()

    # --- Regime Table ---
    st.subheader("Regime by Instrument")

    table_data = []
    for r in regimes_data:
        adjustments = r.get("strategy_adjustments", {})
        table_data.append({
            "Pair": r["pair_display"],
            "Regime": r["regime"].upper(),
            "ADX": round(r["adx"], 1),
            "Trend": r["trend_direction"],
            "Volatility": r["volatility_state"],
            "Vol Ratio": round(r["volatility_ratio"], 2),
            "Confidence": f"{r['confidence']:.0%}",
            "SL Adj": f"{adjustments.get('sl_multiplier_adj', 1.0):.1f}x",
            "TP Adj": f"{adjustments.get('tp_multiplier_adj', 1.0):.1f}x",
            "Size Adj": f"{adjustments.get('position_size_adj', 1.0):.1f}x",
        })

    df_table = pd.DataFrame(table_data)

    def color_regime(val):
        if val == "TRENDING":
            return "background-color: #2d6a4f; color: white"
        elif val == "VOLATILE":
            return "background-color: #9d0208; color: white"
        elif val == "RANGING":
            return "background-color: #495057; color: white"
        return ""

    def color_volatility(val):
        if val == "extreme":
            return "background-color: #9d0208; color: white"
        elif val == "high":
            return "background-color: #e85d04; color: white"
        return ""

    styled = df_table.style.map(color_regime, subset=["Regime"]).map(
        color_volatility, subset=["Volatility"]
    )
    st.dataframe(styled, use_container_width=True)

    st.divider()

    # --- ADX Chart ---
    st.subheader("Trend Strength (ADX)")

    adx_data = pd.DataFrame([
        {"Pair": r["pair_display"], "ADX": r["adx"], "Regime": r["regime"]}
        for r in regimes_data
    ]).sort_values("ADX", ascending=True)

    colors = []
    for _, row in adx_data.iterrows():
        if row["Regime"] == MarketRegime.TRENDING:
            colors.append("#06d6a0")
        elif row["Regime"] == MarketRegime.VOLATILE:
            colors.append("#ef476f")
        else:
            colors.append("#ffd166")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=adx_data["Pair"],
        x=adx_data["ADX"],
        orientation="h",
        marker_color=colors,
    ))
    fig.add_vline(
        x=REGIME_ADX_TRENDING, line_dash="dash", line_color="white",
        annotation_text=f"Trending threshold ({REGIME_ADX_TRENDING})",
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="ADX",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Strategy Impact Explanation ---
    st.subheader("How Regimes Affect Trading")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**TRENDING**")
        st.markdown("""
        - TP widened 1.5x (let winners run)
        - Entry threshold lowered
        - Normal position size
        - Trend-following signals boosted
        """)

    with col2:
        st.markdown("**RANGING**")
        st.markdown("""
        - SL tightened 0.8x
        - TP tightened 0.8x (quick profits)
        - Entry threshold slightly raised
        - Size reduced to 80%
        """)

    with col3:
        st.markdown("**VOLATILE**")
        st.markdown("""
        - SL widened 1.5x (avoid noise)
        - Entry threshold raised significantly
        - Position size halved (capital protection)
        - Only high-confidence signals pass
        """)
