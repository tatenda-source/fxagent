import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from config import ALL_PAIRS, PAIR_DISPLAY, MAX_PORTFOLIO_RISK, CORRELATION_THRESHOLD
from data.storage import Storage
from data.fetcher import ForexFetcher
from risk.portfolio import PortfolioRiskManager


def render():
    st.title("Portfolio Risk Management")

    storage = Storage()
    portfolio_mgr = PortfolioRiskManager()

    # --- Current Portfolio Status ---
    st.subheader("Current Exposure")
    status = portfolio_mgr.get_current_portfolio_risk()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Open Positions", status["open_positions"])
    col2.metric("Total Risk", f"{status['total_risk_pct'] * 100:.1f}%")
    col3.metric("Risk Limit", f"{MAX_PORTFOLIO_RISK * 100:.0f}%")
    col4.metric(
        "Risk Available",
        f"{status['risk_available'] * 100:.1f}%",
        delta=f"{(status['risk_available'] - MAX_PORTFOLIO_RISK) * 100:.1f}%" if status['total_risk_pct'] > 0 else None,
    )

    # Risk gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=status["total_risk_pct"] * 100,
        title={"text": "Portfolio Risk (%)"},
        delta={"reference": MAX_PORTFOLIO_RISK * 100},
        gauge={
            "axis": {"range": [0, MAX_PORTFOLIO_RISK * 100 * 1.5]},
            "bar": {"color": "#06d6a0"},
            "steps": [
                {"range": [0, MAX_PORTFOLIO_RISK * 100 * 0.5], "color": "#2d6a4f"},
                {"range": [MAX_PORTFOLIO_RISK * 100 * 0.5, MAX_PORTFOLIO_RISK * 100], "color": "#ffd166"},
                {"range": [MAX_PORTFOLIO_RISK * 100, MAX_PORTFOLIO_RISK * 100 * 1.5], "color": "#9d0208"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": MAX_PORTFOLIO_RISK * 100,
            },
        },
    ))
    fig_gauge.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Risk breakdown by pair
    if status["risk_by_pair"]:
        st.subheader("Risk by Position")
        risk_df = pd.DataFrame([
            {"Pair": PAIR_DISPLAY.get(p, p), "Risk %": r * 100}
            for p, r in status["risk_by_pair"].items()
        ])
        fig_bar = px.bar(
            risk_df, x="Pair", y="Risk %",
            color="Risk %",
            color_continuous_scale=["#2d6a4f", "#ffd166", "#9d0208"],
        )
        fig_bar.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # --- Correlation Matrix ---
    st.subheader("Cross-Instrument Correlation")
    st.caption(f"Pairs with |correlation| > {CORRELATION_THRESHOLD} are considered correlated and limited to avoid cluster risk.")

    # Fetch data to compute correlations
    fetcher = ForexFetcher()
    ohlcv = {}
    for pair in ALL_PAIRS:
        df = storage.get_ohlcv(pair)
        if df.empty:
            try:
                df = fetcher.fetch_historical(pair, period="6mo", interval="1d")
            except Exception:
                pass
        if not df.empty:
            ohlcv[pair] = df

    if len(ohlcv) >= 2:
        corr = portfolio_mgr.compute_correlation_matrix(ohlcv)
        if not corr.empty:
            # Rename columns/rows for display
            display_names = {p: PAIR_DISPLAY.get(p, p) for p in corr.columns}
            corr_display = corr.rename(columns=display_names, index=display_names)

            fig_heatmap = go.Figure(data=go.Heatmap(
                z=corr_display.values,
                x=corr_display.columns.tolist(),
                y=corr_display.index.tolist(),
                colorscale="RdBu_r",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" for v in row] for row in corr_display.values],
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
            fig_heatmap.update_layout(
                template="plotly_dark",
                height=500,
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)

            # Correlation clusters
            clusters = portfolio_mgr.get_correlation_clusters()
            if clusters:
                st.subheader("Correlation Clusters")
                st.caption("Positions within the same cluster are limited to avoid concentrated risk.")
                for i, cluster in enumerate(clusters):
                    names = [PAIR_DISPLAY.get(p, p) for p in cluster]
                    st.info(f"Cluster {i + 1}: {', '.join(names)}")
    else:
        st.info("Not enough data to compute correlations. Run the pipeline first.")
