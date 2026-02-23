import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="momoFX - Forex AI Trading",
    page_icon="$",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("momoFX")
st.sidebar.caption("Forex AI Trading System")

page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Pair Detail", "Active Signals", "Portfolio Risk",
     "Market Regimes", "Backtesting", "Agent Logs"],
)

if page == "Overview":
    from dashboard.pages.overview import render
    render()
elif page == "Pair Detail":
    from dashboard.pages.pair_detail import render
    render()
elif page == "Active Signals":
    from dashboard.pages.signals import render
    render()
elif page == "Portfolio Risk":
    from dashboard.pages.portfolio import render
    render()
elif page == "Market Regimes":
    from dashboard.pages.regimes import render
    render()
elif page == "Backtesting":
    from dashboard.pages.backtest import render
    render()
elif page == "Agent Logs":
    from dashboard.pages.logs import render
    render()
