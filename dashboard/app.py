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

st.sidebar.divider()

if st.sidebar.button("Run Pipeline", type="primary", use_container_width=True):
    from pipeline.orchestrator import Orchestrator
    orchestrator = Orchestrator()
    with st.spinner("Running pipeline — fetching data, analyzing, predicting..."):
        try:
            result = orchestrator.run_full_pipeline()
            signal_count = len(result.get("signals", []))
            st.sidebar.success(f"Pipeline complete. {signal_count} signals generated.")
        except Exception as e:
            st.sidebar.error(f"Pipeline failed: {e}")

st.sidebar.divider()

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
