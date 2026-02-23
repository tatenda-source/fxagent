import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from data.storage import Storage


def render():
    st.title("Agent Logs & Feedback")

    storage = Storage()

    # Prediction accuracy over time
    st.subheader("Prediction Accuracy")
    predictions = storage.get_predictions(limit=500)
    if not predictions.empty and "actual_price" in predictions.columns:
        valid = predictions.dropna(subset=["actual_price", "predicted_price"])
        if not valid.empty:
            valid["timestamp"] = pd.to_datetime(valid["timestamp"])
            valid["error_pct"] = abs(valid["predicted_price"] - valid["actual_price"]) / valid["actual_price"] * 100
            valid = valid.sort_values("timestamp")

            fig = go.Figure()
            for pair_name in valid["pair"].unique():
                pair_data = valid[valid["pair"] == pair_name]
                fig.add_trace(
                    go.Scatter(x=pair_data["timestamp"], y=pair_data["error_pct"],
                               name=pair_name, mode="lines+markers")
                )
            fig.update_layout(
                template="plotly_dark",
                height=400,
                yaxis_title="Error %",
                xaxis_title="Time",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Predictions exist but no actual prices recorded yet for comparison.")
    else:
        st.info("No prediction data available yet.")

    st.divider()

    # Agent activity logs
    st.subheader("Agent Activity")
    logs = storage.get_agent_logs(limit=200)
    if not logs.empty:
        # Filter
        agents = ["All"] + list(logs["agent_name"].unique())
        selected_agent = st.selectbox("Filter by Agent", agents)

        if selected_agent != "All":
            logs = logs[logs["agent_name"] == selected_agent]

        st.dataframe(logs, use_container_width=True)
    else:
        st.info("No agent logs recorded yet.")

    st.divider()

    # Signal outcomes summary
    st.subheader("Signal Outcomes")
    signals = storage.get_all_signals(limit=200)
    if not signals.empty:
        closed = signals[signals["status"].isin(["TP_HIT", "SL_HIT"])]
        if not closed.empty:
            outcome_counts = closed["status"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=outcome_counts.index.tolist(),
                        values=outcome_counts.values.tolist(),
                        marker_colors=["#2d6a4f", "#9d0208"],
                    )
                ])
                fig.update_layout(template="plotly_dark", height=300, title="Win/Loss Distribution")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("Total Closed", len(closed))
                st.metric("TP Hit", int(outcome_counts.get("TP_HIT", 0)))
                st.metric("SL Hit", int(outcome_counts.get("SL_HIT", 0)))
                st.metric("Total P&L", f"{closed['pnl'].sum():.4f}")
        else:
            st.info("No closed signals yet.")
    else:
        st.info("No signals recorded yet.")
