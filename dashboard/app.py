"""Streamlit dashboard for the Chatbot Analytics system."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import (  # noqa: E402
    compute_overview_metrics,
    get_recent_experiments,
    load_experiments,
    load_dataset,
    compute_intent_distribution,
    compute_flow_summary,
    compute_sentiment_trend,
    compute_sentiment_summary,
)
from src.models.core import DatasetType  # noqa: E402


st.set_page_config(
    page_title="Chatbot Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_overview():
    st.title("Chatbot Analytics Overview")
    st.write("Welcome! Use the sidebar to explore experiments, datasets, and system status.")

    metrics = compute_overview_metrics()
    col1, col2, col3 = st.columns(3)
    col1.metric("Experiments Logged", metrics["experiment_count"])
    col2.metric("Models Tracked", metrics["model_count"])
    col3.metric("Successful Runs", metrics["successful_runs"])

    latest_accuracy = metrics.get("latest_validation_accuracy")
    if latest_accuracy is not None:
        st.metric("Most Recent Validation Accuracy", f"{latest_accuracy:.2%}")

    st.subheader("Recent Experiments")
    recent_experiments = get_recent_experiments()
    if recent_experiments:
        st.dataframe(recent_experiments)
    else:
        st.info("No experiments logged yet. Trigger a training run to populate data.")


def render_experiments():
    experiments = load_experiments()
    st.title("Experiments")

    if not experiments:
        st.info("No experiments recorded.")
        return

    model_ids = sorted({exp.get("model_id") for exp in experiments})
    selected_model = st.selectbox("Filter by Model", options=["All"] + model_ids)

    filtered = (
        experiments
        if selected_model == "All"
        else [exp for exp in experiments if exp.get("model_id") == selected_model]
    )

    st.write(f"Displaying {len(filtered)} experiment(s).")
    st.dataframe(filtered)


def render_intent_distribution():
    st.title("Intent Distribution")
    dataset_type = dataset_selector()

    try:
        dataset = load_dataset(dataset_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    distribution = compute_intent_distribution(dataset)
    if not distribution:
        st.warning("No intent data available for the selected dataset.")
        return

    df = pd.DataFrame(distribution.items(), columns=["intent", "count"]).set_index("intent")
    st.bar_chart(df)
    st.caption(f"Total intents: {len(distribution)} | Conversations: {dataset.size}")


def render_conversation_flow():
    st.title("Conversation Flow")
    dataset_type = dataset_selector()

    try:
        dataset = load_dataset(dataset_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    flow_summary = compute_flow_summary(dataset)

    stats = flow_summary.get("turn_statistics", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Turns", stats.get("average_turns", 0))
    col2.metric("Median Turns", stats.get("median_turns", 0))
    col3.metric("Max Turns", stats.get("max_turns", 0))

    state_distribution = flow_summary.get("state_distribution", {})
    if state_distribution:
        st.subheader("State Distribution")
        df_states = pd.DataFrame(state_distribution.items(), columns=["state", "count"]).set_index("state")
        st.bar_chart(df_states)

    st.subheader("Speaker Transition Matrix")
    transitions = flow_summary.get("transition_matrix", {})
    if transitions:
        df_transitions = pd.DataFrame(
            [
                {"transition": key, "count": value}
                for key, value in transitions.items()
            ]
        )
        st.dataframe(df_transitions)
    else:
        st.info("No transition data available.")


def render_sentiment_trends():
    st.title("Sentiment Trends")
    dataset_type = dataset_selector()
    granularity = st.selectbox("Granularity", options=["hourly", "daily", "conversation"], index=1)

    try:
        dataset = load_dataset(dataset_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    trend = compute_sentiment_trend(dataset, granularity=granularity)
    trend_points = trend.get("trend", [])
    if not trend_points:
        st.warning("No sentiment data available.")
        return

    trend_df = pd.DataFrame(trend_points)
    trend_df = trend_df.rename(columns={"bucket": "Bucket", "average_sentiment": "Sentiment"}).set_index("Bucket")
    st.line_chart(trend_df["Sentiment"])

    st.subheader("Sentiment Summary")
    summary = compute_sentiment_summary(dataset)
    if summary:
        st.json(summary)


def dataset_selector(label: str = "Dataset") -> DatasetType:
    dataset_options = sorted(list(DatasetType), key=lambda dt: dt.value)
    return st.selectbox(label, dataset_options, format_func=lambda dt: dt.value.replace("_", " ").title())


def render_settings():
    st.title("Settings & Help")
    st.write("Configure upcoming features and review documentation links.")
    st.markdown(
        "- **API Docs**: Coming soon\n"
        "- **Model Registry**: Managed via `models/registry`\n"
        "- **Experiment Log**: Stored under `experiments/experiments.json`"
    )


PAGES = {
    "Overview": render_overview,
    "Experiments": render_experiments,
    "Intent Distribution": render_intent_distribution,
    "Conversation Flow": render_conversation_flow,
    "Sentiment Trends": render_sentiment_trends,
    "Settings": render_settings,
}


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if st.sidebar.button("Refresh Data"):
    load_experiments.cache_clear()  # type: ignore[attr-defined]
    load_dataset.cache_clear()  # type: ignore[attr-defined]
    st.experimental_rerun()

renderer = PAGES[selection]
renderer()
