"""Streamlit dashboard for the Chatbot Analytics system."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import streamlit as st
import time


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
    experiments_to_csv,
    build_experiments_pdf,
    analytics_to_csv,
    analytics_to_pdf,
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
        if latest_accuracy < 0.7:
            st.warning("Recent validation accuracy dipped below 70%. Consider retraining or inspecting data quality.")

    st.subheader("Recent Experiments")
    recent_experiments = get_recent_experiments()
    if recent_experiments:
        st.dataframe(recent_experiments)
        csv_bytes = experiments_to_csv(recent_experiments)
        pdf_bytes = build_experiments_pdf(recent_experiments, title="Recent Experiments")
        col_a, col_b = st.columns(2)
        col_a.download_button("Download CSV", data=csv_bytes, file_name="recent_experiments.csv", mime="text/csv")
        col_b.download_button("Download PDF", data=pdf_bytes, file_name="recent_experiments.pdf", mime="application/pdf")
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

    df = pd.DataFrame(experiments)
    df["created_at"] = pd.to_datetime(df.get("created_at"))
    min_date = df["created_at"].min().date() if not df["created_at"].isna().all() else None
    max_date = df["created_at"].max().date() if not df["created_at"].isna().all() else None

    if min_date and max_date:
        start_date, end_date = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        mask = (df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)
        df = df[mask]
        experiments = df.to_dict(orient="records")

    filtered = (
        experiments
        if selected_model == "All"
        else [exp for exp in experiments if exp.get("model_id") == selected_model]
    )

    st.write(f"Displaying {len(filtered)} experiment(s).")
    st.dataframe(filtered)

    csv_bytes = experiments_to_csv(filtered)
    pdf_bytes = build_experiments_pdf(filtered, title="Experiment Export")
    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV", data=csv_bytes, file_name="experiments.csv", mime="text/csv")
    col_b.download_button("Download PDF", data=pdf_bytes, file_name="experiments.pdf", mime="application/pdf")


def render_intent_distribution():
    st.title("Intent Distribution")
    dataset_type = dataset_selector()
    top_n = st.slider("Top intents to display", min_value=5, max_value=50, value=20)

    try:
        dataset = load_dataset(dataset_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    distribution = compute_intent_distribution(dataset)
    if not distribution:
        st.warning("No intent data available for the selected dataset.")
        return

    df = pd.DataFrame(distribution.items(), columns=["intent", "count"]).head(top_n).set_index("intent")
    st.bar_chart(df)
    st.caption(f"Total intents: {len(distribution)} | Conversations: {dataset.size}")

    csv_bytes = analytics_to_csv({"Intent Distribution": df.reset_index().to_dict(orient="records")})
    pdf_bytes = analytics_to_pdf(
        title=f"Intent Distribution – {dataset_type.value}",
        metrics={"Total Intents": len(distribution), "Conversations": dataset.size},
        tables={"Top Intents": df.reset_index().to_dict(orient="records")},
    )
    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV", data=csv_bytes, file_name="intent_distribution.csv", mime="text/csv")
    col_b.download_button("Download PDF", data=pdf_bytes, file_name="intent_distribution.pdf", mime="application/pdf")


def render_conversation_flow():
    st.title("Conversation Flow")
    dataset_type = dataset_selector()
    sample_size = st.slider("Sample conversations", min_value=50, max_value=1000, step=50, value=200)

    try:
        dataset = load_dataset(dataset_type)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    conversation_ids = [conv.id for conv in dataset.conversations[:sample_size]]
    flow_summary = compute_flow_summary(dataset, conversation_ids=conversation_ids)

    stats = flow_summary.get("turn_statistics", {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Turns", stats.get("average_turns", 0))
    col2.metric("Median Turns", stats.get("median_turns", 0))
    col3.metric("Max Turns", stats.get("max_turns", 0))

    state_distribution = flow_summary.get("state_distribution", {})
    df_states = None
    if state_distribution:
        st.subheader("State Distribution")
        df_states = pd.DataFrame(state_distribution.items(), columns=["state", "count"]).set_index("state")
        st.bar_chart(df_states)

    st.subheader("Speaker Transition Matrix")
    transitions = flow_summary.get("transition_matrix", {})
    df_transitions = None
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

    csv_bytes = analytics_to_csv({
        "State Distribution": df_states.reset_index().to_dict(orient="records") if df_states is not None else [],
        "Transitions": df_transitions.to_dict(orient="records") if df_transitions is not None else [],
    })
    pdf_bytes = analytics_to_pdf(
        title=f"Conversation Flow – {dataset_type.value}",
        metrics={
            "Average Turns": stats.get("average_turns", 0),
            "Median Turns": stats.get("median_turns", 0),
            "Max Turns": stats.get("max_turns", 0),
        },
        tables={
            "State Distribution": df_states.reset_index().to_dict(orient="records") if state_distribution else [],
            "Transitions": df_transitions.to_dict(orient="records") if transitions else [],
        },
    )
    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV", data=csv_bytes, file_name="conversation_flow.csv", mime="text/csv")
    col_b.download_button("Download PDF", data=pdf_bytes, file_name="conversation_flow.pdf", mime="application/pdf")


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
        if summary.get("aggregate", {}).get("overall_average_user_sentiment", 0) < -0.2:
            st.error("User sentiment is trending negative. Investigate recent conversations.")

    csv_bytes = analytics_to_csv({"Sentiment Trend": trend_df.reset_index().to_dict(orient="records")})
    pdf_bytes = analytics_to_pdf(
        title=f"Sentiment Trends – {dataset_type.value}",
        metrics={
            "Overall User Sentiment": summary.get("aggregate", {}).get("overall_average_user_sentiment") if summary else "N/A",
            "Overall Assistant Sentiment": summary.get("aggregate", {}).get("overall_average_assistant_sentiment") if summary else "N/A",
        },
        tables={"Trend": trend_df.reset_index().to_dict(orient="records")},
    )
    col_a, col_b = st.columns(2)
    col_a.download_button("Download CSV", data=csv_bytes, file_name="sentiment_trend.csv", mime="text/csv")
    col_b.download_button("Download PDF", data=pdf_bytes, file_name="sentiment_trend.pdf", mime="application/pdf")


def dataset_selector(label: str = "Dataset") -> DatasetType:
    dataset_options = sorted(list(DatasetType), key=lambda dt: dt.value)
    return st.selectbox(label, dataset_options, format_func=lambda dt: dt.value.replace("_", " ").title())


def _rerun_app() -> None:
    """Trigger a Streamlit rerun while remaining compatible with older/newer APIs."""
    for attr in ("experimental_rerun", "rerun"):
        rerun = getattr(st, attr, None)
        if callable(rerun):
            rerun()
            return
    raise RuntimeError("Streamlit rerun function is unavailable; check Streamlit version.")


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

if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()

auto_refresh = st.sidebar.checkbox("Auto refresh", value=st.session_state.get("auto_refresh", False))
interval = st.sidebar.slider("Refresh interval (seconds)", min_value=10, max_value=300, value=60, step=10)
st.session_state["auto_refresh"] = auto_refresh
st.session_state["refresh_interval"] = interval

if st.sidebar.button("Refresh Data"):
    load_experiments.cache_clear()  # type: ignore[attr-defined]
    load_dataset.cache_clear()  # type: ignore[attr-defined]
    st.session_state["last_refresh"] = time.time()
    _rerun_app()

if auto_refresh and (time.time() - st.session_state["last_refresh"]) > interval:
    load_experiments.cache_clear()  # type: ignore[attr-defined]
    load_dataset.cache_clear()  # type: ignore[attr-defined]
    st.session_state["last_refresh"] = time.time()
    _rerun_app()

renderer = PAGES[selection]
renderer()
