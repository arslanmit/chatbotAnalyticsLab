"""Streamlit dashboard for the Chatbot Analytics system."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard import (  # noqa: E402
    compute_overview_metrics,
    get_recent_experiments,
    load_experiments,
)


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
    "Settings": render_settings,
}


st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

if st.sidebar.button("Refresh Data"):
    load_experiments.cache_clear()  # type: ignore[attr-defined]
    st.experimental_rerun()

renderer = PAGES[selection]
renderer()
