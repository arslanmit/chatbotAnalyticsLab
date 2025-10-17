"""
FastAPI application factory for the Chatbot Analytics system.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import datasets, intents, conversations, training, health
from src.api.middleware import RateLimitMiddleware, RequestMetricsMiddleware
from src.api.dependencies import get_metrics_collector


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Chatbot Analytics API",
        description="API surface for dataset processing, intent classification, and analytics.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    metrics_collector = get_metrics_collector()
    app.state.metrics_collector = metrics_collector
    app.add_middleware(RateLimitMiddleware, limit=120, window_seconds=60)
    app.add_middleware(RequestMetricsMiddleware, collector=metrics_collector)

    app.include_router(health.router, tags=["health"])
    app.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
    app.include_router(intents.router, prefix="/intents", tags=["intents"])
    app.include_router(conversations.router, prefix="/conversations", tags=["conversations"])
    app.include_router(training.router, prefix="/training", tags=["training"])

    return app


app = create_app()
