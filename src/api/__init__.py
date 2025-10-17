"""
API package exposing FastAPI application factory.
"""

from src.api.app import create_app, app

__all__ = ["create_app", "app"]
