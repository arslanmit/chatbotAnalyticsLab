"""
Route modules for the Chatbot Analytics API.
"""

from src.api.routes import datasets, intents, conversations, training, health

__all__ = [
    "datasets",
    "intents",
    "conversations",
    "training",
    "health",
]
