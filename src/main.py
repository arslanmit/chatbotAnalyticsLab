"""
Main application entry point for the Chatbot Analytics System.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.utils.logging import setup_logging, get_logger
from src.repositories.database import init_db


def initialize_application():
    """Initialize the application with logging and configuration."""
    # Set up logging
    logger = setup_logging()
    logger.info("Starting Chatbot Analytics System")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize database schema
    init_db()
    logger.info("Database initialized at %s", settings.database.url)
    
    # Create necessary directories
    directories = [
        Path(settings.data.processed_data_dir),
        Path(settings.data.cache_dir),
        Path(settings.model.cache_dir),
        Path("./logs"),
        Path("./data"),
        Path("./models")
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    logger.info("Application initialization complete")
    return logger


if __name__ == "__main__":
    logger = initialize_application()
    logger.info("Chatbot Analytics System is ready")
    
    # Print system information
    print("=" * 50)
    print("Chatbot Analytics and Optimization System")
    print("=" * 50)
    print(f"Environment: {settings.environment}")
    print(f"Dataset Directory: {settings.data.dataset_dir}")
    print(f"API Port: {settings.api.port}")
    print(f"Dashboard Port: {settings.dashboard.port}")
    print("=" * 50)
