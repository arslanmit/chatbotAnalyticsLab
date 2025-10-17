"""
Configuration management for the Chatbot Analytics System.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = "sqlite:///chatbot_analytics.db"
    echo: bool = False
    pool_size: int = 5
    max_overflow: int = 10


@dataclass
class ModelConfig:
    """Model configuration settings."""
    default_model: str = "bert-base-uncased"
    cache_dir: str = "./models/cache"
    max_sequence_length: int = 512
    batch_size: int = 16
    device: str = "auto"  # auto, cpu, cuda


@dataclass
class DataConfig:
    """Data processing configuration."""
    dataset_dir: str = "./Dataset"
    processed_data_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"
    max_conversations_per_dataset: Optional[int] = None
    validation_split: float = 0.15
    test_split: float = 0.15
    backup_dir: str = "./backups"
    backup_format: str = "json"
    backup_retention: int = 5


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    max_request_size: int = 16 * 1024 * 1024  # 16MB


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 8501
    debug: bool = False
    theme: str = "light"
    cache_ttl: int = 300  # seconds


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = "./logs/app.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class AlertConfig:
    """Alerting configuration."""
    cpu_threshold: float = 85.0
    memory_threshold: float = 85.0
    request_latency_threshold_ms: float = 1200.0
    channels: List[str] = field(default_factory=lambda: ["log"])


@dataclass
class Settings:
    """Main application settings."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    
    # Environment settings
    environment: str = "development"
    debug: bool = True
    
    @classmethod
    def from_env(cls) -> 'Settings':
        """Create settings from environment variables."""
        settings = cls()
        
        # Database settings
        if db_url := os.getenv("DATABASE_URL"):
            settings.database.url = db_url
        
        # Model settings
        if model_name := os.getenv("DEFAULT_MODEL"):
            settings.model.default_model = model_name
        if cache_dir := os.getenv("MODEL_CACHE_DIR"):
            settings.model.cache_dir = cache_dir
        
        # Data settings
        if dataset_dir := os.getenv("DATASET_DIR"):
            settings.data.dataset_dir = dataset_dir
        
        # API settings
        if api_host := os.getenv("API_HOST"):
            settings.api.host = api_host
        if api_port := os.getenv("API_PORT"):
            settings.api.port = int(api_port)
        
        # Dashboard settings
        if dashboard_host := os.getenv("DASHBOARD_HOST"):
            settings.dashboard.host = dashboard_host
        if dashboard_port := os.getenv("DASHBOARD_PORT"):
            settings.dashboard.port = int(dashboard_port)
        
        # Logging settings
        if log_level := os.getenv("LOG_LEVEL"):
            settings.logging.level = log_level.upper()
        if log_file := os.getenv("LOG_FILE"):
            settings.logging.file_path = log_file

        if cpu_threshold := os.getenv("ALERT_CPU_THRESHOLD"):
            settings.alerts.cpu_threshold = float(cpu_threshold)
        if memory_threshold := os.getenv("ALERT_MEMORY_THRESHOLD"):
            settings.alerts.memory_threshold = float(memory_threshold)
        if latency_threshold := os.getenv("ALERT_LATENCY_THRESHOLD_MS"):
            settings.alerts.request_latency_threshold_ms = float(latency_threshold)
        if alert_channels := os.getenv("ALERT_CHANNELS"):
            settings.alerts.channels = [channel.strip() for channel in alert_channels.split(",") if channel.strip()]
        
        # Environment
        settings.environment = os.getenv("ENVIRONMENT", "development")
        settings.debug = os.getenv("DEBUG", "true").lower() == "true"
        
        return settings
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Settings':
        """Load settings from a JSON configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        settings = cls()
        
        # Update settings from config data
        if 'database' in config_data:
            for key, value in config_data['database'].items():
                if hasattr(settings.database, key):
                    setattr(settings.database, key, value)
        
        if 'model' in config_data:
            for key, value in config_data['model'].items():
                if hasattr(settings.model, key):
                    setattr(settings.model, key, value)
        
        if 'data' in config_data:
            for key, value in config_data['data'].items():
                if hasattr(settings.data, key):
                    setattr(settings.data, key, value)
        
        if 'api' in config_data:
            for key, value in config_data['api'].items():
                if hasattr(settings.api, key):
                    setattr(settings.api, key, value)
        
        if 'dashboard' in config_data:
            for key, value in config_data['dashboard'].items():
                if hasattr(settings.dashboard, key):
                    setattr(settings.dashboard, key, value)
        
        if 'logging' in config_data:
            for key, value in config_data['logging'].items():
                if hasattr(settings.logging, key):
                    setattr(settings.logging, key, value)

        if 'alerts' in config_data:
            for key, value in config_data['alerts'].items():
                if hasattr(settings.alerts, key):
                    setattr(settings.alerts, key, value)
        
        # Top-level settings
        settings.environment = config_data.get('environment', settings.environment)
        settings.debug = config_data.get('debug', settings.debug)
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            'database': {
                'url': self.database.url,
                'echo': self.database.echo,
                'pool_size': self.database.pool_size,
                'max_overflow': self.database.max_overflow
            },
            'model': {
                'default_model': self.model.default_model,
                'cache_dir': self.model.cache_dir,
                'max_sequence_length': self.model.max_sequence_length,
                'batch_size': self.model.batch_size,
                'device': self.model.device
            },
            'data': {
                'dataset_dir': self.data.dataset_dir,
                'processed_data_dir': self.data.processed_data_dir,
            'cache_dir': self.data.cache_dir,
            'backup_dir': self.data.backup_dir,
            'backup_format': self.data.backup_format,
            'backup_retention': self.data.backup_retention,
                'max_conversations_per_dataset': self.data.max_conversations_per_dataset,
                'validation_split': self.data.validation_split,
                'test_split': self.data.test_split
            },
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'debug': self.api.debug,
                'reload': self.api.reload,
                'workers': self.api.workers,
                'max_request_size': self.api.max_request_size
            },
            'dashboard': {
                'host': self.dashboard.host,
                'port': self.dashboard.port,
                'debug': self.dashboard.debug,
                'theme': self.dashboard.theme,
                'cache_ttl': self.dashboard.cache_ttl
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count
            },
            'alerts': {
                'cpu_threshold': self.alerts.cpu_threshold,
                'memory_threshold': self.alerts.memory_threshold,
                'request_latency_threshold_ms': self.alerts.request_latency_threshold_ms,
                'channels': self.alerts.channels
            },
            'environment': self.environment,
            'debug': self.debug
        }
    
    def save_to_file(self, config_path: str):
        """Save settings to a JSON configuration file."""
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


# Global settings instance
settings = Settings.from_env()
