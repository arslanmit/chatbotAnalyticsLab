import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fastapi.testclient import TestClient

from src.repositories.orm import Base


@pytest.fixture(scope="function", autouse=True)
def _test_environment(monkeypatch, tmp_path):
    test_engine = create_engine("sqlite:///:memory:", future=True)
    TestingSessionLocal = sessionmaker(bind=test_engine, autoflush=False, autocommit=False, future=True)
    Base.metadata.create_all(bind=test_engine)

    monkeypatch.setattr("src.repositories.database.engine", test_engine)
    monkeypatch.setattr("src.repositories.database.SessionLocal", TestingSessionLocal)

    from src.config.settings import settings

    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    settings.data.backup_dir = str(backup_dir)
    settings.data.backup_format = "json"
    settings.data.backup_retention = 2

    settings.alerts.cpu_threshold = 50.0
    settings.alerts.memory_threshold = 50.0
    settings.alerts.request_latency_threshold_ms = 500.0
    settings.alerts.channels = ["log"]

    yield

    Base.metadata.drop_all(bind=test_engine)
    test_engine.dispose()


@pytest.fixture()
def api_client():
    from src.api.app import create_app

    app = create_app()
    with TestClient(app) as client:
        yield client
