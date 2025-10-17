from src.monitoring import evaluate_alerts
from src.config.settings import settings


def test_alert_generation(monkeypatch):
    settings.alerts.cpu_threshold = 10
    settings.alerts.memory_threshold = 10
    settings.alerts.request_latency_threshold_ms = 100

    metrics = {"average_latency_ms": 150,
               "total_requests": 10,
               "total_errors": 0,
               "endpoints": {}}
    system = {
        "system": {
            "cpu_percent": 50,
            "memory_percent": 30,
        }
    }
    alerts = evaluate_alerts(metrics, system)
    names = {alert.name for alert in alerts}
    assert "High Request Latency" in names
    assert "High CPU Usage" in names
    assert "High Memory Usage" in names
