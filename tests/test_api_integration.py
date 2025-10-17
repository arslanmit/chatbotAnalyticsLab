from src.monitoring import collect_system_metrics


def test_health_endpoint(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_health_metrics_endpoint(api_client):
    response = api_client.get("/health/metrics")
    assert response.status_code == 200
    payload = response.json()
    assert "system" in payload
    assert "total_requests" in payload


def test_health_alerts_endpoint(api_client):
    response = api_client.get("/health/alerts", params={"trigger": "false"})
    assert response.status_code == 200
    payload = response.json()
    assert "alerts" in payload
    assert isinstance(payload["alerts"], list)
