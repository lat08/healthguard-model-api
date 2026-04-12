from __future__ import annotations


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "healthy"
    assert "fall" in body["models"]


def test_list_models(client):
    r = client.get("/api/v1/models")
    assert r.status_code == 200
    assert "fall_detection" in r.json()["models"]
