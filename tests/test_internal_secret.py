"""Regression tests for D-013 — verify_internal_secret grace period.

3 scenarios per ADR-005:
1. Env unset -> accept + log (grace).
2. Header missing -> accept + log (grace).
3. Header wrong value -> 401.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_with_secret(mock_fall_service, mock_health_service, mock_sleep_service):
    """Client with HEALTHGUARD_INTERNAL_SECRET configured."""
    with (
        patch("app.routers.fall.fall_service", mock_fall_service),
        patch("app.routers.health.health_service", mock_health_service),
        patch("app.routers.sleep.sleep_service", mock_sleep_service),
        patch("app.routers.system.fall_service", mock_fall_service),
        patch("app.routers.system.health_service", mock_health_service),
        patch("app.routers.system.sleep_service", mock_sleep_service),
        patch("app.main.fall_service", mock_fall_service),
        patch("app.main.health_service", mock_health_service),
        patch("app.main.sleep_service", mock_sleep_service),
        patch("app.config.settings.internal_secret", "test-secret-value"),
    ):
        # Reset grace log flag for clean test.
        import app.dependencies

        app.dependencies._GRACE_LOGGED = False
        from app.main import app

        yield TestClient(app)


@pytest.fixture
def client_no_secret(mock_fall_service, mock_health_service, mock_sleep_service):
    """Client with HEALTHGUARD_INTERNAL_SECRET unset (None)."""
    with (
        patch("app.routers.fall.fall_service", mock_fall_service),
        patch("app.routers.health.health_service", mock_health_service),
        patch("app.routers.sleep.sleep_service", mock_sleep_service),
        patch("app.routers.system.fall_service", mock_fall_service),
        patch("app.routers.system.health_service", mock_health_service),
        patch("app.routers.system.sleep_service", mock_sleep_service),
        patch("app.main.fall_service", mock_fall_service),
        patch("app.main.health_service", mock_health_service),
        patch("app.main.sleep_service", mock_sleep_service),
        patch("app.config.settings.internal_secret", None),
    ):
        import app.dependencies

        app.dependencies._GRACE_LOGGED = False
        from app.main import app

        yield TestClient(app)


class TestGracePeriodEnvUnset:
    """When HEALTHGUARD_INTERNAL_SECRET is not set, all requests accepted (grace)."""

    def test_health_predict_accepted_without_header(
        self, client_no_secret, health_request_payload
    ):
        resp = client_no_secret.post(
            "/api/v1/health/predict", json=health_request_payload
        )
        assert resp.status_code == 200

    def test_fall_predict_accepted_without_header(
        self, client_no_secret, fall_request_payload
    ):
        resp = client_no_secret.post("/api/v1/fall/predict", json=fall_request_payload)
        assert resp.status_code == 200

    def test_sleep_predict_accepted_without_header(
        self, client_no_secret, sleep_request_payload
    ):
        resp = client_no_secret.post(
            "/api/v1/sleep/predict", json=sleep_request_payload
        )
        assert resp.status_code == 200


class TestGracePeriodHeaderMissing:
    """When secret configured but caller sends no header, accept (grace)."""

    def test_health_predict_accepted_missing_header(
        self, client_with_secret, health_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/health/predict", json=health_request_payload
        )
        assert resp.status_code == 200

    def test_sleep_predict_accepted_missing_header(
        self, client_with_secret, sleep_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/sleep/predict", json=sleep_request_payload
        )
        assert resp.status_code == 200


class TestHardRejectWrongSecret:
    """When secret configured and caller sends wrong value, 401."""

    def test_health_predict_rejected_wrong_secret(
        self, client_with_secret, health_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/health/predict",
            json=health_request_payload,
            headers={"X-Internal-Secret": "wrong-value"},
        )
        assert resp.status_code == 401

    def test_fall_predict_rejected_wrong_secret(
        self, client_with_secret, fall_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/fall/predict",
            json=fall_request_payload,
            headers={"X-Internal-Secret": "wrong-value"},
        )
        assert resp.status_code == 401

    def test_sleep_predict_rejected_wrong_secret(
        self, client_with_secret, sleep_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/sleep/predict",
            json=sleep_request_payload,
            headers={"X-Internal-Secret": "wrong-value"},
        )
        assert resp.status_code == 401


class TestCorrectSecret:
    """When secret configured and caller sends correct value, 200."""

    def test_health_predict_accepted_correct_secret(
        self, client_with_secret, health_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/health/predict",
            json=health_request_payload,
            headers={"X-Internal-Secret": "test-secret-value"},
        )
        assert resp.status_code == 200

    def test_fall_predict_accepted_correct_secret(
        self, client_with_secret, fall_request_payload
    ):
        resp = client_with_secret.post(
            "/api/v1/fall/predict",
            json=fall_request_payload,
            headers={"X-Internal-Secret": "test-secret-value"},
        )
        assert resp.status_code == 200


class TestHealthzRename:
    """D-014: /health renamed to /healthz."""

    def test_healthz_returns_200(self, client_no_secret):
        resp = client_no_secret.get("/healthz")
        assert resp.status_code == 200

    def test_old_health_returns_404(self, client_no_secret):
        resp = client_no_secret.get("/health")
        assert resp.status_code in (404, 405)
