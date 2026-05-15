"""ADR-018: Health route 422 structured error contract tests.

Verifies that validation failures (out-of-range, missing fields) return
the canonical ``{"error": {"code": ..., "message": ..., "details": [...]}}``
shape so callers can branch on ``error.code`` instead of regex-matching.
"""

from __future__ import annotations

import os

import pytest


INTERNAL_HEADERS = {"X-Internal-Secret": "test-internal-secret"}


@pytest.fixture(autouse=True)
def _internal_secret_env(monkeypatch):
    monkeypatch.setenv("INTERNAL_SECRET", "test-internal-secret")
    yield


def _valid_record() -> dict:
    return {
        "heart_rate": 80.0,
        "respiratory_rate": 16.0,
        "body_temperature": 36.8,
        "spo2": 98.0,
        "systolic_blood_pressure": 120.0,
        "diastolic_blood_pressure": 80.0,
        "age": 65,
        "gender": 1,
        "weight_kg": 70.0,
        "height_m": 1.7,
        "derived_hrv": 35.0,
        "derived_pulse_pressure": 40.0,
        "derived_bmi": 24.2,
        "derived_map": 93.3,
    }


class TestStructuredValidationError:
    """ADR-018: Pydantic Field violations return VALIDATION_ERROR code."""

    def test_out_of_range_heart_rate_returns_422_validation_error(self, client):
        payload = {"records": [{**_valid_record(), "heart_rate": 300.0}]}
        resp = client.post(
            "/api/v1/health/predict", json=payload, headers=INTERNAL_HEADERS
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "error" in body, body
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert "details" in body["error"]
        assert isinstance(body["error"]["details"], list)

    def test_out_of_range_spo2_details_include_field_name(self, client):
        payload = {"records": [{**_valid_record(), "spo2": 200.0}]}
        resp = client.post(
            "/api/v1/health/predict", json=payload, headers=INTERNAL_HEADERS
        )
        assert resp.status_code == 422
        body = resp.json()
        details = body["error"]["details"]
        assert any("spo2" in d.get("field", "") for d in details), details

    def test_missing_field_returns_422_validation_error(self, client):
        record = _valid_record()
        del record["heart_rate"]
        payload = {"records": [record]}
        resp = client.post(
            "/api/v1/health/predict", json=payload, headers=INTERNAL_HEADERS
        )
        assert resp.status_code == 422
        body = resp.json()
        # Pydantic raises VALIDATION_ERROR for missing required fields at the
        # schema layer (before prepare_inference_frame is reached).
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_legacy_detail_envelope_is_not_present(self, client):
        """Old shape ``{"detail": [...]}`` must not leak through."""
        payload = {"records": [{**_valid_record(), "age": -5}]}
        resp = client.post(
            "/api/v1/health/predict", json=payload, headers=INTERNAL_HEADERS
        )
        assert resp.status_code == 422
        body = resp.json()
        assert "detail" not in body, (
            "Legacy {detail: [...]} envelope must be replaced by "
            "{error: {code, message, details}}"
        )
