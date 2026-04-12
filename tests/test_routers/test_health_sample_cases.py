"""Health sample-cases routes."""

from __future__ import annotations

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(
    not settings.health_sample_cases_path.exists(),
    reason="health iot_sample_cases.json missing",
)


def test_health_sample_cases(client):
    r = client.get("/api/v1/health/sample-cases")
    assert r.status_code == 200
    body = r.json()
    assert body.get("version") == 1
    ids = {c["id"] for c in body["cases"]}
    assert "vitals_normal_young" in ids
    assert "vitals_high_risk_pattern" in ids


@pytest.mark.parametrize(
    "case_id",
    ["vitals_normal_young", "vitals_borderline", "vitals_high_risk_pattern"],
)
def test_health_sample_input_query_case(client, case_id: str):
    r = client.get("/api/v1/health/sample-input", params={"case": case_id})
    assert r.status_code == 200, r.text
    assert "records" in r.json()
    assert len(r.json()["records"]) >= 1


def test_health_sample_input_unknown_case(client):
    r = client.get("/api/v1/health/sample-input", params={"case": "nope"})
    assert r.status_code == 404
