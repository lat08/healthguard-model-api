"""Sleep sample-cases routes."""

from __future__ import annotations

import pytest

from app.config import settings

pytestmark = pytest.mark.skipif(
    not settings.sleep_sample_cases_path.exists(),
    reason="sleep iot_sample_cases.json missing",
)


def test_sleep_sample_cases(client):
    r = client.get("/api/v1/sleep/sample-cases")
    assert r.status_code == 200
    body = r.json()
    assert body.get("version") == 1
    cases = body["cases"]
    assert len(cases) >= 3
    for c in cases:
        assert "id" in c and "request" in c
        assert "records" in c["request"]
        assert len(c["request"]["records"]) >= 1


def test_sleep_sample_input_each_case(client):
    body = client.get("/api/v1/sleep/sample-cases").json()
    for c in body["cases"]:
        cid = c["id"]
        r = client.get("/api/v1/sleep/sample-input", params={"case": cid})
        assert r.status_code == 200, (cid, r.text)
        assert "records" in r.json()
        assert len(r.json()["records"]) >= 1


def test_sleep_sample_input_unknown_case(client):
    r = client.get("/api/v1/sleep/sample-input", params={"case": "nope"})
    assert r.status_code == 404
