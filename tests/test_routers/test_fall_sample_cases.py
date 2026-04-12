"""Fall sample-cases and sample-input?case= routes."""

from __future__ import annotations

import pytest

from app.config import settings


pytestmark = pytest.mark.skipif(
    not settings.fall_sample_cases_path.exists(),
    reason="iot_sample_cases.json missing (run scripts/build_fall_sample_cases.py)",
)


def test_fall_sample_cases_list(client):
    r = client.get("/api/v1/fall/sample-cases")
    assert r.status_code == 200
    body = r.json()
    assert body.get("version") == 1
    ids = {c["id"] for c in body["cases"]}
    assert "upright_stationary" in ids
    assert "fall_impulse_mid" in ids
    for c in body["cases"]:
        assert c.get("intent") in ("not_fall", "fall_like")
        req = c["request"]
        assert len(req["data"]) >= settings.fall_min_sequence_samples


@pytest.mark.parametrize(
    "case_id",
    ["upright_stationary", "walking_like", "fall_impulse_mid", "fall_slump_rotation"],
)
def test_fall_sample_input_query_case(client, case_id: str):
    r = client.get("/api/v1/fall/sample-input", params={"case": case_id})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["window_size"] == 50
    assert len(body["data"]) == 50


def test_fall_sample_input_unknown_case(client):
    r = client.get("/api/v1/fall/sample-input", params={"case": "does_not_exist"})
    assert r.status_code == 404
