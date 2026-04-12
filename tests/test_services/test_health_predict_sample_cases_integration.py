"""Integration: health bundle + each sample case."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.services.health_service import HealthModelService

pytestmark = pytest.mark.skipif(
    not settings.health_bundle_path.exists()
    or not settings.health_sample_cases_path.exists(),
    reason="health bundle or sample cases missing",
)


@pytest.fixture
def loaded_health() -> HealthModelService:
    svc = HealthModelService()
    svc.load()
    assert svc.is_loaded
    return svc


@pytest.mark.parametrize(
    "case_id",
    ["vitals_normal_young", "vitals_borderline", "vitals_high_risk_pattern"],
)
def test_health_predict_sample_case(loaded_health: HealthModelService, case_id: str) -> None:
    doc = json.loads(settings.health_sample_cases_path.read_text(encoding="utf-8-sig"))
    req = next(c["request"] for c in doc["cases"] if c["id"] == case_id)
    out = loaded_health.predict(req["records"])
    assert len(out) == len(req["records"])
    for row in out:
        p = float(row["predicted_health_risk_probability"])
        assert 0.0 <= p <= 1.0
