"""Integration: sleep bundle + each sample case."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.services.sleep_service import SleepModelService

pytestmark = pytest.mark.skipif(
    not settings.sleep_bundle_path.exists()
    or not settings.sleep_sample_cases_path.exists(),
    reason="sleep bundle or sample cases missing",
)


@pytest.fixture
def loaded_sleep() -> SleepModelService:
    svc = SleepModelService()
    svc.load()
    assert svc.is_loaded
    return svc


def test_sleep_predict_each_sample_case(loaded_sleep: SleepModelService) -> None:
    doc = json.loads(settings.sleep_sample_cases_path.read_text(encoding="utf-8-sig"))
    for c in doc["cases"]:
        req = c["request"]
        out = loaded_sleep.predict(req["records"])
        assert len(out) == len(req["records"]), c["id"]
        for row in out:
            s = float(row["predicted_sleep_score"])
            assert 0.0 <= s <= 100.0, c["id"]
