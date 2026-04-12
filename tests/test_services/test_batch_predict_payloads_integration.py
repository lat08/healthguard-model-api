"""Integration: batch JSON files → predict (multi-window / multi-record same user)."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.services.fall_service import FallModelService
from app.services.health_service import HealthModelService
from app.services.sleep_service import SleepModelService

FALL_BATCH = settings.fall_sample_input_path.parent / "cases" / "batch_multi_windows_one_device.json"
HEALTH_BATCH = settings.health_sample_input_path.parent / "cases" / "batch_multi_vitals_same_patient.json"
SLEEP_BATCH = settings.sleep_sample_input_path.parent / "cases" / "batch_multi_nights_one_user.json"


@pytest.mark.skipif(
    not settings.fall_bundle_path.exists() or not FALL_BATCH.exists(),
    reason="fall bundle or batch JSON missing",
)
def test_fall_predict_batch_multi_windows() -> None:
    raw = json.loads(FALL_BATCH.read_text(encoding="utf-8-sig"))
    assert isinstance(raw, list) and len(raw) >= 2
    svc = FallModelService()
    svc.load()
    assert svc.is_loaded
    out = svc.predict(raw)
    assert len(out) == len(raw)
    for row in out:
        assert 0.0 <= float(row["predicted_fall_probability"]) <= 1.0


@pytest.mark.skipif(
    not settings.health_bundle_path.exists() or not HEALTH_BATCH.exists(),
    reason="health bundle or batch JSON missing",
)
def test_health_predict_batch_multi_records() -> None:
    body = json.loads(HEALTH_BATCH.read_text(encoding="utf-8-sig"))
    recs = body["records"]
    assert len(recs) >= 5
    svc = HealthModelService()
    svc.load()
    assert svc.is_loaded
    out = svc.predict(recs)
    assert len(out) == len(recs)


@pytest.mark.skipif(
    not settings.sleep_bundle_path.exists() or not SLEEP_BATCH.exists(),
    reason="sleep bundle or batch JSON missing",
)
def test_sleep_predict_batch_multi_nights() -> None:
    body = json.loads(SLEEP_BATCH.read_text(encoding="utf-8-sig"))
    recs = body["records"]
    assert len(recs) >= 2
    svc = SleepModelService()
    svc.load()
    assert svc.is_loaded
    out = svc.predict(recs)
    assert len(out) == len(recs)
    for row in out:
        assert 0.0 <= float(row["predicted_sleep_score"]) <= 100.0
