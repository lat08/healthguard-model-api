"""Integration: real fall bundle preprocessor + model (guards sklearn / pickle drift)."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.services.fall_service import FallModelService

pytestmark = pytest.mark.skipif(
    not settings.fall_bundle_path.exists(),
    reason="fall_bundle.joblib not present",
)


@pytest.fixture
def loaded_fall() -> FallModelService:
    svc = FallModelService()
    svc.load()
    assert svc.is_loaded
    return svc


def test_fall_predict_preprocessor_and_model(loaded_fall: FallModelService) -> None:
    sample = {
        "timestamp": 1710000000,
        "accel": {"x": 0.42, "y": -0.18, "z": 1.12},
        "gyro": {"x": 30.0, "y": -10.0, "z": 5.0},
        "orientation": {"pitch": 10.0, "roll": 5.0, "yaw": 30.0},
        "environment": {"floor_vibration": 0.2, "room_occupancy": 1.0, "pressure_mat": 100.0},
    }
    data = [{**sample, "timestamp": 1710000000 + i} for i in range(50)]
    payload = {
        "device_id": "wearable_integration_0001",
        "sampling_rate": 50,
        "window_size": 50,
        "data": data,
    }
    out = loaded_fall.predict([payload])
    assert len(out) == 1
    assert "predicted_fall_probability" in out[0]
    assert 0.0 <= float(out[0]["predicted_fall_probability"]) <= 1.0


@pytest.mark.skipif(
    not settings.fall_sample_cases_path.exists(),
    reason="iot_sample_cases.json missing",
)
@pytest.mark.parametrize(
    "case_id",
    ["upright_stationary", "walking_like", "fall_impulse_mid", "fall_slump_rotation"],
)
def test_fall_predict_sample_cases_json(loaded_fall: FallModelService, case_id: str) -> None:
    doc = json.loads(settings.fall_sample_cases_path.read_text(encoding="utf-8-sig"))
    req = next(c["request"] for c in doc["cases"] if c["id"] == case_id)
    out = loaded_fall.predict([req])
    assert len(out) == 1
    p = float(out[0]["predicted_fall_probability"])
    assert 0.0 <= p <= 1.0
