"""Health service logic with mocked bundle (golden probabilities)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services import health_service as hs


@pytest.fixture
def fake_bundle():
    pre = MagicMock()
    pre.transform = MagicMock(return_value=np.zeros((1, 14), dtype=np.float32))
    model = MagicMock()
    model.predict_proba = MagicMock(return_value=np.array([[0.2, 0.8]], dtype=np.float64))
    return {"preprocessor": pre, "model": model}


def test_predict_high_risk_column(fake_bundle):
    svc = hs.HealthModelService()
    svc._bundle = fake_bundle
    svc._loaded = True
    svc._backend = "lightgbm"
    rec = [
        {
            "heart_rate": 100.0,
            "respiratory_rate": 20.0,
            "body_temperature": 37.0,
            "spo2": 97.0,
            "systolic_blood_pressure": 120.0,
            "diastolic_blood_pressure": 80.0,
            "age": 40,
            "gender": 1,
            "weight_kg": 70.0,
            "height_m": 1.75,
            "derived_hrv": 0.1,
            "derived_pulse_pressure": 40.0,
            "derived_bmi": 22.0,
            "derived_map": 90.0,
        }
    ]
    out = svc.predict(rec)
    assert out[0]["predicted_health_risk_probability"] == 0.8
    assert out[0]["predicted_health_risk_label"] == "high_risk"
