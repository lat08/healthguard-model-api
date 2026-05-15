"""ADR-018: Health service synthetic-default contract tests.

Verifies the new ``effective_confidence`` + ``data_quality_warning`` +
``is_synthetic_default`` fields emitted by
``HealthModelService.predict_api`` (XR-003 step 3 — finalises XR-003).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services import health_service as hs


def _build_service_with_fake_bundle(probability: float = 0.8) -> hs.HealthModelService:
    pre = MagicMock()
    pre.transform = MagicMock(return_value=np.zeros((1, 14), dtype=np.float32))
    pre.get_feature_names_out = MagicMock(
        return_value=np.array(hs.FEATURE_ORDER, dtype=object)
    )
    model = MagicMock()
    # Two-column predict_proba so the column-1 (positive) slice picks
    # ``probability``.
    proba = np.array([[1 - probability, probability]], dtype=np.float64)
    model.predict_proba = MagicMock(return_value=proba)

    # SHAP path requires a booster with ``predict(pred_contrib=True)``.
    booster = MagicMock()
    contrib_row = np.zeros(15, dtype=np.float64)
    booster.predict = MagicMock(return_value=np.array([contrib_row]))
    model.booster_ = booster
    model.feature_names_in_ = np.array(hs.FEATURE_ORDER, dtype=object)
    model.n_features_in_ = len(hs.FEATURE_ORDER)

    svc = hs.HealthModelService()
    svc._bundle = {
        "preprocessor": pre,
        "model": model,
        "feature_names": list(hs.FEATURE_ORDER),
    }
    svc._loaded = True
    svc._backend = "lightgbm"
    svc._feature_order = list(hs.FEATURE_ORDER)
    return svc


@pytest.fixture
def clean_record() -> dict:
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
        "is_synthetic_default": False,
        "defaults_applied": [],
    }


class TestSyntheticContract:
    def test_clean_record_emits_no_warning(self, clean_record):
        svc = _build_service_with_fake_bundle(probability=0.7)
        result = svc.predict_api([clean_record])[0]
        prediction = result["prediction"]
        assert prediction["is_synthetic_default"] is False
        assert prediction["data_quality_warning"] is None

    def test_clean_record_effective_equals_raw_confidence(self, clean_record):
        svc = _build_service_with_fake_bundle(probability=0.7)
        result = svc.predict_api([clean_record])[0]
        prediction = result["prediction"]
        assert prediction["confidence"] == 0.7
        assert prediction["effective_confidence"] == 0.7

    def test_synthetic_record_halves_effective_confidence(self, clean_record):
        synthetic = {
            **clean_record,
            "is_synthetic_default": True,
            "defaults_applied": ["derived_hrv", "weight_kg"],
        }
        svc = _build_service_with_fake_bundle(probability=0.8)
        result = svc.predict_api([synthetic])[0]
        prediction = result["prediction"]
        assert prediction["is_synthetic_default"] is True
        assert prediction["confidence"] == 0.8
        assert prediction["effective_confidence"] == 0.4

    def test_synthetic_record_data_quality_warning_lists_fields(self, clean_record):
        synthetic = {
            **clean_record,
            "is_synthetic_default": True,
            "defaults_applied": ["derived_hrv", "weight_kg"],
        }
        svc = _build_service_with_fake_bundle(probability=0.8)
        result = svc.predict_api([synthetic])[0]
        warning = result["prediction"]["data_quality_warning"]
        assert warning is not None
        assert "derived_hrv" in warning
        assert "weight_kg" in warning
        assert "50%" in warning

    def test_synthetic_record_warning_without_fields_uses_generic_copy(
        self, clean_record
    ):
        synthetic = {
            **clean_record,
            "is_synthetic_default": True,
            "defaults_applied": [],
        }
        svc = _build_service_with_fake_bundle(probability=0.6)
        result = svc.predict_api([synthetic])[0]
        warning = result["prediction"]["data_quality_warning"]
        assert warning is not None
        assert "Synthetic defaults applied" in warning


class TestHelperBuildDataQualityWarning:
    def test_returns_none_when_not_synthetic(self):
        assert (
            hs._build_data_quality_warning(is_synthetic=False, defaults_applied=[])
            is None
        )

    def test_returns_none_when_not_synthetic_even_with_fields(self):
        assert (
            hs._build_data_quality_warning(
                is_synthetic=False, defaults_applied=["heart_rate"]
            )
            is None
        )

    def test_returns_field_list_when_synthetic_with_defaults(self):
        msg = hs._build_data_quality_warning(
            is_synthetic=True, defaults_applied=["heart_rate", "spo2"]
        )
        assert msg is not None
        assert "heart_rate" in msg
        assert "spo2" in msg

    def test_returns_generic_copy_when_synthetic_without_defaults(self):
        msg = hs._build_data_quality_warning(
            is_synthetic=True, defaults_applied=[]
        )
        assert msg is not None
        assert "Synthetic defaults" in msg
