"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


def _meta(model_family: str, model_name: str, artifact_path: str) -> dict:
    return {
        "model_family": model_family,
        "model_name": model_name,
        "model_version": "v_current",
        "artifact_type": "python_bundle",
        "artifact_path": artifact_path,
        "timestamp": "2026-04-18T00:00:00+00:00",
        "request_id": "req_test123456",
    }


@pytest.fixture
def single_sensor_sample() -> dict:
    return {
        "timestamp": 1710000000,
        "accel": {"x": 0.42, "y": -0.18, "z": 1.12},
        "gyro": {"x": 30.0, "y": -10.0, "z": 5.0},
        "orientation": {"pitch": 10.0, "roll": 5.0, "yaw": 30.0},
        "environment": {"floor_vibration": 0.2, "room_occupancy": 1.0, "pressure_mat": 100.0},
    }


@pytest.fixture
def fall_request_payload(single_sensor_sample) -> dict:
    data = [{**single_sensor_sample, "timestamp": 1710000000 + i} for i in range(50)]
    return {
        "device_id": "wearable_test_0001",
        "sampling_rate": 50,
        "window_size": 50,
        "data": data,
    }


@pytest.fixture
def single_vital_record() -> dict:
    return {
        "heart_rate": 118,
        "respiratory_rate": 24,
        "body_temperature": 38.2,
        "spo2": 92.5,
        "systolic_blood_pressure": 148,
        "diastolic_blood_pressure": 96,
        "age": 67,
        "gender": 1,
        "weight_kg": 73.5,
        "height_m": 1.68,
        "derived_hrv": 24.0,
        "derived_pulse_pressure": 52,
        "derived_bmi": 26.0,
        "derived_map": 113.3,
    }


@pytest.fixture
def health_request_payload(single_vital_record) -> dict:
    return {"records": [single_vital_record]}


@pytest.fixture
def single_sleep_record() -> dict:
    return {
        "user_id": "user_test_0001",
        "date_recorded": "2024-04-03",
        "sleep_start_timestamp": "2024-04-03 22:36:00",
        "sleep_end_timestamp": "2024-04-04 06:01:00",
        "duration_minutes": 445,
        "sleep_latency_minutes": 2,
        "wake_after_sleep_onset_minutes": 15,
        "sleep_efficiency_pct": 96.2,
        "sleep_stage_deep_pct": 15.6,
        "sleep_stage_light_pct": 53.3,
        "sleep_stage_rem_pct": 15.8,
        "sleep_stage_awake_pct": 15.3,
        "heart_rate_mean_bpm": 62.3,
        "heart_rate_min_bpm": 54.3,
        "heart_rate_max_bpm": 71.3,
        "hrv_rmssd_ms": 31.1,
        "respiration_rate_bpm": 14.4,
        "spo2_mean_pct": 96.7,
        "spo2_min_pct": 94.5,
        "movement_count": 29,
        "snore_events": 0,
        "ambient_noise_db": 38.9,
        "room_temperature_c": 23.7,
        "room_humidity_pct": 57.5,
        "step_count_day": 5139,
        "caffeine_mg": 126,
        "alcohol_units": 0.2,
        "medication_flag": 0,
        "jetlag_hours": -11,
        "timezone": "Europe/London",
        "age": 50,
        "gender": "female",
        "weight_kg": 94.2,
        "height_cm": 166.1,
        "device_model": "AlphaWatch X1",
        "bedtime_consistency_std_min": 37.1,
        "stress_score": 33,
        "activity_before_bed_min": 39,
        "screen_time_before_bed_min": 87,
        "insomnia_flag": 0,
        "apnea_risk_score": 24,
        "nap_duration_minutes": 10,
        "created_at": "2025-10-21 16:41:53.708868",
    }


@pytest.fixture
def sleep_request_payload(single_sleep_record) -> dict:
    return {"records": [single_sleep_record]}


def _mock_fall(loaded: bool = True):
    svc = MagicMock()
    svc.is_loaded = loaded
    svc.backend = "lightgbm" if loaded else "none"
    svc.get_info.return_value = {
        "model_name": "fall_lightgbm_binary",
        "status": "loaded" if loaded else "unavailable",
        "inference_backend": "lightgbm" if loaded else "none",
        "feature_count": 100,
        "min_sequence_samples": 50,
        "thresholds": {"fall_true_at": 0.5, "warning_at": 0.6, "critical_at": 0.85},
    }
    svc.predict.return_value = [
        {
            "device_id": "wearable_test_0001",
            "sample_count": 50,
            "predicted_fall_probability": 0.15,
            "predicted_fall": False,
            "predicted_fall_label": "normal",
            "risk_level": "normal",
            "requires_attention": False,
            "high_priority_alert": False,
            "predicted_activity": None,
            "activity_probability": None,
        }
    ]
    svc.predict_api.return_value = [
        {
            **svc.predict.return_value[0],
            "status": "ok",
            "meta": _meta("fall", "fall_detection", "models/fall/fall_bundle.joblib"),
            "input_ref": {
                "user_id": None,
                "device_id": "wearable_test_0001",
                "event_timestamp": 1710000000,
                "source_file": None,
            },
            "prediction": {
                "prediction_label": "normal",
                "prediction_score": 0.15,
                "prediction_band": "normal",
                "requires_attention": False,
                "high_priority_alert": False,
                "confidence": 0.15,
            },
            "top_features": [
                {
                    "feature": "floor_vibration_mean",
                    "feature_value": 0.2,
                    "impact": 0.12,
                    "direction": "risk_down",
                    "reason": "floor_vibration_mean=0.2 dang lam giam nguy co",
                }
            ],
            "shap": {
                "available": True,
                "output_space": "raw_margin",
                "base_value": 0.0,
                "prediction_value": 0.15,
                "values": [
                    {
                        "feature": "floor_vibration_mean",
                        "feature_value": 0.2,
                        "shap_value": -0.12,
                        "impact": 0.12,
                        "direction": "risk_down",
                    }
                ],
            },
            "explanation": {
                "short_text": "floor_vibration_mean 0.2 la cac tin hieu dang duoc model uu tien theo doi.",
                "clinical_note": "Top features duoc tong hop tu SHAP/native contribution; nen doi chieu them voi boi canh thuc te.",
                "recommended_actions": ["tiep tuc giam sat", "doi chieu neu co bao dong khac"],
            },
        }
    ]
    svc.unavailable_detail = lambda: "Fall detection model is not loaded."
    return svc


def _mock_health(loaded: bool = True):
    svc = MagicMock()
    svc.is_loaded = loaded
    svc.backend = "lightgbm" if loaded else "none"
    svc.get_info.return_value = {
        "model_name": "healthguard_lightgbm",
        "status": "loaded" if loaded else "unavailable",
        "inference_backend": "lightgbm" if loaded else "none",
        "feature_count": 14,
        "thresholds": {"high_risk_true_at": 0.5, "warning_at": 0.35, "critical_at": 0.65},
    }
    svc.predict.return_value = [
        {
            "record_index": 0,
            "predicted_health_risk_probability": 0.88,
            "predicted_health_risk_label": "high_risk",
            "risk_level": "critical",
            "requires_attention": True,
            "high_priority_alert": True,
        }
    ]
    svc.predict_api.return_value = [
        {
            **svc.predict.return_value[0],
            "status": "ok",
            "meta": _meta(
                "healthguard",
                "healthguard",
                "models/healthguard/healthguard_bundle.joblib",
            ),
            "input_ref": {
                "user_id": None,
                "device_id": None,
                "event_timestamp": None,
                "source_file": None,
            },
            "prediction": {
                "prediction_label": "high_risk",
                "prediction_score": 0.88,
                "prediction_band": "critical",
                "requires_attention": True,
                "high_priority_alert": True,
                "confidence": 0.88,
            },
            "top_features": [
                {
                    "feature": "spo2",
                    "feature_value": 92.5,
                    "impact": 0.43,
                    "direction": "risk_up",
                    "reason": "SpO2 thap dang lam tang nguy co",
                }
            ],
            "shap": {
                "available": True,
                "output_space": "raw_margin",
                "base_value": 0.0,
                "prediction_value": 0.88,
                "values": [
                    {
                        "feature": "spo2",
                        "feature_value": 92.5,
                        "shap_value": 0.43,
                        "impact": 0.43,
                        "direction": "risk_up",
                    }
                ],
            },
            "explanation": {
                "short_text": "spo2 92.5 dang lam tang muc do canh bao.",
                "clinical_note": "Top features duoc tong hop tu SHAP/native contribution; nen doi chieu them voi boi canh thuc te.",
                "recommended_actions": ["do lai chi so", "doi chieu trieu chung", "lien he nhan vien y te"],
            },
        }
    ]
    return svc


def _mock_sleep(loaded: bool = True):
    svc = MagicMock()
    svc.is_loaded = loaded
    svc.backend = "catboost" if loaded else "none"
    svc.get_info.return_value = {
        "model_name": "catboost",
        "status": "loaded" if loaded else "unavailable",
        "inference_backend": "catboost" if loaded else "none",
        "feature_count": 42,
        "thresholds": {
            "critical_below": 50,
            "poor_below": 60,
            "fair_below": 75,
            "good_below": 85,
            "attention_below": 60,
            "alert_below": 50,
        },
        "metrics": {},
    }
    svc.predict.return_value = [
        {
            "record_index": 0,
            "predicted_sleep_score": 72.5,
            "predicted_sleep_label": "fair",
            "risk_level": "fair",
            "requires_attention": False,
            "high_priority_alert": False,
        }
    ]
    svc.predict_api.return_value = [
        {
            **svc.predict.return_value[0],
            "status": "ok",
            "meta": _meta("sleep", "sleep_score", "models/Sleep/sleep_score_bundle.joblib"),
            "input_ref": {
                "user_id": "user_test_0001",
                "device_id": None,
                "event_timestamp": "2024-04-04 06:01:00",
                "source_file": None,
            },
            "prediction": {
                "prediction_label": "fair",
                "prediction_score": 72.5,
                "prediction_band": "info",
                "requires_attention": False,
                "high_priority_alert": False,
                "confidence": 0.725,
            },
            "top_features": [
                {
                    "feature": "sleep_efficiency_pct",
                    "feature_value": 96.2,
                    "impact": 0.31,
                    "direction": "risk_down",
                    "reason": "sleep_efficiency_pct=96.2 dang lam giam nguy co",
                }
            ],
            "shap": {
                "available": True,
                "output_space": "prediction",
                "base_value": 70.0,
                "prediction_value": 72.5,
                "values": [
                    {
                        "feature": "sleep_efficiency_pct",
                        "feature_value": 96.2,
                        "shap_value": 0.31,
                        "impact": 0.31,
                        "direction": "risk_down",
                    }
                ],
            },
            "explanation": {
                "short_text": "sleep_efficiency_pct 96.2 dang ho tro sleep score on dinh hon.",
                "clinical_note": "Dien giai duoc tong hop tu SHAP tren score du doan; nen xem cung xu huong nhieu dem lien tiep.",
                "recommended_actions": ["duy tri thoi quen ngu deu", "tiep tuc theo doi xu huong"],
            },
        }
    ]
    return svc


@pytest.fixture
def mock_fall_service():
    return _mock_fall(True)


@pytest.fixture
def mock_health_service():
    return _mock_health(True)


@pytest.fixture
def mock_sleep_service():
    return _mock_sleep(True)


@pytest.fixture
def client(mock_fall_service, mock_health_service, mock_sleep_service):
    with patch("app.routers.fall.fall_service", mock_fall_service), \
         patch("app.routers.health.health_service", mock_health_service), \
         patch("app.routers.sleep.sleep_service", mock_sleep_service), \
         patch("app.routers.system.fall_service", mock_fall_service), \
         patch("app.routers.system.health_service", mock_health_service), \
         patch("app.routers.system.sleep_service", mock_sleep_service), \
         patch("app.main.fall_service", mock_fall_service), \
         patch("app.main.health_service", mock_health_service), \
         patch("app.main.sleep_service", mock_sleep_service):
        from app.main import app
        yield TestClient(app)


@pytest.fixture
def client_unloaded():
    with patch("app.routers.fall.fall_service", _mock_fall(False)), \
         patch("app.routers.health.health_service", _mock_health(False)), \
         patch("app.routers.sleep.sleep_service", _mock_sleep(False)), \
         patch("app.routers.system.fall_service", _mock_fall(False)), \
         patch("app.routers.system.health_service", _mock_health(False)), \
         patch("app.routers.system.sleep_service", _mock_sleep(False)), \
         patch("app.main.fall_service", _mock_fall(False)), \
         patch("app.main.health_service", _mock_health(False)), \
         patch("app.main.sleep_service", _mock_sleep(False)):
        from app.main import app
        yield TestClient(app)
