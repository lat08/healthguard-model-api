"""XR-003 step 1: range validator regression tests.

Verify that out-of-range values on VitalSignsRecord, SleepRecord, and
AccelData/GyroData/OrientationData produce Pydantic ValidationError (422).
"""
import pytest
from pydantic import ValidationError

from app.schemas.health import VitalSignsRecord, HealthPredictionRequest
from app.schemas.sleep import SleepRecord, SleepPredictionRequest
from app.schemas.fall import (
    AccelData,
    GyroData,
    OrientationData,
    FallPredictionRequest,
    SensorSample,
    EnvironmentData,
)


# ---------------------------------------------------------------------------
# VitalSignsRecord range tests
# ---------------------------------------------------------------------------

VALID_VITALS = {
    "heart_rate": 75,
    "respiratory_rate": 16,
    "body_temperature": 36.5,
    "spo2": 97,
    "systolic_blood_pressure": 120,
    "diastolic_blood_pressure": 80,
    "age": 65,
    "gender": 1,
    "weight_kg": 70,
    "height_m": 1.7,
    "derived_hrv": 40,
    "derived_pulse_pressure": 40,
    "derived_bmi": 24,
    "derived_map": 93,
}


def test_valid_vitals_accepted():
    record = VitalSignsRecord(**VALID_VITALS)
    assert record.heart_rate == 75


def test_hr_999_rejected():
    data = {**VALID_VITALS, "heart_rate": 999}
    with pytest.raises(ValidationError) as exc_info:
        VitalSignsRecord(**data)
    assert "heart_rate" in str(exc_info.value)


def test_hr_negative_rejected():
    data = {**VALID_VITALS, "heart_rate": -10}
    with pytest.raises(ValidationError):
        VitalSignsRecord(**data)


def test_spo2_over_100_rejected():
    data = {**VALID_VITALS, "spo2": 105}
    with pytest.raises(ValidationError):
        VitalSignsRecord(**data)


def test_spo2_below_50_rejected():
    data = {**VALID_VITALS, "spo2": 30}
    with pytest.raises(ValidationError):
        VitalSignsRecord(**data)


def test_body_temp_over_45_rejected():
    data = {**VALID_VITALS, "body_temperature": 50}
    with pytest.raises(ValidationError):
        VitalSignsRecord(**data)


def test_health_request_max_length_100():
    """Batch list capped at 100 records."""
    records = [VALID_VITALS] * 101
    with pytest.raises(ValidationError) as exc_info:
        HealthPredictionRequest(records=records)
    assert "too_long" in str(exc_info.value).lower() or "list" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# SleepRecord range tests
# ---------------------------------------------------------------------------

VALID_SLEEP = {
    "user_id": "user_123",
    "date_recorded": "2026-05-14",
    "sleep_start_timestamp": "2026-05-13T23:00:00+07:00",
    "sleep_end_timestamp": "2026-05-14T06:30:00+07:00",
    "duration_minutes": 450,
    "sleep_latency_minutes": 15,
    "wake_after_sleep_onset_minutes": 30,
    "sleep_efficiency_pct": 90,
    "sleep_stage_deep_pct": 20,
    "sleep_stage_light_pct": 50,
    "sleep_stage_rem_pct": 25,
    "sleep_stage_awake_pct": 5,
    "heart_rate_mean_bpm": 60,
    "heart_rate_min_bpm": 50,
    "heart_rate_max_bpm": 80,
    "hrv_rmssd_ms": 45,
    "respiration_rate_bpm": 14,
    "spo2_mean_pct": 96,
    "spo2_min_pct": 92,
    "movement_count": 20,
    "snore_events": 5,
    "ambient_noise_db": 35,
    "room_temperature_c": 24,
    "room_humidity_pct": 55,
    "step_count_day": 8000,
    "caffeine_mg": 200,
    "alcohol_units": 1,
    "medication_flag": 0,
    "jetlag_hours": 0,
    "timezone": "Asia/Ho_Chi_Minh",
    "age": 65,
    "gender": "male",
    "weight_kg": 70,
    "height_cm": 170,
    "device_model": "VSmartwatch_v1",
    "bedtime_consistency_std_min": 30,
    "stress_score": 40,
    "activity_before_bed_min": 20,
    "screen_time_before_bed_min": 45,
    "insomnia_flag": 0,
    "apnea_risk_score": 10,
    "nap_duration_minutes": 0,
    "created_at": "2026-05-14T07:00:00+07:00",
}


def test_valid_sleep_accepted():
    record = SleepRecord(**VALID_SLEEP)
    assert record.user_id == "user_123"


def test_sleep_efficiency_over_100_rejected():
    data = {**VALID_SLEEP, "sleep_efficiency_pct": 150}
    with pytest.raises(ValidationError):
        SleepRecord(**data)


def test_sleep_user_id_too_long_rejected():
    data = {**VALID_SLEEP, "user_id": "x" * 100}
    with pytest.raises(ValidationError):
        SleepRecord(**data)


def test_sleep_request_max_length_100():
    records = [VALID_SLEEP] * 101
    with pytest.raises(ValidationError):
        SleepPredictionRequest(records=records)


# ---------------------------------------------------------------------------
# Fall schema range tests
# ---------------------------------------------------------------------------

def test_accel_over_160_rejected():
    with pytest.raises(ValidationError):
        AccelData(x=200, y=0, z=0)


def test_gyro_over_2000_rejected():
    with pytest.raises(ValidationError):
        GyroData(x=0, y=2500, z=0)


def test_orientation_pitch_over_180_rejected():
    with pytest.raises(ValidationError):
        OrientationData(pitch=200, roll=0, yaw=0)


def test_device_id_max_length_64():
    """device_id string capped at 64 chars."""
    valid_sample = SensorSample(
        timestamp=1000,
        accel=AccelData(x=0, y=0, z=9.8),
        gyro=GyroData(x=0, y=0, z=0),
        orientation=OrientationData(pitch=0, roll=0, yaw=0),
    )
    with pytest.raises(ValidationError):
        FallPredictionRequest(
            device_id="x" * 100,
            data=[valid_sample] * 10,
        )


def test_fall_data_max_length_500():
    """data list capped at 500 samples."""
    valid_sample = SensorSample(
        timestamp=1000,
        accel=AccelData(x=0, y=0, z=9.8),
        gyro=GyroData(x=0, y=0, z=0),
        orientation=OrientationData(pitch=0, roll=0, yaw=0),
    )
    with pytest.raises(ValidationError):
        FallPredictionRequest(
            device_id="test",
            data=[valid_sample] * 501,
        )
