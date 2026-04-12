"""HealthGuard vital signs request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class VitalSignsRecord(BaseModel):
    heart_rate: float
    respiratory_rate: float
    body_temperature: float
    spo2: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    age: int
    gender: int
    weight_kg: float
    height_m: float
    derived_hrv: float
    derived_pulse_pressure: float
    derived_bmi: float
    derived_map: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
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
            ]
        }
    }


class HealthPredictionRequest(BaseModel):
    records: list[VitalSignsRecord] = Field(..., min_length=1)


class HealthPredictionResult(BaseModel):
    record_index: int
    predicted_health_risk_probability: float
    predicted_health_risk_label: str
    risk_level: str
    requires_attention: bool
    high_priority_alert: bool


class HealthPredictionResponse(BaseModel):
    success: bool = True
    results: list[HealthPredictionResult]
    total: int
