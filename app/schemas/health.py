"""HealthGuard vital signs request/response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import (
    InputReference,
    PredictionExplanation,
    PredictionMeta,
    ShapDetails,
    StandardPrediction,
    TopFeature,
)


class VitalSignsRecord(BaseModel):
    """Single vital signs record for health risk inference.

    Range constraints per F6 audit (physiological bounds) + XR-003 contract.
    Pydantic 422 returned with structured error_code on out-of-range input.
    """

    heart_rate: float = Field(ge=20, le=250)
    respiratory_rate: float = Field(ge=0, le=60)
    body_temperature: float = Field(ge=30, le=45)
    spo2: float = Field(ge=50, le=100)
    systolic_blood_pressure: float = Field(ge=40, le=300)
    diastolic_blood_pressure: float = Field(ge=20, le=200)
    age: int = Field(ge=0, le=130)
    gender: int = Field(ge=0, le=1)
    weight_kg: float = Field(ge=1, le=500)
    height_m: float = Field(ge=0.3, le=2.5)
    derived_hrv: float = Field(ge=0, le=500)
    derived_pulse_pressure: float = Field(ge=0, le=200)
    derived_bmi: float = Field(ge=5, le=100)
    derived_map: float = Field(ge=30, le=250)

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
    records: list[VitalSignsRecord] = Field(..., min_length=1, max_length=100)


class HealthPredictionResult(BaseModel):
    record_index: int
    predicted_health_risk_probability: float
    predicted_health_risk_label: str
    risk_level: str
    requires_attention: bool
    high_priority_alert: bool
    status: str = "ok"
    meta: PredictionMeta
    input_ref: InputReference
    prediction: StandardPrediction
    top_features: list[TopFeature] = Field(default_factory=list)
    shap: ShapDetails | None = None
    explanation: PredictionExplanation | None = None


class HealthPredictionResponse(BaseModel):
    success: bool = True
    results: list[HealthPredictionResult]
    total: int
