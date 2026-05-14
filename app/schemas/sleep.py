"""Sleep score request / response schemas."""

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


class SleepRecord(BaseModel):
    """Single sleep session record for sleep quality inference.

    Range constraints per F6 audit + XR-003 contract.
    String fields bounded max_length=64 to prevent DoS via oversized payloads.
    """

    user_id: str = Field(max_length=64)
    date_recorded: str = Field(max_length=32)
    sleep_start_timestamp: str = Field(max_length=32)
    sleep_end_timestamp: str = Field(max_length=32)
    duration_minutes: float = Field(ge=0, le=1440)
    sleep_latency_minutes: float = Field(ge=0, le=600)
    wake_after_sleep_onset_minutes: float = Field(ge=0, le=1440)
    sleep_efficiency_pct: float = Field(ge=0, le=100)
    sleep_stage_deep_pct: float = Field(ge=0, le=100)
    sleep_stage_light_pct: float = Field(ge=0, le=100)
    sleep_stage_rem_pct: float = Field(ge=0, le=100)
    sleep_stage_awake_pct: float = Field(ge=0, le=100)
    heart_rate_mean_bpm: float = Field(ge=0, le=300)
    heart_rate_min_bpm: float = Field(ge=0, le=300)
    heart_rate_max_bpm: float = Field(ge=0, le=300)
    hrv_rmssd_ms: float = Field(ge=0, le=500)
    respiration_rate_bpm: float = Field(ge=0, le=60)
    spo2_mean_pct: float = Field(ge=50, le=100)
    spo2_min_pct: float = Field(ge=50, le=100)
    movement_count: float = Field(ge=0)
    snore_events: float = Field(ge=0)
    ambient_noise_db: float = Field(ge=0, le=150)
    room_temperature_c: float = Field(ge=-10, le=50)
    room_humidity_pct: float = Field(ge=0, le=100)
    step_count_day: float = Field(ge=0, le=100000)
    caffeine_mg: float = Field(ge=0, le=2000)
    alcohol_units: float = Field(ge=0, le=100)
    medication_flag: float = Field(ge=0, le=1)
    jetlag_hours: float = Field(ge=-12, le=12)
    timezone: str = Field(max_length=64)
    age: float = Field(ge=0, le=130)
    gender: str = Field(max_length=16)
    weight_kg: float = Field(ge=1, le=500)
    height_cm: float = Field(ge=30, le=250)
    device_model: str = Field(max_length=64)
    bedtime_consistency_std_min: float = Field(ge=0, le=600)
    stress_score: float = Field(ge=0, le=100)
    activity_before_bed_min: float = Field(ge=0, le=600)
    screen_time_before_bed_min: float = Field(ge=0, le=600)
    insomnia_flag: float = Field(ge=0, le=1)
    apnea_risk_score: float = Field(ge=0, le=100)
    nap_duration_minutes: float = Field(ge=0, le=600)
    created_at: str = Field(max_length=32)


class SleepPredictionRequest(BaseModel):
    records: list[SleepRecord] = Field(..., min_length=1, max_length=100)


class SleepPredictionResult(BaseModel):
    record_index: int
    predicted_sleep_score: float
    predicted_sleep_label: str
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


class SleepPredictionResponse(BaseModel):
    success: bool = True
    results: list[SleepPredictionResult]
    total: int
