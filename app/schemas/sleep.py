"""Sleep score request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SleepRecord(BaseModel):
    user_id: str
    date_recorded: str
    sleep_start_timestamp: str
    sleep_end_timestamp: str
    duration_minutes: float
    sleep_latency_minutes: float
    wake_after_sleep_onset_minutes: float
    sleep_efficiency_pct: float
    sleep_stage_deep_pct: float
    sleep_stage_light_pct: float
    sleep_stage_rem_pct: float
    sleep_stage_awake_pct: float
    heart_rate_mean_bpm: float
    heart_rate_min_bpm: float
    heart_rate_max_bpm: float
    hrv_rmssd_ms: float
    respiration_rate_bpm: float
    spo2_mean_pct: float
    spo2_min_pct: float
    movement_count: float
    snore_events: float
    ambient_noise_db: float
    room_temperature_c: float
    room_humidity_pct: float
    step_count_day: float
    caffeine_mg: float
    alcohol_units: float
    medication_flag: float
    jetlag_hours: float
    timezone: str
    age: float
    gender: str
    weight_kg: float
    height_cm: float
    device_model: str
    bedtime_consistency_std_min: float
    stress_score: float
    activity_before_bed_min: float
    screen_time_before_bed_min: float
    insomnia_flag: float
    apnea_risk_score: float
    nap_duration_minutes: float
    created_at: str


class SleepPredictionRequest(BaseModel):
    records: list[SleepRecord] = Field(..., min_length=1)


class SleepPredictionResult(BaseModel):
    record_index: int
    predicted_sleep_score: float
    predicted_sleep_label: str
    risk_level: str
    requires_attention: bool
    high_priority_alert: bool


class SleepPredictionResponse(BaseModel):
    success: bool = True
    results: list[SleepPredictionResult]
    total: int
