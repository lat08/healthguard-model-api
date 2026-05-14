"""Fall Detection request/response schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.common import (
    InputReference,
    PredictionExplanation,
    PredictionMeta,
    ShapDetails,
    StandardPrediction,
    TopFeature,
)


class AccelData(BaseModel):
    """Accelerometer reading (m/s^2). Typical consumer IMU range +/-16g ~ +/-157 m/s^2."""

    x: float = Field(ge=-160, le=160)
    y: float = Field(ge=-160, le=160)
    z: float = Field(ge=-160, le=160)


class GyroData(BaseModel):
    """Gyroscope reading (degrees per second). Typical range +/-2000 dps."""

    x: float = Field(ge=-2000, le=2000)
    y: float = Field(ge=-2000, le=2000)
    z: float = Field(ge=-2000, le=2000)


class OrientationData(BaseModel):
    """Euler angles (degrees)."""

    pitch: float = Field(ge=-180, le=180)
    roll: float = Field(ge=-180, le=180)
    yaw: float = Field(ge=-180, le=360)


class EnvironmentData(BaseModel):
    floor_vibration: float = Field(default=0.0, ge=0, le=100)
    room_occupancy: float = Field(default=0.0, ge=0, le=10)
    pressure_mat: float = Field(default=0.0, ge=0, le=1)


class SensorSample(BaseModel):
    timestamp: int = Field(ge=0)
    accel: AccelData
    gyro: GyroData
    orientation: OrientationData
    environment: EnvironmentData = Field(default_factory=EnvironmentData)


class FallPredictionRequest(BaseModel):
    device_id: str = Field(default="unknown", max_length=64)
    sampling_rate: int = Field(default=50, ge=1, le=200)
    window_size: int = Field(default=50, ge=1, le=10000)
    data: list[SensorSample] = Field(
        ...,
        min_length=settings.fall_min_sequence_samples,
        max_length=500,
        description=(
            f"IMU window: one object per timestep, length >= {settings.fall_min_sequence_samples}."
        ),
    )


FallPredictPayload = FallPredictionRequest | Annotated[
    list[FallPredictionRequest],
    Field(min_length=1, max_length=100, description="Multiple windows: JSON array of FallPredictionRequest."),
]


class FallPredictionResult(BaseModel):
    device_id: str
    sample_count: int
    predicted_fall_probability: float
    predicted_fall: bool
    predicted_fall_label: str
    risk_level: str
    requires_attention: bool
    high_priority_alert: bool
    predicted_activity: str | None = None
    activity_probability: float | None = None
    status: str = "ok"
    meta: PredictionMeta
    input_ref: InputReference
    prediction: StandardPrediction
    top_features: list[TopFeature] = Field(default_factory=list)
    shap: ShapDetails | None = None
    explanation: PredictionExplanation | None = None


class FallPredictionResponse(BaseModel):
    success: bool = True
    results: list[FallPredictionResult]
    total: int
