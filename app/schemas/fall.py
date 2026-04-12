"""Fall Detection request/response schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from app.config import settings


class AccelData(BaseModel):
    x: float
    y: float
    z: float


class GyroData(BaseModel):
    x: float
    y: float
    z: float


class OrientationData(BaseModel):
    pitch: float
    roll: float
    yaw: float


class EnvironmentData(BaseModel):
    floor_vibration: float = 0.0
    room_occupancy: float = 0.0
    pressure_mat: float = 0.0


class SensorSample(BaseModel):
    timestamp: int
    accel: AccelData
    gyro: GyroData
    orientation: OrientationData
    environment: EnvironmentData = Field(default_factory=EnvironmentData)


class FallPredictionRequest(BaseModel):
    device_id: str = "unknown"
    sampling_rate: int = 50
    window_size: int = 50
    data: list[SensorSample] = Field(
        ...,
        min_length=settings.fall_min_sequence_samples,
        description=(
            f"IMU window: one object per timestep, length >= {settings.fall_min_sequence_samples}."
        ),
    )


FallPredictPayload = FallPredictionRequest | Annotated[
    list[FallPredictionRequest],
    Field(min_length=1, description="Multiple windows: JSON array of FallPredictionRequest."),
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


class FallPredictionResponse(BaseModel):
    success: bool = True
    results: list[FallPredictionResult]
    total: int
