"""Common response schemas shared across all endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    error: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str
    status: str
    inference_backend: str
    feature_count: int
    thresholds: dict
    load_error: str | None = Field(
        default=None,
        description="Startup load failure message when status is unavailable.",
    )


class HealthCheckResponse(BaseModel):
    status: str
    models: dict[str, ModelInfo]
    version: str


class PredictionMeta(BaseModel):
    model_family: str
    model_name: str
    model_version: str
    artifact_type: str
    artifact_path: str
    timestamp: datetime
    request_id: str


class InputReference(BaseModel):
    user_id: str | None = None
    device_id: str | None = None
    event_timestamp: str | int | None = None
    source_file: str | None = None


class StandardPrediction(BaseModel):
    prediction_label: str
    prediction_score: float
    prediction_band: str
    requires_attention: bool
    high_priority_alert: bool
    confidence: float


class TopFeature(BaseModel):
    feature: str
    feature_value: Any | None = None
    impact: float
    direction: Literal["risk_up", "risk_down"]
    reason: str


class ShapFeatureValue(BaseModel):
    feature: str
    feature_value: Any | None = None
    shap_value: float
    impact: float
    direction: Literal["risk_up", "risk_down"]


class ShapDetails(BaseModel):
    available: bool = True
    output_space: str
    base_value: float | None = None
    prediction_value: float | None = None
    values: list[ShapFeatureValue] = Field(default_factory=list)


class PredictionExplanation(BaseModel):
    short_text: str
    clinical_note: str
    recommended_actions: list[str] = Field(default_factory=list)
