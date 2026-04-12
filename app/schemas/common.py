"""Common response schemas shared across all endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, TypeVar

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
