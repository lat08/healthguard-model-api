"""System / health-check endpoints."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from app.config import settings
from app.schemas.common import HealthCheckResponse, ModelInfo
from app.services.fall_service import fall_service
from app.services.health_service import health_service
from app.services.sleep_service import sleep_service

router = APIRouter(tags=["System"])


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@router.get("/healthz", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    fall_info = ModelInfo(**fall_service.get_info())
    health_info = ModelInfo(**health_service.get_info())
    sleep_info = ModelInfo(**sleep_service.get_info())

    models = {"fall": fall_info, "health": health_info, "sleep": sleep_info}

    loaded_count = sum(1 for m in models.values() if m.status == "loaded")
    if loaded_count == len(models):
        status = "healthy"
    elif loaded_count > 0:
        status = "degraded"
    else:
        status = "unhealthy"

    return HealthCheckResponse(
        status=status,
        models=models,
        version=settings.app_version,
    )


@router.get("/api/v1/models", summary="List All Models")
async def list_models():
    return {
        "models": {
            "fall_detection": fall_service.get_info(),
            "health_risk": health_service.get_info(),
            "sleep_score": sleep_service.get_info(),
        }
    }
