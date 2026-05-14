"""HealthGuard vital-signs API endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.dependencies import verify_internal_secret
from app.schemas.health import (
    HealthPredictionRequest,
    HealthPredictionResponse,
    HealthPredictionResult,
)
from app.services.health_service import health_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/health", tags=["Health Risk"])


def _load_health_sample_cases_document() -> dict:
    path = settings.health_sample_cases_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Health sample cases file not found. Run: python scripts/build_health_sample_cases.py",
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


@router.post(
    "/predict",
    response_model=HealthPredictionResponse,
    summary="Predict Health Risk",
    dependencies=[Depends(verify_internal_secret)],
)
async def predict_health(request: HealthPredictionRequest):
    if not health_service.is_loaded:
        raise HTTPException(status_code=503, detail="Health risk model is not loaded.")
    try:
        records = [rec.model_dump() for rec in request.records]
        results = health_service.predict_api(records)
        return HealthPredictionResponse(
            results=[HealthPredictionResult(**r) for r in results],
            total=len(results),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        logger.exception("Health prediction failed")
        raise HTTPException(status_code=500, detail="Prediction error")


@router.post(
    "/predict/batch",
    response_model=HealthPredictionResponse,
    summary="Batch Predict Health Risk",
    dependencies=[Depends(verify_internal_secret)],
)
async def predict_health_batch(request: HealthPredictionRequest):
    return await predict_health(request)


@router.get("/model-info", summary="Health Model Info")
async def health_model_info():
    return health_service.get_info()


@router.get(
    "/sample-cases",
    summary="Health sample cases (low vs high risk vitals)",
)
async def health_sample_cases():
    """Labeled ``records`` payloads for comparing model output across scenarios."""
    return _load_health_sample_cases_document()


@router.get("/sample-input", summary="Health Sample Input")
async def health_sample_input(
    case: str | None = Query(
        None,
        description="Optional: case id from ``iot_sample_cases.json`` (GET /sample-cases).",
    ),
):
    if case:
        doc = _load_health_sample_cases_document()
        for c in doc.get("cases", []):
            if c.get("id") == case:
                req = c.get("request")
                if isinstance(req, dict):
                    return req
        raise HTTPException(
            status_code=404,
            detail=f"Unknown case '{case}'. Use GET /api/v1/health/sample-cases for ids.",
        )
    path = settings.health_sample_input_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample input file not found.")
    return json.loads(path.read_text(encoding="utf-8-sig"))
