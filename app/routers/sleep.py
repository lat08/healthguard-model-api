"""Sleep score API endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.schemas.sleep import (
    SleepPredictionRequest,
    SleepPredictionResponse,
    SleepPredictionResult,
)
from app.services.sleep_service import sleep_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/sleep", tags=["Sleep Score"])


def _load_sleep_sample_cases_document() -> dict:
    path = settings.sleep_sample_cases_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Sleep sample cases file not found. Run: python scripts/build_sleep_sample_cases.py",
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


@router.post("/predict", response_model=SleepPredictionResponse, summary="Predict Sleep Score")
async def predict_sleep(request: SleepPredictionRequest):
    if not sleep_service.is_loaded:
        raise HTTPException(status_code=503, detail="Sleep score model is not loaded.")
    try:
        records = [rec.model_dump() for rec in request.records]
        results = sleep_service.predict_api(records)
        return SleepPredictionResponse(
            results=[SleepPredictionResult(**r) for r in results],
            total=len(results),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Sleep prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@router.post("/predict/batch", response_model=SleepPredictionResponse, summary="Batch Predict Sleep Score")
async def predict_sleep_batch(request: SleepPredictionRequest):
    return await predict_sleep(request)


@router.get("/model-info", summary="Sleep Model Info")
async def sleep_model_info():
    return sleep_service.get_info()


@router.get(
    "/sample-cases",
    summary="Sleep sample cases (good vs poor nights)",
)
async def sleep_sample_cases():
    """Labeled full-night ``records`` for comparing sleep scores across scenarios."""
    return _load_sleep_sample_cases_document()


@router.get("/sample-input", summary="Sleep Sample Input")
async def sleep_sample_input(
    case: str | None = Query(
        None,
        description="Optional: case id from ``iot_sample_cases.json`` (GET /sample-cases).",
    ),
):
    if case:
        doc = _load_sleep_sample_cases_document()
        for c in doc.get("cases", []):
            if c.get("id") == case:
                req = c.get("request")
                if isinstance(req, dict):
                    return req
        raise HTTPException(
            status_code=404,
            detail=f"Unknown case '{case}'. Use GET /api/v1/sleep/sample-cases for ids.",
        )
    path = settings.sleep_sample_input_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample input file not found.")
    return json.loads(path.read_text(encoding="utf-8-sig"))
