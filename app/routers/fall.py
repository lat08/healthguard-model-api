"""Fall Detection API endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException, Query

from app.config import settings
from app.schemas.fall import (
    FallPredictPayload,
    FallPredictionRequest,
    FallPredictionResponse,
    FallPredictionResult,
)
from app.services.fall_service import fall_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/fall", tags=["Fall Detection"])


def _load_fall_sample_cases_document() -> dict:
    path = settings.fall_sample_cases_path
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Fall sample cases file not found. Run: python scripts/build_fall_sample_cases.py",
        )
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _request_to_payload(req: FallPredictionRequest) -> dict:
    return {
        "device_id": req.device_id,
        "sampling_rate": req.sampling_rate,
        "window_size": req.window_size,
        "data": [sample.model_dump() for sample in req.data],
    }


@router.post(
    "/predict",
    response_model=FallPredictionResponse,
    summary="Predict Fall Risk",
)
async def predict_fall(body: FallPredictPayload):
    if not fall_service.is_loaded:
        raise HTTPException(status_code=503, detail=fall_service.unavailable_detail())
    windows: list[FallPredictionRequest] = (
        [body] if isinstance(body, FallPredictionRequest) else list(body)
    )
    try:
        payloads = [_request_to_payload(w) for w in windows]
        results = fall_service.predict(payloads)
        return FallPredictionResponse(
            results=[FallPredictionResult(**r) for r in results],
            total=len(results),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Fall prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")


@router.get("/model-info", summary="Fall Model Info")
async def fall_model_info():
    return fall_service.get_info()


@router.get(
    "/sample-cases",
    summary="Fall sample cases (evaluate fall vs not_fall)",
)
async def fall_sample_cases():
    """Multiple labeled windows: ``cases[].id``, ``intent`` (``not_fall`` | ``fall_like``), and full ``request``."""
    return _load_fall_sample_cases_document()


@router.get("/sample-input", summary="Fall Sample Input (single window)")
async def fall_sample_input(
    case: str | None = Query(
        None,
        description="Optional: case id from ``iot_sample_cases.json`` (see GET /sample-cases).",
    ),
):
    if case:
        doc = _load_fall_sample_cases_document()
        for c in doc.get("cases", []):
            if c.get("id") == case:
                req = c.get("request")
                if not isinstance(req, dict):
                    break
                return req
        raise HTTPException(
            status_code=404,
            detail=f"Unknown case '{case}'. Use GET /api/v1/fall/sample-cases for available ids.",
        )
    path = settings.fall_sample_input_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample input file not found.")
    return json.loads(path.read_text(encoding="utf-8-sig"))
