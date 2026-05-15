"""FastAPI application — joblib bundles under ``models/``."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.routers import fall, health, sleep, system
from app.services.fall_service import fall_service
from app.services.health_service import health_service
from app.services.sleep_service import sleep_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

OPENAPI_DESCRIPTION = (
    "REST API for **HealthGuard** IoT monitoring.\n\n"
    "**Endpoints**\n"
    "- **Fall** — IMU windows → fall probability (LightGBM on aggregated features)\n"
    "- **Health** — 14 vital features → health risk probability (LightGBM)\n"
    "- **Sleep** — sleep session features → score 0–100 + quality label\n\n"
    "**Models (joblib)**\n"
    "- ``models/fall/fall_bundle.joblib``\n"
    "- ``models/healthguard/healthguard_bundle.joblib``\n"
    "- ``models/Sleep/sleep_score_bundle.joblib`` "
    "(+ optional ``sleep_score_preprocessor.joblib`` if not embedded)\n"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models …")
    fall_service.load()
    health_service.load()
    sleep_service.load()
    logger.info(
        "Startup complete — fall=%s  health=%s  sleep=%s",
        fall_service.backend,
        health_service.backend,
        sleep_service.backend,
    )
    yield
    logger.info("Shutting down …")


app = FastAPI(
    title=settings.app_name,
    description=OPENAPI_DESCRIPTION,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """ADR-018: structured 422 response for Pydantic validation failures.

    Replaces the default FastAPI ``{"detail": [...]}`` envelope with a
    machine-parseable shape so callers (mobile BE, IoT sim) can branch
    on ``error.code`` instead of regex-matching the message.
    """
    details = [
        {
            "field": ".".join(str(p) for p in err.get("loc", []) if p != "body"),
            "issue": err.get("msg", "validation error"),
            "type": err.get("type", "value_error"),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request body failed validation",
                "details": details,
            }
        },
    )


app.include_router(system.router)
app.include_router(fall.router)
app.include_router(health.router)
app.include_router(sleep.router)
