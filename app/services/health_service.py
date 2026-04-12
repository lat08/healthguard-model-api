"""Health risk inference from ``healthguard_bundle.joblib`` (preprocessor + LightGBM)."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

FEATURE_ORDER = [
    "heart_rate",
    "respiratory_rate",
    "body_temperature",
    "spo2",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "age",
    "gender",
    "weight_kg",
    "height_m",
    "derived_hrv",
    "derived_pulse_pressure",
    "derived_bmi",
    "derived_map",
]


def prepare_inference_frame(
    records: list[dict] | pd.DataFrame,
    feature_order: list[str] | None = None,
) -> pd.DataFrame:
    order = feature_order or list(FEATURE_ORDER)
    df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    required = set(order)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required keys: {missing}")
    return df[order].copy()


def classify_health_risk(probability: float) -> str:
    t = settings.health_thresholds
    if probability >= t.critical_at:
        return "critical"
    if probability >= t.warning_at:
        return "warning"
    return "normal"


class HealthModelService:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._feature_order: list[str] = list(FEATURE_ORDER)
        self._loaded = False
        self._backend = "none"

    def load(self) -> None:
        try:
            path = settings.health_bundle_path
            if not path.exists():
                raise FileNotFoundError(f"Missing health bundle at {path}")
            self._bundle = joblib.load(path)
            if not isinstance(self._bundle, dict):
                raise TypeError(f"Health bundle must be a dict, got {type(self._bundle)}")
            if "preprocessor" not in self._bundle or "model" not in self._bundle:
                raise KeyError("Health bundle must contain 'preprocessor' and 'model'")
            fn = self._bundle.get("feature_names")
            if isinstance(fn, list) and fn and set(fn) == set(FEATURE_ORDER):
                self._feature_order = [str(x) for x in fn]
            else:
                self._feature_order = list(FEATURE_ORDER)
            mcls = type(self._bundle["model"]).__name__
            self._backend = "lightgbm" if "LGBM" in mcls else "sklearn_classifier"
            self._loaded = True
            logger.info("Health bundle loaded from %s", path)
        except Exception as exc:
            logger.error("Failed to load health bundle: %s", exc)
            self._bundle = None
            self._loaded = False
            self._backend = "none"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def get_info(self) -> dict:
        return {
            "model_name": "healthguard_lightgbm",
            "status": "loaded" if self._loaded else "unavailable",
            "inference_backend": self._backend,
            "feature_count": 14,
            "thresholds": {
                "high_risk_true_at": settings.health_thresholds.high_risk_true_at,
                "warning_at": settings.health_thresholds.warning_at,
                "critical_at": settings.health_thresholds.critical_at,
            },
        }

    def predict(self, records: list[dict]) -> list[dict]:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Health model is not loaded.")

        X = prepare_inference_frame(records, feature_order=self._feature_order)
        prepared = self._bundle["preprocessor"].transform(X)
        proba = self._bundle["model"].predict_proba(prepared)
        idx = 1 if proba.shape[1] > 1 else 0
        t = settings.health_thresholds
        results = []
        for i in range(len(records)):
            prob_f = float(proba[i, idx])
            results.append(
                {
                    "record_index": i,
                    "predicted_health_risk_probability": round(prob_f, 6),
                    "predicted_health_risk_label": (
                        "high_risk" if prob_f >= t.high_risk_true_at else "normal"
                    ),
                    "risk_level": classify_health_risk(prob_f),
                    "requires_attention": prob_f >= t.warning_at,
                    "high_priority_alert": prob_f >= t.critical_at,
                }
            )
        return results


health_service = HealthModelService()
