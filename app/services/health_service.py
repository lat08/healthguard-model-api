"""Health risk inference from ``healthguard_bundle.joblib`` (preprocessor + LightGBM)."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import settings
from app.services.prediction_contract import (
    build_explanation,
    build_shap_payload,
    build_top_features,
    create_request_id,
    make_input_ref,
    make_meta,
)

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

TOP_FEATURE_PRIORITY = [
    "spo2",
    "body_temperature",
    "systolic_blood_pressure",
    "diastolic_blood_pressure",
    "heart_rate",
    "respiratory_rate",
    "derived_map",
]

PATIENT_FACING_EXCLUDED_FEATURES = [
    "age",
    "gender",
    "weight_kg",
    "height_m",
    "derived_bmi",
]

REASON_OVERRIDES = {
    "spo2": "SpO2 thap dang lam tang nguy co",
    "body_temperature": "nhiet do tang dang day muc canh bao len cao hon",
    "systolic_blood_pressure": "huyet ap tam thu dang day muc canh bao len cao hon",
    "diastolic_blood_pressure": "huyet ap tam truong dang day muc canh bao len cao hon",
}


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

    def _prepare_inputs(self, records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Health model is not loaded.")
        X = prepare_inference_frame(records, feature_order=self._feature_order)
        prepared_raw = self._bundle["preprocessor"].transform(X)
        prepared = self._prepared_frame(prepared_raw)
        return X, prepared

    def _prepared_frame(self, prepared: Any) -> pd.DataFrame:
        if isinstance(prepared, pd.DataFrame):
            return prepared
        width = int(getattr(prepared, "shape", (0, 0))[1] or 0)
        cols = self._model_input_columns()
        if len(cols) != width:
            cols = [f"Column_{idx}" for idx in range(width)]
        return pd.DataFrame(prepared, columns=cols)

    def _model_input_columns(self) -> list[str]:
        if not self._loaded or self._bundle is None:
            return [f"Column_{idx}" for idx in range(len(self._feature_order))]
        model = self._bundle["model"]
        names = getattr(model, "feature_names_in_", None)
        if names is not None:
            return [str(name) for name in names]
        count = int(getattr(model, "n_features_in_", len(self._feature_order)) or len(self._feature_order))
        return [f"Column_{idx}" for idx in range(count)]

    def _predict_probabilities(self, prepared: pd.DataFrame) -> np.ndarray:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Health model is not loaded.")
        proba = self._bundle["model"].predict_proba(prepared)
        idx = 1 if proba.shape[1] > 1 else 0
        return np.asarray(proba[:, idx], dtype=np.float64).reshape(-1)

    def _feature_names_out(self) -> list[str]:
        if not self._loaded or self._bundle is None:
            return list(self._feature_order)
        preprocessor = self._bundle["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            return [str(name) for name in preprocessor.get_feature_names_out()]
        return list(self._feature_order)

    def _shap_contributions(self, prepared: pd.DataFrame) -> np.ndarray:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Health model is not loaded.")
        booster = getattr(self._bundle["model"], "booster_", None)
        if booster is None:
            raise RuntimeError("Health model backend does not expose LightGBM SHAP contributions.")
        return np.asarray(booster.predict(prepared, pred_contrib=True), dtype=np.float64)

    def _build_prediction_rows(self, probabilities: np.ndarray) -> list[dict]:
        t = settings.health_thresholds
        results = []
        for i, probability in enumerate(probabilities):
            prob_f = float(probability)
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

    def predict(self, records: list[dict]) -> list[dict]:
        _, prepared = self._prepare_inputs(records)
        probabilities = self._predict_probabilities(prepared)
        return self._build_prediction_rows(probabilities)

    def predict_api(self, records: list[dict], request_id: str | None = None) -> list[dict]:
        X, prepared = self._prepare_inputs(records)
        probabilities = self._predict_probabilities(prepared)
        raw_results = self._build_prediction_rows(probabilities)
        shap_values = self._shap_contributions(prepared)
        feature_names = self._feature_names_out()
        req_id = request_id or create_request_id()

        results: list[dict] = []
        for row, raw_record, raw_result, shap_row in zip(
            X.to_dict(orient="records"),
            records,
            raw_results,
            shap_values,
            strict=False,
        ):
            prediction = {
                "prediction_label": raw_result["predicted_health_risk_label"],
                "prediction_score": raw_result["predicted_health_risk_probability"],
                "prediction_band": "critical"
                if raw_result["predicted_health_risk_label"] == "high_risk"
                else "normal",
                "requires_attention": raw_result["requires_attention"],
                "high_priority_alert": raw_result["high_priority_alert"],
                "confidence": raw_result["predicted_health_risk_probability"],
            }
            shap_payload = build_shap_payload(
                feature_names=feature_names,
                shap_row=shap_row,
                feature_values=row,
                higher_prediction_means_higher_risk=True,
                output_space="raw_margin",
                prediction_value=raw_result["predicted_health_risk_probability"],
            )
            top_features = build_top_features(
                shap_payload=shap_payload,
                preferred_features=TOP_FEATURE_PRIORITY,
                reason_overrides=REASON_OVERRIDES,
                excluded_features=PATIENT_FACING_EXCLUDED_FEATURES,
            )
            results.append(
                {
                    **raw_result,
                    "status": "ok",
                    "meta": make_meta(
                        model_family="healthguard",
                        model_name="healthguard",
                        artifact_path=settings.health_bundle_path,
                        request_id=req_id,
                    ),
                    "input_ref": make_input_ref(
                        user_id=str(raw_record.get("user_id")) if raw_record.get("user_id") else None,
                        device_id=str(raw_record.get("device_id")) if raw_record.get("device_id") else None,
                        event_timestamp=raw_record.get("event_timestamp"),
                    ),
                    "prediction": prediction,
                    "top_features": top_features,
                    "shap": shap_payload,
                    "explanation": build_explanation(
                        model_family="health",
                        prediction=prediction,
                        top_features=top_features,
                    ),
                }
            )
        return results


health_service = HealthModelService()
