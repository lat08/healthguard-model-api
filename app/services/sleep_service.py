"""Sleep score inference from ``sleep_score_bundle.joblib`` + optional preprocessor file."""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

from app.config import settings
from app.services.prediction_contract import (
    build_explanation,
    build_shap_payload,
    build_top_features,
    create_request_id,
    make_input_ref,
    make_meta,
)
from app.services.sklearn_sleep_pickle_compat import patch_sklearn_column_transformer_for_legacy_sleep_pickle
from app.services.sleep_features import prepare_inference_frame

logger = logging.getLogger(__name__)

TOP_FEATURE_PRIORITY = [
    "sleep_efficiency_pct",
    "duration_minutes",
    "stress_score",
    "spo2_mean_pct",
    "sleep_latency_minutes",
    "wake_after_sleep_onset_minutes",
]

REASON_OVERRIDES = {
    "sleep_efficiency_pct": "hieu suat ngu thap dang lam giam sleep score",
    "stress_score": "stress cao dang lam xau danh gia giac ngu",
}

PATIENT_FACING_EXCLUDED_FEATURES = [
    "age",
    "gender",
    "weight_kg",
    "height_cm",
    "bmi",
    "device_model",
    "timezone",
]


def classify_sleep_score(score: float) -> str:
    t = settings.sleep_thresholds
    if score < t.critical_below:
        return "critical"
    if score < t.poor_below:
        return "poor"
    if score < t.fair_below:
        return "fair"
    if score < t.good_below:
        return "good"
    return "excellent"


class SleepModelService:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._preprocessor: Any = None
        self._model: Any = None
        self._metadata: dict = {}
        self._loaded = False
        self._backend = "none"

    def load(self) -> None:
        try:
            bpath = settings.sleep_bundle_path
            if not bpath.exists():
                raise FileNotFoundError(f"Missing sleep bundle at {bpath}")
            patch_sklearn_column_transformer_for_legacy_sleep_pickle()
            self._bundle = joblib.load(bpath)
            if not isinstance(self._bundle, dict):
                raise TypeError(f"Sleep bundle must be a dict, got {type(self._bundle)}")
            if "model" not in self._bundle:
                raise KeyError("Sleep bundle must contain 'model'")

            self._preprocessor = self._bundle.get("preprocessor")
            if self._preprocessor is None and settings.sleep_preprocessor_path.exists():
                patch_sklearn_column_transformer_for_legacy_sleep_pickle()
                self._preprocessor = joblib.load(settings.sleep_preprocessor_path)
            if self._preprocessor is None:
                raise RuntimeError(
                    "Sleep preprocessor missing: embed in bundle or provide sleep_score_preprocessor.joblib"
                )
            self._model = self._bundle["model"]

            if settings.sleep_metadata_path.exists():
                self._metadata = json.loads(settings.sleep_metadata_path.read_text(encoding="utf-8"))

            name = str(self._bundle.get("model_name", "catboost"))
            self._backend = "catboost" if "catboost" in name.lower() else "sklearn_regressor"
            self._loaded = True
            logger.info("Sleep bundle loaded from %s", bpath)
        except Exception as exc:
            logger.error("Failed to load sleep bundle: %s", exc)
            self._bundle = None
            self._preprocessor = None
            self._model = None
            self._loaded = False
            self._backend = "none"

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def get_info(self) -> dict:
        t = settings.sleep_thresholds
        mn = "sleep_score_regressor"
        if self._bundle and self._bundle.get("model_name"):
            mn = str(self._bundle["model_name"])
        return {
            "model_name": mn,
            "status": "loaded" if self._loaded else "unavailable",
            "inference_backend": self._backend,
            "feature_count": 42,
            "thresholds": {
                "critical_below": t.critical_below,
                "poor_below": t.poor_below,
                "fair_below": t.fair_below,
                "good_below": t.good_below,
                "attention_below": t.attention_below,
                "alert_below": t.alert_below,
            },
            "metrics": self._metadata.get("metrics", {}),
        }

    def _prepare_inputs(self, records: list[dict]) -> tuple[pd.DataFrame, np.ndarray, list[dict]]:
        if not self._loaded or self._model is None or self._preprocessor is None:
            raise RuntimeError("Sleep model is not loaded.")

        _, X = prepare_inference_frame(records)
        X_prep = self._preprocessor.transform(X)
        if hasattr(X_prep, "astype"):
            X_prep = X_prep.astype(np.float32)
        else:
            X_prep = np.asarray(X_prep, dtype=np.float32)
        return X, X_prep, X.to_dict(orient="records")

    def _feature_names_out(self) -> list[str]:
        if self._preprocessor is not None and hasattr(self._preprocessor, "get_feature_names_out"):
            return [str(name) for name in self._preprocessor.get_feature_names_out()]
        return []

    def _predict_scores(self, prepared: np.ndarray) -> np.ndarray:
        if not self._loaded or self._model is None:
            raise RuntimeError("Sleep model is not loaded.")
        raw_scores = np.asarray(self._model.predict(prepared), dtype=np.float64).reshape(-1)
        return np.clip(raw_scores, 0, 100)

    def _shap_contributions(self, prepared: np.ndarray) -> np.ndarray:
        if not self._loaded or self._model is None:
            raise RuntimeError("Sleep model is not loaded.")
        pool = Pool(prepared)
        return np.asarray(
            self._model.get_feature_importance(data=pool, type="ShapValues"),
            dtype=np.float64,
        )

    def _build_prediction_rows(self, scores: np.ndarray) -> list[dict]:
        t = settings.sleep_thresholds
        results: list[dict] = []
        for idx, score in enumerate(scores):
            score_f = round(float(score), 2)
            lbl = classify_sleep_score(score_f)
            results.append(
                {
                    "record_index": idx,
                    "predicted_sleep_score": score_f,
                    "predicted_sleep_label": lbl,
                    "risk_level": lbl,
                    "requires_attention": score_f < t.attention_below,
                    "high_priority_alert": score_f < t.alert_below,
                }
            )
        return results

    def predict(self, records: list[dict]) -> list[dict]:
        _, prepared, _ = self._prepare_inputs(records)
        clipped = self._predict_scores(prepared)
        return self._build_prediction_rows(clipped)

    def predict_api(self, records: list[dict], request_id: str | None = None) -> list[dict]:
        _, prepared, feature_rows = self._prepare_inputs(records)
        scores = self._predict_scores(prepared)
        raw_results = self._build_prediction_rows(scores)
        shap_values = self._shap_contributions(prepared)
        feature_names = self._feature_names_out()
        req_id = request_id or create_request_id()

        results: list[dict] = []
        for row, raw_record, raw_result, shap_row in zip(
            feature_rows,
            records,
            raw_results,
            shap_values,
            strict=False,
        ):
            prediction = {
                "prediction_label": raw_result["predicted_sleep_label"],
                "prediction_score": raw_result["predicted_sleep_score"],
                "prediction_band": (
                    "critical"
                    if raw_result["predicted_sleep_score"] < settings.sleep_thresholds.critical_below
                    else "warning"
                    if raw_result["predicted_sleep_score"] < settings.sleep_thresholds.poor_below
                    else "info"
                    if raw_result["predicted_sleep_score"] < settings.sleep_thresholds.fair_below
                    else "normal"
                ),
                "requires_attention": raw_result["requires_attention"],
                "high_priority_alert": raw_result["high_priority_alert"],
                "confidence": round(float(raw_result["predicted_sleep_score"]) / 100.0, 6),
            }
            shap_payload = build_shap_payload(
                feature_names=feature_names,
                shap_row=shap_row,
                feature_values=row,
                higher_prediction_means_higher_risk=False,
                output_space="prediction",
                prediction_value=raw_result["predicted_sleep_score"],
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
                        model_family="sleep",
                        model_name="sleep_score",
                        artifact_path=settings.sleep_bundle_path,
                        request_id=req_id,
                    ),
                    "input_ref": make_input_ref(
                        user_id=str(raw_record.get("user_id")) if raw_record.get("user_id") else None,
                        event_timestamp=raw_record.get("sleep_end_timestamp")
                        or raw_record.get("date_recorded"),
                    ),
                    "prediction": prediction,
                    "top_features": top_features,
                    "shap": shap_payload,
                    "explanation": build_explanation(
                        model_family="sleep",
                        prediction=prediction,
                        top_features=top_features,
                    ),
                }
            )
        return results


sleep_service = SleepModelService()
