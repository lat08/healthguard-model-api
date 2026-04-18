"""Fall detection via ``fall_bundle.joblib`` (LightGBM + sklearn preprocessor)."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np
from xgboost import DMatrix

from app.config import settings
from app.services.fall_featurize import featurize_payloads
from app.services.prediction_contract import (
    build_explanation,
    build_shap_payload,
    build_top_features,
    create_request_id,
    make_input_ref,
    make_meta,
)

logger = logging.getLogger(__name__)

TOP_FEATURE_PRIORITY = [
    "floor_vibration_mean",
    "accel_x_range",
    "accel_mag_max",
    "gyro_mag_max",
    "orientation_dispersion",
    "environment_contact_score",
]

REASON_OVERRIDES = {
    "floor_vibration_mean": "rung san tang trong luc event dang lam tang kha nang te nga",
    "accel_x_range": "bien do gia toc lon dang giong mau te nga",
    "accel_mag_max": "dinh gia toc lon dang day muc canh bao te nga len cao hon",
    "gyro_mag_max": "bien thien con quay lon dang giong event te nga",
}


def classify_fall_risk(probability: float) -> str:
    t = settings.fall_thresholds
    if probability >= t.critical_at:
        return "critical"
    if probability >= t.warning_at:
        return "warning"
    return "normal"


class FallModelService:
    def __init__(self) -> None:
        self._bundle: dict[str, Any] | None = None
        self._loaded = False
        self._backend = "none"
        self._last_load_error: str | None = None

    def load(self) -> None:
        self._last_load_error = None
        try:
            path = settings.fall_bundle_path
            if not path.exists():
                raise FileNotFoundError(f"Missing fall bundle at {path}")
            self._bundle = joblib.load(path)
            if not isinstance(self._bundle, dict):
                raise TypeError(f"Fall bundle must be a dict, got {type(self._bundle)}")
            if "preprocessor" not in self._bundle or "model" not in self._bundle:
                raise KeyError("Fall bundle must contain 'preprocessor' and 'model'")
            mcls = type(self._bundle["model"]).__name__
            if "XGB" in mcls:
                self._backend = "xgboost"
            elif "LGBM" in mcls:
                self._backend = "lightgbm"
            else:
                self._backend = "sklearn"
            self._loaded = True
            logger.info("Fall bundle loaded from %s", path)
        except Exception as exc:
            err = str(exc).strip()
            self._last_load_error = err[:500] if err else type(exc).__name__
            logger.error("Cannot load fall bundle: %s", exc)
            self._bundle = None
            self._backend = "none"
            self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def backend(self) -> str:
        return self._backend

    def unavailable_detail(self) -> str:
        if self._last_load_error:
            return f"Fall detection model is not loaded. Reason: {self._last_load_error}"
        return "Fall detection model is not loaded."

    def get_info(self) -> dict:
        info: dict[str, Any] = {
            "model_name": "fall_binary_classifier",
            "status": "loaded" if self._loaded else "unavailable",
            "inference_backend": self._backend,
            "feature_count": 0,
            "min_sequence_samples": settings.fall_min_sequence_samples,
            "thresholds": {
                "fall_true_at": settings.fall_thresholds.fall_true_at,
                "warning_at": settings.fall_thresholds.warning_at,
                "critical_at": settings.fall_thresholds.critical_at,
            },
        }
        if self._loaded and self._bundle:
            fn = self._bundle.get("feature_names") or []
            meta = self._bundle.get("metadata") or {}
            if isinstance(meta, dict) and meta.get("model_name"):
                info["model_name"] = str(meta["model_name"])
            info["feature_count"] = len(fn) if fn else info["feature_count"]
            if not info["feature_count"]:
                info["feature_count"] = getattr(
                    self._bundle["model"], "n_features_in_", 0
                ) or 0
        if not self._loaded and self._last_load_error:
            info["load_error"] = self._last_load_error
        return info

    def _prepare_inputs(self, payloads: list[dict]) -> tuple[Any, np.ndarray, list[dict]]:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Fall model is not loaded.")

        min_len = settings.fall_min_sequence_samples
        for p in payloads:
            data = p.get("data", [])
            if not data:
                raise ValueError("Fall `data` must be a non-empty list of sensor samples.")
            if len(data) < min_len:
                raise ValueError(
                    f"Fall `data` must contain at least {min_len} timesteps; got {len(data)}."
                )

        feature_names = list(self._bundle.get("feature_names") or [])
        features_df, raw_df = featurize_payloads(payloads, feature_names=feature_names or None)
        prepared = np.asarray(
            self._bundle["preprocessor"].transform(features_df),
            dtype=np.float32,
        )
        return raw_df, prepared, features_df.to_dict(orient="records")

    def _predict_probabilities(self, prepared: np.ndarray) -> np.ndarray:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Fall model is not loaded.")
        return np.asarray(self._bundle["model"].predict_proba(prepared)[:, 1], dtype=np.float64)

    def _feature_names_out(self) -> list[str]:
        if not self._loaded or self._bundle is None:
            return []
        preprocessor = self._bundle["preprocessor"]
        if hasattr(preprocessor, "get_feature_names_out"):
            return [str(name) for name in preprocessor.get_feature_names_out()]
        return [str(name) for name in self._bundle.get("feature_names") or []]

    def _shap_contributions(self, prepared: np.ndarray) -> np.ndarray:
        if not self._loaded or self._bundle is None:
            raise RuntimeError("Fall model is not loaded.")
        feature_names = self._feature_names_out()
        dmatrix = DMatrix(prepared, feature_names=feature_names or None)
        return np.asarray(
            self._bundle["model"].get_booster().predict(dmatrix, pred_contribs=True),
            dtype=np.float64,
        )

    def _build_prediction_rows(self, raw_df: Any, probabilities: np.ndarray) -> list[dict]:
        thr_bundle = self._bundle.get("decision_threshold")
        thr = float(thr_bundle) if thr_bundle is not None else settings.fall_thresholds.fall_true_at
        ft = settings.fall_thresholds

        results = []
        for i, probability in enumerate(probabilities):
            prob_f = float(probability)
            pred = prob_f >= thr
            results.append(
                {
                    "device_id": str(raw_df["device_id"].iloc[i]),
                    "sample_count": int(raw_df["sample_count"].iloc[i]),
                    "predicted_fall_probability": round(prob_f, 6),
                    "predicted_fall": pred,
                    "predicted_fall_label": "fall" if pred else "normal",
                    "risk_level": classify_fall_risk(prob_f),
                    "requires_attention": prob_f >= ft.warning_at,
                    "high_priority_alert": prob_f >= ft.critical_at,
                    "predicted_activity": None,
                    "activity_probability": None,
                }
            )
        return results

    def predict(self, payloads: list[dict]) -> list[dict]:
        raw_df, prepared, _ = self._prepare_inputs(payloads)
        probs = self._predict_probabilities(prepared)
        return self._build_prediction_rows(raw_df, probs)

    def predict_api(self, payloads: list[dict], request_id: str | None = None) -> list[dict]:
        raw_df, prepared, feature_rows = self._prepare_inputs(payloads)
        probabilities = self._predict_probabilities(prepared)
        raw_results = self._build_prediction_rows(raw_df, probabilities)
        shap_values = self._shap_contributions(prepared)
        feature_names = self._feature_names_out()
        req_id = request_id or create_request_id()
        ft = settings.fall_thresholds

        results: list[dict] = []
        for payload, row, raw_result, shap_row in zip(
            payloads,
            feature_rows,
            raw_results,
            shap_values,
            strict=False,
        ):
            probability = float(raw_result["predicted_fall_probability"])
            prediction = {
                "prediction_label": (
                    "critical_fall"
                    if probability >= ft.critical_at
                    else "likely_fall"
                    if probability >= ft.warning_at
                    else "possible_fall"
                    if probability >= ft.fall_true_at
                    else "normal"
                ),
                "prediction_score": probability,
                "prediction_band": classify_fall_risk(probability),
                "requires_attention": raw_result["requires_attention"],
                "high_priority_alert": raw_result["high_priority_alert"],
                "confidence": probability,
            }
            shap_payload = build_shap_payload(
                feature_names=feature_names,
                shap_row=shap_row,
                feature_values=row,
                higher_prediction_means_higher_risk=True,
                output_space="raw_margin",
                prediction_value=probability,
            )
            top_features = build_top_features(
                shap_payload=shap_payload,
                preferred_features=TOP_FEATURE_PRIORITY,
                reason_overrides=REASON_OVERRIDES,
            )
            event_timestamp = None
            if payload.get("data"):
                event_timestamp = payload["data"][0].get("timestamp")
            results.append(
                {
                    **raw_result,
                    "status": "ok",
                    "meta": make_meta(
                        model_family="fall",
                        model_name="fall_detection",
                        artifact_path=settings.fall_bundle_path,
                        request_id=req_id,
                    ),
                    "input_ref": make_input_ref(
                        device_id=str(payload.get("device_id")) if payload.get("device_id") else None,
                        event_timestamp=event_timestamp,
                    ),
                    "prediction": prediction,
                    "top_features": top_features,
                    "shap": shap_payload,
                    "explanation": build_explanation(
                        model_family="fall",
                        prediction=prediction,
                        top_features=top_features,
                    ),
                }
            )
        return results


fall_service = FallModelService()
