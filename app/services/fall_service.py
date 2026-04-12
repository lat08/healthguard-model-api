"""Fall detection via ``fall_bundle.joblib`` (LightGBM + sklearn preprocessor)."""

from __future__ import annotations

import logging
from typing import Any

import joblib
import numpy as np

from app.config import settings
from app.services.fall_featurize import featurize_payloads

logger = logging.getLogger(__name__)


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

    def predict(self, payloads: list[dict]) -> list[dict]:
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
        probs = self._bundle["model"].predict_proba(prepared)[:, 1]
        thr_bundle = self._bundle.get("decision_threshold")
        thr = float(thr_bundle) if thr_bundle is not None else settings.fall_thresholds.fall_true_at
        ft = settings.fall_thresholds

        results = []
        for i in range(len(probs)):
            prob_f = float(probs[i])
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


fall_service = FallModelService()
