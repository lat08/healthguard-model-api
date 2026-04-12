"""Sleep score inference from ``sleep_score_bundle.joblib`` + optional preprocessor file."""

from __future__ import annotations

import json
import logging
from typing import Any

import joblib
import numpy as np

from app.config import settings
from app.services.sklearn_sleep_pickle_compat import patch_sklearn_column_transformer_for_legacy_sleep_pickle
from app.services.sleep_features import prepare_inference_frame

logger = logging.getLogger(__name__)


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

    def predict(self, records: list[dict]) -> list[dict]:
        if not self._loaded or self._model is None or self._preprocessor is None:
            raise RuntimeError("Sleep model is not loaded.")

        _, X = prepare_inference_frame(records)
        X_prep = self._preprocessor.transform(X)
        if hasattr(X_prep, "astype"):
            X_prep = X_prep.astype(np.float32)
        raw_scores = np.asarray(self._model.predict(X_prep), dtype=np.float64).reshape(-1)
        clipped = np.clip(raw_scores, 0, 100)

        t = settings.sleep_thresholds
        results: list[dict] = []
        for idx, score in enumerate(clipped):
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


sleep_service = SleepModelService()
