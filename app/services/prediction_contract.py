"""Helpers for building the extended API response contract with SHAP output."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np

from app.config import BASE_DIR


def create_request_id() -> str:
    return f"req_{uuid4().hex[:12]}"


def make_meta(
    *,
    model_family: str,
    model_name: str,
    artifact_path: Path,
    request_id: str,
    model_version: str = "v_current",
) -> dict[str, Any]:
    return {
        "model_family": model_family,
        "model_name": model_name,
        "model_version": model_version,
        "artifact_type": "python_bundle",
        "artifact_path": _display_path(artifact_path),
        "timestamp": datetime.now(timezone.utc),
        "request_id": request_id,
    }


def make_input_ref(
    *,
    user_id: str | None = None,
    device_id: str | None = None,
    event_timestamp: str | int | None = None,
    source_file: str | None = None,
) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "device_id": device_id,
        "event_timestamp": event_timestamp,
        "source_file": source_file,
    }


def build_shap_payload(
    *,
    feature_names: Sequence[str],
    shap_row: Sequence[float],
    feature_values: Mapping[str, Any],
    higher_prediction_means_higher_risk: bool,
    output_space: str,
    prediction_value: float | None,
) -> dict[str, Any]:
    values = np.asarray(shap_row, dtype=np.float64).reshape(-1)
    base_value: float | None = None
    if values.size == len(feature_names) + 1:
        base_value = round(float(values[-1]), 6)
        values = values[:-1]

    grouped: dict[str, dict[str, Any]] = {}
    for feature_name, shap_value in zip(feature_names, values, strict=False):
        canonical = _canonical_feature_name(feature_name, feature_values)
        entry = grouped.setdefault(
            canonical,
            {
                "feature": canonical,
                "feature_value": _json_scalar(feature_values.get(canonical)),
                "shap_value": 0.0,
            },
        )
        entry["shap_value"] += float(shap_value)

    shap_values = []
    for item in grouped.values():
        shap_value = round(float(item["shap_value"]), 6)
        risk_up = shap_value > 0 if higher_prediction_means_higher_risk else shap_value < 0
        shap_values.append(
            {
                "feature": item["feature"],
                "feature_value": item["feature_value"],
                "shap_value": shap_value,
                "impact": round(abs(shap_value), 6),
                "direction": "risk_up" if risk_up else "risk_down",
            }
        )

    shap_values.sort(key=lambda item: item["impact"], reverse=True)
    return {
        "available": True,
        "output_space": output_space,
        "base_value": base_value,
        "prediction_value": None if prediction_value is None else round(float(prediction_value), 6),
        "values": shap_values,
    }


def build_top_features(
    *,
    shap_payload: Mapping[str, Any],
    preferred_features: Sequence[str],
    reason_overrides: Mapping[str, str],
    excluded_features: Sequence[str] = (),
    limit: int = 5,
) -> list[dict[str, Any]]:
    values = list(shap_payload.get("values") or [])
    preferred = {feature: index for index, feature in enumerate(preferred_features)}
    excluded = set(excluded_features)

    filtered = [
        item
        for item in values
        if item["feature"] not in excluded and float(item["impact"]) > 0
    ]
    preferred_items = [item for item in filtered if item["feature"] in preferred]
    fallback_items = [item for item in filtered if item["feature"] not in preferred]

    preferred_items.sort(
        key=lambda item: (-float(item["impact"]), preferred.get(item["feature"], 999), item["feature"])
    )
    fallback_items.sort(key=lambda item: (-float(item["impact"]), item["feature"]))
    ordered = preferred_items + fallback_items

    top_features: list[dict[str, Any]] = []
    for item in ordered:
        top_features.append(
            {
                "feature": item["feature"],
                "feature_value": item["feature_value"],
                "impact": item["impact"],
                "direction": item["direction"],
                "reason": _build_reason(
                    feature=item["feature"],
                    feature_value=item["feature_value"],
                    direction=item["direction"],
                    reason_overrides=reason_overrides,
                ),
            }
        )
        if len(top_features) >= limit:
            break
    return top_features


def build_explanation(
    *,
    model_family: str,
    prediction: Mapping[str, Any],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if model_family == "sleep":
        return _build_sleep_explanation(prediction=prediction, top_features=top_features)
    return _build_risk_explanation(model_family=model_family, prediction=prediction, top_features=top_features)


def _build_risk_explanation(
    *,
    model_family: str,
    prediction: Mapping[str, Any],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    risk_up = [item for item in top_features if item["direction"] == "risk_up"][:2]
    anchors = risk_up or list(top_features[:2])
    if anchors:
        detail = " va ".join(_feature_phrase(item) for item in anchors)
        if prediction.get("requires_attention"):
            short_text = f"{detail} dang lam tang muc do canh bao."
        else:
            short_text = f"{detail} la cac tin hieu dang duoc model uu tien theo doi."
    else:
        short_text = "Khong co driver noi bat du de tong hop giai thich ngan."

    clinical_note = (
        "Top features duoc tong hop tu SHAP/native contribution; nen doi chieu them voi boi canh thuc te."
    )
    if model_family == "health":
        actions = (
            ["do lai chi so", "doi chieu trieu chung", "lien he nhan vien y te"]
            if prediction.get("high_priority_alert")
            else ["do lai chi so", "theo doi them 30-60 phut"]
            if prediction.get("requires_attention")
            else ["tiep tuc theo doi dinh ky", "duy tri routine do"]
        )
    else:
        actions = (
            ["kiem tra an toan ngay", "xac minh voi nguoi dung", "khoi dong quy trinh canh bao"]
            if prediction.get("high_priority_alert")
            else ["xac minh event", "theo doi them cac cua so lien tiep"]
            if prediction.get("requires_attention")
            else ["tiep tuc giam sat", "doi chieu neu co bao dong khac"]
        )
    return {
        "short_text": short_text,
        "clinical_note": clinical_note,
        "recommended_actions": actions[:3],
    }


def _build_sleep_explanation(
    *,
    prediction: Mapping[str, Any],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchors = list(top_features[:2])
    if anchors:
        detail = " va ".join(_feature_phrase(item) for item in anchors)
        if prediction.get("requires_attention"):
            short_text = f"{detail} dang keo sleep score xuong."
        else:
            short_text = f"{detail} dang ho tro sleep score on dinh hon."
    else:
        short_text = "Khong co driver noi bat du de tong hop giai thich ngan."

    actions = (
        ["xem lai hieu suat ngu", "giam stress va screen time truoc khi ngu", "doi chieu cac yeu to gay giac"]
        if prediction.get("high_priority_alert")
        else ["duy tri gio ngu deu", "theo doi xu huong them vai dem"]
        if prediction.get("requires_attention")
        else ["duy tri thoi quen ngu deu", "tiep tuc theo doi xu huong"]
    )
    return {
        "short_text": short_text,
        "clinical_note": "Dien giai duoc tong hop tu SHAP tren score du doan; nen xem cung xu huong nhieu dem lien tiep.",
        "recommended_actions": actions[:3],
    }


def _build_reason(
    *,
    feature: str,
    feature_value: Any,
    direction: str,
    reason_overrides: Mapping[str, str],
) -> str:
    if direction == "risk_up" and feature in reason_overrides:
        return reason_overrides[feature]
    verb = "tang nguy co" if direction == "risk_up" else "giam nguy co"
    if feature_value is None:
        return f"{feature} dang lam {verb}"
    return f"{feature}={_reason_value(feature_value)} dang lam {verb}"


def _feature_phrase(item: Mapping[str, Any]) -> str:
    if item.get("feature_value") is None:
        return str(item["feature"])
    return f"{item['feature']} {_reason_value(item['feature_value'])}"


def _reason_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(BASE_DIR.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _canonical_feature_name(feature_name: str, feature_values: Mapping[str, Any]) -> str:
    candidate = feature_name.split("__", 1)[1] if "__" in feature_name else feature_name
    if candidate in feature_values:
        return candidate
    matches = [key for key in feature_values if candidate.startswith(f"{key}_")]
    if matches:
        return max(matches, key=len)
    return candidate


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value
