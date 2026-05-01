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
    try:
        from app.services.gemini_explainer import generate_explanation
        gemini_result = generate_explanation(
            model_family=model_family,
            prediction=prediction,
            top_features=top_features,
        )
        if gemini_result:
            return gemini_result
    except Exception:
        pass
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
        detail = " và ".join(_feature_phrase(item) for item in anchors)
        if prediction.get("requires_attention"):
            short_text = f"{detail} đang làm tăng mức độ cảnh báo."
        else:
            short_text = f"{detail} là các tín hiệu đang được model ưu tiên theo dõi."
    else:
        short_text = "Không có yếu tố nổi bật để tổng hợp giải thích ngắn."

    clinical_note = (
        "Các yếu tố hàng đầu được tổng hợp từ SHAP/đóng góp nội tại; nên đối chiếu thêm với bối cảnh thực tế."
    )
    if model_family == "health":
        actions = (
            ["Đo lại chỉ số", "Đối chiếu triệu chứng", "Liên hệ nhân viên y tế"]
            if prediction.get("high_priority_alert")
            else ["Đo lại chỉ số", "Theo dõi thêm 30-60 phút"]
            if prediction.get("requires_attention")
            else ["Tiếp tục theo dõi định kỳ", "Duy trì thói quen đo"]
        )
    else:
        actions = (
            ["Kiểm tra an toàn ngay", "Xác minh với người dùng", "Khởi động quy trình cảnh báo"]
            if prediction.get("high_priority_alert")
            else ["Xác minh sự kiện", "Theo dõi thêm các cửa sổ liên tiếp"]
            if prediction.get("requires_attention")
            else ["Tiếp tục giám sát", "Đối chiếu nếu có báo động khác"]
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
        detail = " và ".join(_feature_phrase(item) for item in anchors)
        if prediction.get("requires_attention"):
            short_text = f"{detail} đang kéo điểm giấc ngủ xuống."
        else:
            short_text = f"{detail} đang hỗ trợ điểm giấc ngủ ổn định hơn."
    else:
        short_text = "Không có yếu tố nổi bật để tổng hợp giải thích ngắn."

    actions = (
        ["Xem lại hiệu suất ngủ", "Giảm stress và screen time trước khi ngủ", "Đối chiếu các yếu tố gây giấc"]
        if prediction.get("high_priority_alert")
        else ["Duy trì giờ ngủ đều", "Theo dõi xu hướng thêm vài đêm"]
        if prediction.get("requires_attention")
        else ["Duy trì thói quen ngủ đều", "Tiếp tục theo dõi xu hướng"]
    )
    return {
        "short_text": short_text,
        "clinical_note": "Diễn giải được tổng hợp từ SHAP trên điểm dự đoán; nên xem cùng xu hướng nhiều đêm liên tiếp.",
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
    verb = "tăng nguy cơ" if direction == "risk_up" else "giảm nguy cơ"
    if feature_value is None:
        return f"{feature} đang làm {verb}"
    return f"{feature}={_reason_value(feature_value)} đang làm {verb}"


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
