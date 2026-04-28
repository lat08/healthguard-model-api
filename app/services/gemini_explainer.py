"""Gemini-powered explanation generator for risk and sleep predictions.

Replaces the rule-based Vietnamese-without-diacritics templates with
natural, properly accented Vietnamese via Gemini 1.5 Flash.

Design principles:
- Fire-and-forget with silent fallback: any Gemini error returns None so
  the caller falls back to the existing template without crashing.
- Sync-only: the rest of the prediction pipeline is sync; no event loop
  juggling needed.
- Single global client reused across requests (thread-safe in google-genai SDK).
- Hard timeout via a background thread join so a slow Gemini call never
  stalls the health inference for more than TIMEOUT_SECONDS.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model selection — gemini-1.5-flash: fast, cheap, stable free tier.
# ---------------------------------------------------------------------------
_GEMINI_MODEL = "gemini-2.5-flash"
_TIMEOUT_SECONDS = 12
_MAX_OUTPUT_TOKENS = 2048
_THINKING_BUDGET = 512

_client_lock = threading.Lock()
_genai_client: Any = None  # google.genai.Client | None


def _get_client() -> Any | None:
    """Lazy-init the Gemini client once; return None if SDK unavailable or key missing."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    with _client_lock:
        if _genai_client is not None:
            return _genai_client
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning("gemini_explainer: GEMINI_API_KEY not set — fallback to template")
            return None
        try:
            from google import genai
            _genai_client = genai.Client(api_key=api_key)
            logger.info("gemini_explainer: client initialised (%s)", _GEMINI_MODEL)
            return _genai_client
        except Exception as exc:
            logger.warning("gemini_explainer: init failed — %s", exc)
            return None


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_RISK_LEVEL_VI = {
    "low": "thấp",
    "medium": "trung bình",
    "high": "cao",
    "critical": "nguy hiểm",
}

_MODEL_FAMILY_VI = {
    "health": "sức khỏe tổng quát",
    "sleep": "giấc ngủ",
    "fall": "ngã",
}


def _format_features(top_features: Sequence[Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for item in top_features[:5]:
        feat = item.get("feature", "?")
        val = item.get("feature_value")
        direction = item.get("direction", "")
        arrow = "▲ tăng rủi ro" if direction == "risk_up" else "▼ giảm rủi ro"
        val_str = f" = {val:.3g}".rstrip("0").rstrip(".") if isinstance(val, float) else (f" = {val}" if val is not None else "")
        lines.append(f"- {feat}{val_str} ({arrow})")
    return "\n".join(lines) if lines else "- (không có feature nổi bật)"


def _build_prompt(
    *,
    model_family: str,
    risk_level: str,
    requires_attention: bool,
    high_priority_alert: bool,
    top_features: Sequence[Mapping[str, Any]],
) -> str:
    domain_vi = _MODEL_FAMILY_VI.get(model_family, model_family)
    level_vi = _RISK_LEVEL_VI.get(risk_level, risk_level)
    urgency = (
        "Cảnh báo KHẨN CẤP — cần hành động ngay."
        if high_priority_alert
        else "Cần theo dõi thêm."
        if requires_attention
        else "Ổn định, duy trì theo dõi định kỳ."
    )
    features_text = _format_features(top_features)

    return f"""Bạn là trợ lý AI y tế cho ứng dụng đồng hồ thông minh sức khỏe.
Dựa trên phân tích SHAP của mô hình {domain_vi}, hãy viết giải thích ngắn gọn bằng tiếng Việt CÓ DẤU đầy đủ.

Thông tin:
- Mức độ rủi ro: {level_vi}
- Trạng thái: {urgency}
- Lĩnh vực: {domain_vi}

Yếu tố ảnh hưởng chính (từ SHAP):
{features_text}

Trả về JSON hợp lệ (KHÔNG có markdown code fence):
{{
  "short_text": "1-2 câu mô tả tình trạng và nguyên nhân chính, tiếng Việt có dấu, dễ hiểu cho người dùng thông thường",
  "clinical_note": "1 câu lưu ý lâm sàng ngắn cho bác sĩ, tiếng Việt có dấu",
  "recommended_actions": ["hành động ngắn 1", "hành động ngắn 2", "hành động ngắn 3"]
}}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_explanation(
    *,
    model_family: str,
    prediction: Mapping[str, Any],
    top_features: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Call Gemini to generate a Vietnamese explanation.

    Returns a dict with keys ``short_text``, ``clinical_note``,
    ``recommended_actions`` on success, or ``None`` if Gemini is
    unavailable / errors so the caller can fall back to the template.
    """
    client = _get_client()
    if client is None:
        return None

    risk_level = str(prediction.get("prediction_band") or prediction.get("risk_level") or "medium")
    requires_attention = bool(prediction.get("requires_attention"))
    high_priority_alert = bool(prediction.get("high_priority_alert"))

    prompt = _build_prompt(
        model_family=model_family,
        risk_level=risk_level,
        requires_attention=requires_attention,
        high_priority_alert=high_priority_alert,
        top_features=top_features,
    )

    result: dict[str, Any] | None = None
    exc_holder: list[Exception] = []

    def _call() -> None:
        try:
            from google.genai import types as genai_types
            response = client.models.generate_content(
                model=_GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=_MAX_OUTPUT_TOKENS,
                    response_mime_type="application/json",
                    thinking_config=genai_types.ThinkingConfig(
                        thinking_budget=_THINKING_BUDGET,
                    ),
                ),
            )
            raw = response.text.strip()
            # Strip accidental markdown fences if model ignores mime type
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            parsed = json.loads(raw)
            nonlocal result
            actions = parsed.get("recommended_actions") or []
            result = {
                "short_text": str(parsed.get("short_text") or "").strip(),
                "clinical_note": str(parsed.get("clinical_note") or "").strip(),
                "recommended_actions": [str(a).strip() for a in actions if str(a).strip()][:3],
            }
        except Exception as exc:
            exc_holder.append(exc)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    t.join(timeout=_TIMEOUT_SECONDS)

    if t.is_alive():
        logger.warning("gemini_explainer: timed out after %ss — falling back", _TIMEOUT_SECONDS)
        return None
    if exc_holder:
        logger.warning("gemini_explainer: error — %s — falling back", exc_holder[0])
        return None
    if result and result.get("short_text"):
        logger.debug("gemini_explainer: OK — %s", result["short_text"][:60])
        return result
    return None
