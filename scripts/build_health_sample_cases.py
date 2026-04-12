#!/usr/bin/env python3
"""Build ``data/runtime/health/iot_sample_cases.json`` — nhiều bản ghi sinh hiệu để đánh giá rủi ro.

    python scripts/build_health_sample_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from write_per_case_json import write_cases_payload_dir  # noqa: E402

OUT = ROOT / "data" / "runtime" / "health" / "iot_sample_cases.json"
_HEALTH_CASES_README = """\
Health — POST /api/v1/health/predict
Mỗi file *.json là body {"records": [...]} đúng API.
Mở file → Ctrl+A → copy → dán Request body trên /docs.
"""


def _record(
    *,
    heart_rate: float,
    respiratory_rate: float,
    body_temperature: float,
    spo2: float,
    systolic_blood_pressure: float,
    diastolic_blood_pressure: float,
    age: int,
    gender: int,
    weight_kg: float,
    height_m: float,
    derived_hrv: float,
    derived_pulse_pressure: float,
    derived_bmi: float,
    derived_map: float,
) -> dict:
    return {
        "heart_rate": heart_rate,
        "respiratory_rate": respiratory_rate,
        "body_temperature": body_temperature,
        "spo2": spo2,
        "systolic_blood_pressure": systolic_blood_pressure,
        "diastolic_blood_pressure": diastolic_blood_pressure,
        "age": age,
        "gender": gender,
        "weight_kg": weight_kg,
        "height_m": height_m,
        "derived_hrv": derived_hrv,
        "derived_pulse_pressure": derived_pulse_pressure,
        "derived_bmi": derived_bmi,
        "derived_map": derived_map,
    }


def main() -> None:
    cases = [
        {
            "id": "vitals_normal_young",
            "intent": "low_risk",
            "description": "Young adult, vitals in typical resting range — expect lower health risk probability.",
            "description_vi": "Người trẻ, chỉ số trong ngưỡng nghỉ ngơi bình thường — thường xác suất rủi ro thấp.",
            "request": {
                "records": [
                    _record(
                        heart_rate=68,
                        respiratory_rate=14,
                        body_temperature=36.6,
                        spo2=98.0,
                        systolic_blood_pressure=118,
                        diastolic_blood_pressure=76,
                        age=28,
                        gender=0,
                        weight_kg=70.0,
                        height_m=1.75,
                        derived_hrv=45.0,
                        derived_pulse_pressure=42,
                        derived_bmi=22.9,
                        derived_map=90.0,
                    )
                ]
            },
        },
        {
            "id": "vitals_borderline",
            "intent": "moderate",
            "description": "Mildly elevated HR/RR and borderline BP — ambiguous zone for models.",
            "description_vi": "HR/RR hơi cao, huyết áp ranh giới — vùng xám cho mô hình.",
            "request": {
                "records": [
                    _record(
                        heart_rate=95,
                        respiratory_rate=20,
                        body_temperature=37.1,
                        spo2=95.5,
                        systolic_blood_pressure=132,
                        diastolic_blood_pressure=86,
                        age=52,
                        gender=1,
                        weight_kg=82.0,
                        height_m=1.70,
                        derived_hrv=28.0,
                        derived_pulse_pressure=46,
                        derived_bmi=28.4,
                        derived_map=101.3,
                    )
                ]
            },
        },
        {
            "id": "vitals_high_risk_pattern",
            "intent": "high_risk",
            "description": "Fever, hypoxia, tachycardia, hypertension — pattern often associated with high risk.",
            "description_vi": "Sốt, SpO2 thấp, tim đập nhanh, HA cao — thường tăng xác suất rủi ro.",
            "request": {
                "records": [
                    _record(
                        heart_rate=118,
                        respiratory_rate=24,
                        body_temperature=38.2,
                        spo2=92.5,
                        systolic_blood_pressure=148,
                        diastolic_blood_pressure=96,
                        age=67,
                        gender=1,
                        weight_kg=73.5,
                        height_m=1.68,
                        derived_hrv=24.0,
                        derived_pulse_pressure=52,
                        derived_bmi=26.0,
                        derived_map=113.3,
                    )
                ]
            },
        },
    ]
    doc = {"version": 1, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cases_dir = OUT.parent / "cases"
    write_cases_payload_dir(cases_dir, cases, readme_body=_HEALTH_CASES_README)
    print("wrote", OUT, "cases:", [c["id"] for c in cases])
    print("wrote", cases_dir, "(per-case JSON for copy-paste)")


if __name__ == "__main__":
    main()
    sys.exit(0)
