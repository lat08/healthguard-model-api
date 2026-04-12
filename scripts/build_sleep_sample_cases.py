#!/usr/bin/env python3
"""Build sleep sample cases from ``smartwatch_sleep_dataset.csv`` + per-file POST bodies.

- ``data/runtime/sleep/iot_sample_cases.json`` — catalog (metadata + ``request``).
- ``data/runtime/sleep/cases/<id>.json`` — chỉ body ``POST /api/v1/sleep/predict`` (copy-paste Swagger).

Ưu tiên CSV: ``data/runtime/sleep/smartwatch_sleep_dataset.csv``, fallback
``data/datasets/smartwatch_sleep_dataset.csv``. Nếu không có CSV, dùng 3 mẫu tay.

    python scripts/build_sleep_sample_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from write_per_case_json import write_cases_payload_dir  # noqa: E402

OUT = ROOT / "data" / "runtime" / "sleep" / "iot_sample_cases.json"
_SLEEP_CASES_README = """\
Sleep — POST /api/v1/sleep/predict
Mỗi file *.json là body {"records": [...]} (một dòng CSV = một bản ghi).
Mở file → Ctrl+A → copy → dán Request body trên /docs.
Nguồn: smartwatch_sleep_dataset.csv (nhãn daily_label / sleep_score chỉ để đối chiếu, không gửi trong request).
"""

# CSV columns that match API ``SleepRecord`` (exclude target / label used only in descriptions).
_DROP_FROM_REQUEST = {"sleep_score", "daily_label"}

_CSV_CANDIDATES = [
    ROOT / "data" / "runtime" / "sleep" / "smartwatch_sleep_dataset.csv",
    ROOT / "data" / "datasets" / "smartwatch_sleep_dataset.csv",
]


def _sleep_record(**kw: float | str | int) -> dict:
    """Synthetic fallback row (full SleepRecord)."""
    base: dict = {
        "user_id": "sample_user",
        "date_recorded": "2026-04-01",
        "sleep_start_timestamp": "2026-04-01 22:45:00",
        "sleep_end_timestamp": "2026-04-02 06:20:00",
        "duration_minutes": 455,
        "sleep_latency_minutes": 12,
        "wake_after_sleep_onset_minutes": 18,
        "sleep_efficiency_pct": 93.8,
        "sleep_stage_deep_pct": 18.2,
        "sleep_stage_light_pct": 51.0,
        "sleep_stage_rem_pct": 21.3,
        "sleep_stage_awake_pct": 9.5,
        "heart_rate_mean_bpm": 61.4,
        "heart_rate_min_bpm": 54.1,
        "heart_rate_max_bpm": 73.2,
        "hrv_rmssd_ms": 34.6,
        "respiration_rate_bpm": 14.1,
        "spo2_mean_pct": 97.2,
        "spo2_min_pct": 95.8,
        "movement_count": 21,
        "snore_events": 1,
        "ambient_noise_db": 29.4,
        "room_temperature_c": 24.0,
        "room_humidity_pct": 49.0,
        "step_count_day": 7640,
        "caffeine_mg": 80,
        "alcohol_units": 0.0,
        "medication_flag": 0,
        "jetlag_hours": 0,
        "timezone": "Asia/Tokyo",
        "age": 31,
        "gender": "female",
        "weight_kg": 58.5,
        "height_cm": 163.0,
        "device_model": "PulsePro 3",
        "bedtime_consistency_std_min": 18.0,
        "stress_score": 27,
        "activity_before_bed_min": 20,
        "screen_time_before_bed_min": 35,
        "insomnia_flag": 0,
        "apnea_risk_score": 14,
        "nap_duration_minutes": 0,
        "created_at": "2026-04-02 06:25:00",
    }
    base.update(kw)
    return base


def _pick_row_indices(n_rows: int) -> list[int]:
    if n_rows <= 0:
        return []
    if n_rows <= 6:
        return list(range(n_rows))
    return sorted({0, n_rows // 2, n_rows - 1})


def _find_sleep_csv() -> Path | None:
    for p in _CSV_CANDIDATES:
        if p.exists():
            return p
    return None


def _row_to_api_record(row: pd.Series, api_cols: list[str]) -> dict:
    """One CSV row → one ``SleepRecord`` dict (JSON-serializable)."""
    rec: dict = {}
    str_cols = {
        "user_id",
        "date_recorded",
        "sleep_start_timestamp",
        "sleep_end_timestamp",
        "timezone",
        "gender",
        "device_model",
        "created_at",
    }
    for col in api_cols:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            rec[col] = "" if col in str_cols else 0.0
            continue
        if col in str_cols:
            rec[col] = str(v)
        elif isinstance(v, (bool, np.bool_)):
            rec[col] = float(bool(v))
        else:
            rec[col] = float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v)
    return rec


def _cases_from_csv(csv_path: Path) -> list[dict]:
    df = pd.read_csv(csv_path)
    if "daily_label" not in df.columns or "sleep_score" not in df.columns:
        return []

    api_cols = [c for c in df.columns if c not in _DROP_FROM_REQUEST]
    cases: list[dict] = []
    for label in ("poor", "fair", "good"):
        sub = df[df["daily_label"] == label].sort_values("sleep_score", kind="mergesort")
        n = len(sub)
        if n == 0:
            continue
        for j, idx in enumerate(_pick_row_indices(n)):
            row = sub.iloc[idx]
            uid = str(row["user_id"]).replace(" ", "_")
            case_id = f"csv_{label}_{j}_{uid}"
            score = float(row["sleep_score"])
            desc = (
                f"Real row from smartwatch_sleep_dataset.csv — "
                f"daily_label={label}, sleep_score={score:.0f} (reference only; not in POST body). "
                f"user_id={uid}."
            )
            desc_vi = (
                f"Dòng thật từ CSV — nhãn daily_label={label}, sleep_score={score:.0f} "
                f"(chỉ để đối chiếu; không gửi trong request). user_id={uid}."
            )
            record = _row_to_api_record(row, api_cols)
            cases.append(
                {
                    "id": case_id,
                    "intent": label,
                    "dataset_sleep_score": round(score, 2),
                    "dataset_daily_label": label,
                    "description": desc,
                    "description_vi": desc_vi,
                    "request": {"records": [record]},
                }
            )
    return cases


def _fallback_handcrafted_cases() -> list[dict]:
    return [
        {
            "id": "sleep_good_hygiene",
            "intent": "good_sleep",
            "description": "Synthetic high efficiency night.",
            "description_vi": "Đêm tổng hợp hiệu suất cao.",
            "request": {
                "records": [
                    _sleep_record(
                        user_id="sample_good_001",
                        sleep_efficiency_pct=96.0,
                        wake_after_sleep_onset_minutes=8,
                        sleep_latency_minutes=5,
                        sleep_stage_deep_pct=22.0,
                        sleep_stage_awake_pct=5.0,
                        movement_count=12,
                        stress_score=18,
                        insomnia_flag=0,
                        screen_time_before_bed_min=20,
                        caffeine_mg=20,
                    )
                ]
            },
        },
        {
            "id": "sleep_average",
            "intent": "moderate_sleep",
            "description": "Synthetic moderate night.",
            "description_vi": "Đêm tổng hợp trung bình.",
            "request": {
                "records": [
                    _sleep_record(
                        user_id="sample_avg_001",
                        sleep_efficiency_pct=84.0,
                        wake_after_sleep_onset_minutes=35,
                        sleep_latency_minutes=22,
                        sleep_stage_deep_pct=14.0,
                        sleep_stage_awake_pct=14.0,
                        movement_count=35,
                        stress_score=45,
                        insomnia_flag=0,
                        apnea_risk_score=28,
                    )
                ]
            },
        },
        {
            "id": "sleep_poor_fragmented",
            "intent": "poor_sleep",
            "description": "Synthetic fragmented poor night.",
            "description_vi": "Đêm tổng hợp phân mảnh / kém.",
            "request": {
                "records": [
                    _sleep_record(
                        user_id="sample_poor_001",
                        sleep_efficiency_pct=62.0,
                        wake_after_sleep_onset_minutes=95,
                        sleep_latency_minutes=55,
                        sleep_stage_deep_pct=6.0,
                        sleep_stage_light_pct=58.0,
                        sleep_stage_rem_pct=10.0,
                        sleep_stage_awake_pct=26.0,
                        movement_count=78,
                        stress_score=72,
                        insomnia_flag=1,
                        screen_time_before_bed_min=120,
                        caffeine_mg=220,
                        alcohol_units=1.5,
                        apnea_risk_score=55,
                        heart_rate_mean_bpm=78.0,
                        hrv_rmssd_ms=18.0,
                    )
                ]
            },
        },
    ]


def main() -> None:
    csv_path = _find_sleep_csv()
    if csv_path is not None:
        cases = _cases_from_csv(csv_path)
        if not cases:
            cases = _fallback_handcrafted_cases()
            print("warn: CSV present but no cases built; using fallback handcrafted")
        else:
            print("built", len(cases), "cases from", csv_path)
    else:
        cases = _fallback_handcrafted_cases()
        print("warn: no smartwatch_sleep_dataset.csv found; using fallback handcrafted")

    doc = {"version": 1, "source_csv": str(csv_path) if csv_path else None, "cases": cases}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    cases_dir = OUT.parent / "cases"
    write_cases_payload_dir(cases_dir, cases, readme_body=_SLEEP_CASES_README)
    print("wrote", OUT)
    print("wrote", cases_dir, f"({len(cases)} files)")


if __name__ == "__main__":
    main()
    sys.exit(0)
