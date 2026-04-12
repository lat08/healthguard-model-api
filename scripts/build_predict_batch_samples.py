#!/usr/bin/env python3
"""Tạo payload batch để test ``POST .../predict`` — nhiều mẫu **cùng một user/device**.

Output (copy-paste Swagger)::

    data/runtime/fall/cases/batch_multi_windows_one_device.json   # JSON array nhiều cửa sổ
    data/runtime/health/cases/batch_multi_vitals_same_patient.json  # {"records": [...]}
    data/runtime/sleep/cases/batch_multi_nights_one_user.json       # {"records": [...]} từ CSV

Chạy sau khi đã có ``iot_sample_cases.json`` (fall) và CSV sleep (tuỳ chọn)::

    python scripts/build_predict_batch_samples.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

FALL_CASES = ROOT / "data" / "runtime" / "fall" / "iot_sample_cases.json"
FALL_OUT = ROOT / "data" / "runtime" / "fall" / "cases" / "batch_multi_windows_one_device.json"

HEALTH_OUT = ROOT / "data" / "runtime" / "health" / "cases" / "batch_multi_vitals_same_patient.json"

SLEEP_OUT = ROOT / "data" / "runtime" / "sleep" / "cases" / "batch_multi_nights_one_user.json"

BATCH_README = ROOT / "data" / "runtime" / "BATCH_PREDICT_SAMPLES.txt"


def _load_sleep_helpers():
    p = Path(__file__).resolve().parent / "build_sleep_sample_cases.py"
    spec = importlib.util.spec_from_file_location("build_sleep_sample_cases", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod


def build_fall_multi_windows() -> None:
    if not FALL_CASES.exists():
        print("skip fall batch:", FALL_CASES, "missing")
        return
    doc = json.loads(FALL_CASES.read_text(encoding="utf-8-sig"))
    cases = doc.get("cases") or []
    windows = []
    for c in cases:
        req = c.get("request")
        if not isinstance(req, dict) or "data" not in req:
            continue
        w = json.loads(json.dumps(req))
        w["device_id"] = "wearable_batch_demo_001"
        windows.append(w)
    if len(windows) < 2:
        print("skip fall batch: not enough cases in", FALL_CASES)
        return
    FALL_OUT.parent.mkdir(parents=True, exist_ok=True)
    FALL_OUT.write_text(json.dumps(windows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", FALL_OUT, f"({len(windows)} windows, same device_id)")


def build_health_multi_records() -> None:
    """Nhiều đọc sinh hiệu liên tiếp — cùng tuổi/giới/tạng, thay đổi HR/BP/SpO2 nhẹ."""
    records = []
    base_age, base_g = 58, 1
    base_w, base_h = 78.0, 1.72
    for i in range(18):
        hr = 68 + (i % 7) * 3 + (0.5 if i % 2 else 0)
        spo2 = 97.0 - min(i, 10) * 0.35
        sbp = 118 + (i % 5) * 4
        dbp = 76 + (i % 4) * 2
        temp = 36.5 + (i % 6) * 0.08
        records.append(
            {
                "heart_rate": round(hr, 1),
                "respiratory_rate": round(14 + (i % 4), 1),
                "body_temperature": round(temp, 2),
                "spo2": round(spo2, 2),
                "systolic_blood_pressure": int(sbp),
                "diastolic_blood_pressure": int(dbp),
                "age": base_age,
                "gender": base_g,
                "weight_kg": base_w,
                "height_m": base_h,
                "derived_hrv": round(32.0 - i * 0.6, 2),
                "derived_pulse_pressure": int(sbp - dbp),
                "derived_bmi": round(base_w / (base_h**2), 2),
                "derived_map": round(dbp + (sbp - dbp) / 3.0, 2),
            }
        )
    payload = {"records": records}
    HEALTH_OUT.parent.mkdir(parents=True, exist_ok=True)
    HEALTH_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", HEALTH_OUT, f"({len(records)} vitals, same patient profile)")


def build_sleep_multi_nights(max_rows: int = 24) -> None:
    mod = _load_sleep_helpers()
    csv_path = mod._find_sleep_csv()
    if csv_path is None:
        print("skip sleep batch: no CSV")
        return
    df = pd.read_csv(csv_path)
    if "user_id" not in df.columns:
        print("skip sleep batch: no user_id column")
        return
    uid = df["user_id"].value_counts().index[0]
    sub = df[df["user_id"] == uid].copy()
    if len(sub) < 2:
        print("skip sleep batch: user has <2 rows", uid)
        return
    sub = sub.sort_values(
        ["date_recorded", "sleep_start_timestamp"],
        kind="mergesort",
    ).head(max_rows)
    api_cols = [c for c in df.columns if c not in mod._DROP_FROM_REQUEST]
    records = [mod._row_to_api_record(sub.iloc[i], api_cols) for i in range(len(sub))]
    for r in records:
        r["user_id"] = str(uid)
    payload = {"records": records}
    SLEEP_OUT.parent.mkdir(parents=True, exist_ok=True)
    SLEEP_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", SLEEP_OUT, f"({len(records)} nights, user_id={uid})")


def write_readme() -> None:
    text = """Batch samples for POST /predict only (not HTTP PATCH). Copy entire file into Swagger Request body.

1) Fall — batch_multi_windows_one_device.json
   Body is a JSON ARRAY [window, window, ...] of IMU windows (same device_id).
   POST /api/v1/fall/predict

2) Health — batch_multi_vitals_same_patient.json
   Body is {"records": [ ... ]} — many vital readings, same age/gender/anthropometrics.
   POST /api/v1/health/predict

3) Sleep — batch_multi_nights_one_user.json
   Body is {"records": [ ... ]} — consecutive nights from one user_id (from CSV).
   POST /api/v1/sleep/predict
"""
    BATCH_README.write_text(text, encoding="utf-8")
    print("wrote", BATCH_README)


def main() -> None:
    build_fall_multi_windows()
    build_health_multi_records()
    build_sleep_multi_nights()
    write_readme()


if __name__ == "__main__":
    main()
    sys.exit(0)
