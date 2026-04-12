#!/usr/bin/env python3
"""Build ``data/runtime/fall/iot_sample_cases.json`` — nhiều cửa sổ 50 mẫu để đánh giá fall vs không fall.

Chạy từ gốc repo::

    python scripts/build_fall_sample_cases.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from write_per_case_json import write_cases_payload_dir  # noqa: E402

OUT = ROOT / "data" / "runtime" / "fall" / "iot_sample_cases.json"
_FALL_CASES_README = """\
Fall — POST /api/v1/fall/predict
Mỗi file *.json là một object cửa sổ (device_id, sampling_rate, window_size, data).
Mở file → Ctrl+A → copy → dán vào Request body trên /docs.
"""
N = 50
TS0 = 1_710_000_000


def _env(floor: float, occ: float, mat: float) -> dict:
    return {
        "floor_vibration": float(floor),
        "room_occupancy": float(occ),
        "pressure_mat": float(mat),
    }


def _sample(
    ts: int,
    ax: float,
    ay: float,
    az: float,
    gx: float,
    gy: float,
    gz: float,
    pitch: float,
    roll: float,
    yaw: float,
    env: dict,
) -> dict:
    return {
        "timestamp": int(ts),
        "accel": {"x": float(ax), "y": float(ay), "z": float(az)},
        "gyro": {"x": float(gx), "y": float(gy), "z": float(gz)},
        "orientation": {"pitch": float(pitch), "roll": float(roll), "yaw": float(yaw)},
        "environment": env,
    }


def build_upright_stationary(rng: np.random.Generator) -> list[dict]:
    """Ít chuyển động, gia tốc quanh trọng trường — kỳ vọng thường là không fall."""
    data = []
    env = _env(0.08, 1.0, 85.0)
    for i in range(N):
        noise = rng.normal(0, 0.02, 6)
        data.append(
            _sample(
                TS0 + i,
                0.05 + noise[0],
                -0.02 + noise[1],
                0.98 + noise[2],
                0.5 + noise[3],
                -0.3 + noise[4],
                0.1 + noise[5],
                2.0 + rng.normal(0, 0.5),
                -1.0 + rng.normal(0, 0.5),
                180.0 + rng.normal(0, 1.0),
                env,
            )
        )
    return data


def build_walking_like(rng: np.random.Generator) -> list[dict]:
    """Dao động điều hòa nhẹ — kỳ vọng thường là hoạt động bình thường, không fall."""
    data = []
    env = _env(0.15, 1.0, 70.0)
    t = np.linspace(0, 4 * np.pi, N)
    for i in range(N):
        data.append(
            _sample(
                TS0 + i,
                0.2 + 0.08 * np.sin(t[i]) + rng.normal(0, 0.015),
                -0.1 + 0.05 * np.cos(t[i]) + rng.normal(0, 0.015),
                0.95 + 0.04 * np.sin(2 * t[i]) + rng.normal(0, 0.02),
                15 + 8 * np.sin(t[i]) + rng.normal(0, 2),
                -8 + 5 * np.cos(t[i]) + rng.normal(0, 2),
                3 + rng.normal(0, 1),
                5 + 3 * np.sin(t[i] * 0.5),
                -2 + 2 * np.cos(t[i] * 0.5),
                30 + i * 2.0,
                env,
            )
        )
    return data


def build_fall_impulse_mid(rng: np.random.Generator) -> list[dict]:
    """Yên ổn → spike gia tốc/gyro giữa cửa sổ → hạ năng lượng — mô phỏng va chạm/té."""
    data = []
    env = _env(0.25, 1.0, 95.0)
    peak_lo, peak_hi = 18, 38
    for i in range(N):
        if i < peak_lo:
            ax, ay, az = 0.06, -0.02, 0.99
            gx, gy, gz = 2.0, -1.0, 0.5
            pitch, roll, yaw = 3.0, -2.0, 10.0
        elif i <= peak_hi:
            phase = (i - peak_lo) / max(peak_hi - peak_lo, 1)
            spike = np.sin(phase * np.pi)
            ax = 0.06 + 2.5 * spike + rng.normal(0, 0.15)
            ay = -0.5 * spike + rng.normal(0, 0.15)
            az = 0.99 - 1.2 * spike + rng.normal(0, 0.2)
            gx = 2 + 180 * spike + rng.normal(0, 15)
            gy = -1 - 120 * spike + rng.normal(0, 15)
            gz = 0.5 + 60 * spike + rng.normal(0, 8)
            pitch = 3 + 40 * spike
            roll = -2 - 25 * spike
            yaw = 10 + 20 * spike
        else:
            ax, ay, az = 0.1 + rng.normal(0, 0.05), 0.05 + rng.normal(0, 0.05), 0.85
            gx, gy, gz = 5.0, -3.0, 1.0
            pitch, roll, yaw = 25.0, -18.0, 25.0
        data.append(_sample(TS0 + i, ax, ay, az, gx, gy, gz, pitch, roll, yaw, env))
    return data


def build_fall_slump_rotation(rng: np.random.Generator) -> list[dict]:
    """Tăng dần nghiêng/roll, gia tốc biến thiên — mô phỏng té dần / slump."""
    data = []
    env = _env(0.2, 1.0, 90.0)
    for i in range(N):
        u = i / max(N - 1, 1)
        roll = -5 - 55 * (u**1.4) + rng.normal(0, 2)
        pitch = 2 + 20 * u + rng.normal(0, 1.5)
        az = 0.98 - 0.35 * u + rng.normal(0, 0.04)
        ax = 0.1 + 0.25 * np.sin(u * np.pi * 3) + rng.normal(0, 0.05)
        ay = -0.05 - 0.2 * u + rng.normal(0, 0.05)
        gx = 5 + 40 * u + rng.normal(0, 4)
        gy = -10 - 30 * u + rng.normal(0, 4)
        gz = 2 + 15 * u + rng.normal(0, 2)
        data.append(
            _sample(TS0 + i, ax, ay, az, gx, gy, gz, pitch, roll, 15.0 + 5 * u, env)
        )
    return data


def _case(
    case_id: str,
    intent: str,
    description: str,
    description_vi: str,
    device_id: str,
    data: list[dict],
) -> dict:
    return {
        "id": case_id,
        "intent": intent,
        "description": description,
        "description_vi": description_vi,
        "request": {
            "device_id": device_id,
            "sampling_rate": 50,
            "window_size": N,
            "data": data,
        },
    }


def main() -> None:
    rng = np.random.default_rng(42)
    cases = [
        _case(
            "upright_stationary",
            "not_fall",
            "Low motion; acceleration near gravity; small IMU noise — expect low fall probability.",
            "Ít chuyển động, gia tốc quanh trọng trường — thường kỳ vọng xác suất fall thấp.",
            "sample_upright_0001",
            build_upright_stationary(rng),
        ),
        _case(
            "walking_like",
            "not_fall",
            "Smooth periodic motion typical of walking — expect not a fall event.",
            "Chuyển động tuần hoàn nhẹ kiểu đi bộ — thường không phải fall.",
            "sample_walk_0001",
            build_walking_like(rng),
        ),
        _case(
            "fall_impulse_mid",
            "fall_like",
            "Quiet stance then strong mid-window accel/gyro spike then damped — impact-like pattern.",
            "Đứng yên rồi spike gia tốc/gyro mạnh giữa cửa sổ rồi giảm — kiểu va đập/té.",
            "sample_impact_0001",
            build_fall_impulse_mid(rng),
        ),
        _case(
            "fall_slump_rotation",
            "fall_like",
            "Progressive roll and tilt with sustained gyro — gradual fall / slump pattern.",
            "Tăng dần roll/nghiêng và gyro kéo dài — mô phỏng té dần / slump.",
            "sample_slump_0001",
            build_fall_slump_rotation(rng),
        ),
    ]
    doc = {
        "version": 1,
        "window_samples": N,
        "cases": cases,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    cases_dir = OUT.parent / "cases"
    write_cases_payload_dir(cases_dir, cases, readme_body=_FALL_CASES_README)
    print("wrote", OUT, "cases:", [c["id"] for c in cases])
    print("wrote", cases_dir, "(per-case JSON for copy-paste)")


if __name__ == "__main__":
    main()
    sys.exit(0)
