"""Fall window featurization — aligned with ``healthguard-ai/models/fall/fall_modeling.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLING_RATE = 50

FALL_LABELS = {"fall_backward", "fall_forward", "fall_side_left", "fall_side_right", "fall_slump"}

SEQUENCE_SIGNAL_COLUMNS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "pitch",
    "roll",
    "yaw",
    "floor_vibration",
    "room_occupancy",
    "pressure_mat",
    "accel_mag",
    "gyro_mag",
    "accel_delta_mag",
    "gyro_delta_mag",
]
STAT_NAMES = ["mean", "std", "min", "max", "median", "q25", "q75", "range", "energy", "slope"]
SENSOR_ONLY_EXCLUDE_PREFIXES = ("floor_vibration_", "room_occupancy_", "pressure_mat_")
SENSOR_ONLY_EXCLUDE_EXACT = {"environment_contact_score"}


def add_frame_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(["sequence_id", "timestep"]).reset_index(drop=True)
    data["accel_mag"] = np.sqrt(data["accel_x"] ** 2 + data["accel_y"] ** 2 + data["accel_z"] ** 2)
    data["gyro_mag"] = np.sqrt(data["gyro_x"] ** 2 + data["gyro_y"] ** 2 + data["gyro_z"] ** 2)
    accel_diff = data.groupby("sequence_id")[["accel_x", "accel_y", "accel_z"]].diff().fillna(0.0)
    gyro_diff = data.groupby("sequence_id")[["gyro_x", "gyro_y", "gyro_z"]].diff().fillna(0.0)
    data["accel_delta_mag"] = np.sqrt((accel_diff**2).sum(axis=1))
    data["gyro_delta_mag"] = np.sqrt((gyro_diff**2).sum(axis=1))
    if "label" in data.columns:
        data["is_fall"] = data["label"].isin(FALL_LABELS).astype(int)
    return data


def summarize_series(values: np.ndarray, timesteps: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    t = np.asarray(timesteps, dtype=float)
    if arr.size == 0:
        return {name: 0.0 for name in STAT_NAMES}
    q25, q75 = np.percentile(arr, [25, 75])
    slope = float(np.polyfit(t, arr, deg=1)[0]) if arr.size > 1 and np.ptp(t) > 0 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "q25": float(q25),
        "q75": float(q75),
        "range": float(arr.max() - arr.min()),
        "energy": float(np.mean(arr**2)),
        "slope": slope,
    }


def extract_sequence_features(group: pd.DataFrame) -> pd.Series:
    timesteps = group["timestep"].to_numpy(dtype=float)
    features: dict[str, float | int | str] = {
        "sequence_id": int(group["sequence_id"].iloc[0]),
        "sequence_length": int(len(group)),
    }
    if "label" in group.columns:
        features["label"] = str(group["label"].iloc[0])
        features["is_fall"] = int(group["is_fall"].max())
    for column in SEQUENCE_SIGNAL_COLUMNS:
        stats = summarize_series(group[column].to_numpy(dtype=float), timesteps)
        for stat_name, value in stats.items():
            features[f"{column}_{stat_name}"] = value
    accel_mag = group["accel_mag"].to_numpy(dtype=float)
    gyro_mag = group["gyro_mag"].to_numpy(dtype=float)
    peak_index = int(np.argmax(accel_mag)) if accel_mag.size else 0
    impact_slice = accel_mag[peak_index:] if accel_mag.size else np.array([0.0])
    features["accel_peak_index_ratio"] = float(peak_index / max(len(accel_mag) - 1, 1))
    features["accel_peak_to_mean"] = float(accel_mag.max() / max(accel_mag.mean(), 1e-6))
    features["gyro_peak_to_mean"] = float(gyro_mag.max() / max(gyro_mag.mean(), 1e-6))
    features["post_impact_accel_mean"] = float(impact_slice[:10].mean())
    features["post_impact_accel_std"] = float(impact_slice[:10].std(ddof=0))
    features["environment_contact_score"] = float(
        group["pressure_mat"].mean() * 0.5
        + group["floor_vibration"].mean() * 0.3
        + group["room_occupancy"].mean() * 0.2
    )
    features["orientation_dispersion"] = float(group[["pitch", "roll", "yaw"]].std(ddof=0).sum())
    features["motion_stability_ratio"] = float(
        (group["accel_delta_mag"].mean() + 1e-6) / (group["accel_mag"].mean() + 1e-6)
    )
    return pd.Series(features)


def build_sequence_dataset(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        extract_sequence_features(group) for _, group in frame_df.groupby("sequence_id", sort=False)
    ]
    return pd.DataFrame(rows).reset_index(drop=True)


def _normalize_sample(sample: dict) -> dict[str, float]:
    accel = sample.get("accel", {})
    gyro = sample.get("gyro", {})
    orientation = sample.get("orientation", {})
    environment = sample.get("environment", {})
    return {
        "accel_x": float(accel.get("x", 0.0)),
        "accel_y": float(accel.get("y", 0.0)),
        "accel_z": float(accel.get("z", 0.0)),
        "gyro_x": float(gyro.get("x", 0.0)),
        "gyro_y": float(gyro.get("y", 0.0)),
        "gyro_z": float(gyro.get("z", 0.0)),
        "pitch": float(orientation.get("pitch", 0.0)),
        "roll": float(orientation.get("roll", 0.0)),
        "yaw": float(orientation.get("yaw", 0.0)),
        "floor_vibration": float(environment.get("floor_vibration", 0.0)),
        "room_occupancy": float(environment.get("room_occupancy", 0.0)),
        "pressure_mat": float(environment.get("pressure_mat", 0.0)),
    }


def featurize_payloads(
    payloads: list[dict],
    feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    raw_rows = []
    for idx, payload in enumerate(payloads):
        device_id = payload.get("device_id", f"device_{idx:04d}")
        samples = payload.get("data", [])
        for timestep, sample in enumerate(samples):
            row = _normalize_sample(sample)
            row["sequence_id"] = idx
            row["timestep"] = timestep
            rows.append(row)
        raw_rows.append(
            {
                "device_id": device_id,
                "sample_count": len(samples),
                "sampling_rate": int(payload.get("sampling_rate", SAMPLING_RATE)),
                "window_size": int(payload.get("window_size", len(samples))),
            }
        )
    frame_df = pd.DataFrame(rows)
    raw_df = pd.DataFrame(raw_rows)
    if frame_df.empty:
        return pd.DataFrame(columns=feature_names or []), raw_df
    frame_df = add_frame_features(frame_df)
    seq_df = build_sequence_dataset(frame_df)
    features_df = seq_df.drop(columns=["sequence_id", "label", "is_fall"], errors="ignore")
    if feature_names is not None:
        features_df = features_df.reindex(columns=feature_names, fill_value=0.0)
    return features_df, raw_df
