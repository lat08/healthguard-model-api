"""Sleep feature engineering — aligned with ``healthguard-ai/models/sleep/sleep_score_modeling.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "sleep_score"
GROUP_COL = "user_id"
DROP_COLS = ["user_id", "daily_label", "created_at"]


def _cyclic_encode(values: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    radians = 2 * np.pi * values / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        },
        index=values.index,
    )


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["date_recorded"] = pd.to_datetime(data["date_recorded"])
    data["sleep_start_timestamp"] = pd.to_datetime(data["sleep_start_timestamp"])
    data["sleep_end_timestamp"] = pd.to_datetime(data["sleep_end_timestamp"])
    data["created_at"] = pd.to_datetime(data["created_at"])

    data["sleep_start_hour"] = (
        data["sleep_start_timestamp"].dt.hour + data["sleep_start_timestamp"].dt.minute / 60.0
    )
    data["sleep_end_hour"] = (
        data["sleep_end_timestamp"].dt.hour + data["sleep_end_timestamp"].dt.minute / 60.0
    )
    data["recorded_weekday"] = data["date_recorded"].dt.dayofweek
    data["recorded_month"] = data["date_recorded"].dt.month
    data["is_weekend_sleep"] = (data["recorded_weekday"] >= 5).astype(int)

    mid_sleep_minutes = (
        (data["sleep_start_timestamp"] + (data["sleep_end_timestamp"] - data["sleep_start_timestamp"]) / 2).dt.hour
        * 60
        + (
            data["sleep_start_timestamp"]
            + (data["sleep_end_timestamp"] - data["sleep_start_timestamp"]) / 2
        ).dt.minute
    )
    data["mid_sleep_hour"] = mid_sleep_minutes / 60.0

    data = pd.concat(
        [
            data,
            _cyclic_encode(data["sleep_start_hour"], 24, "sleep_start"),
            _cyclic_encode(data["sleep_end_hour"], 24, "sleep_end"),
            _cyclic_encode(data["mid_sleep_hour"], 24, "mid_sleep"),
            _cyclic_encode(data["recorded_weekday"], 7, "weekday"),
            _cyclic_encode(data["recorded_month"], 12, "month"),
        ],
        axis=1,
    )

    height_m = data["height_cm"] / 100.0
    data["bmi"] = data["weight_kg"] / (height_m**2)
    data["hr_range"] = data["heart_rate_max_bpm"] - data["heart_rate_min_bpm"]
    data["spo2_range"] = data["spo2_mean_pct"] - data["spo2_min_pct"]
    data["sleep_fragmentation_index"] = (
        data["wake_after_sleep_onset_minutes"] + data["movement_count"]
    ) / data["duration_minutes"].clip(lower=1)
    data["deep_rem_ratio"] = (data["sleep_stage_deep_pct"] + data["sleep_stage_rem_pct"]) / data[
        "sleep_stage_light_pct"
    ].clip(lower=1)
    data["disturbance_load"] = (
        data["ambient_noise_db"]
        + data["screen_time_before_bed_min"] / 10
        + data["caffeine_mg"] / 40
        + data["alcohol_units"] * 3
    )
    data["recovery_index"] = (
        data["hrv_rmssd_ms"] * data["sleep_efficiency_pct"] / 100
    ) / data["stress_score"].clip(lower=1)
    data["behavioral_risk_index"] = (
        data["stress_score"]
        + data["caffeine_mg"] / 10
        + data["screen_time_before_bed_min"] / 5
        + data["jetlag_hours"].abs() * 2
        + data["insomnia_flag"] * 8
        + data["medication_flag"] * 4
    )

    data["user_bedtime_std"] = data.groupby(GROUP_COL)["sleep_start_hour"].transform("std")
    data["user_duration_median"] = data.groupby(GROUP_COL)["duration_minutes"].transform("median")
    data["user_efficiency_median"] = data.groupby(GROUP_COL)["sleep_efficiency_pct"].transform(
        "median"
    )

    data = data.drop(
        columns=[
            "sleep_start_timestamp",
            "sleep_end_timestamp",
            "date_recorded",
            "created_at",
        ]
    )

    return data


def prepare_inference_frame(records: list[dict] | pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    features_df = add_features(raw_df)
    X = features_df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    return raw_df.reset_index(drop=True), X
