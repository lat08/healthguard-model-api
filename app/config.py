"""Application configuration. Inference bundles load from ``models/`` (joblib)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_ROOT = BASE_DIR / "models"
RUNTIME_DATA_DIR = BASE_DIR / "data" / "runtime"
RUNTIME_FALL_DIR = RUNTIME_DATA_DIR / "fall"
RUNTIME_HEALTH_DIR = RUNTIME_DATA_DIR / "health"
RUNTIME_SLEEP_DIR = RUNTIME_DATA_DIR / "sleep"


class FallThresholds(BaseSettings):
    fall_true_at: float = 0.5
    warning_at: float = 0.6
    critical_at: float = 0.85


class HealthThresholds(BaseSettings):
    high_risk_true_at: float = 0.5
    warning_at: float = 0.35
    critical_at: float = 0.65


class SleepThresholds(BaseSettings):
    critical_below: float = 50
    poor_below: float = 60
    fair_below: float = 75
    good_below: float = 85
    attention_below: float = 60
    alert_below: float = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEALTHGUARD_",
        env_nested_delimiter="__",
    )

    app_name: str = Field(default="HealthGuard API")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)

    fall_bundle_path: Path = Field(default=MODELS_ROOT / "fall" / "fall_bundle.joblib")
    fall_min_sequence_samples: int = Field(default=50, ge=1, le=10_000)

    health_bundle_path: Path = Field(
        default=MODELS_ROOT / "healthguard" / "healthguard_bundle.joblib",
    )

    sleep_bundle_path: Path = Field(
        default=MODELS_ROOT / "Sleep" / "sleep_score_bundle.joblib",
    )
    sleep_preprocessor_path: Path = Field(
        default=MODELS_ROOT / "Sleep" / "sleep_score_preprocessor.joblib",
        description="Preprocessor; used when file exists and bundle has no preprocessor.",
    )
    sleep_metadata_path: Path = Field(
        default=MODELS_ROOT / "Sleep" / "sleep_score_metadata.json",
    )

    fall_sample_input_path: Path = Field(default=RUNTIME_FALL_DIR / "iot_sample_input.json")
    fall_sample_cases_path: Path = Field(
        default=RUNTIME_FALL_DIR / "iot_sample_cases.json",
        description="Multi-case fall windows (not_fall vs fall_like) for evaluation.",
    )
    health_sample_input_path: Path = Field(default=RUNTIME_HEALTH_DIR / "iot_sample_input.json")
    health_sample_cases_path: Path = Field(
        default=RUNTIME_HEALTH_DIR / "iot_sample_cases.json",
        description="Multi-case vital records (low vs high risk) for evaluation.",
    )
    sleep_sample_input_path: Path = Field(default=RUNTIME_SLEEP_DIR / "iot_sample_input.json")
    sleep_sample_cases_path: Path = Field(
        default=RUNTIME_SLEEP_DIR / "iot_sample_cases.json",
        description="Multi-case sleep sessions (good vs poor) for evaluation.",
    )

    fall_thresholds: FallThresholds = Field(default_factory=FallThresholds)
    health_thresholds: HealthThresholds = Field(default_factory=HealthThresholds)
    sleep_thresholds: SleepThresholds = Field(default_factory=SleepThresholds)


settings = Settings()
