from __future__ import annotations

from app.services.prediction_contract import build_top_features


def test_build_top_features_excludes_profile_features() -> None:
    shap_payload = {
        "values": [
            {
                "feature": "weight_kg",
                "feature_value": 91.5,
                "shap_value": 0.9,
                "impact": 0.9,
                "direction": "risk_up",
            },
            {
                "feature": "spo2",
                "feature_value": 92.5,
                "shap_value": 0.4,
                "impact": 0.4,
                "direction": "risk_up",
            },
        ]
    }

    top_features = build_top_features(
        shap_payload=shap_payload,
        preferred_features=["spo2"],
        reason_overrides={"spo2": "SpO2 thap dang lam tang nguy co"},
        excluded_features=["weight_kg"],
    )

    assert len(top_features) == 1
    assert top_features[0]["feature"] == "spo2"


def test_build_top_features_orders_preferred_by_impact() -> None:
    shap_payload = {
        "values": [
            {
                "feature": "body_temperature",
                "feature_value": 38.2,
                "shap_value": 0.1,
                "impact": 0.1,
                "direction": "risk_up",
            },
            {
                "feature": "heart_rate",
                "feature_value": 118,
                "shap_value": 5.6,
                "impact": 5.6,
                "direction": "risk_up",
            },
        ]
    }

    top_features = build_top_features(
        shap_payload=shap_payload,
        preferred_features=["body_temperature", "heart_rate"],
        reason_overrides={},
    )

    assert [item["feature"] for item in top_features] == ["heart_rate", "body_temperature"]
