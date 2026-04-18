from __future__ import annotations


def test_health_predict_includes_standard_contract(client, health_request_payload):
    response = client.post("/api/v1/health/predict", json=health_request_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    row = body["results"][0]
    assert row["predicted_health_risk_label"] == "high_risk"
    assert row["status"] == "ok"
    assert row["meta"]["model_family"] == "healthguard"
    assert row["prediction"]["prediction_label"] == "high_risk"
    assert row["top_features"][0]["feature"] == "spo2"
    assert row["shap"]["available"] is True
    assert row["explanation"]["recommended_actions"]


def test_sleep_predict_includes_standard_contract(client, sleep_request_payload):
    response = client.post("/api/v1/sleep/predict", json=sleep_request_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    row = body["results"][0]
    assert row["predicted_sleep_label"] == "fair"
    assert row["status"] == "ok"
    assert row["meta"]["model_family"] == "sleep"
    assert row["input_ref"]["user_id"] == "user_test_0001"
    assert row["prediction"]["prediction_band"] == "info"
    assert row["shap"]["output_space"] == "prediction"


def test_fall_predict_includes_standard_contract(client, fall_request_payload):
    response = client.post("/api/v1/fall/predict", json=fall_request_payload)
    assert response.status_code == 200, response.text
    body = response.json()
    row = body["results"][0]
    assert row["device_id"] == "wearable_test_0001"
    assert row["status"] == "ok"
    assert row["meta"]["model_family"] == "fall"
    assert row["input_ref"]["device_id"] == "wearable_test_0001"
    assert row["prediction"]["prediction_label"] == "normal"
    assert row["shap"]["output_space"] == "raw_margin"
