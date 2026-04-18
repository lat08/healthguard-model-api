# HealthGuard API — Tham chiếu (đồng bộ OpenAPI / Swagger)

Tài liệu này mô tả **cùng contract** với giao diện Swagger tại:

- **Swagger UI:** `http://localhost:8000/docs`
- **OpenAPI JSON (nguồn schema chính thức):** `http://localhost:8000/openapi.json`

Phiên bản OpenAPI do FastAPI sinh: **3.1.0**. Để tái tạo file JSON khi không bật server:

```bash
python -c "import json; from app.main import app; print(json.dumps(app.openapi(), indent=2))" > openapi.json
```

**Content-Type** cho mọi `POST` có body: `application/json`

**Mô tả OpenAPI (`app/main.py` → `OPENAPI_DESCRIPTION`):** có thể ghi tóm tắt kiểu “LightGBM” cho fall; **backend thực tế** lấy từ bundle (`fall_service`: `xgboost` \| `lightgbm` \| …). Chi tiết runtime xem `GET /api/v1/fall/model-info` và `inference_backend` trong `/health`.

**File mẫu trên disk (đồng bộ script):**

| Mục đích | Vị trí |
|----------|--------|
| Hướng dẫn batch `POST /predict` | `data/runtime/BATCH_PREDICT_SAMPLES.txt` |
| Fall: từng case + batch nhiều cửa sổ | `data/runtime/fall/cases/*.json` |
| Health / Sleep tương tự | `data/runtime/health/cases/*.json`, `data/runtime/sleep/cases/*.json` |

Script: `scripts/build_fall_sample_cases.py`, `build_health_sample_cases.py`, `build_sleep_sample_cases.py`, `build_predict_batch_samples.py`.

---

## Mã HTTP & body lỗi

| Mã | Khi nào | Body (thường gặp) |
|----|---------|-------------------|
| **200** | Thành công | Theo schema từng endpoint (OpenAPI `$ref`) |
| **422** | Lỗi validation FastAPI / Pydantic | `HTTPValidationError` — xem mục dưới |
| **422** | Một số route (fall/health/sleep) cũng dùng cho `ValueError` từ service | `{"detail": "<chuỗi mô tả>"}` |
| **503** | `POST .../predict` khi model **chưa load** | Fall: `{"detail": "<fall_service.unavailable_detail()>"}` — thường `"Fall detection model is not loaded."` hoặc kèm `Reason: …` nếu load thất bại. Health: `"Health risk model is not loaded."`. Sleep: `"Sleep score model is not loaded."` |
| **404** | `GET .../sample-input` khi thiếu file mặc định (`iot_sample_input.json`) | `{"detail": "Sample input file not found."}` |
| **404** | `GET .../sample-cases` khi thiếu `iot_sample_cases.json` | Body có `detail` hướng dẫn chạy script build tương ứng (`build_*_sample_cases.py`). |
| **404** | `GET .../sample-input?case=<id>` khi `case` không tồn tại trong catalog | `{"detail": "Unknown case '…'. Use GET /api/v1/<module>/sample-cases for available ids."}` |
| **500** | Lỗi inference không xử lý được | `{"detail": "Prediction error: ..."}` |

### `422` — `HTTPValidationError` (chuẩn FastAPI)

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "records"],
      "msg": "Field required",
      "input": {}
    }
  ]
}
```

### `503` — Ví dụ (Health)

```json
{
  "detail": "Health risk model is not loaded."
}
```

---

## Predict Response Extension (SHAP + Standard Blocks)

`POST /api/v1/fall/predict`, `POST /api/v1/health/predict`, `POST /api/v1/health/predict/batch`,
`POST /api/v1/sleep/predict`, va `POST /api/v1/sleep/predict/batch` van giu top-level envelope:

```json
{
  "success": true,
  "results": [...],
  "total": 1
}
```

Moi item trong `results[]` hien giu cac field legacy (`predicted_*`, `risk_level`, `requires_attention`, ...)
de tranh gay vo client cu, dong thoi duoc bo sung them contract chuan hoa:

- `status`
- `meta`
- `input_ref`
- `prediction`
- `top_features`
- `shap`
- `explanation`

### `shap`

`shap` la block contribution day du de frontend/dashboard/doc audit dung lai:

```json
{
  "available": true,
  "output_space": "raw_margin | prediction",
  "base_value": -0.1523,
  "prediction_value": 0.81,
  "values": [
    {
      "feature": "spo2",
      "feature_value": 92.5,
      "shap_value": 0.4281,
      "impact": 0.4281,
      "direction": "risk_up"
    }
  ]
}
```

Ghi chu:

- `top_features` duoc rut gon tu `shap.values` de UI hien nhanh.
- `top_features` uu tien cac yeu to co the hanh dong/doi chieu duoc va an bot cac bien ho so tinh
  nhu `age`, `gender`, `weight`, `height`, `device_model`, `timezone` khi render patient-facing.
- `output_space = raw_margin` thuong ap dung cho classifier native contribution (LightGBM/XGBoost), nen
  `base_value + sum(shap_value)` khong nhat thiet bang probability dang hien o `prediction_value`.
- `output_space = prediction` duoc dung khi contribution da nam cung khong gian voi diem du doan
  (vi du sleep score).

## System

`app/routers/system.py`: `GET /` redirect tới `/docs` (`include_in_schema=False`).

### `GET /health` — Health Check

| | |
|--|--|
| **OpenAPI** | `operationId`: `health_check_health_get` |
| **Response 200** | `application/json` → schema `#/components/schemas/HealthCheckResponse` |

**Các trường (`HealthCheckResponse`):** `status` (string), `models` (object — mỗi giá trị là `ModelInfo`), `version` (string).

**`ModelInfo`:** `model_name`, `status`, `inference_backend`, `feature_count`, `thresholds` (object), tùy chọn `load_error` (khi `status` = `unavailable` và có lỗi load).

**Lưu ý (`GET /health`):** `fall_service.get_info()` còn trả `min_sequence_samples` (ngưỡng tối thiểu mẫu IMU), nhưng khi bọc `ModelInfo` các khóa **không** khai báo trong schema sẽ **bị bỏ** (`extra="ignore"`). Để đọc đủ dict (gồm `min_sequence_samples`, `load_error`), dùng **`GET /api/v1/fall/model-info`**.

#### Sample output (200)

Khi cả ba model đều không load (ví dụ môi trường test thiếu artifact):

```json
{
  "status": "unhealthy",
  "models": {
    "fall": {
      "model_name": "fall_lightgbm_binary",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 0,
      "thresholds": {
        "fall_true_at": 0.5,
        "warning_at": 0.6,
        "critical_at": 0.85
      }
    },
    "health": {
      "model_name": "healthguard_lightgbm",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 14,
      "thresholds": {
        "high_risk_true_at": 0.5,
        "warning_at": 0.35,
        "critical_at": 0.65
      }
    },
    "sleep": {
      "model_name": "catboost",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 42,
      "thresholds": {
        "critical_below": 50.0,
        "poor_below": 60.0,
        "fair_below": 75.0,
        "good_below": 85.0,
        "attention_below": 60.0,
        "alert_below": 50.0
      }
    }
  },
  "version": "1.0.0"
}
```

Khi đủ model load: `status` có thể là `healthy` hoặc `degraded`. `inference_backend` **phụ thuộc artifact**: fall có thể `xgboost` hoặc `lightgbm` (theo class model trong bundle); health thường `lightgbm`; sleep thường `catboost` hoặc `sklearn_regressor`. Artifact mặc định: `models/...` (`.joblib`). Khi không load được: `none`.

---

### `GET /api/v1/models` — List All Models

| | |
|--|--|
| **OpenAPI** | `operationId`: `list_models_api_v1_models_get` |
| **Response 200** | `application/json` — trong OpenAPI schema là `{}` (object tự do); thực tế luôn có khóa `models`. |

#### Sample output (200)

```json
{
  "models": {
    "fall_detection": {
      "model_name": "fall_lightgbm_binary",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 0,
      "thresholds": {
        "fall_true_at": 0.5,
        "warning_at": 0.6,
        "critical_at": 0.85
      }
    },
    "health_risk": {
      "model_name": "healthguard_lightgbm",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 14,
      "thresholds": {
        "high_risk_true_at": 0.5,
        "warning_at": 0.35,
        "critical_at": 0.65
      }
    },
    "sleep_score": {
      "model_name": "catboost",
      "status": "unavailable",
      "inference_backend": "none",
      "feature_count": 42,
      "thresholds": {
        "critical_below": 50.0,
        "poor_below": 60.0,
        "fair_below": 75.0,
        "good_below": 85.0,
        "attention_below": 60.0,
        "alert_below": 50.0
      },
      "metrics": {}
    }
  }
}
```

**Ghi chú:** `sleep_score` có thêm `metrics` (object) khi có `sleep_score_metadata.json`. **`list_models` không bọc `ModelInfo`:** mỗi entry là dict đầy đủ từ `get_info()` (fall có thể có `min_sequence_samples`, `load_error` khi lỗi load). OpenAPI vẫn có thể khai báo schema lỏng — xem response thực tế trên `/docs`.

---

## Fall Detection — `/api/v1/fall`

Schema request/response: `#/components/schemas/FallPredictionResponse`, `FallPredictionRequest`, `SensorSample`, … (body `POST /predict` là **oneOf**: một `FallPredictionRequest` hoặc mảng các `FallPredictionRequest`.)

### `POST /api/v1/fall/predict` — Predict Fall Risk (một hoặc nhiều cửa sổ)

| | |
|--|--|
| **OpenAPI** | `operationId`: `predict_fall_api_v1_fall_predict_post` |
| **Body** | Một object `FallPredictionRequest` **hoặc** JSON array `[FallPredictionRequest, …]` (tối thiểu 1 phần tử). Không dùng wrapper `payloads`. |
| **200** | `#/components/schemas/FallPredictionResponse` |
| **422** | `#/components/schemas/HTTPValidationError` |
| **503** | `{"detail": "Fall detection model is not loaded."}` |

**`FallPredictionRequest`:** `device_id` (string, default `"unknown"`), `sampling_rate` (int, default `50`), `window_size` (int, default `50`), **`data`** (array, độ dài ≥ `min_sequence_samples`, mặc định 50) — phần tử là `SensorSample`.

**`SensorSample` (bắt buộc):** `timestamp` (integer), `accel` `{x,y,z}`, `gyro` `{x,y,z}`, `orientation` `{pitch,roll,yaw}`; `environment` (optional, default 0).

#### Sample input (POST) — một cửa sổ

Cấu trúc dưới đây chỉ minh họa **một bước** trong `data`; request thật cần **đủ số bước** (mặc định ≥ 50). Dùng `GET /api/v1/fall/sample-input` để có payload hợp lệ.

```json
{
  "device_id": "wearable_fall_0001",
  "sampling_rate": 50,
  "window_size": 50,
  "data": [
    {
      "timestamp": 1710000000,
      "accel": { "x": 0.42, "y": -0.18, "z": 1.12 },
      "gyro": { "x": 30.0, "y": -10.0, "z": 5.0 },
      "orientation": { "pitch": 10.0, "roll": 5.0, "yaw": 30.0 },
      "environment": { "floor_vibration": 0.2, "room_occupancy": 1.0, "pressure_mat": 100.0 }
    }
  ]
}
```

#### Sample output (200) — khi model **đã load** (cấu trúc cố định; số liệu phụ thuộc input + backend)

```json
{
  "success": true,
  "results": [
    {
      "device_id": "wearable_fall_0001",
      "sample_count": 50,
      "predicted_fall_probability": 0.12,
      "predicted_fall": false,
      "predicted_fall_label": "normal",
      "risk_level": "normal",
      "requires_attention": false,
      "high_priority_alert": false,
      "predicted_activity": null,
      "activity_probability": null
    }
  ],
  "total": 1
}
```

*(Giá trị `predicted_fall_probability` / nhãn chỉ là minh họa — trên `/docs` bạn bấm “Try it out” để xem đúng output môi trường của bạn.)*

#### Batch (nhiều cửa sổ)

Gửi **JSON array** `[window1, window2, …]`; mỗi phần tử cùng schema `FallPredictionRequest` (mỗi `data` đủ `min_sequence_samples` bước — mặc định **50**, cấu hình `HEALTHGUARD_FALL_MIN_SEQUENCE_SAMPLES` / `settings.fall_min_sequence_samples`). Có thể dùng file **`data/runtime/fall/cases/batch_multi_windows_one_device.json`** (mảng nhiều cửa sổ, cùng `device_id`) — sinh bởi `python scripts/build_predict_batch_samples.py`.

`results` trả về theo **thứ tự** mảng; `total` = số cửa sổ.

---

### `GET /api/v1/fall/model-info` — Fall Model Info

| | |
|--|--|
| **OpenAPI** | `operationId`: `fall_model_info_api_v1_fall_model_info_get` |
| **200** | Schema `{}` trong OpenAPI — thực tế là dict từ `fall_service.get_info()` |

#### Sample output (200)

```json
{
  "model_name": "fall_binary_classifier",
  "status": "loaded",
  "inference_backend": "xgboost",
  "feature_count": 100,
  "min_sequence_samples": 50,
  "thresholds": {
    "fall_true_at": 0.5,
    "warning_at": 0.6,
    "critical_at": 0.85
  }
}
```

*`model_name` / `feature_count` / `inference_backend` lấy từ bundle (`metadata`, `feature_names`, class model).*

---

### `GET /api/v1/fall/sample-cases` — Fall sample cases (catalog)

| | |
|--|--|
| **OpenAPI** | `operationId`: `fall_sample_cases_api_v1_fall_sample_cases_get` |
| **200** | JSON từ `HEALTHGUARD_FALL_SAMPLE_CASES_PATH` (mặc định `data/runtime/fall/iot_sample_cases.json`): `version`, `window_samples`, **`cases`** — mỗi phần tử có `id`, `intent` (`not_fall` \| `fall_like`), `description`, `description_vi`, **`request`** (một `FallPredictionRequest` đủ `data`). |
| **404** | Thiếu file → `detail` gợi ý chạy `python scripts/build_fall_sample_cases.py`. |

Thư mục **`data/runtime/fall/cases/`**: mỗi file `{case_id}.json` chỉ là body **một** cửa sổ (copy-paste `POST /predict`). Script trên cũng tạo các file này.

---

### `GET /api/v1/fall/sample-input` — Fall Sample Input (một cửa sổ)

| | |
|--|--|
| **OpenAPI** | `operationId`: `fall_sample_input_api_v1_fall_sample_input_get` |
| **Query** | `case` (optional, string): nếu có — trả đúng `cases[].request` có `cases[].id == case` (xem `GET /sample-cases`). Nếu không có — đọc file mặc định. |
| **200 (không `case`)** | JSON từ `HEALTHGUARD_FALL_SAMPLE_INPUT_PATH` (mặc định `data/runtime/fall/iot_sample_input.json`). |
| **200 (có `case`)** | Một object `FallPredictionRequest` từ catalog. |
| **404** | Thiếu file mặc định; hoặc `case` không khớp id nào. |

---

## Health Risk — `/api/v1/health`

Schema: `HealthPredictionRequest`, `HealthPredictionResponse`, `VitalSignsRecord`, …

### `POST /api/v1/health/predict` — Predict Health Risk

| | |
|--|--|
| **OpenAPI** | `operationId`: `predict_health_api_v1_health_predict_post` |
| **Body** | `#/components/schemas/HealthPredictionRequest` |
| **200** | `#/components/schemas/HealthPredictionResponse` |
| **503** | `{"detail": "Health risk model is not loaded."}` |

**`HealthPredictionRequest`:** `records` (array, min 1) — `VitalSignsRecord`.

**`VitalSignsRecord` (14 field, bắt buộc):**  
`heart_rate`, `respiratory_rate`, `body_temperature`, `spo2`, `systolic_blood_pressure`, `diastolic_blood_pressure`, `age`, `gender` (0 nữ / 1 nam), `weight_kg`, `height_m`, `derived_hrv`, `derived_pulse_pressure`, `derived_bmi`, `derived_map`.

#### Sample input (POST) — trùng `examples` trong OpenAPI cho `VitalSignsRecord`

```json
{
  "records": [
    {
      "heart_rate": 118,
      "respiratory_rate": 24,
      "body_temperature": 38.2,
      "spo2": 92.5,
      "systolic_blood_pressure": 148,
      "diastolic_blood_pressure": 96,
      "age": 67,
      "gender": 1,
      "weight_kg": 73.5,
      "height_m": 1.68,
      "derived_hrv": 24.0,
      "derived_pulse_pressure": 52,
      "derived_bmi": 26.0,
      "derived_map": 113.3
    }
  ]
}
```

#### Sample output (200) — minh họa khi bundle LightGBM đã load

```json
{
  "success": true,
  "results": [
    {
      "record_index": 0,
      "predicted_health_risk_probability": 0.88,
      "predicted_health_risk_label": "high_risk",
      "risk_level": "critical",
      "requires_attention": true,
      "high_priority_alert": true
    }
  ],
  "total": 1
}
```

---

### `POST /api/v1/health/predict/batch` — Batch Predict Health Risk

| | |
|--|--|
| **OpenAPI** | `operationId`: `predict_health_batch_api_v1_health_predict_batch_post` |
| **Body** | Giống `HealthPredictionRequest` |
| **200** | `HealthPredictionResponse` |

#### Sample input (POST)

Hai bản ghi (có thể copy từ `data/runtime/health/iot_sample_input.json`):

```json
{
  "records": [
    {
      "heart_rate": 60.0,
      "respiratory_rate": 12.0,
      "body_temperature": 36.86,
      "spo2": 95.7,
      "systolic_blood_pressure": 124.0,
      "diastolic_blood_pressure": 86.0,
      "age": 37,
      "gender": 0,
      "weight_kg": 91.54,
      "height_m": 1.679,
      "derived_hrv": 0.121,
      "derived_pulse_pressure": 38.0,
      "derived_bmi": 32.46,
      "derived_map": 98.67
    },
    {
      "heart_rate": 63.0,
      "respiratory_rate": 18.0,
      "body_temperature": 36.51,
      "spo2": 96.69,
      "systolic_blood_pressure": 126.0,
      "diastolic_blood_pressure": 84.0,
      "age": 77,
      "gender": 1,
      "weight_kg": 50.7,
      "height_m": 1.993,
      "derived_hrv": 0.117,
      "derived_pulse_pressure": 42.0,
      "derived_bmi": 12.77,
      "derived_map": 98.0
    }
  ]
}
```

#### Sample output (200)

`results` có 2 phần tử (`record_index` 0 và 1), `total`: `2`.

---

### `GET /api/v1/health/model-info` — Health Model Info

**200:** dict `health_service.get_info()`: `model_name`, `status`, `inference_backend`, `feature_count` (14), `thresholds`.

---

### `GET /api/v1/health/sample-cases` — Health sample cases

| | |
|--|--|
| **OpenAPI** | `operationId`: `health_sample_cases_api_v1_health_sample_cases_get` |
| **200** | `HEALTHGUARD_HEALTH_SAMPLE_CASES_PATH` → mặc định `data/runtime/health/iot_sample_cases.json`: `version`, **`cases`** (`id`, `intent`, `description`, `description_vi`, `request` với `records`). |
| **404** | Thiếu file → `detail` gợi ý `python scripts/build_health_sample_cases.py`. |

**`data/runtime/health/cases/`**: file `{id}.json` = body `POST /predict` một scenario. File **`batch_multi_vitals_same_patient.json`**: nhiều `records` cùng hồ sơ — sinh bởi `python scripts/build_predict_batch_samples.py`.

---

### `GET /api/v1/health/sample-input` — Health Sample Input

| | |
|--|--|
| **OpenAPI** | `operationId`: `health_sample_input_api_v1_health_sample_input_get` |
| **Query** | `case` (optional): chọn `cases[].id` từ `GET /sample-cases`. |
| **200** | Một object `HealthPredictionRequest` (`{"records":[...]}`) — từ file mặc định hoặc từ catalog. |
| **404** | Thiếu `iot_sample_input.json` hoặc `case` không hợp lệ. |

Mặc định đọc `HEALTHGUARD_HEALTH_SAMPLE_INPUT_PATH` → `data/runtime/health/iot_sample_input.json`.

---

## Sleep Score — `/api/v1/sleep`

Schema: `SleepPredictionRequest`, `SleepPredictionResponse`, `SleepRecord`, …

### `POST /api/v1/sleep/predict` — Predict Sleep Score

| | |
|--|--|
| **OpenAPI** | `operationId`: `predict_sleep_api_v1_sleep_predict_post` |
| **Body** | `#/components/schemas/SleepPredictionRequest` |
| **200** | `#/components/schemas/SleepPredictionResponse` |
| **503** | `{"detail": "Sleep score model is not loaded."}` |

**`SleepPredictionRequest`:** `records` (array, min 1) — mỗi phần tử `SleepRecord` (**42 field**).

#### Sample input (POST) — một bản ghi (trùng `data/runtime/sleep/iot_sample_input.json`)

```json
{
  "records": [
    {
      "user_id": "user_00332",
      "date_recorded": "2024-04-03",
      "sleep_start_timestamp": "2024-04-03 22:36:00",
      "sleep_end_timestamp": "2024-04-04 06:01:00",
      "duration_minutes": 445,
      "sleep_latency_minutes": 2,
      "wake_after_sleep_onset_minutes": 15,
      "sleep_efficiency_pct": 96.2,
      "sleep_stage_deep_pct": 15.6,
      "sleep_stage_light_pct": 53.3,
      "sleep_stage_rem_pct": 15.8,
      "sleep_stage_awake_pct": 15.3,
      "heart_rate_mean_bpm": 62.3,
      "heart_rate_min_bpm": 54.3,
      "heart_rate_max_bpm": 71.3,
      "hrv_rmssd_ms": 31.1,
      "respiration_rate_bpm": 14.4,
      "spo2_mean_pct": 96.7,
      "spo2_min_pct": 94.5,
      "movement_count": 29,
      "snore_events": 0,
      "ambient_noise_db": 38.9,
      "room_temperature_c": 23.7,
      "room_humidity_pct": 57.5,
      "step_count_day": 5139,
      "caffeine_mg": 126,
      "alcohol_units": 0.2,
      "medication_flag": 0,
      "jetlag_hours": -11,
      "timezone": "Europe/London",
      "age": 50,
      "gender": "female",
      "weight_kg": 94.2,
      "height_cm": 166.1,
      "device_model": "AlphaWatch X1",
      "bedtime_consistency_std_min": 37.1,
      "stress_score": 33,
      "activity_before_bed_min": 39,
      "screen_time_before_bed_min": 87,
      "insomnia_flag": 0,
      "apnea_risk_score": 24,
      "nap_duration_minutes": 10,
      "created_at": "2025-10-21 16:41:53.708868"
    }
  ]
}
```

#### Sample output (200) — minh họa khi bundle + preprocessor đã load

```json
{
  "success": true,
  "results": [
    {
      "record_index": 0,
      "predicted_sleep_score": 72.5,
      "predicted_sleep_label": "fair",
      "risk_level": "fair",
      "requires_attention": false,
      "high_priority_alert": false
    }
  ],
  "total": 1
}
```

`predicted_sleep_label` / `risk_level`: `critical` \| `poor` \| `fair` \| `good` \| `excellent`.

---

### `POST /api/v1/sleep/predict/batch` — Batch Predict Sleep Score

| | |
|--|--|
| **OpenAPI** | `operationId`: `predict_sleep_batch_api_v1_sleep_predict_batch_post` |
| **Body** | Giống `SleepPredictionRequest` |
| **200** | `SleepPredictionResponse` |

Nhiều phần tử trong `records` → nhiều phần tử trong `results`, `total` = số bản ghi.

---

### `GET /api/v1/sleep/model-info` — Sleep Model Info

**200:** dict `sleep_service.get_info()`: `model_name` (từ bundle hoặc mặc định), `status`, `inference_backend`, `feature_count` (42), `thresholds`, **`metrics`** (object; từ `sleep_score_metadata.json` nếu có).

---

### `GET /api/v1/sleep/sample-cases` — Sleep sample cases

| | |
|--|--|
| **OpenAPI** | `operationId`: `sleep_sample_cases_api_v1_sleep_sample_cases_get` |
| **200** | `HEALTHGUARD_SLEEP_SAMPLE_CASES_PATH` → mặc định `data/runtime/sleep/iot_sample_cases.json`. Có thể có **`source_csv`** (đường dẫn file đã dùng khi build). **`cases`**: mỗi phần tử có `id`, `intent` (nhãn kịch bản / `daily_label` từ CSV), `description`, `description_vi`, **`request`**. Các case sinh từ CSV có thêm **`dataset_sleep_score`**, **`dataset_daily_label`** (chỉ trong catalog — **không** nằm trong `request` gửi `POST /predict`). |
| **404** | Thiếu file → `detail` gợi ý `python scripts/build_sleep_sample_cases.py`. |

**`data/runtime/sleep/cases/`**: `{id}.json` = một scenario; **`batch_multi_nights_one_user.json`** = nhiều đêm một `user_id` (script `build_predict_batch_samples.py`).

---

### `GET /api/v1/sleep/sample-input` — Sleep Sample Input

| | |
|--|--|
| **OpenAPI** | `operationId`: `sleep_sample_input_api_v1_sleep_sample_input_get` |
| **Query** | `case` (optional): chọn theo `GET /sample-cases`. |
| **200** | `SleepPredictionRequest` từ file mặc định hoặc catalog. |
| **404** | Thiếu `iot_sample_input.json` hoặc `case` không hợp lệ. |

Mặc định: `HEALTHGUARD_SLEEP_SAMPLE_INPUT_PATH` → `data/runtime/sleep/iot_sample_input.json`.

---

## Biến môi trường — đường dẫn & ngưỡng (`app/config.py`)

Tiền tố **`HEALTHGUARD_`** (xem `Settings` trong `app/config.py`). Ví dụ override đường dẫn:

| Cài đặt (field) | Biến môi trường tương ứng |
|-----------------|---------------------------|
| `fall_bundle_path` | `HEALTHGUARD_FALL_BUNDLE_PATH` |
| `fall_min_sequence_samples` | `HEALTHGUARD_FALL_MIN_SEQUENCE_SAMPLES` |
| `fall_sample_input_path` | `HEALTHGUARD_FALL_SAMPLE_INPUT_PATH` |
| `fall_sample_cases_path` | `HEALTHGUARD_FALL_SAMPLE_CASES_PATH` |
| `health_bundle_path` | `HEALTHGUARD_HEALTH_BUNDLE_PATH` |
| `health_sample_input_path` | `HEALTHGUARD_HEALTH_SAMPLE_INPUT_PATH` |
| `health_sample_cases_path` | `HEALTHGUARD_HEALTH_SAMPLE_CASES_PATH` |
| `sleep_bundle_path` | `HEALTHGUARD_SLEEP_BUNDLE_PATH` |
| `sleep_preprocessor_path` | `HEALTHGUARD_SLEEP_PREPROCESSOR_PATH` |
| `sleep_metadata_path` | `HEALTHGUARD_SLEEP_METADATA_PATH` |
| `sleep_sample_input_path` | `HEALTHGUARD_SLEEP_SAMPLE_INPUT_PATH` |
| `sleep_sample_cases_path` | `HEALTHGUARD_SLEEP_SAMPLE_CASES_PATH` |

Ngưỡng (nested): ví dụ `HEALTHGUARD_FALL_THRESHOLDS__FALL_TRUE_AT`, `HEALTHGUARD_HEALTH_THRESHOLDS__WARNING_AT`, `HEALTHGUARD_SLEEP_THRESHOLDS__FAIR_BELOW`, … (`__` phân tách nested).

---

## Liên kết mã nguồn

| Thành phần | File |
|------------|------|
| Ứng dụng FastAPI | `app/main.py` |
| Package | `app/__init__.py` |
| Schema Pydantic | `app/schemas/common.py`, `fall.py`, `health.py`, `sleep.py` |
| Router | `app/routers/system.py`, `fall.py`, `health.py`, `sleep.py` |
| Cấu hình | `app/config.py` |
| Inference | `app/services/fall_service.py`, `fall_featurize.py`, `health_service.py`, `sleep_service.py`, `sleep_features.py`, `sklearn_sleep_pickle_compat.py` |
| Mẫu JSON / CSV build | `scripts/build_*_sample_cases.py`, `build_predict_batch_samples.py`, `write_per_case_json.py` |

**Lưu ý:** Một số route `GET` `model-info` / `sample-input` / `sample-cases` khai báo response schema rỗng `{}` hoặc tự do trong OpenAPI nhưng runtime trả dict/JSON như tài liệu này — xác nhận trên `/docs` hoặc `openapi.json`.

---

## Artifact & môi trường

- Model inference chỉ dùng **joblib** trong **`models/`** (gốc repo): `models/fall/fall_bundle.joblib`, `models/healthguard/healthguard_bundle.joblib`, `models/Sleep/sleep_score_bundle.joblib`. Sleep: preprocessor lấy từ bundle nếu có; nếu không — load thêm `models/Sleep/sleep_score_preprocessor.joblib` (`settings.sleep_preprocessor_path`). Metadata tùy chọn: `models/Sleep/sleep_score_metadata.json`.
- Override đường dẫn bằng biến môi trường `HEALTHGUARD_*` (mục trên).
- **scikit-learn / pickle:** bundle health/fall/sleep thường được pickle với **scikit-learn 1.6.x**. Runtime **sklearn ≥ 1.7** có thể gây lỗi khi transform (ví dụ `SimpleImputer` thiếu `_fill_dtype`) hoặc lỗi load sleep. **`requirements.txt`** ghim `scikit-learn>=1.6.1,<1.7.0`; giữ đúng khi deploy. Sleep: `app/services/sklearn_sleep_pickle_compat.py` bổ sung `_RemainderColsList` cũ trước `joblib.load` để tương thích pickle `ColumnTransformer` trên sklearn mới hơn (vẫn nên khớp phiên bản sklearn khi có thể).
- **XGBoost (fall):** nếu model pickle từ bản XGBoost cũ, có thể thấy cảnh báo khi load — khuyến nghị lâu dài: export `Booster.save_model` rồi load lại theo tài liệu XGBoost.

---

## Phụ lục — `operationId` (trùng Swagger / OpenAPI)

| Method | Path | `operationId` |
|--------|------|-----------------|
| `GET` | `/health` | `health_check_health_get` |
| `GET` | `/api/v1/models` | `list_models_api_v1_models_get` |
| `POST` | `/api/v1/fall/predict` | `predict_fall_api_v1_fall_predict_post` |
| `GET` | `/api/v1/fall/model-info` | `fall_model_info_api_v1_fall_model_info_get` |
| `GET` | `/api/v1/fall/sample-cases` | `fall_sample_cases_api_v1_fall_sample_cases_get` |
| `GET` | `/api/v1/fall/sample-input` | `fall_sample_input_api_v1_fall_sample_input_get` |
| `POST` | `/api/v1/health/predict` | `predict_health_api_v1_health_predict_post` |
| `POST` | `/api/v1/health/predict/batch` | `predict_health_batch_api_v1_health_predict_batch_post` |
| `GET` | `/api/v1/health/model-info` | `health_model_info_api_v1_health_model_info_get` |
| `GET` | `/api/v1/health/sample-cases` | `health_sample_cases_api_v1_health_sample_cases_get` |
| `GET` | `/api/v1/health/sample-input` | `health_sample_input_api_v1_health_sample_input_get` |
| `POST` | `/api/v1/sleep/predict` | `predict_sleep_api_v1_sleep_predict_post` |
| `POST` | `/api/v1/sleep/predict/batch` | `predict_sleep_batch_api_v1_sleep_predict_batch_post` |
| `GET` | `/api/v1/sleep/model-info` | `sleep_model_info_api_v1_sleep_model_info_get` |
| `GET` | `/api/v1/sleep/sample-cases` | `sleep_sample_cases_api_v1_sleep_sample_cases_get` |
| `GET` | `/api/v1/sleep/sample-input` | `sleep_sample_input_api_v1_sleep_sample_input_get` |

*(Route `GET /` redirect `/docs` có `include_in_schema=False` — thường không xuất hiện trong OpenAPI.)*
