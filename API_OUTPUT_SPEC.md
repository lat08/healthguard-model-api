# API_OUTPUT_SPEC.md

## Muc tieu

Tai lieu nay chuan hoa output API cho 3 model dang deploy:

- `sleep_score_bundle.joblib`
- `healthguard_bundle.joblib`
- `fall_bundle.joblib`

Muc tieu:

- 3 model tra ve cung 1 format de backend de tich hop
- co du `prediction + top_features + explanation`
- de noi them dashboard, Gemini, alert engine, audit log

## 1. Nguyen tac chung

Moi endpoint suy luan nen tra ve 1 object JSON co 6 lop:

1. `meta`
2. `input_ref`
3. `prediction`
4. `top_features`
5. `shap`
6. `explanation`

Neu co loi, van nen giu response object on dinh va them:

- `status = error`
- `error_code`
- `error_message`

## 2. Contract output chuan

```json
{
  "status": "ok",
  "meta": {
    "model_family": "sleep | healthguard | fall",
    "model_name": "sleep_score | healthguard | fall_detection",
    "model_version": "v_current",
    "artifact_type": "python_bundle",
    "artifact_path": "Modelok/...bundle.joblib",
    "timestamp": "2026-04-17T00:00:00+07:00",
    "request_id": "req_123456"
  },
  "input_ref": {
    "user_id": "U1001",
    "device_id": "device_01",
    "event_timestamp": "2026-04-17T00:00:00+07:00",
    "source_file": "data/example.csv"
  },
  "prediction": {
    "prediction_label": "normal",
    "prediction_score": 0.81,
    "prediction_band": "warning",
    "requires_attention": true,
    "high_priority_alert": false,
    "confidence": 0.81
  },
  "top_features": [
    {
      "feature": "spo2",
      "feature_value": 92.5,
      "impact": 0.4281,
      "direction": "risk_up",
      "reason": "chi so nay dang lam tang muc do nguy hiem"
    }
  ],
  "shap": {
    "available": true,
    "output_space": "raw_margin",
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
  },
  "explanation": {
    "short_text": "SpO2 92.5% va nhiet do 38.2C dang lam tang nguy co.",
    "clinical_note": "Can doi chieu them trieu chung va chat luong phep do.",
    "recommended_actions": [
      "do lai chi so",
      "doi chieu trieu chung"
    ]
  }
}
```

## 3. Y nghia cac truong

### 3.1. `status`

- `ok`: suy luan thanh cong
- `error`: loi trong pipeline

### 3.2. `meta`

- `model_family`: nhom model `sleep`, `healthguard`, `fall`
- `model_name`: ten logic de backend route
- `model_version`: version artifact hoac business version
- `artifact_type`: hien tai nen la `python_bundle`
- `artifact_path`: duong dan artifact hien dung
- `timestamp`: thoi diem tra ket qua
- `request_id`: id de trace log

### 3.3. `input_ref`

Khong can tra lai full raw input trong response production. Nen chi tra:

- `user_id`
- `device_id`
- `event_timestamp`
- `source_file` hoac `source_type`

Neu mode debug moi them `raw_input`.

### 3.4. `prediction`

Truong dung chung:

- `prediction_label`: nhan sau cung
- `prediction_score`: diem chinh de sap xep/ranking
- `prediction_band`: `normal | info | warning | critical`
- `requires_attention`
- `high_priority_alert`
- `confidence`

### 3.5. `top_features`

Day la danh sach feature quan trong nhat tu SHAP/native contribution.

Moi item nen co:

- `feature`: ten de nguoi doc hieu
- `feature_value`: gia tri thuc te
- `impact`: do lon anh huong, luon la so duong
- `direction`: `risk_up` hoac `risk_down`
- `reason`: 1 cau ngan gon de hien thi UI/Gemini

Khuyen nghi:

- tra top `3` hoac top `5`
- sap xep theo `impact` giam dan
- uu tien feature co the hanh dong/doi chieu duoc; khong dua cac bien ho so tinh len man hinh chinh
  nhu `age`, `gender`, `weight`, `height` neu van con cac chi so sinh hoc/hanh vi co y nghia hon

### 3.6. `explanation`

Phan nay la text da tong hop de API frontend/Gemini hien thi.

- `short_text`: 1-2 cau ngan, bam sat chi so
- `clinical_note`: ghi chu cho nguoi co chuyen mon
- `recommended_actions`: toi da 2-3 action

### 3.7. `shap`

Phan nay giu du lieu contribution day du de backend/frontend co the render dashboard, waterfall, debug, audit.

- `available`: model co tra duoc SHAP/native contribution hay khong
- `output_space`: khong gian output cua SHAP, vi du `raw_margin` hoac `prediction`
- `base_value`: gia tri nen cua model trong output space tuong ung
- `prediction_value`: score/probability backend dang tra ra de client doi chieu
- `values`: danh sach contribution theo feature

Moi item trong `values` nen co:

- `feature`
- `feature_value`
- `shap_value`: gia tri SHAP co dau
- `impact`: `abs(shap_value)`
- `direction`: `risk_up` hoac `risk_down`

Luu y:

- voi model phan loai, `base_value + sum(shap_value)` co the nam o `raw_margin`, khong nhat thiet bang probability
- `top_features` nen duoc sinh tu `shap.values` sau khi sap xep theo `impact`

## 4. Mapping rieng cho tung model

## 4.1. Sleep

### Prediction mapping

```json
{
  "prediction_label": "poor",
  "prediction_score": 58.6,
  "prediction_band": "warning",
  "requires_attention": true,
  "high_priority_alert": false,
  "confidence": 0.586
}
```

Quy uoc:

- `prediction_label` = `critical | poor | fair | good | excellent`
- `prediction_score` = `predicted_sleep_score`
- `confidence` = `predicted_sleep_score / 100`
- `prediction_band`
  - `critical` neu score `< 50`
  - `warning` neu score `50-<60`
  - `info` neu score `60-<75`
  - `normal` neu score `>= 75`

### Top features khuyen nghi

Feature Sleep nen uu tien hien thi:

- `sleep_efficiency_pct`
- `duration_minutes`
- `stress_score`
- `spo2_mean_pct`
- `sleep_latency_minutes`
- `wake_after_sleep_onset_minutes`

Vi du:

```json
[
  {
    "feature": "sleep_efficiency_pct",
    "feature_value": 72.5,
    "impact": 1.48,
    "direction": "risk_up",
    "reason": "hieu suat ngu thap dang lam giam sleep score"
  },
  {
    "feature": "stress_score",
    "feature_value": 81,
    "impact": 0.56,
    "direction": "risk_up",
    "reason": "stress cao dang lam xau danh gia giac ngu"
  }
]
```

## 4.2. HealthGuard

### Prediction mapping

```json
{
  "prediction_label": "high_risk",
  "prediction_score": 0.812,
  "prediction_band": "critical",
  "requires_attention": true,
  "high_priority_alert": true,
  "confidence": 0.812
}
```

Quy uoc:

- `prediction_label` = output hien tai `normal | high_risk`
- `prediction_score` = `predicted_health_risk_probability`
- `confidence` = probability
- `prediction_band`
  - `critical` neu `high_risk`
  - `normal` neu `normal`

Neu sau nay co `warning` semantic that:

- `normal | warning | critical`

### Top features khuyen nghi

Feature HealthGuard nen uu tien hien thi:

- `spo2`
- `body_temperature`
- `systolic_blood_pressure`
- `diastolic_blood_pressure`
- `heart_rate`
- `respiratory_rate`
- `derived_bmi`
- `derived_map`

Vi du:

```json
[
  {
    "feature": "spo2",
    "feature_value": 92.5,
    "impact": 0.43,
    "direction": "risk_up",
    "reason": "SpO2 thap dang lam tang nguy co"
  },
  {
    "feature": "body_temperature",
    "feature_value": 38.2,
    "impact": 0.21,
    "direction": "risk_up",
    "reason": "nhiet do tang dang day muc canh bao len cao hon"
  }
]
```

## 4.3. Fall

### Prediction mapping

```json
{
  "prediction_label": "critical_fall",
  "prediction_score": 0.91,
  "prediction_band": "critical",
  "requires_attention": true,
  "high_priority_alert": true,
  "confidence": 0.91
}
```

Quy uoc:

- `prediction_label` = `normal | possible_fall | likely_fall | critical_fall`
- `prediction_score` = `predicted_fall_probability`
- `confidence` = probability
- `prediction_band`
  - `normal` neu `< 0.50`
  - `warning` neu `0.50-<0.85`
  - `critical` neu `>= 0.85`

### Top features khuyen nghi

Feature Fall hien tai la feature engineered, nen backend/UI nen hien thi ten de doc:

- `floor_vibration_mean`
- `accel_x_range`
- `accel_mag_max`
- `gyro_mag_max`
- `orientation_dispersion`
- `environment_contact_score`

Vi du:

```json
[
  {
    "feature": "floor_vibration_mean",
    "feature_value": 0.68,
    "impact": 0.90,
    "direction": "risk_up",
    "reason": "rung san tang trong luc event dang lam tang kha nang te nga"
  },
  {
    "feature": "accel_x_range",
    "feature_value": 4.99,
    "impact": 0.52,
    "direction": "risk_up",
    "reason": "bien do gia toc lon dang giong mau te nga"
  }
]
```

## 5. Quy tac tao `reason` cho top_features

Nen tao `reason` theo template, khong de frontend/Gemini tu phan doan hoan toan.

Template khuyen nghi:

- neu `direction = risk_up`
  - `"{{feature}}={{value}} dang lam tang nguy co"`
- neu `direction = risk_down`
  - `"{{feature}}={{value}} dang lam giam nguy co"`

Co the them bang map rieng:

- `spo2` -> `"SpO2 thap dang lam tang nguy co"`
- `body_temperature` -> `"nhiet do tang dang day muc canh bao len cao hon"`
- `sleep_efficiency_pct` -> `"hieu suat ngu thap dang lam giam sleep score"`
- `floor_vibration_mean` -> `"rung san tang dang giong event te nga"`

## 6. Contract loi chuan

```json
{
  "status": "error",
  "meta": {
    "model_family": "healthguard",
    "model_name": "healthguard",
    "model_version": "v_current",
    "artifact_type": "python_bundle",
    "artifact_path": "Modelok/healthguard/healthguard_bundle.joblib",
    "timestamp": "2026-04-17T00:00:00+07:00",
    "request_id": "req_123456"
  },
  "error_code": "INFERENCE_FAILED",
  "error_message": "cannot compute prediction",
  "prediction": null,
  "top_features": [],
  "explanation": null
}
```

## 7. Goi y endpoint

Khuyen nghi 3 endpoint rieng:

- `POST /api/v1/predict/sleep`
- `POST /api/v1/predict/healthguard`
- `POST /api/v1/predict/fall`

Hoac 1 endpoint chung:

- `POST /api/v1/predict`

Neu dung endpoint chung, request nen co:

```json
{
  "model_family": "healthguard",
  "payload": { ... }
}
```

## 8. Ban toi gian de backend tich hop ngay

Neu can ra nhanh, toi thieu hay giu:

```json
{
  "status": "ok",
  "prediction": {
    "prediction_label": "high_risk",
    "prediction_score": 0.812,
    "prediction_band": "critical",
    "requires_attention": true,
    "high_priority_alert": true,
    "confidence": 0.812
  },
  "top_features": [
    {
      "feature": "spo2",
      "feature_value": 92.5,
      "impact": 0.4281,
      "direction": "risk_up",
      "reason": "SpO2 thap dang lam tang nguy co"
    }
  ],
  "explanation": {
    "short_text": "SpO2 92.5% va nhiet do 38.2C dang lam tang nguy co.",
    "clinical_note": "Can doi chieu them trieu chung va chat luong phep do.",
    "recommended_actions": [
      "do lai chi so",
      "doi chieu trieu chung"
    ]
  }
}
```

## 9. Ket luan

Neu backend cua ban bam theo spec nay thi ca 3 model `.joblib` se tra ve cung 1 ngon ngu:

- `prediction` de business logic xu ly
- `top_features` de SHAP/native contribution hien thi
- `explanation` de UI va Gemini dung lai

Day la format khuyen nghi de dua 3 model vao API production ma khong can moi model mot contract rieng.
