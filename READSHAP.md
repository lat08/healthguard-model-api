# READSHAP

Tai lieu nay mo ta cach doc block `shap` moi trong output API va mot so ham helper de frontend/backend dung lai nhanh.

## 1. Vi tri SHAP trong response

Predict endpoints van tra:

```json
{
  "success": true,
  "results": [
    {
      "status": "ok",
      "meta": { "...": "..." },
      "input_ref": { "...": "..." },
      "prediction": { "...": "..." },
      "top_features": [],
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
      "explanation": { "...": "..." }
    }
  ],
  "total": 1
}
```

## 2. Y nghia nhanh

- `shap.available`: model co tra duoc contribution hay khong.
- `shap.output_space`: `raw_margin` hoac `prediction`.
- `shap.base_value`: baseline cua model trong output space tuong ung.
- `shap.prediction_value`: score/probability da show trong `prediction`.
- `shap.values[]`: danh sach contribution day du theo feature.

Quy uoc hien tai:

- `shap.values` giu day du tat ca feature de audit/debug.
- `top_features` la lop hien thi patient-facing/clinician-facing da duoc loc bot cac bien ho so tinh
  va it hanh dong duoc, vi du `weight_kg`, `height_m`, `height_cm`, `age`, `gender`, `device_model`, `timezone`.
- Neu can xem toan bo contribution, dung `shap.values`; neu can UI gon cho nguoi dung, dung `top_features`.

Luu y:

- Neu `output_space = raw_margin`, tong `base_value + sum(shap_value)` la raw output cua model, khong nhat thiet bang probability.
- `top_features` la ban rut gon tu `shap.values`, phu hop de render UI nhanh.

## 3. Python helper

```python
from typing import Any


def get_shap_block(api_response: dict[str, Any], index: int = 0) -> dict[str, Any] | None:
    results = api_response.get("results") or []
    if index >= len(results):
        return None
    return results[index].get("shap")


def top_shap_values(api_response: dict[str, Any], index: int = 0, limit: int = 5) -> list[dict[str, Any]]:
    shap = get_shap_block(api_response, index=index) or {}
    values = shap.get("values") or []
    return sorted(values, key=lambda item: float(item.get("impact", 0.0)), reverse=True)[:limit]


def risk_up_features(api_response: dict[str, Any], index: int = 0) -> list[dict[str, Any]]:
    return [
        item
        for item in top_shap_values(api_response, index=index, limit=999)
        if item.get("direction") == "risk_up"
    ]
```

## 4. TypeScript / JavaScript helper

```ts
type ShapValue = {
  feature: string;
  feature_value: unknown;
  shap_value: number;
  impact: number;
  direction: "risk_up" | "risk_down";
};

type PredictResult = {
  prediction?: { prediction_label?: string; prediction_score?: number };
  top_features?: Array<Record<string, unknown>>;
  shap?: {
    available?: boolean;
    output_space?: "raw_margin" | "prediction" | string;
    base_value?: number | null;
    prediction_value?: number | null;
    values?: ShapValue[];
  };
};

export function getShap(result: PredictResult): ShapValue[] {
  return [...(result.shap?.values ?? [])].sort((a, b) => b.impact - a.impact);
}

export function getRiskDrivers(result: PredictResult, limit = 5): ShapValue[] {
  return getShap(result)
    .filter((item) => item.direction === "risk_up")
    .slice(0, limit);
}

export function getTopFeatureNames(result: PredictResult, limit = 3): string[] {
  return getShap(result)
    .slice(0, limit)
    .map((item) => item.feature);
}
```

## 5. Goi y render UI

- Man hinh tong quan: dung `top_features`.
- Drawer/detail/debug: dung `shap.values`.
- Canh bao nhanh: uu tien `direction = risk_up`.
- Neu can waterfall chart: dung `base_value` + `shap.values`, nhung nho kiem tra `output_space`.

## 6. Goi y tich hop

- Backend can log `meta.request_id` cung `prediction` va `shap.values[:5]`.
- Frontend nen fallback ve `top_features` neu `shap.available = false`.
- Khong nen tu suy dien probability tu `raw_margin` neu model dang tra `output_space = raw_margin`.

## 7. Ghi chu thiet ke

Rule loc `top_features` duoc chon theo huong user-centered/actionable explanations:

- uu tien cac chi so co the theo doi hoac can thiep duoc
- giam xuat hien bien profile/demographic bat bien trong man hinh chinh
- van giu contribution day du trong `shap.values` de phuc vu audit, fairness review, va debug
