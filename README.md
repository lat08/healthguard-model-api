# HealthGuard API (model_be)

Backend **FastAPI** — suy luận **Fall / Health / Sleep** từ file model trong **`models/`** (joblib).

## Chạy nhanh (Windows, PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- **Swagger:** http://127.0.0.1:8000/docs  
- **Trạng thái model:** http://127.0.0.1:8000/health  

Cần **Python 3.12**. Chi tiết API: **`docs/API_REFERENCE.md`**.

---

## Thư mục `data/runtime/.../cases` là gì?

Đây là **file JSON mẫu** để bạn **mở → copy toàn bộ → dán vào ô Request** trên Swagger khi gọi `POST /predict` (không cần tự gõ từ đầu).

| Thư mục | Dùng cho API | Nội dung mỗi file |
|----------|----------------|-------------------|
| **`data/runtime/fall/cases/`** | `POST /api/v1/fall/predict` | **Một cửa sổ IMU** (một object JSON: `device_id`, `data` gồm đủ mẫu, …). |
| **`data/runtime/health/cases/`** | `POST /api/v1/health/predict` | **Một lần gọi** dạng `{"records":[…]}` — thường 1–n bản ghi sinh hiệu. |
| **`data/runtime/sleep/cases/`** | `POST /api/v1/sleep/predict` | **Một lần gọi** dạng `{"records":[…]}` — thường một (hoặc nhiều) đêm ngủ. |

Một số file tên **`batch_*.json`** là **nhiều mẫu trong một lần gọi** (ví dụ nhiều đêm ngủ cùng một `user_id` trong `batch_multi_nights_one_user.json`). Vẫn dùng **`POST /predict`** trên Swagger — **không** phải phương thức HTTP **`PATCH`** (repo này không có API PATCH cho predict).

Hướng dẫn thêm: `data/runtime/BATCH_PREDICT_SAMPLES.txt`.

Các file này được **tạo lại** khi chạy script bên dưới (không sửa tay thường xuyên).

---

## Script (trong venv, từ gốc repo)

```powershell
python scripts\inspect_modelok.py
python scripts\build_runtime_samples.py
python scripts\build_fall_sample_cases.py
python scripts\build_health_sample_cases.py
python scripts\build_sleep_sample_cases.py
python scripts\build_predict_batch_samples.py
```

- **`inspect_modelok.py`** — thử load bundle trong `models/`.  
- **`build_runtime_samples.py`** — copy CSV mẫu vào `data/runtime/` (`--dataset-profile v1` hoặc `v2`).  
- **`build_*_sample_cases.py`** — sinh `iot_sample_cases.json` + nội dung trong các thư mục **`cases/`** ở trên.  
- **`build_predict_batch_samples.py`** — sinh thêm file **`batch_*.json`** trong `cases/`.

---

## Test

```powershell
pytest
```

---

## Ghi chú ngắn

- PowerShell chặn script venv: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`  
- Giữ **`pip install -r requirements.txt`** (sklearn đã ghim tương thích bundle).  
- Linux/mac: `source .venv/bin/activate`, đổi `python scripts\...` thành `python scripts/...`.
