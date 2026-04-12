Sleep — POST /api/v1/sleep/predict
Mỗi file *.json là body {"records": [...]} (một dòng CSV = một bản ghi).
Mở file → Ctrl+A → copy → dán Request body trên /docs.
Nguồn: smartwatch_sleep_dataset.csv (nhãn daily_label / sleep_score chỉ để đối chiếu, không gửi trong request).
