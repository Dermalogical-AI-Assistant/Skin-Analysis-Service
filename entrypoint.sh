#!/bin/bash
set -e

# Kiểm tra xem models đã được tải chưa
MODEL_DIR_YOLO="./app/ml_models/yolo/weights"
MODEL_DIR_MOBILEVIT="./app/ml_models/mobileViT/weights"

# Tạo thư mục nếu chưa tồn tại
mkdir -p $MODEL_DIR_YOLO
mkdir -p $MODEL_DIR_MOBILEVIT

# Kiểm tra xem model đã tồn tại chưa
if [ ! -f "${MODEL_DIR_YOLO}/best.pt" ] || [ ! -f "${MODEL_DIR_MOBILEVIT}/best_model.keras" ]; then
    echo "Models are not found. Downloading models..."
    python set_up_ml_mode.py
    echo "Models downloaded successfully."
else
    echo "Models already exist. Skipping download."
fi

# Chạy ứng dụng FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 4002