#!/bin/bash
set -e

python set_up_ml_mode.py

# Chạy ứng dụng FastAPI
exec uvicorn app.main:app --host 0.0.0.0 --port 4002