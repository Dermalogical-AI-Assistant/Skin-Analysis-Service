# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the requirements.txt to utilize cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app ./app
COPY set_up_ml_mode.py .

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Create directories for models
RUN mkdir -p ./app/ml_models/yolo/weights \
    && mkdir -p ./app/ml_models/mobileViT/weights \
    && mkdir -p ./app/ml_models/convnext

# Expose the default FastAPI port
EXPOSE 4002

# Use the entrypoint script
ENTRYPOINT ["./entrypoint.sh"]