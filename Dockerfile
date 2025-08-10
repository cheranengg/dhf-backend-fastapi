# Dockerfile (CPU)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps for torch/transformers + wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# CPU wheels only (no CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app /app/app

# cache dirs (optional)
ENV TRANSFORMERS_CACHE=/cache/hf HF_HOME=/cache/hf TORCH_HOME=/cache/torch
RUN mkdir -p /cache/hf /cache/torch

# Read $PORT from Cloud Run
ENV PORT=8080
EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8080"]
