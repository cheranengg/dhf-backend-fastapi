# Dockerfile (CPU, Cloud Run)
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps for scipy/numpy/faiss/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/

# CPU wheels only (no CUDA)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App code
COPY app /app/app

# Optional caches to speed up HF/Torch pulls
ENV TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch
RUN mkdir -p /cache/hf /cache/torch

# Cloud Run expects PORT env
ENV PORT=8080
EXPOSE 8080

# One worker on Cloud Run; keepalive helps with readiness
ENV UVICORN_WORKERS=1
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8080", "--timeout-keep-alive","65"]
