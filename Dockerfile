# syntax=docker/dockerfile:1.7
# GPU-friendly base; PyTorch CUDA libs will come from wheels
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (FAISS/numpy/torch use libgomp1); git for HF pulls if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- deps layer (cacheable) ----------
COPY requirements.txt /app/requirements.txt

# Install CUDA-enabled PyTorch first to avoid resolver conflicts
# cu121 wheels include the user-space CUDA runtime
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 && \
    pip install --no-cache-dir -r requirements.txt

# ---------- app code ----------
COPY app /app/app

# ---------- runtime caches ----------
ENV HF_HOME=/cache/hf \
    TRANSFORMERS_CACHE=/cache/hf \
    TORCH_HOME=/cache/torch \
    HF_HUB_ENABLE_HF_TRANSFER=1
RUN mkdir -p /cache/hf /cache/torch

# ---------- OPTIONAL: model warmup (no-op if local model dirs absent) ----------
COPY app/utils/warmup.py /app/app/utils/warmup.py
ENV TM_MODEL_DIR=/app/app/models/mistral_finetuned_Trace_Matrix \
    DVP_MODEL_DIR=/app/app/models/mistral_finetuned_Design_Verification_Protocol
RUN python -u - <<'PY'
import sys
try:
    import app.utils.warmup as W
    W.run()
    print("Warmup finished.")
except Exception as e:
    print("Warmup skipped:", e, file=sys.stderr)
PY

# ---------- defaults (can be overridden in Cloud Run) ----------
# These let you switch between stub vs real models. Keep ON for GPU run.
ENV USE_HA_MODEL=1 \
    USE_DVP_MODEL=1 \
    USE_TM_MODEL=1

# Cloud Run provides $PORT; default for local runs
ENV PORT=8080
EXPOSE 8080

# One worker is safer for big models; keepalive helps readiness
ENV UVICORN_WORKERS=1
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8080", "--timeout-keep-alive","65"]
