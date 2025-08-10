# syntax=docker/dockerfile:1.7
# Dockerfile (CPU, Cloud Run, cached)

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# system deps for scipy/numpy/faiss/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- deps layer (cacheable) ----------
COPY requirements.txt /app/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---------- app code ----------
COPY app /app/app

# ---------- runtime caches ----------
ENV TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch
RUN mkdir -p /cache/hf /cache/torch

# ---------- OPTIONAL: model warmup (safe to keep; no-op if folders absent) ----------
# If you've committed fine-tuned model dirs under app/models, this verifies they load
COPY app/utils/warmup.py /app/app/utils/warmup.py
# Point to local folders if you have them (adjust names or leave as-is)
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

# ---------- serve ----------
# Cloud Run provides $PORT; we default to 8080 for local runs
ENV PORT=8080
EXPOSE 8080

# One worker is fine on Cloud Run; keepalive helps readiness checks
ENV UVICORN_WORKERS=1
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0","--port","8080", "--timeout-keep-alive","65"]
