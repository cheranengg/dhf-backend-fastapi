# Base with CUDA 12.1 + cuDNN8 runtime
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Make python/pip the defaults
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python -m pip install --upgrade pip

# Workdir
WORKDIR /app

# Copy and install Python deps (except torch — install below with CUDA wheels)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install CUDA-enabled PyTorch (matches CUDA 12.1 base)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch torchvision torchaudio

# Copy app code
COPY app /app/app

# (Optional) Set model/cache dirs; helpful for cold starts
ENV TRANSFORMERS_CACHE=/cache/hf \
    HF_HOME=/cache/hf \
    TORCH_HOME=/cache/torch
RUN mkdir -p /cache/hf /cache/torch

# Expose port used by uvicorn
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "app.main:app", "--host","0.0.0.0", "--port","8080"]
