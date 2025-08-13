# 1) CUDA Runtime + PyTorch（已含對應 CUDA）
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# 2) 系統相依
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) 工作目錄
WORKDIR /app

# 4) 安裝依賴
COPY ../src/requirements.txt /app/requirements.txt

ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TRANSFORMERS_VERBOSITY=error

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 5) 複製程式碼
COPY ../src /app

# 6) 暴露 FastAPI 端口
EXPOSE 5000

# 7) GPU 小優化（可選）
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TORCH_CUDNN_V8_API_DISABLED=1

# 8) 啟動服務（單進程，GPU 安全）
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--no-server-header"]
