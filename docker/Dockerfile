# 1) CUDA Runtime + PyTorch（已含對應 CUDA）
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# 2) 系統相依：libsndfile for soundfile；基本編譯工具給 webrtcvad（有時不需要，但保險）
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 3) 工作目錄
WORKDIR /app

# 4) 複製 requirements 並安裝
#    注意：請確保 docker build 的 context 在 docker 目錄，且 ../src 路徑有效
COPY ../src/requirements.txt /app/requirements.txt

# 避免安裝 CPU 版 torch：requirements.txt 內不要有 torch
ENV PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# 可選：加速下載（需要網路環境允許）
# RUN pip install -U hf-transfer && \
#     echo "export HF_HUB_ENABLE_HF_TRANSFER=1" >> /etc/environment
# ENV HF_HUB_ENABLE_HF_TRANSFER=1

# 5) 複製程式碼
COPY ../src /app

# 6) 暴露 FastAPI 端口
EXPOSE 5000

# 7) 一些 GPU 小優化（可選）
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    TORCH_CUDNN_V8_API_DISABLED=1

# 8) 啟動服務
# 若你的主程式檔名就是 app.py 且內有 app=FastAPI()，維持如下
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000", "--no-server-header"]
