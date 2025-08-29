# =======================
# Stage A: .py → .pyc 編譯器
# 目的：把應用程式的 .py 預編譯為 sourceless .pyc，加速冷啟動
# =======================
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime AS pyc-builder
WORKDIR /build

# 複製原始碼（注意路徑：假設 Dockerfile 與 src 平行）
COPY ../src /build/app

# 產出「平放」的 .pyc（-b），用 checked-hash 驗證；再刪 .py 與空 __pycache__
RUN python -m compileall -f -j 0 --invalidation-mode=checked-hash -b /build/app \
 && find /build/app -type f -name "*.py" -delete \
 && find /build/app -type d -name "__pycache__" -empty -delete

# =======================
# Stage B: Final runtime
# =======================
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime
WORKDIR /app

# 系統相依（音訊 I/O；也保留 build-essential 以便 webrtcvad 這類套件需要本機建置）
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      libsndfile1 ffmpeg ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

# 主要環境設定（延續 phi4 風格）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    CUDA_DEVICE_MAX_CONNECTIONS=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    PYTHONPATH=/app

# 先安裝依賴（用快取；requirements 變動才會重裝）
COPY ../src/requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir --prefer-binary -r /app/requirements.txt

# 放入已編好的 .pyc-only 程式碼（沒有 .py）
COPY --from=pyc-builder /build/app /app

# 對外服務埠（與 uvicorn --port 對應）
EXPOSE 8000

# 單進程、GPU 友好、關閉 server header（與 phi4 一致）
# 確認應用主模組為 app 且內有 app = FastAPI()
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
