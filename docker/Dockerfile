# 1. 使用包含 CUDA 開發工具的 Docker Image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 2. 設定工作目錄
WORKDIR /app

# 3. 複製 requirements.txt 並安裝 Python 依賴
COPY ../src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4. 安裝 Flash Attention 2 (可選，用於加速生成)
#RUN pip install -U flash-attn --no-build-isolation

# 5. 將當前目錄的內容複製到工作目錄中
COPY ../src /app

# 6. 暴露 FastAPI 端口
EXPOSE 5000

# 7. 啟動 API 服務
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
