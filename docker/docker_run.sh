#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 1) 匯入 .env 讓主機 bash 能展開 ${...}
set -a
. ./config/.env
set +a

# 2) 防呆
: "${HUGGINGFACE_API_KEY:?HUGGINGFACE_API_KEY 未設定（$ENV_FILE）}"    # 沒設就直接中止
mkdir -p "$PWD/models"

# 3) 起容器（仍保留 --env-file 給容器環境）
docker rm -f qwen25-asr 2>/dev/null || true
docker run -d --name qwen25-asr \
  --gpus all \
  --ipc=host \
  --network pgi_llm_network \
  --network-alias qwen25-asr \
  --env-file ./config/.env \
  -e HF_HOME=/models \
  -e HUGGINGFACE_HUB_CACHE=/models/hub \
  -v "$PWD/models:/models" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /etc/timezone:/etc/timezone:ro \
  --restart unless-stopped \
  qwen2.5-omni-7b