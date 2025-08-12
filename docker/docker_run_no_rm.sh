docker run --gpus all -d --name qwen2.5-omni-7b \
    -v /etc/localtime:/etc/localtime:ro \
    -v /etc/timezone:/etc/timezone:ro \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HOME=/root/.cache/huggingface \
    -e TRANSFORMERS_CACHE=/root/.cache/huggingface \
    -p 5005:5000 \
    --network llm_network \
    qwen2.5-omni-7b
