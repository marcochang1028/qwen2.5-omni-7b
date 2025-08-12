import os
import logging
import threading
import time
import gc
from dotenv import load_dotenv
import torch
import librosa
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor

# 載入 .env 文件
load_dotenv()

# 初始化 logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(level=log_level, format=log_format, force=True)
logger = logging.getLogger(__name__)

# 取得 Hugging Face API Key 與模型名稱
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not huggingface_api_key:
    raise RuntimeError("Hugging Face API Key 未設定，請在 .env 檔案中加入 HUGGINGFACE_API_KEY")

model_name = "Qwen/Qwen2.5-Omni-7B"

# 模型與處理器變數（初始為 None，等有請求時才載入）
model = None
processor = None

# 設定模型閒置時間（秒），並確保轉成 int
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", 300))
last_used_time = None  # 記錄最後使用時間
timeout_thread = None  # 計時器線程

# FastAPI 應用
app = FastAPI()

def monitor_idle_time():
    """
    每 10 秒檢查一次，若超過指定閒置時間，釋放模型與 GPU 資源。
    """
    global model, processor, last_used_time
    while True:
        time.sleep(10)  # 每 10 秒檢查一次
        if model and last_used_time and (time.time() - last_used_time > IDLE_TIMEOUT):
            logger.info("模型閒置超時，釋放 GPU...")
            # 刪除模型與處理器
            del model
            del processor
            # 強制垃圾回收
            gc.collect()
            # 清空 PyTorch CUDA 緩存
            torch.cuda.empty_cache()
            # 回收共享記憶體
            torch.cuda.ipc_collect()

            # 重置變數
            model = None
            processor = None
            last_used_time = None
            logger.info("模型已成功釋放")

def start_timeout_monitor():
    """
    啟動監控閒置時間的執行緒（只需啟動一次）。
    """
    global timeout_thread
    if timeout_thread is None:
        timeout_thread = threading.Thread(target=monitor_idle_time, daemon=True)
        timeout_thread.start()

def load_model():
    """
    有請求時才載入模型；若已載入則只更新最後使用時間。
    """
    global model, processor, last_used_time
    if model is None:
        try:
            logger.info("載入處理器與模型...")
            processor = AutoProcessor.from_pretrained(model_name)
            model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16
            )
            model.to("cuda:0")  # 強制移動至 GPU 0
            logger.info("模型載入完成")
        except Exception as e:
            logger.error(f"模型載入失敗: {e}")
            raise RuntimeError(f"Failed to load model. Error: {e}")

    last_used_time = time.time()  # 更新最後使用時間
    start_timeout_monitor()       # 確保閒置監控已啟動

@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    prompt_message: str = "請僅輸出音檔中的語音內容，不要加入其他資訊。"
):
    """
    上傳音訊文件，並使用 Qwen2.5-Omni-7B 進行轉錄。
    """
    logger.info("開始轉錄音訊")

    if not file:
        logger.error("未提供檔案")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # 確保模型已載入
        load_model()

        # 讀取音訊內容
        logger.info("讀取上傳的音訊內容...")
        audio_data = await file.read()
        logger.info(f"音訊資料長度: {len(audio_data)} bytes")

        # 使用 librosa 讀取音檔
        audio, _ = librosa.load(BytesIO(audio_data), sr=processor.feature_extractor.sampling_rate)
        logger.info(f"音訊形狀: {audio.shape} (單聲道)")

        # 建立 Qwen2.5-Omni 的對話模板
        logger.info("準備聊天模板...")
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt_message}
                ]
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        logger.info("聊天模板建置完成")

        # 準備模型輸入
        logger.info("準備模型輸入...")
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to("cuda:0") for k, v in inputs.items()}  # 移動到 GPU
        logger.info("模型輸入準備完成")

        # 進行推理
        logger.info("開始模型推理...")
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, return_audio=False)
        logger.info("推理完成，處理結果...")

        # 只取出新增生成的部分
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        # 轉錄結果
        transcription = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        logger.info(f"音訊轉錄完成: {transcription}")

        return JSONResponse(content={'transcription': transcription})

    except Exception as e:
        logger.error(f"處理音訊時出現錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}")

@app.post("/audio/chat")
async def chat_with_audio(
    file: UploadFile = File(...),
    message: str = "請分析這個音檔的內容",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    speaker: str = "Chelsie"
):
    """
    上傳音訊文件，並與 Qwen2.5-Omni-7B 進行對話。
    """
    logger.info("開始音訊對話")

    if not file:
        logger.error("未提供檔案")
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # 確保模型已載入
        load_model()

        # 讀取音訊內容
        logger.info("讀取上傳的音訊內容...")
        audio_data = await file.read()
        logger.info(f"音訊資料長度: {len(audio_data)} bytes")

        # 使用 librosa 讀取音檔
        audio, _ = librosa.load(BytesIO(audio_data), sr=processor.feature_extractor.sampling_rate)
        logger.info(f"音訊形狀: {audio.shape} (單聲道)")

        # 建立 Qwen2.5-Omni 的對話模板
        logger.info("準備聊天模板...")
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": message}
                ]
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        logger.info("聊天模板建置完成")

        # 準備模型輸入
        logger.info("準備模型輸入...")
        inputs = processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  # 移動到模型所在設備
        logger.info("模型輸入準備完成")

        # 進行推理
        logger.info("開始模型推理...")
        generate_ids, audio_output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature, return_audio=True, speaker=speaker)
        logger.info("推理完成，處理結果...")

        # 只取出新增生成的部分
        generate_ids = generate_ids[:, inputs["input_ids"].size(1):]

        # 文字回應
        text_response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        logger.info(f"對話完成: {text_response}")

        return JSONResponse(content={
            'text_response': text_response,
            'audio_available': audio_output is not None
        })

    except Exception as e:
        logger.error(f"處理音訊對話時出現錯誤: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing audio chat: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
