import os
import logging
import threading
import time
import gc
import tempfile
from io import BytesIO

import soundfile as sf
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

# ===== Basic setup =====
load_dotenv()
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger("qwen2.5-omni-asr-basic")

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY 未設定")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))

app = FastAPI()
processor = None
thinker = None
last_used_time = None
timeout_thread = None

# ===== Helpers =====
def _touch():
    global last_used_time
    last_used_time = time.time()

def _idle_monitor():
    global processor, thinker, last_used_time
    while True:
        time.sleep(10)
        if last_used_time and (time.time() - last_used_time > IDLE_TIMEOUT):
            try:
                if thinker is not None: del thinker
                if processor is not None: del processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            finally:
                thinker = None
                processor = None
                last_used_time = None
                logger.info("資源已釋放")

def _start_monitor_once():
    global timeout_thread
    if timeout_thread is None:
        timeout_thread = threading.Thread(target=_idle_monitor, daemon=True)
        timeout_thread.start()

def _load_proc():
    global processor
    if processor is None:
        logger.info("載入 Processor ...")
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        logger.info("Processor 載入完成")

def _load_thinker():
    global thinker
    if thinker is None:
        logger.info("載入 Thinker 模型 ...")
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype="auto", device_map="auto", token=HF_TOKEN
        )
        logger.info("Thinker 模型載入完成")

def _bytes_to_wav_path(data: bytes) -> str:
    # 讀任何格式 -> 單聲道 WAV，交給 Processor 做該做的事
    audio, sr = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    if hasattr(audio, "ndim") and audio.ndim > 1:
        audio = audio.mean(axis=1)
    fd, path = tempfile.mkstemp(suffix=".wav"); os.close(fd)
    sf.write(path, audio, sr)
    return path

def _eos_ids(proc):
    tok = proc.tokenizer
    out = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None: out.append(im_end)
    if tok.eos_token_id is not None: out.append(tok.eos_token_id)
    # 去重
    return list(dict.fromkeys(out))

def _transcribe_once(wav_path: str, proc, model, max_new_tokens: int) -> str:
    # === vLLM 風格 prompt（官方 system + user: audio + 指令），讓 apply_chat_template 產生:
    # <|im_start|>system ... <|im_end|>\n<|im_start|>user\n<AUDIO TOKENS> 指令<|im_end|>\n<|im_start|>assistant
    default_system = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                      "capable of perceiving auditory and visual inputs, as well as generating text and speech.")
    conversations = [
        {"role": "system", "content": [{"type": "text", "text": default_system}]},
        {"role": "user", "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text", "text": "ASR task: transcribe the audio verbatim. Output only the words actually spoken. If no speech, output [NO-SPEECH]."}
        ]},
    ]

    inputs = proc.apply_chat_template(
        conversations,
        add_generation_prompt=True,   # 添加 <|im_start|>assistant
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,                 # 不取樣，避免離題
        eos_token_id=_eos_ids(proc)      # 靠 <|im_end|>/eos 結束
    )

    # 只解碼新增段
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()

# ===== API =====
@app.post("/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...), max_new_tokens: int = 512):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        _load_proc()
        _load_thinker()

        data = await file.read()
        wav_path = _bytes_to_wav_path(data)

        transcript = _transcribe_once(wav_path, processor, thinker, max_new_tokens)

        try: os.remove(wav_path)
        except: pass

        _touch(); _start_monitor_once()
        return JSONResponse(content={"transcription": transcript})
    except Exception as e:
        logger.exception("ASR 發生錯誤")
        raise HTTPException(status_code=500, detail=f"Error processing audio file: {e}") from e

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    uvicorn.run(app, host="0.0.0.0", port=5000)
