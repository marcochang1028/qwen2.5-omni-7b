import os
import logging
import threading
import time
import gc
import tempfile
from io import BytesIO
from typing import List, Tuple

import numpy as np
import soundfile as sf
import webrtcvad
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    StoppingCriteria, StoppingCriteriaList,
)

# ===============================
# Qwen2.5-Omni-7B ASR 專注強化版
# - 只用 Thinker（文字輸出）。無 Talker、無語音回覆。
# - 正確多模態餵法：apply_chat_template + {"type":"audio","path":...}
# - VAD 語音端點偵測：webrtcvad（16k/mono/16-bit PCM）
# - 長段落安全切片：max_seg_sec + seg_overlap_sec
# - 文字智慧合併：段與段的重疊去重（suffix/prefix 比對）
# - 生成：thinker_do_sample=False（穩定）
# - 空閒釋放：GPU/記憶體清理
# ===============================

load_dotenv()

# 停止條件類別：避免輸出對話標籤和續寫提示
class StopOnStrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings):
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.generated = ""

    def __call__(self, input_ids, scores, **kwargs):
        # 只解碼新增部分，避免每步全量解碼
        new_text = self.tokenizer.decode(input_ids[0][-1:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        self.generated += new_text
        for s in self.stop_strings:
            if self.generated.endswith(s):
                return True
        return False

# 設定 Transformers 詳細程度以避免警告
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(level=log_level, format=log_format, force=True)
logger = logging.getLogger("qwen2.5-omni-asr-vad")

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY 未設定")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")

# ====== VAD 與切片參數 ======
# VAD aggressiveness: 0(最寬鬆)~3(最嚴格)。越大越容易切斷噪音/弱音。
VAD_AGGR = int(os.getenv("ASR_VAD_AGGR", "2"))

# VAD 分段合併條件
MIN_SPEECH_SEC   = float(os.getenv("ASR_MIN_SPEECH_SEC", "0.3"))   # 小於此長度的語音段丟掉
MAX_SILENCE_SEC  = float(os.getenv("ASR_MAX_SILENCE_SEC", "0.5"))  # 中間靜音 <= 這個就把段合併
PAD_SEC          = float(os.getenv("ASR_PAD_SEC", "0.2"))          # 每段前後補一點，減少截字

# 若單一 VAD 段過長，再次切片避免一次生成過長
MAX_SEG_SEC      = float(os.getenv("ASR_MAX_SEG_SEC", "45.0"))  # 單段最多秒數
SEG_OVERLAP_SEC  = float(os.getenv("ASR_SEG_OVERLAP_SEC", "1.0"))

# 生成參數
MAX_NEW_TOKENS   = int(os.getenv("ASR_MAX_NEW_TOKENS", "512"))

# 閒置釋放
IDLE_TIMEOUT     = int(os.getenv("IDLE_TIMEOUT", "300"))
last_used_time   = None
timeout_thread   = None

app = FastAPI()

processor = None
thinker_model = None

# -------------------- 工具函式 --------------------
def _touch_last_used():
    global last_used_time
    last_used_time = time.time()

def monitor_idle_time():
    global processor, thinker_model, last_used_time
    while True:
        time.sleep(10)
        if last_used_time and (time.time() - last_used_time > IDLE_TIMEOUT):
            logger.info("模型閒置超時，釋放 GPU/CPU 記憶體...")
            try:
                if thinker_model is not None:
                    del thinker_model
                if processor is not None:
                    del processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            finally:
                thinker_model = None
                processor = None
                last_used_time = None
                logger.info("資源釋放完成")

def start_timeout_monitor():
    global timeout_thread
    if timeout_thread is None:
        timeout_thread = threading.Thread(target=monitor_idle_time, daemon=True)
        timeout_thread.start()

def _load_processor():
    global processor
    if processor is None:
        logger.info("載入 Qwen2_5OmniProcessor ...")
        processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        logger.info("Processor 載入完成")

def _load_thinker_model():
    global thinker_model
    if thinker_model is None:
        logger.info("載入 Thinker（text-only）模型 ...")
        thinker_model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
            token=HF_TOKEN,
        )
        logger.info("Thinker 模型載入完成")

def _bytes_to_audio(data: bytes) -> Tuple[np.ndarray, int]:
    """讀取任意格式到 (mono float32, sr)。"""
    audio, sr = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr

def _resample_mono_float32(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """簡單線性內插重取樣（僅用於 VAD，品質足夠）。"""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    n_target = int(round(len(audio) * target_sr / orig_sr))
    if n_target <= 1 or len(audio) <= 1:
        return np.zeros(0, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=n_target, endpoint=True)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    return resampled

def _float32_to_pcm16(x: np.ndarray) -> bytes:
    """float32(-1~1) -> int16 PCM bytes。"""
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def _frame_generator(pcm16: bytes, sample_rate: int, frame_ms: int = 20):
    """產生固定幀長的 bytes 片段給 webrtcvad（20ms/10ms/30ms 任一）。"""
    n = int(sample_rate * (frame_ms / 1000.0)) * 2  # 2 bytes/sample
    for i in range(0, len(pcm16), n):
        chunk = pcm16[i:i+n]
        if len(chunk) == n:
            yield chunk

def _vad_segments(audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    """
    用 webrtcvad 產生語音區間（在 16k/mono/16-bit PCM 上運作）。
    回傳: 在 16k 的 sample index 的 (start, end) 清單。
    """
    target_sr = 16000
    vad = webrtcvad.Vad(VAD_AGGR)

    # 轉到 16k mono
    audio_16k = _resample_mono_float32(audio, sr, target_sr)
    if audio_16k.size == 0:
        return []

    pcm16 = _float32_to_pcm16(audio_16k)

    frame_ms = 20
    frames = list(_frame_generator(pcm16, target_sr, frame_ms))
    if not frames:
        return []

    # 將每幀標記是否有聲
    voiced = [vad.is_speech(f, target_sr) for f in frames]
    frame_len_samples = int(target_sr * (frame_ms / 1000.0))

    segs: List[Tuple[int, int]] = []
    in_speech = False
    start_idx = 0
    last_voiced_idx = -1

    max_silence_frames = int(MAX_SILENCE_SEC / (frame_ms / 1000.0))
    min_speech_frames = max(1, int(MIN_SPEECH_SEC / (frame_ms / 1000.0)))

    for i, v in enumerate(voiced):
        if v and not in_speech:
            # 進入語音段
            in_speech = True
            start_idx = i
            last_voiced_idx = i
        elif v and in_speech:
            last_voiced_idx = i
        elif (not v) and in_speech:
            # 靜音中，但先不立刻結束，看靜音累積
            if (i - last_voiced_idx) > max_silence_frames:
                # 結束語音段
                length_frames = last_voiced_idx - start_idx + 1
                if length_frames >= min_speech_frames:
                    seg_start = start_idx * frame_len_samples
                    seg_end = (last_voiced_idx + 1) * frame_len_samples
                    segs.append((seg_start, seg_end))
                in_speech = False

    # 收尾：若結束時仍在語音段
    if in_speech:
        length_frames = last_voiced_idx - start_idx + 1
        if length_frames >= min_speech_frames:
            seg_start = start_idx * frame_len_samples
            seg_end = (last_voiced_idx + 1) * frame_len_samples
            segs.append((seg_start, seg_end))

    # 加上前後 padding
    pad_samps = int(PAD_SEC * target_sr)
    segs = [(max(0, s - pad_samps), min(len(audio_16k), e + pad_samps)) for s, e in segs]

    # 合併相鄰/重疊的段
    merged: List[Tuple[int, int]] = []
    for s, e in sorted(segs):
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    return merged  # 以 16k 為座標

def _map_16k_to_orig(segs_16k: List[Tuple[int, int]], orig_sr: int, target_sr: int = 16000) -> List[Tuple[int, int]]:
    """把 16k 的 sample index 區間轉回原始取樣率的 index。"""
    mapped = []
    ratio = orig_sr / float(target_sr)
    for s, e in segs_16k:
        mapped.append((int(round(s * ratio)), int(round(e * ratio))))
    return mapped

def _split_long_segment(seg: Tuple[int, int], sr: int, max_sec: float, overlap_sec: float) -> List[Tuple[int, int]]:
    """把太長的語音段切成小片段，含重疊。seg 是原始 sr 的 sample index 區間。"""
    s, e = seg
    max_samps = int(max_sec * sr)
    ovl_samps = int(overlap_sec * sr)
    out = []
    cur = s
    while cur < e:
        end = min(cur + max_samps, e)
        out.append((cur, end))
        if end == e:
            break
        cur = max(end - ovl_samps, s)
    return out

def _write_wav_segment(audio: np.ndarray, sr: int, seg: Tuple[int, int]) -> str:
    """把原始音訊的某段寫成暫存 WAV，回傳路徑。"""
    s, e = seg
    s = max(0, s); e = min(len(audio), e)
    segment = audio[s:e]
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp_path, segment, sr)
    return tmp_path

def _qwen_transcribe_wav(wav_path: str, model, proc, max_new_tokens: int) -> str:
    """
    單段轉錄（保留官方 system prompt；user 內加嚴格 ASR 指示；加入停止準則）
    """
    conversations = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": wav_path},
                {
                    "type": "text",
                    "text": (
                        "ASR task: Transcribe the audio verbatim. "
                        "Output ONLY the words actually spoken in the audio. "
                        "Do NOT include any extra text, explanations, labels, or role tags such as 'Human:', 'HUMAN:', "
                        "'User:', 'Assistant:', 'System:'. "
                        "Do NOT invent, continue, or request to 'continue writing'. "
                        "If there is non-speech or silence only, return [NO-SPEECH]."
                    )
                },
            ],
        },
    ]

    inputs = proc.apply_chat_template(
        conversations,
        add_generation_prompt=True,   # 依官方範例維持 True
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # 停止條件：一旦模型開始產生對話式標籤或續寫提示，就立刻停
    stop_strings = [
        "\nHuman:", "Human:", "\nHUMAN", "HUMAN:",
        "\nUser:", "User:",
        "\nAssistant:", "Assistant:", "ASSISTANT:",
        "\nSystem:", "System:",
        "请续写下面这段话", "請續寫下面這段話", "請繼續", "继续", "續寫"
    ]
    stopping_criteria = StoppingCriteriaList([StopOnStrings(proc.tokenizer, stop_strings)])

    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,             # 穩定、可重現
        temperature=0.0,
        repetition_penalty=1.05,     # 抑制重複段落（可視需要調整/移除）
        stopping_criteria=stopping_criteria
    )

    text = proc.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text.strip()

def _normalize_ws(s: str) -> str:
    return " ".join(s.split())

def _merge_texts_with_overlap(pieces: List[str], min_overlap_chars: int = 10, max_probe: int = 80) -> str:
    """
    逐段合併，若前一段的尾端與下一段的開頭有重疊，去重合併。
    - 探測重疊長度從 min_overlap_chars..max_probe 遞減尋找最長匹配。
    - 簡單且實用，不引入 heavy NLP 對齊。
    """
    out = ""
    for idx, cur in enumerate(pieces):
        if idx == 0:
            out = _normalize_ws(cur)
            continue
        prev = out
        cur_n = _normalize_ws(cur)
        best = 0
        # 取 prev 的尾端與 cur 的開頭做匹配
        for k in range(min(len(prev), max_probe), min_overlap_chars - 1, -1):
            if prev[-k:] == cur_n[:k]:
                best = k
                break
        if best > 0:
            out = prev + cur_n[best:]
        else:
            # 沒有重疊，直接空格銜接
            if prev and not prev.endswith(" "):
                out = prev + " " + cur_n
            else:
                out = prev + cur_n
    return out.strip()

# -------------------- FastAPI 端點 --------------------
@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    高品質 ASR：
    1) webrtcvad 產生語音段（含合併、padding）
    2) 對過長段再切片（帶重疊）
    3) 逐片呼叫 Qwen Thinker 轉錄
    4) 文字重疊去重合併
    """
    logger.info("開始轉錄音訊")
    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        _load_processor()
        _load_thinker_model()

        raw = await file.read()
        logger.info("收到音訊 bytes=%d", len(raw))
        audio, sr = _bytes_to_audio(raw)

        # === 1) VAD 產生語音段（先在 16k 上切，再映回原 sr）===
        segs_16k = _vad_segments(audio, sr)
        if not segs_16k:
            logger.info("VAD 未偵測到語音，直接嘗試整檔切片轉錄")
            # Fallback：整檔當一段
            segs_orig = [(0, len(audio))]
        else:
            segs_orig = _map_16k_to_orig(segs_16k, sr)

        logger.info("VAD 段數：%d", len(segs_orig))

        # === 2) 對每個 VAD 段長度過長時再切片（含重疊）===
        final_segs: List[Tuple[int, int]] = []
        for seg in segs_orig:
            s, e = seg
            dur = (e - s) / float(sr)
            if dur > MAX_SEG_SEC:
                parts = _split_long_segment(seg, sr, MAX_SEG_SEC, SEG_OVERLAP_SEC)
                final_segs.extend(parts)
            else:
                final_segs.append(seg)

        logger.info("切片後段數：%d", len(final_segs))

        # === 3) 逐段轉錄 ===
        texts: List[str] = []
        tmp_paths: List[str] = []
        try:
            for i, seg in enumerate(final_segs, 1):
                wav_path = _write_wav_segment(audio, sr, seg)
                tmp_paths.append(wav_path)
                seg_text = _qwen_transcribe_wav(wav_path, thinker_model, processor, max_new_tokens)
                texts.append(seg_text)
                logger.debug("段 %d 轉錄字數=%d", i, len(seg_text))
        finally:
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass

        # === 4) 文字重疊智慧合併 ===
        transcription = _merge_texts_with_overlap(texts)

        _touch_last_used()
        start_timeout_monitor()
        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        logger.exception("ASR 流程發生錯誤")
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
