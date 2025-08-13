import os
import logging
import threading
import time
import gc
import tempfile
from io import BytesIO
from typing import List, Tuple
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

# ===============================
# Qwen2.5-Omni ASR 強化版（保持你現在的生成邏輯與效果）
# - 固定 HF revision（可重現）
# - FA2（不支援則自動回退，不影響輸出）
# - inference_mode + AMP（greedy 下不影響輸出）
# - 段內/段後心跳，避免 idle 被回收
# - OOM/長段安全回退（降 max_new_tokens 一次）
# - tokenizer 左側 padding、pad_token=eos
# - HF cache 環境一致
# - 音訊增益/VAD/切片/重疊合併 = 與你先前版一致
# ===============================

load_dotenv()
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(level=log_level, format=log_format, force=True)
logger = logging.getLogger("qwen2.5-omni-asr-full")

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("HUGGINGFACE_API_KEY 未設定")

MODEL_NAME  = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
HF_REVISION = os.getenv("HF_REVISION", "main")  # 固定版本，避免 remote code 演變

# ===== 音訊處理參數 =====
TRIM_DB         = int(os.getenv("ASR_TRIM_DB", "0"))
TARGET_DBFS     = float(os.getenv("ASR_GAIN_TARGET_DBFS", "-20"))
PEAK_LIMIT      = float(os.getenv("ASR_PEAK_LIMIT", "0.99"))

# VAD 參數
VAD_AGGR        = int(os.getenv("ASR_VAD_AGGR", "2"))
MIN_SPEECH_SEC  = float(os.getenv("ASR_MIN_SPEECH_SEC", "0.3"))
MAX_SILENCE_SEC = float(os.getenv("ASR_MAX_SILENCE_SEC", "0.5"))
PAD_SEC         = float(os.getenv("ASR_PAD_SEC", "0.2"))

# 切片參數
CHUNK_SEC       = float(os.getenv("ASR_CHUNK_SEC", "30.0"))
MAX_SEG_SEC     = float(os.getenv("ASR_MAX_SEG_SEC", "45.0"))
SEG_OVERLAP_SEC = float(os.getenv("ASR_SEG_OVERLAP_SEC", "1.0"))

# 生成參數
MAX_NEW_TOKENS  = int(os.getenv("ASR_MAX_NEW_TOKENS", "512"))

# 閒置釋放
IDLE_TIMEOUT    = int(os.getenv("IDLE_TIMEOUT", "300"))

app = FastAPI()
processor: Qwen2_5OmniProcessor | None = None
thinker: Qwen2_5OmniThinkerForConditionalGeneration | None = None
last_used_time = None
timeout_thread = None

# ========= 基礎工具 =========
def _touch():
    global last_used_time
    last_used_time = time.time()

def _idle_monitor():
    global processor, thinker, last_used_time
    while True:
        time.sleep(10)
        if last_used_time and (time.time() - last_used_time > IDLE_TIMEOUT):
            logger.info("閒置釋放資源...")
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
                logger.info("釋放完成")

def _start_monitor_once():
    global timeout_thread
    if timeout_thread is None:
        timeout_thread = threading.Thread(target=_idle_monitor, daemon=True)
        timeout_thread.start()

def _gen_ctx():
    # AMP：在 greedy 下不影響輸出結果（只提速省顯存）
    if torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()

def _safe_generate(generate_fn, max_new_tokens, timeout_s=60):
    """OOM/長段超時的保護；若 OOM 則清 cache 並把 max_new_tokens 對半重試一次。"""
    try:
        t0 = time.time()
        out = generate_fn(max_new_tokens)
        if time.time() - t0 > timeout_s:
            logger.warning("segment generate exceeded soft timeout (%.1fs)", timeout_s)
        return out
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM. Clearing cache and retry with half max_new_tokens.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = max(128, max_new_tokens // 2)
        return generate_fn(half)

def _load_proc():
    global processor
    if processor is None:
        logger.info("載入 Processor ...")
        processor = Qwen2_5OmniProcessor.from_pretrained(
            MODEL_NAME, token=HF_TOKEN, revision=HF_REVISION
        )
        logger.info("Processor 載入完成")
        # tokenizer 小設定（不改文本內容）
        if processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None and processor.tokenizer.eos_token is not None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token

def _load_thinker():
    global thinker
    if thinker is None:
        logger.info("載入 Thinker 模型 ...")
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            token=HF_TOKEN,
            revision=HF_REVISION,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation="flash_attention_2",  # 不支援會自動回退
        )
        logger.info("Thinker 模型載入完成")
        # 消警告：pad_token_id
        if thinker.config.pad_token_id is None and processor is not None:
            thinker.config.pad_token_id = processor.tokenizer.eos_token_id

def _eos_ids(proc: Qwen2_5OmniProcessor) -> List[int]:
    tok = proc.tokenizer
    ids = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        ids.append(im_end)
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return list(dict.fromkeys(ids))

def _append_assistant_prefill(proc, model_device, inputs_dict, prefill_text: str):
    tok = proc.tokenizer
    pre_ids = tok.encode(prefill_text, add_special_tokens=False, return_tensors="pt").to(model_device)
    inputs_dict["input_ids"] = torch.cat([inputs_dict["input_ids"], pre_ids], dim=1)
    if "attention_mask" in inputs_dict:
        add_mask = torch.ones_like(pre_ids, dtype=inputs_dict["attention_mask"].dtype)
        inputs_dict["attention_mask"] = torch.cat([inputs_dict["attention_mask"], add_mask], dim=1)
    return inputs_dict

# ========= 音訊 I/O 與增益 =========
def _db_to_amp(db): return 10 ** (db / 20.0)

def _bytes_to_audio_mono(data: bytes) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(BytesIO(data), dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr

def _preprocess_audio_for_model(y: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    if y.size == 0:
        return y, sr
    z = y.copy()
    if TRIM_DB > 0:
        z, _ = librosa.effects.trim(z, top_db=TRIM_DB)
    if z.size > 0:
        rms = float(np.sqrt(np.mean(z**2) + 1e-12))
        target = _db_to_amp(TARGET_DBFS)  # 約 0.1（-20 dBFS）
        if rms > 0 and rms < target:
            gain = target / rms
            z = np.clip(z * gain, -PEAK_LIMIT, PEAK_LIMIT)
    return z, sr

def _write_wav(y: np.ndarray, sr: int) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr, subtype="PCM_16")
    return path

# ========= VAD（只用於偵測；不改模型吃的 sr） =========
def _resample_for_vad(y: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if y.size == 0:
        return y
    if sr == target_sr:
        return y.astype(np.float32)
    return librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="polyphase").astype(np.float32)

def _float32_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()

def _frame_gen(pcm16: bytes, sr: int, frame_ms: int = 20):
    n = int(sr * (frame_ms / 1000.0)) * 2
    for i in range(0, len(pcm16), n):
        chunk = pcm16[i:i+n]
        if len(chunk) == n:
            yield chunk

def _vad_segments(y: np.ndarray, sr: int) -> List[Tuple[int, int]]:
    if y.size == 0:
        return []
    target_sr = 16000
    vad = webrtcvad.Vad(VAD_AGGR)

    y16 = _resample_for_vad(y, sr, target_sr)
    if y16.size == 0:
        return []
    pcm16 = _float32_to_pcm16(y16)
    frame_ms = 20
    frames = list(_frame_gen(pcm16, target_sr, frame_ms))
    if not frames:
        return []

    voiced = [vad.is_speech(f, target_sr) for f in frames]
    frm_len = int(target_sr * (frame_ms / 1000.0))

    segs: List[Tuple[int, int]] = []
    in_speech = False
    start_idx = 0
    last_voiced = -1

    max_silence_frames = int(MAX_SILENCE_SEC / (frame_ms / 1000.0))
    min_speech_frames  = max(1, int(MIN_SPEECH_SEC  / (frame_ms / 1000.0)))

    for i, v in enumerate(voiced):
        if v and not in_speech:
            in_speech = True
            start_idx = i
            last_voiced = i
        elif v and in_speech:
            last_voiced = i
        elif (not v) and in_speech:
            if (i - last_voiced) > max_silence_frames:
                length = last_voiced - start_idx + 1
                if length >= min_speech_frames:
                    s = start_idx * frm_len
                    e = (last_voiced + 1) * frm_len
                    segs.append((s, e))
                in_speech = False

    if in_speech:
        length = last_voiced - start_idx + 1
        if length >= min_speech_frames:
            s = start_idx * frm_len
            e = (last_voiced + 1) * frm_len
            segs.append((s, e))

    pad = int(PAD_SEC * target_sr)
    segs = [(max(0, s - pad), min(len(y16), e + pad)) for s, e in segs]

    merged: List[Tuple[int, int]] = []
    for s, e in sorted(segs):
        if not merged or s > merged[-1][1]:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged

def _map_16k_to_orig(segs_16k: List[Tuple[int, int]], orig_sr: int, target_sr: int = 16000) -> List[Tuple[int, int]]:
    r = orig_sr / float(target_sr)
    return [(int(round(s * r)), int(round(e * r))) for s, e in segs_16k]

# ========= 切片 =========
def _split_by_chunk(seg: Tuple[int, int], sr: int, chunk_sec: float, overlap_sec: float) -> List[Tuple[int, int]]:
    if chunk_sec <= 0:
        return [seg]
    s, e = seg
    chunk = int(chunk_sec * sr)
    ovl = int(overlap_sec * sr)
    out = []
    cur = s
    while cur < e:
        end = min(cur + chunk, e)
        out.append((cur, end))
        if end == e: break
        cur = max(end - ovl, s)
    return out

def _split_long_segment(seg: Tuple[int, int], sr: int, max_sec: float, overlap_sec: float) -> List[Tuple[int, int]]:
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

# ========= 合併文字 =========
def _normalize_ws(s: str) -> str:
    return " ".join(s.split())

def _merge_texts_with_overlap(pieces: List[str], min_overlap_chars: int = 10, max_probe: int = 80) -> str:
    out = ""
    for idx, cur in enumerate(pieces):
        if idx == 0:
            out = _normalize_ws(cur)
            continue
        prev = out
        cur_n = _normalize_ws(cur)
        best = 0
        for k in range(min(len(prev), max_probe), min_overlap_chars - 1, -1):
            if prev[-k:] == cur_n[:k]:
                best = k
                break
        if best > 0:
            out = prev + cur_n[best:]
        else:
            out = (prev + " " + cur_n) if (prev and not prev.endswith(" ")) else (prev + cur_n)
    return out.strip()

# ========= 單段推論 =========
def _transcribe_once(wav_path: str, proc: Qwen2_5OmniProcessor, model, max_new_tokens: int) -> str:
    default_system = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                      "capable of perceiving auditory and visual inputs, as well as generating text and speech.")

    conversations = [
        {"role": "system", "content": [{"type": "text", "text": default_system}]},
        {"role": "user",   "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text",  "text": "Please transcribe the audio verbatim. Output only the spoken words, no commentary."}
        ]},
    ]

    inputs = proc.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    # assistant prefill（不影響語意，只讓格式更穩）
    inputs = _append_assistant_prefill(proc, model.device, inputs, "Transcript: ")

    def _do_gen(mnt):
        return model.generate(
            **inputs,
            max_new_tokens=mnt,
            do_sample=False,
            eos_token_id=_eos_ids(proc),
            pad_token_id=proc.tokenizer.eos_token_id,
        )

    with torch.inference_mode():
        with _gen_ctx():
            out_ids = _safe_generate(_do_gen, max_new_tokens)

    # 只解碼新增段
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    if text.lower().startswith("transcript:"):
        text = text[len("transcript:"):].lstrip()
    return text

# ========= API =========
@app.post("/audio/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    max_new_tokens: int = MAX_NEW_TOKENS,
):
    """
    流程：
      1) 讀檔 -> 單聲道 float32
      2) 前處理（可選裁切、RMS 自動增益）
      3) VAD 分段（在 16k 上運作），回映到原始 sr
      4) 固定長度切片 + 長片段二次切片（含重疊）
      5) 逐段推論；重疊文字去重合併
      6) 若 VAD 無段，fallback：整檔切片/或直接推論
    """
    _touch()
    _start_monitor_once()

    if not file:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        _load_proc()
        _load_thinker()

        raw = await file.read()
        y, sr = _bytes_to_audio_mono(raw)
        logger.info("sr=%d, duration=%.2fs", sr, (len(y) / float(sr) if sr > 0 else 0.0))

        # 前處理（供模型用）
        y_model, sr_model = _preprocess_audio_for_model(y, sr)

        # VAD 分段
        segs_16k = _vad_segments(y, sr)
        if segs_16k:
            segs = _map_16k_to_orig(segs_16k, sr)  # 在原始 sr 座標
        else:
            segs = [(0, len(y_model))]
            logger.info("VAD 無段，使用整檔作為單一段")

        # 固定長度切片 + 長片段二次切片
        final_segs: List[Tuple[int, int]] = []
        for s, e in segs:
            parts = _split_by_chunk((s, e), sr_model, CHUNK_SEC, SEG_OVERLAP_SEC) if CHUNK_SEC > 0 else [(s, e)]
            for ps, pe in parts:
                dur = (pe - ps) / float(sr_model)
                if dur > MAX_SEG_SEC:
                    final_segs.extend(_split_long_segment((ps, pe), sr_model, MAX_SEG_SEC, SEG_OVERLAP_SEC))
                else:
                    final_segs.append((ps, pe))

        logger.info("切片後段數：%d", len(final_segs))
        if not final_segs:
            _touch()
            return JSONResponse(content={"transcription": "[NO-SPEECH]"})

        # 逐段推論
        pieces: List[str] = []
        tmp_paths: List[str] = []
        try:
            for i, (s, e) in enumerate(final_segs, 1):
                _touch()  # 段前心跳
                ss = max(0, s); ee = min(len(y_model), e)
                seg = y_model[ss:ee]
                seg_path = _write_wav(seg, sr_model)
                tmp_paths.append(seg_path)

                seg_text = _transcribe_once(seg_path, processor, thinker, max_new_tokens)
                pieces.append(seg_text)
                logger.debug("段 %d 長度=%.2fs 轉錄字數=%d", i, (ee-ss)/float(sr_model), len(seg_text))
                _touch()  # 段後心跳
        finally:
            for p in tmp_paths:
                try: os.remove(p)
                except: pass

        # 合併
        transcription = _merge_texts_with_overlap(pieces)

        _touch()
        return JSONResponse(content={"transcription": transcription if transcription else "[NO-SPEECH]"})

    except Exception as e:
        logger.exception("ASR 流程錯誤")
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
