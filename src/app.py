import os
import logging
import threading
import time
import gc
import tempfile
import asyncio
from typing import List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import soundfile as sf
import librosa
import webrtcvad
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from dotenv import load_dotenv

from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

# ===============================
# Env & Logging（與 Phi-4 一致）
# ===============================
load_dotenv("/app/config/.env", override=False)
load_dotenv(override=False)

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
HF_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
log_format = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(level=log_level, format=log_format, force=True)
logger = logging.getLogger("qwen2.5-omni-asr")

HF_TOKEN = os.getenv("HUGGINGFACE_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("Hugging Face API Key 未設定，請在 config/.env 檔案中加入 HUGGINGFACE_API_KEY")

# ========== HF 與模型 ==========
MODEL_NAME  = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Omni-7B")
HF_REVISION = os.getenv("HF_REVISION", "main")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 服務/運行參數（與 Phi-4 對齊） ==========
IDLE_TIMEOUT = int(os.getenv("IDLE_TIMEOUT", "300"))
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "2"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "512"))
TRANSCRIBE_TIMEOUT = float(os.getenv("ASR_TIMEOUT", "600"))

# ===== 音訊處理參數 =====
TRIM_DB         = int(os.getenv("ASR_TRIM_DB", "0"))
TARGET_DBFS     = float(os.getenv("ASR_GAIN_TARGET_DBFS", "-20"))
PEAK_LIMIT      = float(os.getenv("ASR_PEAK_LIMIT", "0.99"))

# VAD：預設值（實際由 _pick_vad_aggr 動態調整）
VAD_AGGR        = int(os.getenv("ASR_VAD_AGGR", "2"))
MIN_SPEECH_SEC  = float(os.getenv("ASR_MIN_SPEECH_SEC", "0.3"))
MAX_SILENCE_SEC = float(os.getenv("ASR_MAX_SILENCE_SEC", "0.5"))
PAD_SEC         = float(os.getenv("ASR_PAD_SEC", "0.2"))

CHUNK_SEC       = float(os.getenv("ASR_CHUNK_SEC", "30.0"))
MAX_SEG_SEC     = float(os.getenv("ASR_MAX_SEG_SEC", "45.0"))
SEG_OVERLAP_SEC = float(os.getenv("ASR_SEG_OVERLAP_SEC", "1.0"))

SILENCE_DB_THRESHOLD = float(os.getenv("ASR_SILENCE_DB_THRESHOLD", "-50.0"))

# ===== 生成參數 =====
MAX_NEW_TOKENS_DEFAULT = int(os.getenv("ASR_MAX_NEW_TOKENS", "512"))

# ========= Global State（與 Phi-4 結構一致）=========
app = FastAPI()

processor: Optional[Qwen2_5OmniProcessor] = None
thinker: Optional[Qwen2_5OmniThinkerForConditionalGeneration] = None

model_lock = threading.Lock()

last_used_time: Optional[float] = None
last_used_lock = threading.Lock()

active_requests = 0
active_requests_lock = threading.Lock()

timeout_thread: Optional[threading.Thread] = None
timeout_thread_started = False
timeout_thread_lock = threading.Lock()

# asyncio semaphore 於 startup 綁定當前 loop
inference_semaphore: Optional[asyncio.Semaphore] = None


# ========= 共用小工具（與 Phi-4 對齊）=========
def _touch():
    global last_used_time
    with last_used_lock:
        last_used_time = time.time()

def _active_inc():
    global active_requests
    with active_requests_lock:
        active_requests += 1
        logger.debug(f"active++ -> {active_requests}")
    _touch()

def _active_dec():
    global active_requests
    with active_requests_lock:
        active_requests = max(0, active_requests - 1)
        logger.debug(f"active-- -> {active_requests}")
    _touch()

def _safe_from_pretrained(cls, *args, **kwargs):
    """支援 token / use_auth_token 的相容載入；可選 cache_dir。"""
    try:
        return cls.from_pretrained(*args, **kwargs)
    except TypeError:
        if "token" in kwargs:
            kwargs = dict(kwargs)
            kwargs["use_auth_token"] = kwargs.pop("token")
        return cls.from_pretrained(*args, **kwargs)

def monitor_idle_time():
    """每 10 秒檢查一次，若超過指定閒置時間且沒有活躍請求則釋放資源。"""
    global processor, thinker, last_used_time
    while True:
        time.sleep(10)
        with active_requests_lock:
            no_active = (active_requests == 0)
        with last_used_lock:
            last_time = last_used_time

        should_free = (
            (processor is not None or thinker is not None)
            and no_active
            and last_time is not None
            and (time.time() - last_time > IDLE_TIMEOUT)
        )
        if should_free:
            logger.info("ASR 模型閒置超時且無活躍請求，開始釋放資源...")
            with model_lock:
                with active_requests_lock:
                    no_active2 = (active_requests == 0)
                with last_used_lock:
                    last_time2 = last_used_time
                if ((processor is not None or thinker is not None)
                        and no_active2 and last_time2 is not None
                        and (time.time() - last_time2 > IDLE_TIMEOUT)):
                    try:
                        if thinker is not None:
                            del thinker
                        if processor is not None:
                            del processor
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                            logger.info("GPU 資源已清理")
                        else:
                            logger.info("CPU 環境，無需清理 GPU 資源")
                    except Exception:
                        logger.exception("釋放 ASR 模型資源失敗")
                    finally:
                        processor = None
                        thinker = None
                        with last_used_lock:
                            last_used_time = None
                        logger.info("ASR 模型已成功釋放")

def start_timeout_monitor():
    global timeout_thread, timeout_thread_started
    with timeout_thread_lock:
        if timeout_thread_started:
            return
        timeout_thread = threading.Thread(target=monitor_idle_time, daemon=True)
        timeout_thread.start()
        timeout_thread_started = True
        logger.info("監控執行緒已啟動")


# ========= 模型載入（Lazy + 單卡上機）=========
def load_model():
    """
    Lazy load Qwen2.5-Omni：
    - 明確停用 device_map="auto"，整個模型放到單一裝置（GPU/CPU）
    - 一次性 GPU 優化（TF32 / cudnn / eval / requires_grad=False）
    """
    global processor, thinker

    if processor is None or thinker is None:
        with model_lock:
            if processor is None:
                logger.info("載入 Processor ...")
                common_kwargs = {"revision": HF_REVISION, "token": HF_TOKEN, "trust_remote_code": True}
                if HF_CACHE_DIR:
                    common_kwargs["cache_dir"] = HF_CACHE_DIR
                processor_local = _safe_from_pretrained(
                    Qwen2_5OmniProcessor, MODEL_NAME, **common_kwargs
                )
                if processor_local.tokenizer is not None:
                    processor_local.tokenizer.padding_side = "left"
                    if processor_local.tokenizer.pad_token_id is None and processor_local.tokenizer.eos_token is not None:
                        processor_local.tokenizer.pad_token = processor_local.tokenizer.eos_token
                processor = processor_local
                logger.info("Processor 載入完成")

            if thinker is None:
                logger.info("載入 Thinker 模型 ...")
                model_kwargs = dict(
                    revision=HF_REVISION,
                    token=HF_TOKEN,
                    trust_remote_code=True,
                    torch_dtype="auto",
                    device_map=None,                # 關掉分片
                    low_cpu_mem_usage=False,        # 避免殘留 meta 權重
                    attn_implementation="sdpa",
                )
                if HF_CACHE_DIR:
                    model_kwargs["cache_dir"] = HF_CACHE_DIR
                thinker_local = _safe_from_pretrained(
                    Qwen2_5OmniThinkerForConditionalGeneration, MODEL_NAME, **model_kwargs
                )

                # 明確上到目標裝置
                thinker_local.to(DEVICE)

                if torch.cuda.is_available():
                    try:
                        torch.set_float32_matmul_precision('high')
                    except Exception:
                        logger.debug("set_float32_matmul_precision 失敗，忽略。")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cudnn.benchmark = True
                    if hasattr(thinker_local, "eval"):
                        try:
                            thinker_local.eval()
                        except Exception:
                            logger.debug("model.eval() 失敗，忽略。")
                    try:
                        thinker_local.gradient_checkpointing_disable()
                    except Exception:
                        pass
                    for p in thinker_local.parameters():
                        p.requires_grad = False

                if thinker_local.config.pad_token_id is None and processor is not None:
                    thinker_local.config.pad_token_id = processor.tokenizer.eos_token_id

                thinker = thinker_local
                logger.info(f"Thinker 模型載入完成（device={DEVICE.type}）")

    _touch()
    start_timeout_monitor()


# ========= 音訊 I/O/前處理 =========
def _db_to_amp(db): return 10 ** (db / 20.0)

def _file_to_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=False)
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
        target = _db_to_amp(TARGET_DBFS)
        if rms > 0 and rms < target:
            gain = target / rms
            z = np.clip(z * gain, -PEAK_LIMIT, PEAK_LIMIT)
    return z, sr

def _write_wav(y: np.ndarray, sr: int) -> str:
    """
    將切片寫為 16k/mono/PCM16，保證與 WebRTC-VAD/大多數 ASR-friendly 設定一致。
    """
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="polyphase")
        sr = 16000
    if hasattr(y, "ndim") and y.ndim > 1:
        y = y.mean(axis=1)
    if y.size > 0:
        y = np.clip(y, -1.0, 1.0)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, y, sr, subtype="PCM_16")
    return path


# ========= VAD（動態等級 + 16k 偵測 + 映回原 sr）=========
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

def _estimate_snr_db(y: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    粗估 SNR 與整體音量(dBFS)：
    - 噪聲估計用絕對振幅的低分位數（5th percentile）
    """
    if y.size == 0:
        return -120.0, -120.0
    y16 = _resample_for_vad(y, sr, 16000)
    rms = float(np.sqrt(np.mean(y16**2) + 1e-12))
    noise_floor = float(np.percentile(np.abs(y16), 5))
    noise_rms = max(noise_floor / np.sqrt(2.0), 1e-8)
    snr_db = 20.0 * np.log10(max(rms, 1e-8) / noise_rms)
    vol_db = 20.0 * np.log10(rms + 1e-8)
    return snr_db, vol_db

def _pick_vad_aggr(y: np.ndarray, sr: int, fallback: int = 2) -> int:
    """
    依 SNR/音量選擇 VAD 等級（1/2/3）：
    - 很安靜或低 SNR -> 用 3（最激進）
    - 一般 -> 2
    - 很清楚/大聲 -> 1（最鬆）
    """
    try:
        snr_db, vol_db = _estimate_snr_db(y, sr)
        if vol_db < -40.0 or snr_db < 5.0:
            return 3
        if vol_db < -30.0 or snr_db < 10.0:
            return 2
        return 1
    except Exception:
        return fallback

def _vad_segments(y: np.ndarray, sr: int, aggr: Optional[int] = None) -> List[Tuple[int, int]]:
    if y.size == 0:
        return []
    target_sr = 16000
    if aggr is None:
        aggr = VAD_AGGR
    vad = webrtcvad.Vad(int(np.clip(aggr, 0, 3)))

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
        if end == e:
            break
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


# ========= 生成參數：以片段秒數估 token 上限 + 每段 max_time =========
TOKENS_PER_SECOND = float(os.getenv("ASR_TOKENS_PER_SECOND", "12"))
def _cap_tokens_by_duration(sec: float, hard_cap: int) -> int:
    # 與 phi4 對齊：每秒上限 ~12 tokens，最少 64，且不超過呼叫者設定的 hard_cap
    return max(64, min(hard_cap, int(sec * TOKENS_PER_SECOND)))


# ========= Qwen 生成工具 =========
def _gen_ctx():
    # 修正 deprecated：改用 torch.amp.autocast("cuda", ...)
    if torch.cuda.is_available():
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return nullcontext()

def _safe_generate(generate_fn, max_new_tokens, max_time_s: Optional[float] = None, timeout_s: float = 60):
    """
    OOM/超時保護：
    - OOM 清 cache 並把 max_new_tokens 對半重試一次
    - 軟超時警告（不會中止，但會記錄）
    """
    try:
        t0 = time.time()
        out = generate_fn(max_new_tokens, max_time_s)
        if time.time() - t0 > timeout_s:
            logger.warning("segment generate exceeded soft timeout (%.1fs)", timeout_s)
        return out
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM. Clearing cache and retry with half max_new_tokens.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        half = max(128, max_new_tokens // 2)
        return generate_fn(half, max_time_s)

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

def _should_skip_silence(audio_array: np.ndarray, sr: int, db_threshold: float = None) -> bool:
    if audio_array.size == 0:
        return True
    if db_threshold is None:
        db_threshold = SILENCE_DB_THRESHOLD
    rms = np.sqrt(np.mean(audio_array**2))
    if rms == 0:
        return True
    rms_db = 20 * np.log10(rms + 1e-12)
    return rms_db < db_threshold

def _transcribe_once(
    wav_path: str,
    proc: Qwen2_5OmniProcessor,
    model,
    max_new_tokens: int,
    max_time: Optional[float],
) -> str:
    # 官方預設 system（不加 zh-TW 或 code-switching 指示）
    default_system = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
                      "capable of perceiving auditory and visual inputs, as well as generating text and speech.")

    conversations = [
        {"role": "system", "content": [{"type": "text", "text": default_system}]},
        {"role": "user",   "content": [
            {"type": "audio", "path": wav_path},
            {"type": "text",  "text": "Please transcribe the audio verbatim. Output only the spoken words, no commentary."}
        ]},
    ]

    # 明確使用實際運算裝置（單卡模式下就是 DEVICE）
    model_device = DEVICE

    inputs = proc.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True
    ).to(model_device)

    # assistant prefill（不影響語意，只讓格式更穩）
    inputs = _append_assistant_prefill(proc, model_device, inputs, "Transcript: ")

    def _do_gen(mnt, mtime):
        return model.generate(
            **inputs,
            max_new_tokens=mnt,
            do_sample=False,
            eos_token_id=_eos_ids(proc),
            pad_token_id=proc.tokenizer.eos_token_id,
            max_time=mtime,  # 防單段「掛住」
        )

    with torch.inference_mode():
        with _gen_ctx():
            out_ids = _safe_generate(_do_gen, max_new_tokens, max_time_s=max_time, timeout_s=90.0)

    # 只解碼新增段
    gen_only = out_ids[:, inputs["input_ids"].shape[1]:]
    text = proc.batch_decode(gen_only, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    if text.lower().startswith("transcript:"):
        text = text[len("transcript:"):].lstrip()
    return text


# ========= Heavy work（同步函式，丟給 threadpool 執行）=========
def _do_transcribe(temp_path: str,
                   max_new_tokens: int,
                   segment_duration: float,
                   overlap_duration: float) -> str:
    load_model()

    # 讀音訊
    y, sr = _file_to_audio_mono(temp_path)
    dur_total = (len(y) / float(sr) if sr > 0 else 0.0)
    logger.info("sr=%d, duration=%.2fs", sr, dur_total)

    # 前處理（不改 sr）
    y_model, sr_model = _preprocess_audio_for_model(y, sr)

    # 動態挑 VAD 等級（依 SNR/音量）
    aggr = _pick_vad_aggr(y_model, sr_model, fallback=VAD_AGGR)
    logger.info(f"VAD aggressiveness picked: {aggr}")

    # VAD 在 16k 偵測 -> 座標映回原 sr
    segs_16k = _vad_segments(y_model, sr_model, aggr=aggr)
    if segs_16k:
        segs = _map_16k_to_orig(segs_16k, sr_model)
    else:
        segs = [(0, len(y_model))]
        logger.info("VAD 無段，使用整檔作為單一段")

    # 固定切 + 二次切（在原始 sr_model 座標上運作）
    final_segs: List[Tuple[int, int]] = []
    for s, e in segs:
        parts = _split_by_chunk((s, e), sr_model, segment_duration, overlap_duration) if segment_duration > 0 else [(s, e)]
        for ps, pe in parts:
            dur = (pe - ps) / float(sr_model)
            if dur > MAX_SEG_SEC:
                final_segs.extend(_split_long_segment((ps, pe), sr_model, MAX_SEG_SEC, overlap_duration))
            else:
                final_segs.append((ps, pe))
    logger.info("切片後段數：%d", len(final_segs))
    if not final_segs:
        return "[NO-SPEECH]"

    # 逐段推論
    pieces: List[str] = []
    tmp_paths: List[str] = []
    try:
        for i, (s, e) in enumerate(final_segs, 1):
            ss = max(0, s); ee = min(len(y_model), e)
            seg = y_model[ss:ee]
            seg_dur_s = (ee - ss) / float(sr_model)

            # 靜音快篩
            if _should_skip_silence(seg, sr_model):
                logger.debug(f"略過靜音段落 idx={i}")
                pieces.append("")
                continue

            # 以秒數估 token 上限 & 段落 max_time
            seg_max_new = _cap_tokens_by_duration(seg_dur_s, max_new_tokens)
            seg_max_time = min(90.0, max(10.0, seg_dur_s * 3.0))  # 10~90 秒

            seg_path = _write_wav(seg, sr_model)  # 轉 16k/mono/PCM16
            tmp_paths.append(seg_path)

            seg_text = _transcribe_once(seg_path, processor, thinker, seg_max_new, seg_max_time)
            pieces.append(seg_text)
            logger.debug("段 %d 長度=%.2fs tokens_cap=%d max_time=%.1fs 轉錄字數=%d",
                         i, seg_dur_s, seg_max_new, seg_max_time, len(seg_text))
            _touch()  # 長段保活
    finally:
        for p in tmp_paths:
            try:
                os.remove(p)
            except Exception:
                pass

    transcription = _merge_texts_with_overlap(pieces)
    _touch()
    return transcription if transcription else "[NO-SPEECH]"


# ========= FastAPI lifecycle =========
@app.on_event("startup")
async def _startup():
    global inference_semaphore
    inference_semaphore = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
    logger.info(f"Semaphore 初始化：MAX_CONCURRENT_INFERENCES={MAX_CONCURRENT_INFERENCES}")

@app.on_event("shutdown")
async def _shutdown():
    global processor, thinker
    with model_lock:
        if processor is not None or thinker is not None:
            logger.info("Shutdown：釋放 ASR 模型與 GPU 資源...")
            try:
                if thinker is not None: del thinker
                if processor is not None: del processor
                thinker = None; processor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                logger.exception("Shutdown 清理失敗")
            else:
                logger.info("Shutdown 清理完成")


# ========= Health / Status =========
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time(), "service": "qwen2.5-omni-asr"}

@app.get("/ready")
async def ready():
    model_ready = (processor is not None and thinker is not None)
    with active_requests_lock:
        active_count = active_requests
    status_info = {
        "ready": model_ready,
        "model_loaded": model_ready,
        "active_requests": active_count,
        "models_dir": HF_CACHE_DIR,
        "device": DEVICE.type,
        "timestamp": time.time(),
        "service": "qwen2.5-omni-asr"
    }
    return JSONResponse(status_code=200, content=status_info)

@app.get("/status")
async def status():
    with active_requests_lock:
        active_count = active_requests
    with last_used_lock:
        last_used = last_used_time
    info = {
        "model_loaded": (processor is not None and thinker is not None),
        "active_requests": active_count,
        "last_used_time": last_used,
        "idle_timeout": IDLE_TIMEOUT,
        "max_concurrent_inferences": MAX_CONCURRENT_INFERENCES,
        "max_upload_mb": MAX_UPLOAD_MB,
        "cuda_available": torch.cuda.is_available(),
        "models_dir": HF_CACHE_DIR,
        "device": DEVICE.type,
        "timestamp": time.time(),
        "service": "qwen2.5-omni-asr"
    }
    return info


# ========= Main API =========
@app.post("/audio/transcribe")
async def transcribe_audio(file: UploadFile = File(...),
                           max_new_tokens: int = MAX_NEW_TOKENS_DEFAULT,
                           segment_duration: float = CHUNK_SEC,
                           overlap_duration: float = SEG_OVERLAP_SEC):
    """上傳音訊檔，進行語音轉錄。"""
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    _active_inc()
    temp_path = None
    try:
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        tmp = tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=file_ext)
        temp_path = tmp.name

        chunk_size = 1024 * 1024  # 1MB
        total_bytes = 0
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            tmp.write(chunk)
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_MB * 1024 * 1024:
                tmp.close()
                try:
                    os.unlink(temp_path)
                except Exception:
                    logger.warning("刪除超限暫存檔失敗")
                raise HTTPException(status_code=413, detail="File too large")
        tmp.close()
        logger.info(f"音訊已寫入暫存檔：{temp_path}，大小 {total_bytes} bytes")

        assert inference_semaphore is not None, "Semaphore 未初始化"
        async with inference_semaphore:
            logger.info("取得推理鎖，開始轉錄")
            transcription = await asyncio.wait_for(
                run_in_threadpool(_do_transcribe, temp_path, max_new_tokens, segment_duration, overlap_duration),
                timeout=TRANSCRIBE_TIMEOUT
            )

        _touch()
        return JSONResponse(content={"transcription": transcription})

    except asyncio.TimeoutError:
        logger.warning("轉錄超時")
        raise HTTPException(status_code=504, detail="Transcription timeout")
    except HTTPException:
        raise
    except Exception:
        logger.exception("轉錄發生錯誤")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            logger.warning("清理暫存檔失敗")
        _active_dec()


if __name__ == "__main__":
    import uvicorn
    # 保留 lazy-load（不在 startup 強制 load_model）
    uvicorn.run(app, host="0.0.0.0", port=8000)
