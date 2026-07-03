import os
import io
import sys
import json
import glob
import uuid
import threading
import torch
import hashlib
import argparse
import time
import psutil
import numpy as np
import soundfile as sf
import torchaudio as _ta
from pydub import AudioSegment
from fastapi import FastAPI, HTTPException, status, Request, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager

# MPS fallback for ops not yet ported to Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Add parent and local path to sys.path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "BigVGAN"))

# ── Device override for F5-TTS (must happen BEFORE F5-TTS imports) ──
_VAGDHENU_DEVICE = os.environ.get("VAGDHENU_DEVICE", None)
if _VAGDHENU_DEVICE:
    import f5_tts.infer.utils_infer as _f5_pre_patch
    _f5_pre_patch.device = _VAGDHENU_DEVICE
    print(f"[Device] Forced F5-TTS device to: {_VAGDHENU_DEVICE}", flush=True)

import bigvgan
from f5_tts.model import DiT
from f5_tts.infer.utils_infer import load_model, load_vocoder, infer_process, preprocess_ref_audio_text

# Set stable torchaudio backend
try:
    _ta.set_audio_backend("soundfile")
except AttributeError:
    pass

# Global paths
CHAMP = os.environ.get("CHAMP_ROOT", os.path.join(REPO, "models"))
BANK_PATH = os.path.join(HERE, "reference_bank", "bank.json")
DEFAULT_VOICE = os.path.join(CHAMP, "voice_steer_ema_2026-06-17.pt")
DEFAULT_VOC = os.path.join(CHAMP, "voc_bigvgan_EMA_2026-06-11.pth")

# Global Constants
VAGDHENU_VERSION = "1.1.0"  # bump on model/config changes
BATCH_MAX = 20               # conservative for 24GB shared server
SR = 24000
FALLBACK_METER = "vasantatilakā"
CFG_MODEL = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

# ── Cloudflare R2 Setup ───────────────────────────────────────────────────────────────
import boto3
from botocore.config import Config

R2_ACCOUNT_ID = os.environ.get("R2_ACCOUNT_ID") or os.environ.get("AKASHA_R2_ACCOUNT_ID")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID") or os.environ.get("AKASHA_R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY") or os.environ.get("AKASHA_R2_SECRET_ACCESS_KEY")
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME") or os.environ.get("AKASHA_R2_BUCKET")
R2_PUBLIC_URL = os.environ.get("R2_PUBLIC_URL") or os.environ.get("AKASHA_R2_PUBLIC_DOMAIN")

r2_client = None
if R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY:
    endpoint_url = f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
    r2_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4")
    )
    print(f"[R2] Connected to bucket: {R2_BUCKET_NAME}", flush=True)
else:
    print("[R2] Missing credentials. Operating in local fallback mode.", flush=True)

# ── Sanskrit Sandhi Preprocessing (Prep_text helpers) ───────────────────────────────
from src.prep_text import model_text_sandhi, model_text

def _basetext(p, no_sandhi):
    """Apply Vagdhenu text normalization: sandhi-aware or raw model_text."""
    return model_text_sandhi(p, echo_final=False) if not no_sandhi else model_text(p)


# ── Audio Metadata Embedding ─────────────────────────────────────────────────────────
def _embed_metadata(audio_bytes: bytes, meta: dict, fmt: str) -> bytes:
    """
    Embed Brahma provenance metadata directly into the audio file.

    Strategy per format:
      - WAV: Inject a custom 'brm1' RIFF sub-chunk with JSON metadata.
             RIFF readers skip unknown chunks, so files remain playable.
      - MP3: Leave untouched for now (MP3 frames don't support arbitrary binary
             chunks without ID3v2, and mutagen chokes on soundfile WAV→MP3 output).
             Metadata lives in the API response + R2 object metadata instead.

    The 'brm1' chunk format: 4-char ID, 4-byte LE size, N bytes UTF-8 JSON.
    """
    if fmt != "wav":
        return audio_bytes  # MP3: use API response + R2 metadata for provenance

    meta_json = json.dumps(meta, ensure_ascii=False).encode("utf-8")
    # RIFF sub-chunk: 4-byte ID + 4-byte LE size + data (+ pad byte if odd)
    chunk_id = b"brm1"
    data_size = len(meta_json)
    pad = b"" if data_size % 2 == 0 else b"\x00"
    brm1_chunk = chunk_id + data_size.to_bytes(4, "little") + meta_json + pad

    # Inject before the 'data' sub-chunk (standard WAV layout: RIFF header → fmt chunk → data chunk)
    data_marker = b"data"
    idx = audio_bytes.find(data_marker)
    if idx == -1:
        return audio_bytes  # unexpected format, skip

    # Update RIFF total size (bytes 4-7, LE int32)
    riff_size = int.from_bytes(audio_bytes[4:8], "little")
    new_riff_size = riff_size + len(brm1_chunk)
    header = audio_bytes[:4] + new_riff_size.to_bytes(4, "little") + audio_bytes[8:idx]

    return header + brm1_chunk + audio_bytes[idx:]

def _satva(text):
    return text.replace("स् त", "ष् त").replace("स् थ", "ष् थ")

def _anusvara_m(text):
    return text.replace("म् ", "ं ").replace("म्।", "ं।").replace("म्॥", "ं॥").replace("म् ", "ं ")

def _danda_fix(text):
    return text.replace("।", " ।").replace("॥", " ॥")

def _hna_metathesis(text):
    return text.replace("हन्", "न्ह").replace("हम्", "म्ह")

def _vocalic_l(text):
    return text.replace("लृ", "ल्ऋ")

def _rep_depths(arr):
    # simple repetition checker for autoprime heuristic
    if not arr: return 0, 0
    mono, di = 1, 1
    for i in range(len(arr)-1):
        if arr[i] == arr[i+1]: mono += 1
    for i in range(len(arr)-3):
        if arr[i:i+2] == arr[i+2:i+4]: di += 1
    return mono, di

def _aksharas(text):
    # extract approximate akshara tokens
    return [x for x in text if x not in " ।॥.,;:!?‌‍"]

def n_aksharas(text):
    return len(_aksharas(text))

def _ends_halant(txt):
    _VIRAMA = "्"
    t = txt.rstrip(" ।॥|.,;:!?‌‍")
    return len(t) > 0 and t[-1] in _VIRAMA

# ── Model Wrapper & Lifecycle ────────────────────────────────────────────────────────
class TTSModels:
    def __init__(self):
        self.device = "cpu"
        self.cfm = None
        self.cap = None
        self.g = None
        self.vocab = None
        self.bank = {}
        self.lut = {}
        self.primes = {}
        self.ref_cache = {}

    def load(self, device: str = None):
        self.backend = os.environ.get("VAGDHENU_INFERENCE_BACKEND", "local")
        if self.backend == "replicate":
            print(f"[Model] Using Replicate backend ({os.environ.get('VAGDHENU_REPLICATE_MODEL')}). Skipping local model loading.", flush=True)
            # Load reference bank only
            if os.path.exists(BANK_PATH):
                self.bank = json.load(open(BANK_PATH, encoding="utf-8"))
                self.primes = self.bank.get("repeat_primes", {})
                for k, v in self.bank.items():
                    if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                        continue
                    self.lut[k] = v
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        # Force F5-TTS to use our device (it has its own module-level default)
        import f5_tts.infer.utils_infer as f5_utils
        f5_utils.device = self.device
        
        print(f"[Model] Loading models onto target device: {self.device}...", flush=True)

        # 1. Resolve vocab.txt
        vocab_cands = [
            os.path.join(CHAMP, "vocab.txt"),
            os.path.join(HERE, "reference_bank", "vocab.txt")
        ] + glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--ai4bharat--IndicF5/snapshots/*/checkpoints/vocab.txt"))
        
        self.vocab = next((v for v in vocab_cands if v and os.path.exists(v)), None)
        if self.vocab is None:
            raise RuntimeError("vocab.txt not found. Run scripts/download_weights.py first.")

        # 2. Load F5-TTS DiT Model
        self.cfm = load_model(DiT, CFG_MODEL, mel_spec_type="vocos", vocab_file=self.vocab, device=self.device)
        ck = torch.load(DEFAULT_VOICE, map_location="cpu", weights_only=True)
        ema = {k.replace("ema_model.", ""): v for k, v in ck["ema_model_state_dict"].items() if k not in ("initted", "step")}
        self.cfm.load_state_dict(ema, strict=False)
        self.cfm.eval()

        # 3. Load Vocos Vocoder (needed by F5-TTS internally)
        real_voc = load_vocoder("vocos", device=self.device)
        class Cap:
            def __init__(self, r):
                self.r = r
                self.last = None
            def decode(self, m):
                self.last = m.detach().cpu().numpy()
                return self.r.decode(m)
        self.cap = Cap(real_voc)

        # 4. Load BigVGAN Vocoder (for high fidelity waveform generation)
        self.g = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
        bsd = torch.load(DEFAULT_VOC, map_location="cpu")
        bsd = bsd.get("model", bsd)
        self.g.load_state_dict(bsd)
        self.g.remove_weight_norm()
        self.g = self.g.to(self.device).eval()
        for p in self.g.parameters():
            p.requires_grad = False

        # 5. Load reference bank
        if os.path.exists(BANK_PATH):
            self.bank = json.load(open(BANK_PATH, encoding="utf-8"))
            self.primes = self.bank.get("repeat_primes", {})
            for k, v in self.bank.items():
                if k.startswith("_") or not isinstance(v, dict) or "wav" not in v:
                    continue
                self.lut[k.lower()] = v
                self.lut[v["wav"].replace(".wav", "").lower()] = v
        else:
            print(f"[Warning] Reference bank not found at {BANK_PATH}", flush=True)

        print("[Model] All models loaded successfully.", flush=True)

    def get_ref(self, meter: str):
        key = meter.lower().replace(".wav", "")
        if key in self.ref_cache:
            return self.ref_cache[key]
        if key not in self.lut:
            if FALLBACK_METER not in self.lut:
                raise RuntimeError(f"Meter '{meter}' not in bank, and fallback '{FALLBACK_METER}' missing.")
            key = FALLBACK_METER
        
        e = self.lut[key]
        bdir = os.path.join(HERE, "reference_bank")
        ref_wav = os.path.join(bdir, e["wav"])
        ref_text = e["ref_text"]
        sps = float(e.get("sec_per_syll", 0.26))
        
        ref_audio, ref_t = preprocess_ref_audio_text(ref_wav, ref_text, clip_short=True)
        ra, sr = _ta.load(ref_audio)
        ref_len = ra.shape[-1] / sr
        
        val = (ref_audio, ref_t, sps, ref_len)
        self.ref_cache[key] = val
        return val

    def bvgan_decode(self, mel):
        m = torch.from_numpy(mel).to(self.device)
        with torch.no_grad():
            if m.dim() == 3 and m.shape[1] != 100 and m.shape[2] == 100:
                m = m.transpose(1, 2)
            return self.g(m).squeeze().cpu().numpy().astype(np.float32)

models = TTSModels()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    device_override = os.environ.get("VAGDHENU_DEVICE", None)
    models.load(device_override)
    yield
    # Cleanup on shutdown (if any)
    pass

app = FastAPI(
    title="Vāgdhenu Sanskrit TTS API",
    description="Microservice wrapper for metrical Sanskrit audio generation",
    version="1.0.0",
    lifespan=lifespan
)

# ── API Key Security Check ────────────────────────────────────────────────────────────
ADMIN_TOKEN = os.environ.get("VAGDHENU_API_KEY", "")
PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}

async def verify_api_key(request: Request):
    if not ADMIN_TOKEN:
        return  # No token configured = open access (dev mode)
    if request.url.path in PUBLIC_PATHS or request.url.path.startswith("/out/"):
        return  # Public paths
    api_key = request.headers.get("X-API-Key", "")
    auth_header = request.headers.get("Authorization", "")
    if api_key == ADMIN_TOKEN:
        return
    if auth_header.startswith("Bearer ") and auth_header[7:] == ADMIN_TOKEN:
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key. Provide X-API-Key or Authorization: Bearer header."
    )

if not ADMIN_TOKEN:
    print("[Warning] VAGDHENU_API_KEY is NOT set. API running WITHOUT authentication.", flush=True)
else:
    print(f"[Auth] API key authentication enabled.", flush=True)

# ── Async Job Queue (in-memory) ──────────────────────────────────────────────────────
_jobs: dict = {}
_job_queue: list = []
_jobs_lock = threading.Lock()
_worker_busy = False

def _process_job(job_id: str, req: "TTSRequest"):
    """Background worker: run inference, store result in _jobs."""
    global _worker_busy
    try:
        with _jobs_lock:
            _jobs[job_id]["status"] = "running"
        
        padas = _resolve_padas(req)
        if not padas:
            raise ValueError("No valid text/padas")
        
        req_format = req.format.lower()
        mode = req.mode.lower()
        
        hash_str = _compute_hash(padas, req)
        cache_key = f"vagdhenu/{hash_str}.{req_format}"
        
        cached = _check_cache(cache_key, req_format)
        if cached is not None:
            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                _jobs[job_id]["result"] = cached.model_dump()
            return
        
        final_audio, dur, timestamps_data = _render_padas(req, padas)
        
        wav_io = io.BytesIO()
        sf.write(wav_io, final_audio, SR, format="WAV")
        audio_bytes = wav_io.getvalue()
        
        if req_format == "mp3":
            audio_bytes = convert_wav_to_mp3(audio_bytes)
        
        meta = {
            "urn": req.urn or "",
            "mode": mode,
            "meter": req.meter if mode != "japa" else "gadya",
            "speed": req.speed,
            "seed": req.seed,
            "repeat": req.repeat,
            "format": req_format,
            "no_sandhi": req.no_sandhi,
            "text": padas[0][:200] if padas else "",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        } if req.embed_metadata else None
        
        result = _upload_and_respond(audio_bytes, cache_key, dur, timestamps_data, req_format, meta)
        
        with _jobs_lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _jobs[job_id]["result"] = result.model_dump()
    except Exception as e:
        import traceback
        traceback.print_exc()
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            _jobs[job_id]["error"] = str(e)
    finally:
        _worker_busy = False
        _try_dequeue()

def _try_dequeue():
    """Start next queued job if worker is idle."""
    global _worker_busy
    with _jobs_lock:
        if _worker_busy or not _job_queue:
            return
        _worker_busy = True
        job_id = _job_queue.pop(0)
    threading.Thread(target=_process_job, args=(job_id, _jobs[job_id]["request"]), daemon=True).start()


# ── API Models ────────────────────────────────────────────────────────────────────────
class PadTimestamp(BaseModel):
    pada: str = Field(..., description="The original pada text.")
    start: float = Field(..., description="Start time in seconds from audio beginning.")
    end: float = Field(..., description="End time in seconds from audio beginning.")

class TTSRequest(BaseModel):
    text: Optional[str] = Field(None, description="Raw shloka text. Will be split by danda/newlines if padas is not specified.")
    padas: Optional[List[str]] = Field(None, description="Array of padas (hemistichs/lines) to render sequentially.")
    meter: str = Field("anushtubh", description="Sanskrit meter name from reference bank. Ignored when mode='japa' (uses 'gadya' for flat prosody).")
    speed: float = Field(0.90, description="Pace override (lower is slower/elongated). For japa mode, 0.70–0.85 recommended.")
    seed: int = Field(50, description="Random seed for inference variance.")
    no_sandhi: bool = Field(False, description="Disable automatic Sanskrit sandhi phonology processing.")
    format: str = Field("wav", description="Target audio format ('wav' or 'mp3').")
    mode: str = Field("parayana", description="Render mode: 'parayana' (metered chanting) or 'japa' (flat meditative repetition).")
    repeat: int = Field(1, ge=1, le=1080, description="Number of times to repeat the mantra text. Only used in japa mode. Max 1080 (10 malas).")
    embed_metadata: bool = Field(True, description="Embed Brahma provenance metadata as ID3 tags in the audio file.")
    urn: Optional[str] = Field(None, description="Optional Brahma Vakya URN for audio provenance tracking.")
    reverb: Optional[str] = Field(None, description="Optional reverb preset: 'temple', 'cave', 'studio'.")
    chorus: bool = Field(False, description="Apply a multi-voice chorus chanting effect.")
    tanpura: Optional[str] = Field(None, description="Optional background drone: 'C#', 'D', 'G', etc.")
    pause_duration: Optional[float] = Field(None, description="Optional custom pause duration in seconds between padas.")

class TTSResponse(BaseModel):
    url: str = Field(..., description="The direct URL to download/stream the generated audio.")
    dur: float = Field(..., description="Duration of the generated audio in seconds.")
    cached: bool = Field(..., description="True if the request resolved to a pre-existing cache file.")
    timestamps: List[PadTimestamp] = Field(default_factory=list, description="Per-pada start/end timestamps for karaoke sync.")
    content_hash: str = Field("", description="SHA-256 hash of the audio bytes (empty when cached).")
    vagdhenu_version: str = Field(VAGDHENU_VERSION, description="Vagdhenu model version that produced this audio.")

class SegmentRequest(BaseModel):
    text: str = Field(..., description="Devanagari text to split into padas.")
    meter: Optional[str] = Field(None, description="Optional meter hint (e.g. 'anushtubh' = 4 padas). Enables syllable-count fallback splitting.")

# ── Shared render pipeline ───────────────────────────────────────────────────────────
def _resolve_padas(req: TTSRequest) -> List[str]:
    """Extract padas from request (explicit or text-split)."""
    if req.padas:
        return req.padas
    if req.text:
        raw_lines = [line.strip() for line in req.text.replace("॥", "।").split("।") if line.strip()]
        return raw_lines if raw_lines else [req.text.strip()]
    return []

def _compute_hash(padas: List[str], req: TTSRequest) -> str:
    payload = req.model_dump()
    payload["padas"] = padas
    if "meter" in payload and payload["meter"]:
        payload["meter"] = payload["meter"].lower()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

def _preprocess_pieces(padas: List[str], no_sandhi: bool) -> List[str]:
    """Apply sandhi, satva, anusvara, danda, metathesis, vocalic-l normalization."""
    pieces = [_basetext(p, no_sandhi) for p in padas]
    if not no_sandhi:
        pieces = [_satva(x) for x in pieces]
    pieces = [_danda_fix(_anusvara_m(x)) for x in pieces]
    pieces = [_hna_metathesis(x) for x in pieces]
    pieces = [_vocalic_l(x) for x in pieces]
    return pieces

def _compute_timestamps(bseg: List[np.ndarray], gaps: List[np.ndarray], original_padas: List[str], sr: int = SR) -> List[dict]:
    """Compute per-pada start/end timestamps from rendered segments."""
    timestamps = []
    cumulative = 0.0
    for idx, seg in enumerate(bseg):
        seg_dur = len(seg) / sr
        timestamps.append({
            "pada": original_padas[idx].strip(),
            "start": round(cumulative, 3),
            "end": round(cumulative + seg_dur, 3)
        })
        cumulative += seg_dur
        if idx < len(bseg) - 1 and idx < len(gaps):
            cumulative += len(gaps[idx]) / sr
    return timestamps

def _apply_dsp_and_stitch(bseg: List[np.ndarray], GAPS: List[np.ndarray], padas: List[str], req: TTSRequest):
    # Stitch
    if len(bseg) == 1:
        final_audio = bseg[0]
    else:
        stitched = []
        last = len(bseg) - 1
        for idx, s in enumerate(bseg):
            stitched.append(s)
            if idx < last:
                stitched.append(GAPS[idx])
        final_audio = np.concatenate(stitched)

    # 1. Apply Chorus if requested
    if req.chorus:
        from src.dsp import apply_chorus
        final_audio = apply_chorus(final_audio, SR)

    # 2. Apply Reverb if requested
    if req.reverb and req.reverb.lower().strip() in ("temple", "cave", "studio"):
        from src.dsp import apply_reverb
        final_audio = apply_reverb(final_audio, SR, req.reverb.lower().strip())

    # 3. Apply Tanpura Background Drone if requested
    if req.tanpura:
        from src.dsp import synthesize_drone, mix_drone
        drone_duration = len(final_audio) / float(SR)
        drone = synthesize_drone(drone_duration, SR, req.tanpura)
        final_audio = mix_drone(final_audio, drone, volume_db=-24)

    dur = float(len(final_audio) / SR)
    timestamps_data = _compute_timestamps(bseg, GAPS, padas)
    return final_audio, dur, timestamps_data

def _replicate_infer(ref_audio: str, ref_text: str, gen_text: str, speed: float, nfe_step: int, cfg_strength: float, fix_duration: Optional[float]):
    """Helper to run inference on Replicate GPU worker"""
    import replicate
    import requests
    
    model_version = os.environ.get("VAGDHENU_REPLICATE_MODEL")
    if not model_version:
        raise ValueError("VAGDHENU_REPLICATE_MODEL environment variable must be set to use Replicate backend.")
        
    print(f"[Replicate] Running inference for: '{gen_text[:40]}...' using model {model_version}...", flush=True)
    
    # Initialize Replicate client with a 5-minute timeout to handle cold starts
    client = replicate.Client(api_token=os.environ.get("REPLICATE_API_TOKEN"), timeout=300.0)
    
    # Open local reference audio file; replicate library will upload it automatically
    with open(ref_audio, "rb") as f:
        inputs = {
            "ref_audio": f,
            "ref_text": ref_text,
            "gen_text": gen_text,
            "speed": speed,
            "nfe_step": nfe_step,
            "cfg_strength": cfg_strength,
        }
        if fix_duration is not None:
            inputs["fix_duration"] = fix_duration
            
        output = client.run(model_version, input=inputs, wait=300)
        
    # Handle Replicate SDK output:
    #   SDK ≥1.0: returns FileOutput (has .read() method)
    #   SDK <1.0: returns a URL string
    if hasattr(output, 'read'):
        # FileOutput — read bytes directly (no extra HTTP call)
        audio_bytes = output.read()
    elif isinstance(output, str):
        # Legacy URL string
        resp = requests.get(output)
        resp.raise_for_status()
        audio_bytes = resp.content
    else:
        # Possibly a list with a single FileOutput
        item = output[0] if isinstance(output, list) else output
        audio_bytes = item.read() if hasattr(item, 'read') else requests.get(str(item)).content
    
    # Read the audio bytes back into a numpy array
    audio_io = io.BytesIO(audio_bytes)
    y, sr = sf.read(audio_io)
    return y.astype(np.float32)

def _render_padas(req: TTSRequest, padas: List[str]):
    """Core render: returns (final_audio, dur, timestamps). Does NOT handle caching or upload.

    Supports two modes:
      - 'parayana': metered chanting with reference-clip prosody (default)
      - 'japa': flat meditative repetition, uses 'gadya' meter, lower cfg_strength
    """
    is_japa = req.mode == "japa" and req.repeat > 1

    # ── Japa mode: repeat a single mantra N times ──
    if is_japa:
        mantra_text = padas[0] if padas else ""
        if not mantra_text:
            raise ValueError("Japa mode requires non-empty mantra text.")

        # Use 'gadya' (prose) reference for flat intonation — no metrical contour
        japa_meter = "gadya"
        try:
            ref_audio, ref_t, sps, ref_len = models.get_ref(japa_meter)
        except Exception:
            # Fallback to anushtubh if gadya not in bank
            ref_audio, ref_t, sps, ref_len = models.get_ref("anushtubh")

        processed = _preprocess_pieces([mantra_text], req.no_sandhi)[0]
        n_syl = n_aksharas(processed)

        # Japa parameters: slower, lower cfg for flatter prosody, wider gap
        japa_speed = req.speed if req.speed <= 0.95 else 0.78
        japa_cfg = 1.5  # lower = less reference-driven contour
        japa_gap_s = req.pause_duration if req.pause_duration is not None else 1.2  # breathing room between repetitions

        bseg = []
        japa_pada_labels = []
        for rep in range(req.repeat):
            _fixd = (ref_len + n_syl * sps) if (sps > 0 and n_syl) else None
            if models.backend == "replicate":
                y = _replicate_infer(
                    ref_audio=ref_audio,
                    ref_text=ref_t,
                    gen_text=processed,
                    speed=japa_speed,
                    nfe_step=64,
                    cfg_strength=japa_cfg,
                    fix_duration=_fixd
                )
            else:
                torch.manual_seed(req.seed + rep)
                w, sr, _ = infer_process(
                    ref_audio, ref_t, processed, models.cfm, models.cap,
                    mel_spec_type="vocos", speed=japa_speed,
                    nfe_step=64, cfg_strength=japa_cfg,
                    device=models.device, fix_duration=_fixd
                )
                w = np.array(w, dtype=np.float32)
                if np.abs(w).max() > 1.5:
                    w = w / 32768.0
                y = models.bvgan_decode(models.cap.last)
                mx = np.abs(y).max()
                y = y / mx * 0.97 if mx > 1 else y
            bseg.append(y)
            japa_pada_labels.append(f"{mantra_text.strip()}  ({rep + 1}/{req.repeat})")

        # Generate gaps between repetitions
        japa_gaps = [np.zeros(int(japa_gap_s * SR), dtype=np.float32) for _ in range(len(bseg) - 1)]
        return _apply_dsp_and_stitch(bseg, japa_gaps, japa_pada_labels, req)

    # ── Parayana mode: original metered chanting ──
    ref_audio, ref_t, sps, ref_len = models.get_ref(req.meter)
    processed_pieces = _preprocess_pieces(padas, req.no_sandhi)

    # Priming
    mono = max((_rep_depths(_aksharas(x))[0] for x in processed_pieces), default=1)
    di = max((_rep_depths(_aksharas(x))[1] for x in processed_pieces), default=1)
    _ra, _rt = ref_audio, ref_t
    _pick = None
    if di >= 3:
        _pick = next((k for k in ["prime_jaya","prime_chata"] if k in models.primes and models.primes[k].get("di_max",0)>=di), None) \
                or next((k for k,v in models.primes.items() if isinstance(v,dict) and v.get("di_max",0)>=di), None)
    if _pick is None and mono >= 2 and "prime_mono" in models.primes and models.primes["prime_mono"].get("mono_max",0) >= mono:
        _pick = "prime_mono"
    bdir = os.path.join(HERE, "reference_bank")
    if _pick:
        _pv = models.primes[_pick]
        _ra, _rt = preprocess_ref_audio_text(os.path.join(bdir, _pv["wav"]), _pv["ref_text"], clip_short=True)
        _prb, _psr = _ta.load(_ra)
        ref_len = _prb.shape[-1] / _psr

    NSYLL = [n_aksharas(x) for x in processed_pieces]
    gap_duration = req.pause_duration if req.pause_duration is not None else 0.55
    gap_halant_duration = 0.20
    GAPS = [np.zeros(int(gap_duration*SR) + (int(gap_halant_duration*SR) if _ends_halant(_p) else 0), dtype=np.float32) for _p in processed_pieces]

    bseg = []
    for i, p in enumerate(processed_pieces):
        _fixd = (ref_len + NSYLL[i]*sps) if (sps > 0 and NSYLL) else None
        if models.backend == "replicate":
            y = _replicate_infer(
                ref_audio=_ra,
                ref_text=_rt,
                gen_text=p,
                speed=req.speed,
                nfe_step=64,
                cfg_strength=3.0,
                fix_duration=_fixd
            )
        else:
            au = None
            for att in range(4):
                torch.manual_seed(req.seed + att)
                w, sr, _ = infer_process(
                    _ra, _rt, p, models.cfm, models.cap,
                    mel_spec_type="vocos", speed=req.speed,
                    nfe_step=64, cfg_strength=3.0,
                    device=models.device, fix_duration=_fixd
                )
                w = np.array(w, dtype=np.float32)
                if np.abs(w).max() > 1.5:
                    w = w / 32768.0
                if float(np.sqrt((w**2).mean())) > 0.04:
                    au = w
                    break
            if au is None:
                au = w
            y = models.bvgan_decode(models.cap.last)
            mx = np.abs(y).max()
            y = y / mx * 0.97 if mx > 1 else y
        bseg.append(y)

    return _apply_dsp_and_stitch(bseg, GAPS, padas, req)

def _upload_and_respond(audio_bytes: bytes, cache_key: str, dur: float, timestamps_data: List[dict], req_format: str, meta: dict = None):
    """Upload to R2 (or save local), return TTSResponse with content_hash."""
    # Compute content hash BEFORE metadata embedding (hash of raw audio, not tagged version)
    raw_hash = hashlib.sha256(audio_bytes).hexdigest()
    
    # Embed Brahma provenance metadata into audio file
    if meta:
        audio_bytes = _embed_metadata(audio_bytes, meta, req_format)
    
    timestamps_json = json.dumps(timestamps_data, ensure_ascii=False)
    if r2_client and R2_BUCKET_NAME:
        r2_client.put_object(
            Bucket=R2_BUCKET_NAME,
            Key=cache_key,
            Body=audio_bytes,
            ContentType=f"audio/{req_format}",
            Metadata={"duration": str(dur), "timestamps": timestamps_json, "content_hash": raw_hash, "vagdhenu_version": VAGDHENU_VERSION}
        )
        public_url = f"{R2_PUBLIC_URL}/{cache_key}" if R2_PUBLIC_URL else f"/out/{os.path.basename(cache_key)}"
    else:
        local_dir = os.path.join(REPO, "out")
        os.makedirs(local_dir, exist_ok=True)
        local_fname = cache_key.split("/")[-1] if "/" in cache_key else cache_key
        local_path = os.path.join(local_dir, local_fname)
        with open(local_path, "wb") as f:
            f.write(audio_bytes)
        public_url = f"/out/{local_fname}"
    return TTSResponse(url=public_url, dur=dur, cached=False, timestamps=[PadTimestamp(**ts) for ts in timestamps_data], content_hash=raw_hash, vagdhenu_version=VAGDHENU_VERSION)

def _check_cache(cache_key: str, req_format: str):
    """Check R2 or local cache. Returns TTSResponse if found, else None."""
    if r2_client and R2_BUCKET_NAME:
        try:
            head_res = r2_client.head_object(Bucket=R2_BUCKET_NAME, Key=cache_key)
            meta = head_res.get("Metadata", {})
            dur = float(meta.get("duration", 0.0))
            ts_raw = meta.get("timestamps", "[]")
            timestamps_data = json.loads(ts_raw) if ts_raw else []
            public_url = f"{R2_PUBLIC_URL}/{cache_key}" if R2_PUBLIC_URL else f"/out/{os.path.basename(cache_key)}"
            return TTSResponse(url=public_url, dur=dur, cached=True, timestamps=[PadTimestamp(**ts) for ts in timestamps_data])
        except Exception:
            pass
    else:
        local_fname = cache_key.split("/")[-1] if "/" in cache_key else cache_key
        local_path = os.path.join(REPO, "out", local_fname)
        if os.path.exists(local_path):
            try:
                import wave
                with wave.open(local_path, "rb") as wav_file:
                    dur = wav_file.getnframes() / float(wav_file.getframerate())
            except Exception:
                dur = 0.0
            return TTSResponse(url=f"/out/{local_fname}", dur=dur, cached=True, timestamps=[])
    return None

# ── Controller logic ──────────────────────────────────────────────────────────────────
@app.post("/tts", response_model=TTSResponse, status_code=status.HTTP_200_OK)
async def generate_tts(req: TTSRequest, _ = Depends(verify_api_key)):
    padas = _resolve_padas(req)
    if not padas:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Either 'text' or 'padas' must be provided with valid non-empty Sanskrit text.")

    req_format = req.format.lower()
    if req_format not in ("wav", "mp3"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported format. Only 'wav' and 'mp3' are supported.")

    mode = req.mode.lower()
    if mode not in ("parayana", "japa"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Mode must be 'parayana' or 'japa'.")

    hash_str = _compute_hash(padas, req)
    cache_key = f"vagdhenu/{hash_str}.{req_format}"

    cached = _check_cache(cache_key, req_format)
    if cached is not None:
        return cached

    try:
        final_audio, dur, timestamps_data = _render_padas(req, padas)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(e)}")

    wav_io = io.BytesIO()
    sf.write(wav_io, final_audio, SR, format="WAV")
    audio_bytes = wav_io.getvalue()

    if req_format == "mp3":
        try:
            audio_bytes = convert_wav_to_mp3(audio_bytes)
        except Exception as e:
            print(f"[Error] MP3 conversion failed: {e}", flush=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Audio format conversion to MP3 failed. Error: {e}")

    # Build metadata for embedding
    meta = {
        "urn": req.urn or "",
        "mode": mode,
        "meter": req.meter if mode != "japa" else "gadya",
        "speed": req.speed,
        "seed": req.seed,
        "repeat": req.repeat,
        "format": req_format,
        "no_sandhi": req.no_sandhi,
        "text": padas[0][:200] if padas else "",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    } if req.embed_metadata else None

    return _upload_and_respond(audio_bytes, cache_key, dur, timestamps_data, req_format, meta)

@app.post("/tts/batch", response_model=List[TTSResponse], status_code=status.HTTP_200_OK)
async def generate_tts_batch(reqs: List[TTSRequest], _ = Depends(verify_api_key)):
    """Batch render multiple TTS requests. Models loaded once; renders sequentially."""
    if not reqs:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Batch must contain at least one request.")
    if len(reqs) > BATCH_MAX:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Batch limited to {BATCH_MAX} requests (server constraint).")

    results: List[TTSResponse] = []
    for idx, req in enumerate(reqs):
        padas = _resolve_padas(req)
        if not padas:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item {idx}: either 'text' or 'padas' must be provided.")
        req_format = req.format.lower()
        if req_format not in ("wav", "mp3"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item {idx}: unsupported format '{req_format}'.")
        mode = req.mode.lower()
        if mode not in ("parayana", "japa"):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Item {idx}: mode must be 'parayana' or 'japa'.")

        hash_str = _compute_hash(padas, req)
        cache_key = f"vagdhenu/{hash_str}.{req_format}"

        cached = _check_cache(cache_key, req_format)
        if cached is not None:
            results.append(cached)
            continue

        try:
            final_audio, dur, timestamps_data = _render_padas(req, padas)
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Item {idx} inference failed: {str(e)}")

        wav_io = io.BytesIO()
        sf.write(wav_io, final_audio, SR, format="WAV")
        audio_bytes = wav_io.getvalue()

        if req_format == "mp3":
            try:
                audio_bytes = convert_wav_to_mp3(audio_bytes)
            except Exception as e:
                print(f"[Error] MP3 conversion failed for item {idx}: {e}", flush=True)
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Item {idx}: MP3 conversion failed.")

        meta = {
            "urn": req.urn or "",
            "mode": mode,
            "meter": req.meter if mode != "japa" else "gadya",
            "speed": req.speed,
            "seed": req.seed,
            "repeat": req.repeat,
            "format": req_format,
            "no_sandhi": req.no_sandhi,
            "text": padas[0][:200] if padas else "",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        } if req.embed_metadata else None

        results.append(_upload_and_respond(audio_bytes, cache_key, dur, timestamps_data, req_format, meta))

    return results

# ── Pada Segmentation ───────────────────────────────────────────────────────────────

# Meter → expected pada count + syllables per pada (for syllable-based fallback splitting)
_METER_PADA_MAP: dict = {
    "anuṣṭubh": (4, 8),
    "anushtubh": (4, 8),
    "pramāṇikā": (4, 8),
    "pramanika": (4, 8),
    "vasantatilakā": (4, 14),
    "vasantatilaka": (4, 14),
    "upajāti": (4, 11),
    "upajati": (4, 11),
    "indravajrā": (4, 11),
    "indravajra": (4, 11),
    "upendravajrā": (4, 11),
    "upendravajra": (4, 11),
    "vaṃśastha": (4, 12),
    "vamshastha": (4, 12),
    "rathoddhatā": (4, 11),
    "rathoddhata": (4, 11),
    "śālinī": (4, 11),
    "shalini": (4, 11),
    "indravaṃśā": (4, 12),
    "indravamsha": (4, 12),
    "drutavilambita": (4, 12),
    "bhujaṅgaprayāta": (4, 12),
    "bhujangaprayata": (4, 12),
    "mālinī": (4, 15),
    "malini": (4, 15),
    "śārdūlavikrīḍita": (4, 19),
    "shardulavikridita": (4, 19),
    "sragdharā": (4, 21),
    "sragdhara": (4, 21),
}

def _split_by_syllables(text: str, syllables_per_pada: int, expected_count: int) -> List[str]:
    """Split a cleaned text string into equal-syllable padas.

    Uses the same n_aksharas() logic as the rest of Vagdhenu for consistent syllable counting.
    Walks through the text character by character, calling n_aksharas() on cumulative substrings
    to detect syllable boundaries. This is O(n²) but only called as a fallback for edge cases.
    """
    cleaned = text.replace(" ", "").replace("।", "").replace("॥", "").replace("\n", "").strip()
    if not cleaned:
        return [text.strip()]
    total_syl = n_aksharas(cleaned)
    if total_syl == 0 or syllables_per_pada <= 0:
        return [text.strip()]
    # Walk through text; cut when cumulative n_aksharas reaches the threshold
    padas = []
    start = 0
    for end in range(1, len(cleaned) + 1):
        if end == len(cleaned):
            # Last chunk
            padas.append(cleaned[start:end].strip())
            break
        cum = n_aksharas(cleaned[start:end])
        if cum >= syllables_per_pada and len(padas) < expected_count - 1:
            padas.append(cleaned[start:end].strip())
            start = end
            if len(padas) >= expected_count:
                remaining = cleaned[start:].strip()
                if remaining:
                    padas.append(remaining)
                break
    return padas if padas else [text.strip()]

class SegmentResponse(BaseModel):
    padas: List[str] = Field(..., description="Split pada strings.")
    count: int = Field(..., description="Number of padas.")
    method: str = Field(..., description="Splitting method: 'danda', 'newline', 'syllable', or 'raw'.")

@app.post("/tts/segment", response_model=SegmentResponse, status_code=status.HTTP_200_OK)
async def segment_text(req: SegmentRequest, _ = Depends(verify_api_key)):
    """Split Sanskrit text into padas for TTS rendering.

    Tries in order:
    1. Split by danda (। / ॥) — standard verse boundary
    2. Split by newline
    3. If meter is provided and count doesn't match expected, syllable-based split
    4. Fallback: return as single pada
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Text must not be empty.")

    # Method 1: danda split
    danda_parts = [p.strip() for p in text.replace("॥", "।").split("।") if p.strip()]
    if len(danda_parts) > 1:
        return SegmentResponse(padas=danda_parts, count=len(danda_parts), method="danda")

    # Method 2: newline split
    newline_parts = [p.strip() for p in text.split("\n") if p.strip()]
    if len(newline_parts) > 1:
        return SegmentResponse(padas=newline_parts, count=len(newline_parts), method="newline")

    # Method 3: syllable-based with meter hint
    if req.meter:
        m = req.meter.lower().strip()
        if m in _METER_PADA_MAP:
            expected_count, syl_per = _METER_PADA_MAP[m]
            # Only apply if the full text is long enough for the expected count
            total_syl = n_aksharas(text)
            if total_syl >= expected_count * syl_per * 0.7:  # allow 30% tolerance
                syll_padas = _split_by_syllables(text, syl_per, expected_count)
                if len(syll_padas) > 1:
                    return SegmentResponse(padas=syll_padas, count=len(syll_padas), method="syllable")

    # Method 4: fallback — return as-is
    return SegmentResponse(padas=[text], count=1, method="raw")


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    models_loaded: bool
    meters_available: int
    batch_max: int
    memory_used_mb: float
    memory_total_mb: float
    gpu_available: bool

@app.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check():
    """Server health + capacity introspection for orchestration layer."""
    mem = psutil.virtual_memory()
    return HealthResponse(
        status="healthy",
        version=VAGDHENU_VERSION,
        device=models.device,
        models_loaded=models.cfm is not None,
        meters_available=len(models.bank),
        batch_max=BATCH_MAX,
        memory_used_mb=round(mem.used / (1024 * 1024), 1),
        memory_total_mb=round(mem.total / (1024 * 1024), 1),
        gpu_available=torch.cuda.is_available(),
    )

# Helper function to convert WAV bytes to MP3
def convert_wav_to_mp3(wav_bytes: bytes) -> bytes:
    wav_io = io.BytesIO(wav_bytes)
    audio = AudioSegment.from_wav(wav_io)
    mp3_io = io.BytesIO()
    audio.export(mp3_io, format="mp3", bitrate="192k")
    return mp3_io.getvalue()


# ── Async Job Endpoints ───────────────────────────────────────────────────────────────
class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    result: Optional[TTSResponse] = None
    error: Optional[str] = None

@app.post("/tts/async", response_model=JobSubmitResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_async_tts(req: TTSRequest, _=Depends(verify_api_key)):
    """
    Submit a TTS job for async processing. Returns immediately with a job_id.
    Poll GET /tts/jobs/{job_id} for result.

    This endpoint avoids proxy timeouts — ideal for CPU servers where inference
    takes minutes. Jobs are processed sequentially via a background thread.
    """
    padas = _resolve_padas(req)
    if not padas:
        raise HTTPException(status_code=400, detail="Either 'text' or 'padas' must be provided.")
    mode = req.mode.lower()
    if mode not in ("parayana", "japa"):
        raise HTTPException(status_code=400, detail="Mode must be 'parayana' or 'japa'.")

    job_id = str(uuid.uuid4())[:12]
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": now,
            "completed_at": None,
            "result": None,
            "error": None,
            "request": req,
        }
        _job_queue.append(job_id)
        queue_pos = len(_job_queue)
    
    _try_dequeue()
    return JobSubmitResponse(
        job_id=job_id,
        status="queued",
        message=f"Job queued (position {queue_pos}). Poll /tts/jobs/{job_id} for status."
    )

@app.get("/tts/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of an async TTS job."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        result=TTSResponse(**job["result"]) if job.get("result") else None,
        error=job.get("error"),
    )

@app.get("/tts/jobs")
async def list_jobs(limit: int = 20):
    """List recent jobs (max 50)."""
    with _jobs_lock:
        items = sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:min(limit, 50)]
    return {"count": len(items), "jobs": items}
