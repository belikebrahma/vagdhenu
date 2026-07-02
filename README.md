# Vāgdhenu — Sanskrit Chant TTS

*"The wish-cow of speech."* A production-grade, single-speaker **Sanskrit chant (pārāyaṇa) text-to-speech** system — it *chants* classical ślokas with metrically-aware durations and tradition-faithful melodic contour, not flat read-aloud.

> **MOS ~4.6** (expert listener). Conjuncts — including retroflex aspirates (ṣṭ, ḍḍh, …) — render 100% correctly, the class earlier architectures could not crack. Used to produce **MBTN** (32 YouTube videos, 17h 34m) and the **Śrīmad Bhāgavatam** (16,017 verses, audio app + 31 karaoke videos).

[ **[Project page + live demo](https://prathosh.in/vagdhenu/)** · [Model weights → HF](https://huggingface.co/prathoshap/vagdhenu) · [Demo → HF Space](https://huggingface.co/spaces/prathoshap/vagdhenu-demo) · Tech report → `docs/TECH_REPORT.md` ]

## Demos (rendered with this system)
- **Mahābhārata Tātparya Nirṇaya (MBTN)** — full chant series: [YouTube playlist](https://www.youtube.com/playlist?list=PLL1s8qiaGy0IP0G_PhlwaGA5EOfzoKrV_)
- **Śrīmad Bhāgavatam** — karaoke-video series: [YouTube playlist](https://www.youtube.com/playlist?list=PLDiYyVdyo2Sc)

Developed and maintained by **Prof. Prathosh, Indian Institute of Science, Bengaluru.**

## How it works
- **Backbone:** IndicF5 / F5-TTS — a flow-matching **DiT** (OT-CFM mel-infilling, ~337M params, *no* native duration or pitch head). Sanskrit is routed through **Kannada script** (Devanagari triggers Hindi schwa-deletion).
- **Vocoder:** NVIDIA **BigVGAN-v2**, fine-tuned on F5 vocos-mel (mandatory — vocos shivers on long vowels).
- **Prosody:** F5's content fidelity is bulletproof but its prosody is *text-driven, not designable*. The working levers are **the reference clip** (voice + swara + pace, via the *half-reference rule*) and a **voice-steering fine-tune**. (See `docs/TECH_REPORT.md` §14 for the full account — this is the central architectural finding.)
- **Text frontend (`src/prep_text.py`)** — the most reusable piece: Deva→SLP1→Kannada routing, internal visarga sandhi (utva/rutva/lopa/satva), homorganic anusvāra, vocalic-ṝ handling, daṇḍa-final rules, meter/gaṇa (L/G) detection.

## Layout
```
src/         text frontend, meter detection, inference, post-gate, reference bank
pipeline/    data-prep (cut→pair→train) + build/assemble/QC
demo/        Gradio app (HF ZeroGPU)
docs/        scrubbed technical report + frontend/pipeline references
examples/    sample inputs + rendered outputs
scripts/     env setup + weight download
```

## Install & quickstart
Requires **Python 3.10** and a **CUDA 12.1 GPU**.
```bash
bash scripts/setup.sh    # torch+cu121, deps, BigVGAN, and downloads weights -> models/
# render a Devanagari verse (+ meter) to a chanted wav:
python src/render.py --shard examples/sample_shard.json --results /tmp/res.json --outdir out
# -> out/sample_anushtubh.wav
```
The batch renderer takes a shard JSON: `[{"id","meter","padas":[devanagari…],"seed","out"}]`. For one-off single-verse renders see `src/render_production.py`. `CHAMP_ROOT` env overrides the weights dir (default `models/`).

## API Reference

Vāgdhenu exposes a production-grade FastAPI server. When running the server, interactive Swagger OpenAPI documentation is available at `http://localhost:8000/docs`.

### 1. TTS Synthesis Endpoint (`POST /tts`)

Render Sanskrit text to high-quality audio with advanced digital signal processing (DSP) options.

#### Request Body Schema (JSON)

| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `text` | `str` | *Required* | Sanskrit text in Devanagari to chant. |
| `padas` | `List[str]` | `None` | Alternative to `text`; explicit pre-split lines/quarters of the verse. |
| `meter` | `str` | `"anushtubh"` | Reference voice pattern template from catalog. Supports `"anushtubh"`, `"tristubh"`, `"jagati"`, `"gayatri"`, `"vasantatilaka"`, `"adiyogi"`, etc. |
| `speed` | `float` | `0.90` | Chanting speed multiplier (recommended range: `0.75` - `1.05`). |
| `seed` | `int` | `42` | Random seed for the CFM infilling process (determines melodic micro-variations). |
| `no_sandhi` | `bool` | `False` | Disable sandhi preprocessing frontend (forces exact character pronunciation). |
| `format` | `str` | `"wav"` | Output audio encoding: `"wav"` or `"mp3"`. |
| `mode` | `str` | `"parayana"` | Chanting style mode: `"parayana"` (metered chanting) or `"japa"` (meditative mantra repetitions). |
| `repeat` | `int` | `1` | Number of repetitions (primarily used in `"japa"` mode). |
| `urn` | `str` | `None` | Unique reference key stored in response metadata. |
| `embed_metadata` | `bool` | `True` | Embed standard metadata tags in the generated audio files. |
| **`pause_duration`** | `float` | `None` | Custom silent breathing gap (in seconds) between sections (defaults: `0.55s` for parayana, `1.2s` for japa). |
| **`reverb`** | `str` | `None` | Reverb preset: `"temple"` (spacious stone hall), `"cave"` (deep resonance), `"studio"` (subtle warmth), or `None`. |
| **`chorus`** | `bool` | `False` | Enable chorus effect (group chanting simulation via pitch/delay offsets). |
| **`tanpura`** | `str` | `None` | Synthesize and mix a continuous Indian string drone in the background. Pitch keys: `"C"`, `"C#"`, `"D"`, `"D#"`, `"E"`, `"F"`, `"F#"`, `"G"`, `"G#"`, `"A"`, `"A#"`, `"B"`. |

#### Response Schema (JSON)

```json
{
  "url": "/out/6a2cd840def2f89c020ca510b10386729be5ef6d01661b34d8aa844c4ee0ebd6.wav",
  "dur": 2.592,
  "cached": false,
  "timestamps": [
    {
      "pada": "ॐ नमः शिवाय",
      "start": 0.0,
      "end": 2.592
    }
  ],
  "content_hash": "8ca582cfdd55f3fdc2a2739a4d629e82557ea3489e79bb6825c4daa003d55ca1",
  "vagdhenu_version": "1.1.0"
}
```

---

### 2. Batch Synthesis Endpoint (`POST /tts/batch`)

Render a sequence of TTS requests in a single transaction. Under the hood, models are kept active to render the requests sequentially.

* **URL**: `POST /tts/batch`
* **Request Payload**: A list of TTS Request objects `List[TTSRequest]`
* **Batch Limit Constraint**: Enforced at **20 requests max per batch call** (reduced from 100 to ensure safety on shared CPU/GPU environments).

---

### 3. Segment Text Endpoint (`POST /tts/segment`)

Decompose Devanagari text into metered lines (padas) without rendering audio.

* **URL**: `POST /tts/segment`
* **Request Payload**: `{"text": "ॐ नमः शिवाय"}`
* **Response**: `{"padas": ["ॐ नमः शिवाय"], "count": 1, "method": "raw"}`

---

### 4. Health & Capacity Endpoint (`GET /health`)

Capacity introspection and status checks. Used by the orchestration layer (`Vaani`) before enqueuing requests.

* **URL**: `GET /health`
* **Response**:
  ```json
  {
    "status": "healthy",
    "version": "1.1.0",
    "device": "cpu",
    "models_loaded": true,
    "meters_available": 6,
    "batch_max": 20,
    "memory_used_mb": 4210.5,
    "memory_total_mb": 4918.4,
    "gpu_available": false
  }
  ```

---

### 5. Audio Metadata Embedding (`brm1` RIFF Chunk)

Every generated WAV file is self-describing. When `embed_metadata` is `true` (default), Vāgdhenu injects a custom `brm1` RIFF sub-chunk before the audio sample data payload containing standard JSON metadata:

```
  Offset  Chunk     Content
  ──────  ────────  ──────────────────────────────────────
  0       RIFF hdr  Standard WAV header
  36      brm1      259 bytes: {urn, mode, meter, speed,
                    seed, repeat, format, text, generated_at}
  ...     data      Audio samples
```

* **Why this matters**: In case of a database loss, the complete configuration (text, parameters, voice profile, and original URN) can be reconstructed entirely from the `.wav` files alone.

## Case studies
- **MBTN** (Mahābhārata Tātparya Nirṇaya) — 32-adhyāya *video* deliverable (Devanagari + Kannada karaoke, tanpura), shipped.
- **Śrīmad Bhāgavatam** — 12 skandhas, ~18k verses, *audio* app + a 31-video 3-script (Devanāgarī · Kannada · IAST) karaoke series. Sanskrit text gratefully acknowledged to **Poornaprajna Samshodhana Mandiram, Bengaluru**.

## Attribution & licenses
- Code: **Apache-2.0** (`LICENSE`).
- Built on **AI4Bharat IndicF5** (MIT), **NVIDIA BigVGAN-v2**, and **F5-TTS** — see their licenses; weights redistributed per those terms.
- Model weights + intended-use/ethics note: see the HF model card.

## Ethics / intended use
Single-speaker synthesis of sacred Sanskrit recitation, for pārāyaṇa/study/accessibility. The voice is the author's own. Please use responsibly; do not impersonate.

## Citation
*(BibTeX added with the arXiv report.)*
