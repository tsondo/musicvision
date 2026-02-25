# MusicVision — Project Status

**Last updated:** 2026-02-25
**Branch:** `main`
**Codebase:** ~5,200 lines across 34 Python modules + 4 test files

---

## What This Project Is

MusicVision is a Python pipeline that takes an AI-generated (or real) song and produces a music video:

```
Song (WAV) + Lyrics → Scenes → Reference Images → Video Clips → Assembled Video
```

The pipeline is designed around AI-generated music from **AceStep** (a text-conditioned music generation model), which produces songs alongside structured metadata (BPM, key, lyrics with section markers). The pipeline ingests that metadata natively.

**Primary output:** A rough-cut MP4 + an EDL/FCPXML timeline for DaVinci Resolve finishing.

**Target hardware:** Dual-GPU workstation (RTX 5080 32 GB + RTX 3080 Ti 12 GB), with VRAM-tiered fallbacks for single-GPU and lighter configs.

---

## Pipeline Overview

### Stage 1 — Intake
**Status: Fully implemented and testable**

Orchestrated by `run_intake()` in `intake/pipeline.py`.

Steps:
1. Load audio, resolve duration (from AceStep metadata or ffprobe)
2. BPM detection via librosa beat tracker
3. Optional vocal separation (MelBandRoFormer or Demucs) to improve transcription quality
4. Transcription via Whisper-large-v3 (word-level timestamps)
5. LLM scene segmentation — Claude analyzes lyrics + timestamps → JSON scene list, boundaries snapped to beats
6. Rule-based fallback if no LLM available
7. Slice audio into per-scene WAV segments
8. Save `scenes.json`

**What works today without GPU:**
- BPM detection, beat tracking
- Scene segmentation (rule-based, no LLM)
- AceStep JSON import
- All persistence (project.yaml, scenes.json)

**What requires GPU:**
- Whisper transcription (can skip with pre-provided lyrics)
- Vocal separation (MelBandRoFormer / Demucs)

**What requires API key (LLM_BACKEND=anthropic):**
- LLM-assisted scene segmentation (fallback available without it)

---

### Stage 2 — Reference Image Generation
**Status: Fully implemented; requires GPU + HuggingFace token**

Each scene gets a reference still image (FLUX.1-dev or Z-Image) that the HuMo video model uses as its visual anchor.

**Image prompt generation:**
- Claude (or local vLLM) writes a 2–4 sentence cinematic description per scene
- Injected with style sheet: character descriptions, visual style, color palette, settings
- User can override any prompt via API/scenes.json

**Engine (`FluxEngine`, `ZImageEngine`):**
- VRAM-tiered loading: 4 tiers from bf16 (28 GB free) down to quantized sequential offload (<8 GB)
- FP8 quantization (Ada/Hopper GPUs, compute cap ≥ 8.9); falls back to INT8 on older GPUs
- `optimum-quanto` for quantization
- Optional project-level LoRA (style); optional per-scene LoRA (character)
- FLUX.1-dev is gated — requires accepted HuggingFace terms + `HUGGINGFACE_TOKEN`
- Z-Image is open-weight, lighter (~12 GB fp16), no token required

**Design note:** Both image engines implement the same `ImageEngine` base class. `create_engine()` factory selects by `ImageGenConfig.model`. Adding new image models later requires only a new engine class + one line in the factory.

---

### Stage 3 — Video Generation
**Status: Infrastructure complete; inference layer pending HuMo source release**

This is where the pipeline currently has a gap.

#### What IS fully built

**Tiered weight system** (`video/weight_registry.py`):
- 6 precision tiers mapped to HuggingFace repos
- `download_all_for_tier()` downloads DiT weights + shared T5/VAE/Whisper
- `weight_status()` checks presence locally
- `MUSICVISION_WEIGHTS_DIR` overrides cache location

**Tiered model loaders** (`video/model_loader.py`):
| Tier | Model | Format | DiT VRAM | Min GPU |
|------|-------|--------|----------|---------|
| `fp16` | HuMo 17B | FP16 safetensors | ~34 GB | 2× GPU (FSDP) |
| `fp8_scaled` | HuMo 17B | FP8 e4m3fn scaled (Kijai) | ~18 GB | 16 GB |
| `gguf_q8` | HuMo 17B | GGUF Q8_0 | ~18.5 GB | 20 GB |
| `gguf_q6` | HuMo 17B | GGUF Q6_K | ~14.4 GB | 16 GB |
| `gguf_q4` | HuMo 17B | GGUF Q4_K_M | ~11.5 GB | 12 GB |
| `preview` | HuMo 1.7B | FP16 safetensors | ~3.4 GB | 8 GB |

Each loader (`FP16Loader`, `FP8ScaledLoader`, `GGUFLoader`, `Preview1_7BLoader`) handles weight format, device placement, and quantized linear layer dispatch. All implement `BaseHumoLoader`.

**Block swap manager** (`video/block_swap.py`):
- HuMo 17B has ~40 transformer blocks
- `BlockSwapManager` keeps N blocks on GPU, rest on CPU, swapping sequentially during the denoising loop
- `block_swap_count=20` → ~35% VRAM reduction at ~15–20% speed cost
- Controlled by `HumoConfig.block_swap_count`

**Engine orchestration** (`video/humo_engine.py`):
- `HumoEngine.load()` — downloads missing weights, instantiates loader, applies block swap
- `HumoEngine.generate()` — accepts `HumoInput` (prompt + reference image + audio + output path)
- `HumoEngine.generate_scene()` — splits scenes > 3.88 s into sub-clips automatically
- `HumoEngine.unload()` — full VRAM teardown
- Video prompt generation: Claude/vLLM writes Qwen2.5-VL-style dense captions (the style HuMo was trained on)

#### What is STUBBED (the gap)

The four core TIA inference methods inside `HumoEngine` return `None` with log warnings:

```python
def _encode_text(self, prompt):
    # TODO: implement using wan.modules.t5.WanT5Encoder
    return None

def _encode_image(self, image_path):
    # TODO: implement using wan.modules.vae.WanVideoVAE + HuMoEmbeds
    return None

def _denoise(self, n_frames, text_embeds, image_latent, audio_embeds, seed):
    # TODO: implement using WanModel forward() + Flow Matching scheduler
    return None

def _decode_latent(self, latent):
    # TODO: implement using wan.modules.vae.WanVideoVAE.decode()
    return None
```

Each stub has exact references to the source files needed:
- `wan.modules.t5.WanT5Encoder` from `Wan-AI/Wan2.1-T2V-1.3B`
- `wan.modules.vae.WanVideoVAE` from the same repo
- `WanModel` forward signature from `bytedance-research/HuMo`
- `kijai/ComfyUI-WanVideoWrapper/nodes_sampler.py` — denoising loop reference implementation
- `kijai/ComfyUI-WanVideoWrapper/nodes.py` — HuMoEmbeds (image + audio conditioning) reference

`_encode_audio()` via Whisper is partially implemented (feature extraction works; the forward pass through `self._bundle.whisper` will work once the bundle is populated).

**Note:** `_save_mp4()` is implemented (torchvision.io.write_video + imageio fallback). The full pipeline will work end-to-end once the four inference stubs are filled in.

**What's needed to close this gap:**
1. Access to `bytedance-research/HuMo` repo (currently gated/pending release) for the exact `WanModel` forward signature and TIA conditioning hooks
2. Or: port from kijai's ComfyUI wrapper (he reverse-engineered the same interface)
3. Estimated effort once source is available: 1–2 days of adapter code

---

### Stage 4 — Assembly
**Status: Fully implemented and testable (requires ffmpeg in PATH)**

`assemble_rough_cut()` in `assembly/concatenator.py`:
- Sorts scenes by order
- Resolves single clips or merged sub-clips
- Concatenates via ffmpeg concat demuxer (no re-encode, fast)
- Muxes original song audio back onto the video
- Output: `output/rough_cut.mp4`

`export_edl()` → `output/timeline.edl` (CMX 3600, DaVinci Resolve compatible)
`export_fcpxml()` → `output/timeline.fcpxml` (FCPXML 1.10, DaVinci Resolve 18+ / Final Cut Pro)

---

## API and CLI

### REST API (`musicvision serve ./my-project`)

FastAPI server with 25+ endpoints. Full project lifecycle:

```
POST   /api/projects/create
POST   /api/projects/open
GET    /api/projects/config
PUT    /api/projects/config
PUT    /api/projects/config/style-sheet
PUT    /api/projects/config/humo
PUT    /api/projects/config/image-gen

POST   /api/upload/audio
POST   /api/upload/lyrics
POST   /api/upload/acestep-json

GET    /api/scenes
GET    /api/scenes/{id}
PATCH  /api/scenes/{id}
POST   /api/scenes/approve-all

POST   /api/pipeline/intake
POST   /api/pipeline/generate-images
POST   /api/pipeline/generate-videos
POST   /api/pipeline/assemble
```

`/docs` shows auto-generated Swagger UI. CORS is open for localhost:5173 (Vite) and localhost:3000 (CRA) — ready for a React frontend.

### CLI (`musicvision`)

```
musicvision create <dir> --name "My Video"
musicvision serve <dir> [--port 8000]
musicvision info <dir>
musicvision detect-hardware              # GPU info + recommended HuMo tier
musicvision download-weights --tier fp8_scaled
musicvision generate-video --project <dir> [--tier gguf_q4] [--block-swap 20] [--scene-ids ...]
```

---

## Data Model

Everything flows through Pydantic v2 models. No raw dict manipulation anywhere.

### Key models

**`ProjectConfig`** (saved as `project.yaml`):
```
name, created
song: SongInfo (audio_file, bpm, duration, keyscale, AceStep metadata)
style_sheet: StyleSheet (visual_style, color_palette, characters[], props[], settings[])
humo: HumoConfig (tier, resolution, scale_a, scale_t, denoising_steps, block_swap_count)
image_gen: ImageGenConfig (model, quant, steps, guidance_scale, lora_path)
vocal_separation: VocalSeparationConfig (method, demucs_model, roformer_model)
```

**`SceneList`** (saved as `scenes.json`):
```
scenes[]:
  id, order, time_start, time_end, type (vocal/instrumental)
  lyrics, audio_segment, audio_segment_vocal
  image_prompt, image_prompt_user_override, reference_image, image_status
  video_prompt, video_prompt_user_override, video_clip, video_status
  sub_clips[] (for scenes > 3.88 s)
  characters[], props[], settings[]   (references to style sheet IDs)
  notes
```

**StyleSheet** is the project's visual identity — a reusable reference that all LLM prompt calls pull from. Characters, props, and settings can have reference images and LoRA paths.

---

## Design Decisions Worth Discussing

### 1. HuMo as the video model
HuMo (ByteDance) is a Text+Image+Audio → Video model — the only openly-available model that takes a reference image AND audio as conditioning inputs simultaneously. This is essential for music videos: the video must be visually consistent with the singer and synchronized with the audio.

Alternative: Wan2.1 T2V (text-only video) or Kling/Sora (commercial APIs). HuMo was chosen specifically for TIA mode.

**Risk:** The HuMo source is partially gated. The inference stubs are the one blocking item.

### 2. FLUX vs Z-Image for reference images
FLUX.1-dev produces highest quality but is gated and VRAM-heavy. Z-Image (Tongyi-MAI 6B) is open-weight, lighter, and faster — good for rapid iteration. The factory pattern (`create_engine()`) makes them interchangeable via config.

### 3. Two-GPU split
GPU0 (RTX 5080 32 GB) handles DiT/UNet compute. GPU1 (RTX 3080 Ti 12 GB) handles T5 text encoder, VAE, Whisper encoder — all the smaller models. This is the pattern from kijai's ComfyUI wrapper.

Block swap allows the 17B HuMo DiT to fit in fewer GPU gigabytes by sequentially swapping transformer blocks between CPU and GPU during the denoising loop.

### 4. LLM backend abstraction
`llm.py` supports Anthropic (Claude) as the default cloud backend, and any OpenAI-compatible local endpoint (vLLM, Ollama, LM Studio) via env var switch. Used for scene segmentation, image prompts, and video prompts. A rule-based scene segmentation fallback requires no LLM at all.

**vLLM candidate models (for local inference on 28 GB VRAM):**
- `Qwen/Qwen2.5-32B-Instruct-AWQ` (~18 GB, best quality)
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (~12 GB 4-bit, fast)
- `Qwen/Qwen2.5-14B-Instruct` (~28 GB bf16)

### 5. Vocal separation choice
MelBandRoFormer (via `audio-separator`) is best for live/mixed recordings. Demucs htdemucs is often better for synthetically generated music (the primary use case here). Both are available; method is selectable per project.

The vocal stem is used for Whisper transcription (cleaner input = better word timestamps) and then sliced per-scene to provide a cleaner audio input for HuMo.

### 6. Sub-clip architecture
HuMo hard limit is 97 frames @ 25 fps = 3.88 seconds per clip. Longer scenes are automatically split by `generate_scene()` into sequential sub-clips with pre-sliced audio segments. The assembly stage merges them back transparently.

### 7. DaVinci Resolve integration
The pipeline outputs a rough-cut MP4 for immediate preview, plus CMX 3600 EDL and FCPXML 1.10 timelines for professional grading and finishing in DaVinci Resolve. This lets the AI output be a starting point for human editing rather than a final product.

---

## Approximate Level of Effort (Completed)

| Component | Estimated Effort |
|-----------|-----------------|
| Data models (Pydantic) | 1 day |
| Project service + paths | 0.5 day |
| Intake pipeline (BPM, Whisper, segmentation) | 2 days |
| LLM abstraction (llm.py) | 0.5 day |
| Prompt generators (image + video) | 1 day |
| FLUX engine (VRAM-tiered loading, LoRA) | 2 days |
| Z-Image engine | 0.5 day |
| HuMo infrastructure (weight registry, block swap, loaders, engine) | 3 days |
| Assembly (concat, EDL, FCPXML) | 1 day |
| FastAPI + CLI | 1 day |
| Tests | 1 day |
| **Total** | **~13–14 developer-days** |

**Remaining to full functionality:**
- HuMo inference implementation (~1–2 days, blocked on wan.modules access)
- End-to-end integration testing on GPU (~1 day)
- Optional: React frontend (~3–5 days)

---

## Environment Setup

```bash
# Core install
uv sync

# With ML deps (GPU workstation)
uv sync --extra ml
pip install "audio-separator[gpu]"   # MelBandRoFormer (conflicts with onnxruntime)

# Runtime
cp .env.example .env   # fill in ANTHROPIC_API_KEY, HUGGINGFACE_TOKEN

# First run (downloads HuMo weights ~14–34 GB depending on tier)
musicvision download-weights --tier gguf_q6

# Start pipeline
musicvision create ./my-video --name "My Video"
musicvision serve ./my-video
# → open http://localhost:8000/docs
```

**Required env vars:**
| Variable | Required For | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | LLM prompts (or use vLLM) | — |
| `HUGGINGFACE_TOKEN` | FLUX.1-dev, HuMo weights | — |
| `LLM_BACKEND` | `anthropic` or `openai` | `anthropic` |
| `OPENAI_BASE_URL` | Local vLLM server | — |
| `MUSICVISION_WEIGHTS_DIR` | Custom weight cache | `~/.cache/musicvision/weights/` |

---

## Tests

```bash
uv run pytest tests/ -v
```

| File | Coverage | GPU needed? |
|------|---------|------------|
| `tests/test_core.py` (26 tests) | Models, HumoTier, FluxConfig, ProjectService, timecode | No |
| `tests/test_intake.py` (~15 tests) | Rule-based segmentation, lyrics parsing | No |
| `tests/test_image_engine.py` (~20 tests) | Config compat, engine interface, LoRA loading | No (mocked) |
| `tests/test_video_engine.py` (~20 tests) | Constants, config, device map, block swap | No (mocked) |

All tests are GPU-free (torch lazy-imported, GPU behavior mocked).

---

## What's Not Built Yet

- **Frontend** — no React/Vue UI. Pipeline is driven entirely via API or CLI today.
- **HuMo inference implementation** — the four stubs (`_encode_text`, `_encode_image`, `_denoise`, `_decode_latent`). The entire surrounding infrastructure is complete.
- **LoRA training** — the pipeline supports LoRA paths for characters but doesn't train them.
- **Scene approval UI** — the API endpoints exist (`PATCH /api/scenes/{id}`, `POST /api/scenes/approve-all`) but there's no web UI to show images side-by-side for review.
- **Beat-locked scene boundaries** — implemented in segmentation but not yet exposed as a visual timeline in any UI.
- **Seed propagation** — `HumoInput` accepts a seed but the denoising stub doesn't use it yet.
