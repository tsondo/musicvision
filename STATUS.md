# MusicVision — Project Status

**Last updated:** 2026-02-25
**Branch:** `main` — commit `bdd8553`
**Codebase:** ~9,200 lines across 41 Python modules + scripts

---

## What This Project Is

MusicVision is a Python pipeline that takes an AI-generated (or real) song and produces a music video:

```
Song (WAV) + Lyrics → Scenes → Reference Images → Video Clips → Assembled Video
```

The pipeline is designed around AI-generated music from **AceStep** (a text-conditioned music generation model), which produces songs alongside structured metadata (BPM, key, lyrics with section markers). The pipeline ingests that metadata natively.

**Primary output:** A rough-cut MP4 + an EDL/FCPXML timeline for DaVinci Resolve finishing.

**Target hardware:** Dual-GPU workstation (RTX 5090 32 GB + RTX 4080 16 GB), with VRAM-tiered fallbacks for single-GPU and lighter configs.

---

## Current Status Summary

| Stage | Status |
|-------|--------|
| Intake (audio, BPM, Whisper, segmentation) | ✅ Complete |
| Reference image generation (FLUX) | ✅ Complete |
| Video generation (HuMo TIA) | ✅ Complete — **pending first GPU test** |
| Assembly (rough cut, EDL, FCPXML) | ✅ Complete |
| API + CLI | ✅ Complete |
| React frontend | ❌ Not built |

**All four pipeline stages are code-complete.** The next milestone is a first GPU integration test on the workstation with a real song.

---

## Pipeline Stages

### Stage 1 — Intake
**Status: Complete**

Orchestrated by `run_intake()` in `intake/pipeline.py`.

1. Load audio, resolve duration (from AceStep metadata or ffprobe)
2. BPM detection via librosa beat tracker
3. Optional vocal separation (MelBandRoFormer or Demucs) → cleaner Whisper input
4. Transcription via Whisper-large-v3 (word-level timestamps)
5. LLM scene segmentation — analyzes lyrics + timestamps → JSON scene list, boundaries snapped to beats
6. Rule-based fallback if no LLM available (`segment_scenes_simple()`)
7. Slice audio into per-scene WAV segments
8. Save `scenes.json`

**No GPU needed for:** BPM detection, rule-based segmentation, AceStep JSON import, all persistence.
**GPU needed for:** Whisper transcription (skippable with pre-provided lyrics), vocal separation.
**LLM needed for:** LLM-assisted segmentation (rule-based fallback available without it).

---

### Stage 2 — Reference Image Generation
**Status: Complete**

Each scene gets a FLUX reference still image used as the HuMo visual anchor.

**Prompt generation (`imaging/prompt_generator.py`):**
- LLM (Claude or vLLM) writes a 2–4 sentence cinematic description per scene
- Injected with style sheet: character descriptions, visual style, color palette, settings
- **If no LLM available:** interactive terminal prompt (shows scene context + guidelines)
- **If non-interactive (piped):** auto-template built from style sheet fields
- User can override any prompt via `image_prompt_user_override` in `scenes.json`

**Engine (`FluxEngine`):**
- VRAM-tiered: 4 tiers from bf16 (≥28 GB free) down to quantized sequential offload (<8 GB)
- FP8 quantization on Ada/Hopper (compute cap ≥ 8.9); auto-falls back to INT8
- Optional project-level and per-scene LoRA
- FLUX.1-dev is gated (requires `HUGGINGFACE_TOKEN`); FLUX.1-schnell is open

---

### Stage 3 — Video Generation
**Status: Complete — pending first GPU test**

**All inference stubs have been replaced.** The pipeline is no longer blocked on `wan.modules` from bytedance-research/HuMo. A self-contained PyTorch implementation was written without that dependency.

#### New files (added Feb 2026)

| File | Purpose |
|------|---------|
| `video/wan_model.py` | Self-contained WanModel DiT — WanRoPE 3D, AdaLN blocks, AudioProjModel, AudioCrossAttentionWrapper |
| `video/scheduler.py` | FlowMatchScheduler — Euler steps, shift=5.0 sigma warping |
| `video/wan_t5.py` | WanT5Encoder — wraps HuggingFace T5EncoderModel with Wan-AI .pth key remapping |
| `video/wan_vae.py` | WanVideoVAE — CausalConv3d encoder/decoder, temporal stride 4, spatial stride 8 |
| `video/audio_encoder.py` | HumoAudioEncoder — Whisper → 5-band features → sliding windows → [1,F,8,5,1280] |

#### TIA denoising loop

Dual CFG with three DiT passes per step:

```
v_cond      = DiT(z + img_pos,  t,  pos_text,  audio)
v_audio_neg = DiT(z + img_pos,  t,  pos_text,  zeros)
v_text_neg  = DiT(z + img_neg,  t,  neg_text,  audio)

v_pred = v_text_neg
       + scale_a  × (v_cond - v_audio_neg)
       + (scale_t - 2.0) × (v_audio_neg - v_text_neg)
```

Block swap supported: `pre_blocks()` → `BlockSwapManager.execute_block()` × N → `post_blocks()`.

#### Tiered weight system

| Tier | Format | DiT VRAM | Auto-downloads? |
|------|--------|----------|----------------|
| `fp16` | FP16 safetensors | ~34 GB | Yes (token required) |
| `fp8_scaled` | FP8 e4m3fn scaled | ~18 GB | Yes (token required) |
| `gguf_q8/q6/q4` | GGUF | 11–18.5 GB | Yes (token required) |
| `preview` | FP16 safetensors (1.7B) | ~3.4 GB | Yes (token required) |
| T5 encoder | .pth (Wan-AI) | ~10 GB on GPU1 | **Yes — no token needed** |
| VAE | .pth (Wan-AI) | ~0.4 GB on GPU1 | **Yes — no token needed** |
| Whisper large-v3 | safetensors | ~1.5 GB on GPU1 | **Yes — no token needed** (HF transformers cache) |

DiT weights require `HUGGINGFACE_TOKEN`. Shared weights (T5, VAE, Whisper) are open models and auto-download silently on first use.

#### Video prompt generation (`video/prompt_generator.py`)

- LLM writes dense Qwen2.5-VL-style captions (HuMo's training style)
- Same LLM-unavailable fallback as image prompts: interactive input → auto-template
- Auto-template anchors on image prompt, style sheet, and scene type

#### Scene splitting

HuMo hard limit: 97 frames @ 25 fps = 3.88 s. `generate_scene()` automatically splits longer scenes into sub-clips with pre-sliced audio. Sub-clip continuity (`HumoConfig.sub_clip_continuity=True`): last frame of sub-clip N becomes reference image for sub-clip N+1.

#### Known risk areas for first GPU test

1. **T5 key names** — Wan-AI `.pth` may use different key names than the remap table in `wan_t5._remap_wan_t5_keys()`. Fix: print first 20 checkpoint keys, patch the table. `strict=False` prevents hard failure.
2. **VAE architecture** — inferred from Wan2.1 conventions; channel counts or layer names may differ. Fix: log missing keys from `load_state_dict(strict=False)`, adjust `_build()`.
3. **FP8 scale key suffix** — three fallback patterns exist in `_patch_fp8_linears`; a fourth may be needed.
4. **GGUF key mapping** — `_gguf_name_to_pt_key()` covers common llama.cpp conventions; exporter-specific names may need additional entries.

---

### Stage 4 — Assembly
**Status: Complete**

`assemble_rough_cut()` → `output/rough_cut.mp4`
- Sorts scenes, merges sub-clips, concatenates via ffmpeg concat demuxer (no re-encode)
- Muxes original song audio back

`export_edl()` → `output/timeline.edl` (CMX 3600, DaVinci Resolve)
`export_fcpxml()` → `output/timeline.fcpxml` (FCPXML 1.10, DaVinci Resolve 18+ / FCP)

---

## LLM Availability

Both prompt generators gracefully degrade when no LLM is configured:

| Situation | Behaviour |
|-----------|-----------|
| LLM configured and reachable | Normal LLM call |
| No key / unreachable | Interactive terminal prompt (shows scene context + guidelines) |
| Non-interactive (piped) | Auto-template from style sheet fields |
| Manual override in scenes.json | Override always wins, LLM never called |

`llm_available()` in `llm.py` checks credentials without a network call.

**Backends:**
- `LLM_BACKEND=anthropic` (default) — `ANTHROPIC_API_KEY`
- `LLM_BACKEND=openai` — vLLM: `OPENAI_BASE_URL` + `OPENAI_MODEL`

**vLLM candidates for local inference (28 GB VRAM: RTX 4080 16 GB + RTX 5070 12 GB):**
- `Qwen/Qwen2.5-32B-Instruct-AWQ` (~18 GB, best quality)
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (~12 GB 4-bit, fast)
- `Qwen/Qwen2.5-14B-Instruct` (~28 GB bf16)

---

## API and CLI

### REST API (`musicvision serve ./my-project`)

FastAPI, 25+ endpoints. Auto-generated Swagger UI at `/docs`. CORS open for localhost:5173 / localhost:3000.

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

### CLI (`musicvision`)

```bash
musicvision create <dir> --name "My Video"
musicvision serve <dir> [--port 8000]
musicvision info <dir>
musicvision detect-hardware              # GPU info + recommended HuMo tier
musicvision download-weights --tier fp8_scaled
musicvision generate-video --project <dir> [--tier gguf_q4] [--block-swap 20] [--scene-ids ...]
```

---

## Data Model

Everything flows through Pydantic v2 models. No raw dict manipulation.

**`ProjectConfig`** (`project.yaml`):
```
name, created
song: SongInfo (audio_file, bpm, duration, keyscale, AceStep metadata)
style_sheet: StyleSheet (visual_style, color_palette, characters[], props[], settings[])
humo: HumoConfig (tier, resolution, scale_a, scale_t, denoising_steps, block_swap_count, sub_clip_continuity)
image_gen: ImageGenConfig (model, quant, steps, guidance_scale, lora_path)
vocal_separation: VocalSeparationConfig (method, demucs_model, roformer_model)
```

**`SceneList`** (`scenes.json`):
```
scenes[]:
  id, order, time_start, time_end, type (vocal/instrumental)
  lyrics, audio_segment, audio_segment_vocal
  image_prompt, image_prompt_user_override, reference_image, image_status
  video_prompt, video_prompt_user_override, video_clip, video_status
  sub_clips[] (for scenes > 3.88 s)
  characters[], props[], settings[]
  notes
```

---

## Tests

```bash
# CPU unit tests (no GPU needed):
uv run pytest tests/ -v
python scripts/test_humo_inference.py   # 11/11 passing

# GPU integration test (run on workstation):
python scripts/test_gpu_pipeline.py --audio song.wav --image ref.png --tier fp8_scaled --steps 30
```

| File | What it covers | GPU? |
|------|---------------|------|
| `tests/test_core.py` (26 tests) | Models, HumoTier, FluxConfig, ProjectService, timecode | No |
| `tests/test_intake.py` (~15 tests) | Rule-based segmentation, lyrics parsing | No |
| `tests/test_image_engine.py` (~20 tests) | Config compat, engine interface, LoRA loading | No (mocked) |
| `tests/test_video_engine.py` (~20 tests) | Constants, config, device map, block swap | No (mocked) |
| `scripts/test_humo_inference.py` (11 tests) | WanModel forward, RoPE, AudioProjModel, FlowMatchScheduler, pre/post_blocks identity | No (CPU) |
| `scripts/test_gpu_pipeline.py` | Single clip, generate_scene() sub-splits, assemble_rough_cut() | **Yes** |

---

## Environment Setup

```bash
# Core install
uv sync

# With ML deps (GPU workstation)
uv sync --extra ml
pip install "audio-separator[gpu]"   # MelBandRoFormer (separate due to onnxruntime conflict)

# Runtime config
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY (or configure vLLM), HUGGINGFACE_TOKEN (for DiT weights)

# Shared weights (T5, VAE, Whisper) auto-download on first engine.load()
# DiT weights require HUGGINGFACE_TOKEN — download explicitly or let engine.load() handle it:
musicvision download-weights --tier fp8_scaled

# Start pipeline
musicvision create ./my-video --name "My Video"
musicvision serve ./my-video
# → http://localhost:8000/docs
```

**Required env vars:**

| Variable | Required For | Notes |
|----------|-------------|-------|
| `ANTHROPIC_API_KEY` | LLM prompts | Not required — interactive/auto-template fallback available |
| `HUGGINGFACE_TOKEN` | HuMo DiT weights, FLUX.1-dev | Not required for shared weights (T5/VAE/Whisper auto-download open) |
| `LLM_BACKEND` | Backend selection | Default: `anthropic` |
| `OPENAI_BASE_URL` | Local vLLM | Required if `LLM_BACKEND=openai` |
| `OPENAI_MODEL` | Local vLLM | Required if `LLM_BACKEND=openai` |
| `MUSICVISION_WEIGHTS_DIR` | Custom weight cache | Default: `~/.cache/musicvision/weights/` |

---

## What's Not Built Yet

- **React frontend** — pipeline driven via API/CLI today (Swagger UI at `/docs`)
- **LoRA training** — pipeline supports LoRA paths for characters but doesn't train them
- **Scene approval UI** — API endpoints exist but no web UI for side-by-side image review
- **Seed propagation** — `HumoInput` accepts a seed; wired through to the denoising loop

---

## Design Notes

### HuMo as the video model
HuMo (ByteDance, Apache 2.0) is the only openly-available model that takes reference image + audio as simultaneous conditioning inputs — essential for music videos (visual consistency + A/V sync). TIA mode: Text + Image + Audio → Video.

### Two-GPU split
GPU0 (RTX 5090 32 GB) handles the DiT compute. GPU1 (RTX 4080 16 GB) handles T5, VAE, Whisper — all fit simultaneously in 16 GB. Block swap allows the 17B DiT to run in less VRAM by sequentially swapping transformer blocks between CPU and GPU.

### Vocal separation
The vocal stem is used **only for Whisper transcription** (cleaner input = better timestamps). HuMo receives the **full mix** from `segments/` — not the isolated stem. HuMo was trained on mixed audio; isolated vocals degrade A/V sync.

### Sub-clip continuity
After generating sub-clip N, the last frame is extracted via ffmpeg and used as the reference image for sub-clip N+1. Prevents visual discontinuity across a long scene's sub-clips.

### DaVinci Resolve integration
Rough-cut MP4 for immediate preview + FCPXML 1.10 / CMX 3600 EDL for professional finishing. AI output is a starting point for human editing, not a final product.
