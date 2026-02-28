# MusicVision — Project Status

**Last updated:** 2026-02-27
**Branch:** `main`

---

## What This Project Is

MusicVision is a Python pipeline that takes an AI-generated (or real) song and produces a music video:

```
Song (WAV) + Lyrics → Scenes → Reference Images → Video Clips → Assembled Video
```

The pipeline is designed around AI-generated music from **AceStep** (a text-conditioned music generation model), which produces songs alongside structured metadata (BPM, key, lyrics with section markers). The pipeline ingests that metadata natively.

**Primary output:** A rough-cut MP4 + an EDL/FCPXML timeline for DaVinci Resolve finishing.

**Target hardware:** Dual-GPU inference workstation (RTX 5090 32 GB + RTX 4080 16 GB) for image/video generation, with VRAM-tiered fallbacks for lighter configs. Separate RTX 3090 Ti (24 GB) available for local vLLM inference.

---

## Current Status Summary

| Pipeline Stage | Status |
|----------------|--------|
| Intake (audio, BPM, Whisper, segmentation) | ✅ Complete |
| Reference image generation (FLUX) | ✅ Complete |
| Video generation (HuMo TIA) | ✅ Complete — **GPU tested, producing video** |
| Assembly (rough cut, EDL, FCPXML) | ✅ Complete |

| Infrastructure | Status |
|----------------|--------|
| API + CLI | ✅ Complete |
| Frontend (React or Gradio) | ❌ Not started |
| Progress/status feedback (SSE/WebSocket) | ❌ Not started |

**All four pipeline stages are code-complete and GPU-tested.** The pipeline produces recognizable video with audio sync. Quality tuning is ongoing (color artifacts, higher step counts).

---

## Critical Design Decisions

> **HuMo receives the full audio mix, not isolated vocals.** The vocal stem from Kim_Vocal_2 / Demucs is used *only* to improve Whisper transcription accuracy. HuMo was trained on mixed audio — feeding it isolated vocals degrades A/V sync. Audio segments in `segments/` are always full-mix; `segments_vocal/` is consumed only by the transcription step.

> **Sub-clip continuity uses last-frame chaining.** For scenes longer than 3.88s, after generating sub-clip N, ffmpeg extracts the last frame and uses it as the reference image for sub-clip N+1. This prevents visual discontinuity within a scene but means sub-clips must generate sequentially.

> **FLUX and HuMo never run simultaneously.** They occupy the same VRAM (DiT on GPU0). The pipeline enforces sequential stage execution. Model weights are fully unloaded between stages.

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
- FP8 quantization on Ada/Hopper/Blackwell (compute cap ≥ 8.9); auto-falls back to INT8
- Optional project-level and per-scene LoRA
- FLUX.1-dev is gated (requires `HUGGINGFACE_TOKEN`); FLUX.1-schnell is open

---

### Stage 3 — Video Generation
**Status: Complete — GPU tested**

**All inference stubs have been replaced.** The pipeline is no longer blocked on `wan.modules` from bytedance-research/HuMo. A self-contained PyTorch implementation was written without that dependency.

**GPU test results (2026-02-27):** FAST preset (LoRA + 384p + 6 steps + UniPC) generates a 3.72s clip in ~52s on RTX 5090. Output shows recognizable content with spatial/temporal coherence. 10 inference bugs were found and fixed during bring-up (see `humo_debugging.md` in project memory).

#### New files (added Feb 2026)

| File | Purpose |
|------|---------|
| `video/wan_model.py` | Self-contained WanModel DiT — WanRoPE 3D, AdaLN blocks, AudioProjModel, AudioCrossAttentionWrapper |
| `video/scheduler.py` | FlowMatchScheduler (Euler) + FlowMatchUniPCScheduler (default, higher-order) |
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

#### GPU test history

All four risk areas from initial bring-up have been resolved. 10 bugs were found and fixed:
- Bugs 1-3: Reference frame positioning (last temporal position, not first)
- Bug 4-5: CFG formula corrections (negative conditioning, time-adaptive switching)
- Bug 6: Sigma shift 5.0 → 8.0 to match ComfyUI workflow
- Bug 7: FP8 input scaling (dynamic per-tensor, not hardcoded 1.0)
- Bug 8-9: Diagnostic logging for missing keys and denoising health
- Bug 10 (root cause of noise): Timestep not scaled ×1000
- Sampler switch: Euler → UniPC (matches ComfyUI `uni_pc` sampler)

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

**vLLM candidates for local inference (RTX 3090 Ti, 24 GB VRAM):**
- `Qwen/Qwen2.5-32B-Instruct-AWQ` (~18 GB, best quality)
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (~12 GB 4-bit, fast)

---

## API and CLI

### REST API (`musicvision serve ./my-project`)

FastAPI, 25+ endpoints. Auto-generated Swagger UI at `/docs`. CORS open for local development (Vite and CRA default ports).

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
| `tests/test_intake.py` | Rule-based segmentation, lyrics parsing | No |
| `tests/test_image_engine.py` | Config compat, engine interface, LoRA loading | No (mocked) |
| `tests/test_video_engine.py` | Constants, config, device map, block swap | No (mocked) |
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

**Pinned dependency constraints** (from HuMo requirements):
- `torch==2.10.0` with CUDA 12.8 (upgraded for RTX 5090 sm_120 support)
- `flash_attn==2.6.3`
- Python 3.11+
- See `pyproject.toml` for the full lockfile.

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

### Frontend & UI
- **React frontend** — pipeline driven entirely via REST API and CLI today; Swagger UI at `/docs` for manual testing
- **Gradio UI** — originally planned as the storyboard interface; not started. Core pipeline modules are UI-agnostic by design.
- **Scene approval UI** — API endpoints for scene CRUD and approval exist, but no visual side-by-side review interface

### Pipeline Features
- **Progress/status tracking** — no WebSocket or SSE for long-running generation jobs; API endpoints are synchronous. A 50-scene video gen blocks for hours with no progress feedback.
- **Partial failure recovery** — if video generation fails at scene 22 of 50, there's no automatic resume. The exception propagates to the API caller or CLI, and scenes 23–50 are never attempted. Already-generated clips (scenes 1–21) are saved to disk and remain valid. Workaround: use `--scene-ids` CLI flag to regenerate specific scenes. Needs a proper job/resume model with per-scene status tracking.
- **Scene reordering** — scenes are ordered by `order` field in scenes.json, but no API endpoint or UI to drag-reorder and renumber.
- **Transitions** — hard cuts only between scenes. No crossfades, AI-generated transitions, or blending.
- **Batch parallelism** — scenes generate sequentially. No concurrent generation across multiple GPUs or cloud workers.

### Model & Generation
- **LoRA training** — pipeline accepts LoRA paths for character consistency but doesn't include training workflows
- **Seed reproducibility** — `HumoInput.seed` is wired: random seed generated when None, both `torch.manual_seed` and `torch.cuda.manual_seed` called, seed recorded in `HumoOutput.seed_used`. Remaining gap: GPU test needed to confirm fully deterministic output (CUDNN non-determinism).
- **Render time estimation** — no estimated time display for users. 3× DiT passes per step × 50 steps × N scenes can take many hours; users need to know upfront.

### Export
- **Audio-only export** — no way to export just the segmented audio clips without running image/video gen
- **Rough cut preview without full render** — storyboard slideshow (images + audio, no video) would let users validate before committing to HuMo render time

---

## Design Notes

See also **Critical Design Decisions** above for the most important architectural constraints.

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
