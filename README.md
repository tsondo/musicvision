<p align="center"><img src="MusicVision.png" alt="MusicVision — AI Music Video Generator" width="480"></p>

<h3 align="center">Open-Source AI Music Video Generator for Consumer GPUs</h3>

<p align="center">
  Turn any song into a lip-synced music video using AI — fully local, no cloud APIs required.
</p>

<p align="center">
  <a href="#how-it-works">How It Works</a> · <a href="#quick-start">Quick Start</a> · <a href="#installation">Installation</a> · <a href="#hardware-requirements">Hardware</a> · <a href="docs/PIPELINE_SPEC.md">Pipeline Spec</a> · <a href="docs/STATUS.md">Status</a>
</p>

---

## What Is MusicVision?

MusicVision is an open-source Python pipeline that generates AI music videos from a song file and a character reference image. It segments your song into scenes, generates a storyboard, renders lip-synced video clips, upscales them, and assembles a rough cut ready for editing in DaVinci Resolve.

```
song.wav + lyrics.txt + reference.png
  → scene segmentation → storyboard → lip-synced video clips → upscaled → rough cut MP4 + FCPXML
```

Everything runs locally on consumer NVIDIA GPUs. No cloud services, no per-minute billing, no data leaving your machine.

### Key Features

- **Lip-synced video generation** — characters sing along to the music with accurate mouth movements, facial expressions, and body motion driven by the audio
- **Multiple AI video engines** — HunyuanVideo-Avatar (lip sync), LTX-Video 2 (cinematic), HuMo (audio-reactive)
- **Multiple image engines** — Z-Image (ungated, fast) and FLUX (LoRA support for character consistency)
- **Three upscalers** — SeedVR2 (faces), LTX Spatial (latent-space), Real-ESRGAN (fast preview)
- **Frame-accurate audio sync** — integer frame math eliminates drift; original uncut audio in final assembly
- **Sub-clip chaining** — scenes longer than the engine's max clip length are automatically split with visual continuity (last frame → next reference image)
- **Professional export** — rough cut MP4, EDL, and FCPXML 1.10 for DaVinci Resolve
- **Iterative workflow** — review and regenerate individual scenes via React GUI or CLI; approve only what you like
- **Per-scene engine selection** — use different video engines for different scenes in the same project
- **Fully local LLM option** — vLLM with Qwen2.5-32B on a LAN server for scene segmentation and prompt generation, or use Claude API, or skip LLM entirely with auto-templates
- **Config-driven projects** — every project is a YAML + JSON directory; reproducible, version-controllable, shareable

### Current Status

All five pipeline stages are code-complete and GPU-tested. The React storyboard GUI is functional. End-to-end pipeline test passed (2026-03-01). See [STATUS.md](docs/STATUS.md) for full details.

---

## How It Works

MusicVision wraps multiple AI models into a five-stage pipeline with user review at each step:

1. **Intake & Segmentation** — Whisper transcription + LLM-assisted segmentation splits the song into 2–10 second scenes aligned to musical phrasing and section boundaries. AceStep JSON metadata (BPM, section markers) is auto-imported if available.

2. **Image Generation & Storyboard** — Z-Image or FLUX generates a reference image for each scene. Users review the storyboard grid and regenerate individual scenes until satisfied.

3. **Video Generation** — HunyuanVideo-Avatar, LTX-Video 2, or HuMo renders each scene. Long scenes are split into sub-clips with visual continuity chaining. Each engine has draft and production presets.

4. **Upscaling** — Per-engine upscaler selection: LTX Spatial for LTX-2 output (latent-space), SeedVR2 for HVA/HuMo (pixel-space face detail), Real-ESRGAN for fast preview. Configurable target resolution (720p–4K, default 1080p).

5. **Assembly & Export** — Clips concatenated with the original uncut audio, exported as MP4 + EDL + FCPXML for DaVinci Resolve. Assembly enforces a duration assertion within one frame tolerance.

### Video Engine Comparison

| Engine | Lip Sync | Audio Input | Max Clip | Resolution | Best For |
|--------|----------|-------------|----------|------------|----------|
| **HunyuanVideo-Avatar** | Native | Full mix | 5.16s (129 frames @ 25fps) | 320p–704p | Singing scenes, character performance |
| **LTX-Video 2** | Post-process | Audio+video unified | Configurable | Up to 720p | Cinematic scenes, non-vocal |
| **HuMo** | Native | Full mix (TIA mode) | 3.88s (97 frames @ 25fps) | Up to 720p | Audio-reactive motion |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/tsondo/musicvision.git
cd musicvision
bash setup_env.sh            # installs Python 3.11, PyTorch + CUDA, all deps

# 2. Configure
cp .env.example .env         # edit: set HUGGINGFACE_TOKEN (for FLUX), optionally ANTHROPIC_API_KEY

# 3. Create a project
musicvision create ./my-video --name "My Music Video"
musicvision import-audio --project ./my-video --audio song.wav --lyrics lyrics.txt

# 4. Run the pipeline
musicvision intake --project ./my-video --skip-transcription
musicvision generate-images --project ./my-video --model z-image-turbo
musicvision generate-video --project ./my-video --engine hunyuan_avatar
musicvision upscale --project ./my-video --resolution 1080p
musicvision assemble --project ./my-video
# → my-video/output/rough_cut.mp4

# Or use the GUI instead:
musicvision serve                                # API server (no project — create from frontend)
cd frontend && npm install && npm run dev        # React UI at http://localhost:5173
```

---

## Hardware Requirements

### Linux / WSL2 / Windows (CUDA)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Primary GPU (GPU0)** | 20 GB VRAM | RTX 5090 32 GB |
| **Secondary GPU (GPU1)** | 12 GB VRAM | RTX 4080 16 GB |
| **RAM** | 32 GB | 64 GB |
| **Storage** | 100 GB SSD | 500 GB NVMe |
| **CUDA** | 12.8+ | 12.8+ |
| **PyTorch** | 2.6+ | 2.10.x |

GPU0 runs DiT/UNet inference (FLUX, video engines). GPU1 handles text encoders, VAE, Whisper, and audio separator. Models are fully unloaded between stages — FLUX and video engines never run simultaneously.

A single-GPU setup works if the card has ≥32 GB VRAM. The two-GPU split is a consumer hardware optimization, not a requirement.

### Cloud (A100 / H100 / H200)

Single-GPU A100 80 GB or H100 80 GB runs the full FP16 model without splitting. No multi-GPU complexity needed. Minor gaps remain in tier auto-selection for high-VRAM single GPUs — see [future_plans.md](docs/future_plans.md).

### Apple Silicon (MPS) — Planned

M-series Mac support is planned but not yet implemented. Blocking issues include RoPE float64/complex128 ops and FP8 unavailability on MPS. See [future_plans.md](docs/future_plans.md) for the roadmap.

### Optional: Local LLM Server

A separate GPU (e.g. RTX 3090 Ti 24 GB) can run [vLLM](https://github.com/vllm-project/vllm) with Qwen2.5-32B-AWQ for scene segmentation and prompt generation, eliminating the Claude API dependency entirely. This is optional — the pipeline works without it using auto-templates or the Claude API.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

### Automated Setup

```bash
git clone https://github.com/tsondo/musicvision.git
cd musicvision
bash setup_env.sh
```

The script installs `uv` if missing, creates a Python 3.11 venv, installs PyTorch with CUDA, installs MusicVision in editable mode, and runs the test suite.

### Manual Setup

```bash
git clone https://github.com/tsondo/musicvision.git
cd musicvision

# Create venv and install base deps
uv sync --extra dev

# Install PyTorch with CUDA 12.8
uv run pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Audio separator (separate install due to onnxruntime conflict)
pip install "audio-separator[gpu]"
```

`flash_attn` is **not required**. PyTorch's native SDPA provides equivalent performance on modern GPUs. Install it separately only if you want it:

```bash
pip install flash-attn --no-build-isolation
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Required | Notes |
|----------|----------|-------|
| `HUGGINGFACE_TOKEN` | For FLUX.1-dev (gated) | Not needed for Z-Image (ungated) |
| `ANTHROPIC_API_KEY` | For Claude LLM prompts | Not needed if using vLLM or auto-templates |
| `LLM_BACKEND` | No (default: `anthropic`) | Set to `openai` for vLLM |
| `OPENAI_BASE_URL` | If using vLLM | e.g. `http://192.168.1.100:8000/v1` |
| `OPENAI_MODEL` | If using vLLM | e.g. `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| `MUSICVISION_WEIGHTS_DIR` | No | Override model cache dir (default: `~/.cache/musicvision/weights`) |
| `HVA_REPO_DIR` | For HunyuanVideo-Avatar | Path to cloned HVA repo |
| `SEEDVR2_REPO_DIR` | For SeedVR2 upscaler | Path to cloned SeedVR repo |

---

## GUI

The React frontend provides a storyboard-based workflow:

- **Scene grid** — lyrics, reference images, video clips, prompts, and approval status per scene
- **Preview panel** — full-size image and video playback with regeneration controls
- **Per-scene engine selection** — choose a different video engine for each scene
- **Per-scene regeneration** — re-render individual scenes without restarting the whole pipeline
- **Model management** — select and switch between image/video models
- **Dark theme**

```bash
musicvision serve ./my-video          # start API server
cd frontend && npm install && npm run dev   # React dev server at http://localhost:5173
```

The API server also provides Swagger UI at `http://localhost:8000/docs` for direct API access.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `musicvision create <dir> --name "…"` | Create a new project |
| `musicvision import-audio --project DIR --audio song.wav [--lyrics lyrics.txt]` | Import audio and lyrics |
| `musicvision intake --project DIR [--llm] [--skip-transcription] [--vocal-separation]` | Stage 1: audio analysis + segmentation |
| `musicvision generate-images --project DIR [--model MODEL] [--scene-ids ID…]` | Stage 2: generate reference images |
| `musicvision generate-video --project DIR [--engine ENGINE] [--tier TIER] [--scene-ids ID…]` | Stage 3: generate video clips |
| `musicvision upscale --project DIR [--resolution RES] [--upscaler TYPE] [--scene-ids ID…]` | Stage 4: upscale clips |
| `musicvision assemble --project DIR [--approved-only] [--no-edl] [--no-fcpxml]` | Stage 5: assemble rough cut + export |
| `musicvision info <dir>` | Show project status |
| `musicvision serve [dir] [--port 8000]` | Start API + GUI server |
| `musicvision detect-hardware` | Print GPU info and recommended tier |
| `musicvision download-weights --tier TIER [--token TOKEN]` | Download model weights |

**Image models:** `flux-dev`, `flux-schnell`, `z-image`, `z-image-turbo`
**Video engines:** `hunyuan_avatar`, `ltx_video`, `humo`
**Upscalers:** `ltx_spatial`, `seedvr2`, `real_esrgan`
**Resolutions:** `720p`, `1080p`, `1440p`, `4k`

---

## Project Structure

Each MusicVision project is a self-contained directory:

```
my-video/
├── project.yaml          # Config: engines, style sheet, generation params
├── scenes.json           # Scene list with timestamps, prompts, approval status
├── input/                # Source audio + lyrics
├── assets/               # Characters, props, settings, LoRAs
├── segments/             # Per-scene audio (full mix → video engines)
├── segments_vocal/       # Vocal stems (→ Whisper transcription)
├── images/               # Reference images per scene
├── clips/                # Generated video clips + sub-clips
└── output/               # rough_cut.mp4, timeline.edl, timeline.fcpxml
```

All intermediate artifacts are saved. You can re-enter the pipeline at any stage, regenerate individual scenes, and the final assembly always uses the original uncut audio.

---

## Testing

```bash
# Unit tests — ~257 tests, no GPU, < 10 seconds
uv run pytest tests/ -v

# LLM prompt tests (requires vLLM server)
python scripts/test_vllm_prompts.py

# GPU image generation (Z-Image + FLUX)
python scripts/test_image_gen.py

# GPU video generation (HuMo)
python scripts/test_gpu_pipeline.py --tier fp8_scaled --steps 6

# HunyuanVideo-Avatar standalone
python scripts/test_hva_standalone.py
```

See [TESTING.md](docs/TESTING.md) for the full test strategy.

---

## Documentation

| Document | Description |
|----------|-------------|
| [PIPELINE_SPEC.md](docs/PIPELINE_SPEC.md) | Full pipeline specification with API endpoints and frame math |
| [STATUS.md](docs/STATUS.md) | Current implementation status and what's built |
| [HUMO_REFERENCE.md](docs/HUMO_REFERENCE.md) | HuMo model internals, TIA mode, prompt guidelines |
| [TESTING.md](docs/TESTING.md) | Two-layer test strategy (unit + integration) |
| [MUSICVISION_GPU_TEST.md](docs/MUSICVISION_GPU_TEST.md) | GPU integration test setup guide |
| [OOM_RESILIENCE_PLAN.md](docs/OOM_RESILIENCE_PLAN.md) | OOM resilience strategy and implementation status |
| [LIP_SYNC_POST.md](docs/LIP_SYNC_POST.md) | LatentSync lip sync post-processing spec |
| [future_plans.md](docs/future_plans.md) | Long-term vision: story bible → manga → animation |
| [FIXLOG.md](docs/FIXLOG.md) | Checkpoint loading fix history |

---

## How MusicVision Compares

MusicVision occupies a unique niche: end-to-end music video generation running fully local on consumer GPUs.

| Capability | MusicVision | ViMax | LTX-2 (model) | Music2Video |
|---|---|---|---|---|
| Lip-synced video from audio | Yes (HVA) | No | Yes (built-in) | No (VQGAN) |
| Scene segmentation from lyrics | Yes | Yes (from scripts) | No | Partial |
| Runs fully local | Yes | No (cloud APIs) | Partial (28 GB+) | Yes |
| Consumer dual-GPU support | Yes | N/A | No | No |
| Sub-clip chaining for continuity | Yes | No | No | No |
| DaVinci Resolve export (FCPXML) | Yes | No | No | No |
| Multiple engine backends | Yes (3 engines) | Yes (cloud) | Single model | Single model |
| Storyboard GUI with per-scene control | Yes (React) | Yes | No | No |

---

## Built With

- [HunyuanVideo-Avatar](https://github.com/tencent/HunyuanVideo) (Tencent) — audio-driven video generation with lip sync
- [LTX-Video 2](https://github.com/Lightricks/LTX-Video) (Lightricks) — unified audio+video generation
- [HuMo](https://github.com/Phantom-video/HuMo) (ByteDance) — audio-conditioned video with TIA mode
- [FLUX](https://github.com/black-forest-labs/flux) (Black Forest Labs) — text-to-image with LoRA support
- [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (Tongyi) — fast ungated image generation
- [SeedVR2](https://github.com/ByteDance/SeedVR2) (ByteDance) — face-aware video upscaling
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) — fast general-purpose upscaling
- [Whisper](https://github.com/openai/whisper) (OpenAI) — speech transcription and alignment
- [Kim_Vocal_2 / Demucs](https://github.com/facebookresearch/demucs) — vocal separation
- [FastAPI](https://fastapi.tiangolo.com/) + [React](https://react.dev/) — API server and storyboard GUI
- [ffmpeg](https://ffmpeg.org/) — audio slicing, video concatenation, muxing
- [vLLM](https://github.com/vllm-project/vllm) — local LLM serving (optional)

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for personal, academic, and research use. Commercial licenses available; see [LICENSE-COMMERCIAL](LICENSE-COMMERCIAL) for details.
