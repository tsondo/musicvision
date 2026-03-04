<p align="center"><img src="MusicVision.png" alt="MusicVision" width="480"></p>

AI-powered music video production pipeline with **lip-synced video generation**. Feed it a song and a character reference image — get back video clips where the character sings along to the music, fully timed and ready for editing in DaVinci Resolve.

```
audio + lyrics + reference image  →  scene segmentation  →  storyboard  →  lip-synced video clips  →  rough cut MP4 + EDL/FCPXML
```

**Status:** All five pipeline stages are code-complete and GPU-tested. Full CLI pipeline + React scene review GUI. Three video engines: [HunyuanVideo-Avatar](https://github.com/tencent/HunyuanVideo) (primary, lip sync), [LTX-Video 2](https://github.com/Lightricks/LTX-Video) (cinematic), and [HuMo](https://github.com/Phantom-video/HuMo) (experimental). Three upscalers: SeedVR2 (faces), LTX Spatial (latent), Real-ESRGAN (fast). Two image engines: [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (ungated) and [FLUX](https://github.com/black-forest-labs/flux).

---

## How It Works

MusicVision wraps multiple AI video and image generation models into an iterative, user-controlled workflow.

The primary video engine is [HunyuanVideo-Avatar](https://github.com/tencent/HunyuanVideo) (Tencent), which generates audio-driven video with lip sync, facial expressions, and body motion from a single reference image. [HuMo](https://github.com/Phantom-video/HuMo) (ByteDance) is available as an experimental alternative. For reference images, [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (ungated, fast) and [FLUX](https://github.com/black-forest-labs/flux) are both supported.

### Pipeline Stages

1. **Intake & Segmentation** — Song audio + lyrics are split into 2–10 second scenes using Whisper transcription and LLM-assisted segmentation that respects musical phrasing and section boundaries.
2. **Image Generation & Storyboard** — Z-Image or FLUX generates reference images for each scene. Users review and iterate on individual scenes until satisfied.
3. **Video Generation** — HunyuanVideo-Avatar, LTX-Video 2, or HuMo renders each scene as a video clip. Long scenes are automatically split into sub-clips with visual continuity (last frame of clip N becomes the reference image for clip N+1).
4. **Upscaling** — Per-engine upscaler selection: LTX Spatial for LTX-2 output (latent-space), SeedVR2 for HVA/HuMo (pixel-space faces), Real-ESRGAN for fast preview. Target resolution configurable (720p–4K, default 1080p).
5. **Assembly & Export** — Clips are concatenated (preferring upscaled versions) with the original audio, exported as a rough cut MP4 plus EDL and FCPXML project files for DaVinci Resolve.

---

## Video Engine Presets

### HunyuanVideo-Avatar (primary)

| Setting | Resolution | Steps | ~Time/Clip (5.16s) | Notes |
|---------|-----------|-------|---------------------|-------|
| **Draft** | 320p | 10 | ~5 min | Fast iteration, good quality |
| **Production** | 704p | 30 | ~2 hours | Full quality, lip sync |

Audio-driven generation with lip sync from a single reference image. Fixed clip length of 5.16s (129 frames @ 25fps). Runs in a separate subprocess/venv. BF16 with block-level CPU offloading on ≤32GB VRAM.

### HuMo (experimental)

| Preset | Resolution | Steps | CFG | LoRA | ~Time/Clip |
|--------|-----------|-------|-----|------|------------|
| **FAST** | 688×384 | 6 | 1.0 | Lightx2V distill | ~1 min |
| **FULL** | 1280×720 | 30 | dual | None | ~45 min |

Currently produces noisy output — deprioritized in favor of HVA.

---

## Hardware Requirements

### Linux / Windows (CUDA)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Primary GPU (GPU0) | 20 GB VRAM | RTX 5090 32 GB |
| Secondary GPU (GPU1) | 12 GB VRAM | RTX 4080 16 GB |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| CUDA | 12.8+ | 12.8+ |

**GPU allocation:**
- **GPU0** (primary): DiT compute for FLUX image generation and HuMo video generation
- **GPU1** (secondary): T5 text encoder (~10 GB), VAE (~0.4 GB), Whisper (~1.5 GB), audio separator (~0.5 GB)
- FLUX and HuMo run sequentially (different stages), never simultaneously. Weights fully unloaded between stages.

**Optional LLM server:** A separate GPU (e.g. RTX 3090 Ti 24 GB) can run [vLLM](https://github.com/vllm-project/vllm) for local prompt generation, eliminating the Claude API dependency.

### Apple Silicon (MPS)

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Chip | M1 | M3 Max or M4 Max |
| Unified RAM | 24 GB | 36 GB+ |
| Storage | 100 GB SSD | 500 GB NVMe |

HuMo is limited to the `preview` (1.7B) tier on MPS. See [PLATFORM_SUPPORT_PLAN.md](docs/PLATFORM_SUPPORT_PLAN.md) for details.

### Cloud (A100 / H100 / H200)

Single-GPU A100 80 GB or H100 80 GB can run the full FP16 model without splitting. See [PLATFORM_SUPPORT_PLAN.md](docs/PLATFORM_SUPPORT_PLAN.md) for cloud-specific configuration.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone git@github.com:tsondo/musicvision.git
cd musicvision

# Install (CPU-only deps + dev tools)
uv sync --extra dev

# Install ML deps with CUDA 12.8
uv run pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Audio separator (separate due to onnxruntime conflict)
pip install "audio-separator[gpu]"

# Or use the full setup script
bash setup_env.sh
```

`flash_attn` is **optional**. The vendored DiT uses PyTorch's native SDPA as a fallback, which provides equivalent performance on modern GPUs. Install separately if desired:

```bash
pip install flash-attn --no-build-isolation
```

---

## Quick Start

```bash
# Configure environment
cp .env.example .env
# Set HUGGINGFACE_TOKEN (for FLUX weights) and optionally ANTHROPIC_API_KEY (for LLM prompts)

# Create project and import audio
musicvision create ./my-video --name "My Music Video"
musicvision import-audio --project ./my-video --audio song.wav --lyrics lyrics.txt

# Run the pipeline
musicvision intake --project ./my-video --skip-transcription    # segmentation (use --llm for LLM-assisted)
musicvision generate-images --project ./my-video --model z-image-turbo  # reference images
musicvision generate-video --project ./my-video --engine hunyuan_avatar # lip-synced video clips
musicvision upscale --project ./my-video --resolution 1080p     # upscale to 1080p
musicvision assemble --project ./my-video                       # rough cut + EDL/FCPXML
# → output/rough_cut.mp4

# Or use the GUI:
musicvision serve                     # start with no project — create/open from the frontend
cd frontend && npm install && npm run dev  # → http://localhost:5173

# Or start the API server with an existing project:
musicvision serve ./my-video          # → Swagger UI at http://localhost:8000/docs
```

### Required Environment Variables

| Variable | Required For | Notes |
|----------|-------------|-------|
| `HUGGINGFACE_TOKEN` | HuMo DiT weights, FLUX.1-dev | Not required for Z-Image (ungated) or shared weights (T5/VAE/Whisper) |
| `ANTHROPIC_API_KEY` | LLM prompts | Not required — vLLM or auto-template fallback available |
| `LLM_BACKEND` | Backend selection | Default: `anthropic`. Set to `openai` for vLLM. |
| `OPENAI_BASE_URL` | Local vLLM | Required if `LLM_BACKEND=openai` |
| `OPENAI_MODEL` | Local vLLM | Required if `LLM_BACKEND=openai` |

### Model Weight Locations

All weight paths have sensible defaults. Override them to point at a shared network drive so team members don't each download ~100 GB of weights.

| Variable | Default | Controls |
|----------|---------|----------|
| `MUSICVISION_WEIGHTS_DIR` | `~/.cache/musicvision/weights` | HuMo DiT, T5, VAE, Whisper, LoRA |
| `HF_HOME` | `~/.cache/huggingface` | FLUX, Z-Image, transformers models (standard HuggingFace env var) |
| `HVA_REPO_DIR` | _(must be set)_ | HunyuanVideo-Avatar repo + weights |
| `HVA_VENV_PYTHON` | `$HVA_REPO_DIR/.venv/bin/python` | HVA venv python (auto-derived if not set) |

Example team setup with a shared NFS mount:
```bash
# .env — shared across the team
MUSICVISION_WEIGHTS_DIR=/mnt/models/musicvision
HF_HOME=/mnt/models/huggingface
HVA_REPO_DIR=/mnt/models/HunyuanVideoAvatar
```

---

## Project Directory Structure

```
my-video/
├── project.yaml          # Project config (style sheet, model settings, quality preset)
├── scenes.json           # Scene list with all prompts and approval state
├── input/
│   ├── song.wav
│   ├── lyrics.txt
│   └── lyrics_whisper.txt
├── segments/             # Per-scene audio slices (full mix — fed to HuMo for lip sync)
├── segments_vocal/       # Per-scene vocal stems (Whisper transcription only)
├── images/               # FLUX reference images (scene_001.png …)
├── clips/                # Video clips (scene_001.mp4 …)
│   └── sub/              # Sub-clips for long scenes + continuity frames
├── clips_upscaled/       # Upscaled video clips (scene_001.mp4 …)
│   └── sub/              # Upscaled sub-clips
├── assets/
│   ├── characters/       # Character reference images
│   ├── loras/            # Character LoRA weights + Lightx2V distillation LoRA
│   └── settings/
└── output/
    ├── rough_cut.mp4     # Full song, all scenes assembled with original audio
    ├── timeline.edl      # CMX 3600 format
    └── timeline.fcpxml   # FCPXML 1.10 for DaVinci Resolve 18+ / Final Cut Pro
```

---

## Style Sheet

Each project has a persistent visual identity defined in `project.yaml`:

```yaml
style_sheet:
  visual_style: "Cinematic 35mm film, desaturated tones, shallow depth of field"
  color_palette: "Muted blues and amber, high contrast shadows"
  characters:
    - id: protagonist
      description: "Young woman, mid-20s, dark curly hair, worn leather jacket"
      reference_image: assets/characters/protagonist_ref.png
      lora_path: assets/loras/protagonist.safetensors
      lora_weight: 0.8
  settings:
    - id: rooftop
      description: "City rooftop at dusk, neon signs in background, gravel surface"
```

Style sheet elements are automatically injected into every image and video prompt for consistency across scenes.

---

## AceStep Integration

If your song was generated with [AceStep](https://github.com/ace-step/ace-step), upload the companion `.json` file. MusicVision auto-imports BPM, key, section-marked lyrics, and genre/mood captions to inform segmentation and prompt generation.

---

## Frontend (Scene Review GUI)

The React frontend provides a full visual workflow for managing the pipeline. Start the backend and frontend:

```bash
musicvision serve               # starts API on :8000 (no project needed)
cd frontend && npm run dev      # starts Vite dev server on :5173
```

Vite proxies `/api` and `/files` to `localhost:8000`.

**Features:**

- **Project creation/opening** — Create new projects or open existing ones from the UI (no need to pass a directory to `musicvision serve`)
- **Filesystem browser** — Browse the server's local filesystem to import audio and lyrics by path
- **Pipeline bar** — 5-step progress bar (Import → Intake → Images → Videos → Upscale) with model/engine selectors, resolution picker, and batch generation controls
- **Storyboard** — Scene-by-scene view with editable prompts, image thumbnails, and inline video preview
- **Per-scene generation** — Generate or regenerate individual scene images and videos with independent model/seed controls
- **Lip sync toggle** — Per-scene checkbox; when off, a silent audio segment is generated for non-vocal scenes
- **Auto-save** — Prompt edits are debounced and saved to the backend automatically
- **Inline video preview** — Click a storyboard thumbnail to toggle between the reference image and the generated video clip
- **Sub-clip auto-join** — Multi-segment scenes are automatically joined into a single preview clip

**Stack:** React 19, TypeScript, Vite 6, plain CSS (no UI framework). Dark theme.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `musicvision create <dir> --name "…"` | Create a new project |
| `musicvision import-audio --project DIR --audio song.wav [--lyrics lyrics.txt]` | Import audio and lyrics |
| `musicvision intake --project DIR [--llm] [--skip-transcription] [--vocal-separation]` | Stage 1: audio analysis + segmentation |
| `musicvision generate-images --project DIR [--model MODEL] [--scene-ids ID…]` | Stage 2: generate reference images |
| `musicvision generate-video --project DIR [--engine ENGINE] [--tier TIER] [--scene-ids ID…]` | Stage 3: generate video clips |
| `musicvision upscale --project DIR [--resolution RES] [--upscaler TYPE] [--scene-ids ID…]` | Stage 4: upscale video clips |
| `musicvision assemble --project DIR [--approved-only] [--no-edl] [--no-fcpxml]` | Stage 5: assemble rough cut + export |
| `musicvision info <dir>` | Show project status |
| `musicvision serve [dir] [--port 8000]` | Start API server (directory optional) |
| `musicvision detect-hardware` | Print GPU info and recommended tier |
| `musicvision download-weights --tier TIER [--token TOKEN]` | Download HuMo weights |

Image models: `flux-dev`, `flux-schnell`, `z-image`, `z-image-turbo`. Video engines: `hunyuan_avatar`, `ltx_video`, `humo`. Upscalers: `ltx_spatial`, `seedvr2`, `real_esrgan`. Resolutions: `720p`, `1080p`, `1440p`, `4k`.

---

## Testing

```bash
# CPU unit tests — 250 tests, no GPU needed, < 10 seconds
uv run pytest tests/ -v

# LLM prompt tests (requires vLLM server on LAN)
python scripts/test_vllm_prompts.py

# GPU image generation test (Z-Image + FLUX)
python scripts/test_image_gen.py

# GPU video generation test (HuMo)
python scripts/test_gpu_pipeline.py --tier fp8_scaled --steps 6

# HunyuanVideo-Avatar standalone test
python scripts/test_hva_standalone.py
```

See [TESTING.md](docs/TESTING.md) for the full test strategy and [MUSICVISION_GPU_TEST.md](docs/MUSICVISION_GPU_TEST.md) for GPU test setup.

---

## Documentation

| Document | Contents |
|----------|----------|
| [PIPELINE_SPEC.md](docs/PIPELINE_SPEC.md) | Full pipeline specification, API endpoints, CLI commands |
| [HUMO_REFERENCE.md](docs/HUMO_REFERENCE.md) | HuMo model details, TIA mode, prompt guidelines |
| [TESTING.md](docs/TESTING.md) | Two-layer test strategy |
| [MUSICVISION_GPU_TEST.md](docs/MUSICVISION_GPU_TEST.md) | GPU integration test guide |
| [PLATFORM_SUPPORT_PLAN.md](docs/PLATFORM_SUPPORT_PLAN.md) | Apple Silicon + cloud GPU support plan |
| [STATUS.md](docs/STATUS.md) | Current implementation status |
| [FIXLOG.md](docs/FIXLOG.md) | Checkpoint loading fix history |

---

## License

PolyForm Noncommercial License 1.0.0

