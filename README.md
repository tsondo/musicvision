# MusicVision

AI-powered music video production pipeline. Feed it a song; get back a fully timed rough cut with reference images, video clips, and a DaVinci Resolve-ready timeline.

```
audio + lyrics  →  scene segmentation  →  reference images  →  video clips  →  rough cut MP4 + EDL/FCPXML
```

**Status:** All four pipeline stages are code-complete. Next milestone: first GPU integration test. No frontend yet — interaction is via REST API (Swagger UI) or CLI.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Primary GPU (GPU0) | 16 GB VRAM | RTX 5090 32 GB |
| Secondary GPU (GPU1) | 12 GB VRAM | RTX 4080 16 GB |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| CUDA | 12.4+ | 12.4+ |

**GPU allocation:**
- **GPU0** (primary): DiT compute for FLUX image generation and HuMo video generation
- **GPU1** (secondary): T5 text encoder (~10 GB), VAE (~0.4 GB), Whisper (~1.5 GB), audio separator (~0.5 GB)
- FLUX and HuMo run sequentially (different stages), never simultaneously. Weights fully unloaded between stages.

**Optional LLM server:** A separate GPU (e.g. RTX 3090 Ti 24 GB) can run vLLM for local prompt generation, eliminating the Claude API dependency.

---

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone
git clone git@github.com:tsondo/musicvision.git
cd musicvision

# Install (CPU-only deps + dev tools)
uv sync --extra dev

# Install ML deps with CUDA 12.4 (run separately due to index URL)
uv run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

uv run pip install flash_attn==2.6.3

# Audio separator (separate due to onnxruntime conflict)
pip install "audio-separator[gpu]"

# Or use the full setup script
bash setup_env.sh
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Required For | Notes |
|----------|-------------|-------|
| `ANTHROPIC_API_KEY` | LLM prompts (cloud) | Not required — interactive/auto-template fallback available |
| `HUGGINGFACE_TOKEN` | HuMo DiT weights, FLUX.1-dev | Not required for shared weights (T5/VAE/Whisper auto-download) |
| `LLM_BACKEND` | Backend selection | Default: `anthropic`. Set to `openai` for vLLM. |
| `OPENAI_BASE_URL` | Local vLLM | Required if `LLM_BACKEND=openai` |
| `OPENAI_MODEL` | Local vLLM | Required if `LLM_BACKEND=openai` |
| `MUSICVISION_WEIGHTS_DIR` | Custom weight cache | Default: `~/.cache/musicvision/weights/` |

### Anthropic (cloud, default)

```env
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

### Local vLLM (LAN, OpenAI-compatible)

```env
LLM_BACKEND=openai
OPENAI_BASE_URL=http://192.168.1.100:8000/v1
OPENAI_API_KEY=vllm
OPENAI_MODEL=Qwen/Qwen2.5-32B-Instruct-AWQ
```

**Recommended local LLM models (RTX 3090 Ti 24 GB):**

| Model | VRAM | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~18 GB | Best quality |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | ~12 GB 4-bit | Fast, capable |

LLM is optional for all pipeline stages — without one, the pipeline falls back to interactive terminal prompts or auto-templates built from style sheet fields.

---

## Usage

### CLI

```bash
# Create a new project
musicvision create /path/to/my-video --name "My Song Title"

# Start the API server for a project
musicvision serve /path/to/my-video --port 8000

# Show project status
musicvision info /path/to/my-video

# Detect GPUs and recommend HuMo tier
musicvision detect-hardware

# Download model weights for a specific tier
musicvision download-weights --tier fp8_scaled

# Generate video for specific scenes
musicvision generate-video --project /path/to/my-video --tier fp8_scaled --scene-ids scene_001 scene_002
```

### API

With the server running, visit `http://localhost:8000/docs` for the interactive Swagger UI.

#### Pipeline stages

```
POST /api/upload/audio              Upload song file
POST /api/upload/lyrics             Upload lyrics text
POST /api/upload/acestep-json       Upload AceStep metadata (optional)
POST /api/pipeline/intake           Stage 1: BPM + transcription + scene segmentation
POST /api/pipeline/generate-images  Stage 2: Reference image generation (FLUX)
POST /api/pipeline/generate-videos  Stage 3: Video clip generation (HuMo)
POST /api/pipeline/assemble         Stage 4: Rough cut + EDL/FCPXML export
```

#### Scene review

```
GET    /api/scenes               List all scenes
GET    /api/scenes/{id}          Get a single scene
PATCH  /api/scenes/{id}          Override prompts, set approval status
POST   /api/scenes/approve-all   Bulk approve
```

#### Project configuration

```
GET    /api/projects/config              Full project config
PUT    /api/projects/config/style-sheet  Update style sheet
PUT    /api/projects/config/humo         Update HuMo settings
PUT    /api/projects/config/image-gen    Update FLUX settings
```

---

## Pipeline

### Stage 1 — Intake & Segmentation

1. Detects BPM and beat grid via librosa
2. Optionally separates vocals (MelBandRoFormer or Demucs) for cleaner transcription
3. Transcribes audio with Whisper large-v3 (word-level timestamps)
4. LLM segments lyrics into 2–10 second scenes on phrase/section boundaries (rule-based fallback available)
5. Snaps boundaries to nearest beat (0.15s tolerance)
6. Slices audio into per-scene `.wav` segments (full mix for HuMo, vocal stem for Whisper only)

> **Important:** HuMo receives the full audio mix, not isolated vocals. The vocal stem is used only to improve Whisper transcription accuracy. HuMo was trained on mixed audio — feeding it isolated vocals degrades A/V sync.

### Stage 2 — Reference Images

1. LLM generates FLUX image prompts per scene (style sheet injected)
2. FLUX dev/schnell generates reference stills (one per scene)
3. Optional LoRA for character consistency
4. User reviews and approves/rejects; can override prompts and regenerate per scene

### Stage 3 — Video Clips

1. LLM generates dense HuMo video prompts (Qwen2.5-VL caption style)
2. HuMo TIA mode generates clips using text + reference image + audio segment (full mix)
3. Scenes longer than 3.88s are automatically split into sub-clips with last-frame continuity chaining
4. User reviews and approves; can override prompts and regenerate per scene

### Stage 4 — Assembly

1. Approved clips are concatenated in scene order (sub-clips merged first)
2. Original audio is muxed back via ffmpeg concat demuxer (no re-encode)
3. Exports:
   - `output/rough_cut.mp4` — ready-to-view video
   - `output/timeline.edl` — CMX 3600 for DaVinci Resolve
   - `output/timeline.fcpxml` — FCPXML 1.10 for DaVinci Resolve 18+ / Final Cut Pro

---

## HuMo Weight Tiers

The video engine supports multiple precision tiers to fit different VRAM budgets:

| Tier | Format | DiT VRAM | Quality | Notes |
|------|--------|----------|---------|-------|
| `fp16` | FP16 safetensors | ~34 GB | Best | Needs block swap on <40 GB GPUs |
| `fp8_scaled` | FP8 e4m3fn scaled | ~18 GB | Near-best | Recommended for RTX 5090 |
| `gguf_q8` | GGUF 8-bit | ~18.5 GB | Good | |
| `gguf_q6` | GGUF 6-bit | ~14 GB | Good | |
| `gguf_q4` | GGUF 4-bit | ~11 GB | Acceptable | Fastest iteration |
| `preview` | FP16 (1.7B model) | ~3.4 GB | Lower visual quality | Nearly identical A/V sync to 17B |

DiT weights require `HUGGINGFACE_TOKEN`. Shared weights (T5, VAE, Whisper) are open models and auto-download on first use.

---

## GPU Integration Test

A self-contained test script validates the full HuMo pipeline on your hardware — no real song or reference image needed.

```bash
# Quick start (auto-detects GPU, synthesizes test assets)
python scripts/test_gpu_pipeline.py

# Override tier and reduce steps for faster iteration
python scripts/test_gpu_pipeline.py --tier gguf_q4 --steps 20

# Use block swap if tight on VRAM
python scripts/test_gpu_pipeline.py --block-swap 20

# Use real assets
python scripts/test_gpu_pipeline.py --audio ~/music/song.wav --image ~/photos/ref.png
```

The test runs four phases: hardware detection → single clip smoke test → sub-clip splitting → assembly. See [MUSICVISION_GPU_TEST.md](MUSICVISION_GPU_TEST.md) for the full guide and troubleshooting.

---

## Project Directory Structure

```
my-video/
├── project.yaml          # Project config (style sheet, model settings)
├── scenes.json           # Scene list with all prompts and approval state
├── input/
│   ├── song.wav
│   ├── lyrics.txt        # Optional provided lyrics
│   └── lyrics_whisper.txt  # Whisper transcription output
├── segments/             # Per-scene audio slices (full mix — fed to HuMo)
├── segments_vocal/       # Per-scene vocal stems (Whisper transcription only)
├── images/               # FLUX reference images (scene_001.png …)
├── clips/                # HuMo video clips (scene_001.mp4 …)
│   └── sub/              # Sub-clips for long scenes + continuity frames
├── assets/
│   ├── characters/       # Character reference images
│   ├── props/
│   ├── settings/
│   └── loras/            # Character LoRA weights
└── output/
    ├── rough_cut.mp4
    ├── timeline.edl
    └── timeline.fcpxml
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
  props:
    - id: vintage_mic
      description: "Chrome vintage ribbon microphone on a black stand"
  settings:
    - id: rooftop
      description: "City rooftop at dusk, neon signs in background, gravel surface"
```

Style sheet elements are automatically injected into every image and video prompt.

---

## AceStep Integration

If your song was generated with [AceStep](https://github.com/ace-step/ace-step), upload the companion `.json` file via `POST /api/upload/acestep-json`. MusicVision auto-imports:

- BPM, key, duration
- Section-marked lyrics (`[verse]`, `[chorus]`, etc.)
- Genre/mood caption (used as context for segmentation and prompts)

---

## Testing

```bash
# CPU unit tests (no GPU needed)
uv run pytest tests/ -v

# HuMo inference unit tests (CPU, no weights needed)
python scripts/test_humo_inference.py

# GPU integration test (run on workstation with GPU)
python scripts/test_gpu_pipeline.py --tier fp8_scaled --steps 30
```

| Test Suite | What It Covers | GPU Required? |
|------------|---------------|---------------|
| `tests/test_core.py` (26 tests) | Models, tiers, config, ProjectService, timecode | No |
| `tests/test_intake.py` | Rule-based segmentation, lyrics parsing | No |
| `tests/test_image_engine.py` | Config compat, engine interface, LoRA loading | No (mocked) |
| `tests/test_video_engine.py` | Constants, config, device map, block swap | No (mocked) |
| `scripts/test_humo_inference.py` (11 tests) | WanModel forward, RoPE, AudioProjModel, scheduler | No (CPU) |
| `scripts/test_gpu_pipeline.py` | Full end-to-end: load → generate → split → assemble | **Yes** |

```bash
# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

---

## Tech Stack

| Role | Library / Model |
|------|----------------|
| Video generation | [HuMo](https://github.com/Phantom-video/HuMo) TIA (ByteDance, Apache 2.0) |
| Image generation | FLUX dev / schnell (diffusers) |
| LLM (cloud) | Anthropic Claude |
| LLM (local) | vLLM — OpenAI-compatible API |
| Transcription | OpenAI Whisper large-v3 |
| Vocal separation | MelBandRoFormer / Demucs |
| Audio analysis | librosa |
| AV processing | ffmpeg |
| API | FastAPI + uvicorn |
| Data models | Pydantic v2 |
| Package manager | uv |
