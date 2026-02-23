# MusicVision

AI-powered music video production pipeline. Feed it a song; get back a fully timed rough cut with reference images, video clips, and a DaVinci Resolve-ready timeline.

```
audio + lyrics  →  scene segmentation  →  reference images  →  video clips  →  rough cut MP4 + EDL/FCPXML
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Primary GPU | 16 GB VRAM | RTX 5080 32 GB |
| Secondary GPU | 12 GB VRAM | RTX 3080 Ti 12 GB |
| RAM | 32 GB | 64 GB |
| Storage | 100 GB SSD | 500 GB NVMe |
| CUDA | 12.4+ | 12.4+ |

**GPU allocation:**
- Primary: FLUX image generation, HuMo video DiT/UNet
- Secondary: Whisper transcription, text encoders, VAE

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

# Or use the full setup script
bash setup_env.sh
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

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

**Recommended models for ~28 GB total VRAM (e.g. RTX 4080 16 GB + RTX 5070 12 GB):**

| Model | VRAM | Notes |
|-------|------|-------|
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | ~18 GB | Best quality |
| `mistralai/Mistral-Small-3.1-24B-Instruct-2503` | ~12 GB | Fast, capable |
| `Qwen/Qwen2.5-14B-Instruct` | ~28 GB bf16 | Fastest, no quantisation |

Start vLLM with tensor parallelism across both GPUs:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --tensor-parallel-size 2
```

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
```

### API

With the server running, visit `http://localhost:8000/docs` for the interactive API explorer.

#### Pipeline stages

```
POST /api/upload/audio           Upload song file
POST /api/upload/acestep-json    Upload AceStep metadata (optional)
POST /api/pipeline/intake        Stage 1: BPM + transcription + scene segmentation
POST /api/pipeline/generate-images  Stage 2: Reference image generation (FLUX)
POST /api/pipeline/generate-videos  Stage 3: Video clip generation (HuMo)
POST /api/pipeline/assemble      Stage 4: Rough cut + EDL/FCPXML export
```

#### Scene review

```
GET    /api/scenes               List all scenes
GET    /api/scenes/{id}          Get a single scene
PATCH  /api/scenes/{id}          Override prompts, set approval status
POST   /api/scenes/approve-all   Bulk approve
```

---

## Pipeline

### Stage 1 — Intake & Segmentation

1. Detects BPM and beat grid via librosa
2. Transcribes audio with Whisper large-v3 (word-level timestamps)
3. Optionally aligns provided lyrics with Whisper timing
4. Calls LLM to segment into 2–10 second scenes on phrase/section boundaries
5. Snaps boundaries to nearest beat
6. Slices audio into per-scene `.wav` segments

### Stage 2 — Reference Images

1. LLM generates FLUX image prompts per scene (style sheet injected)
2. FLUX dev/schnell generates reference stills (one per scene)
3. User reviews and approves/rejects; can override prompts

### Stage 3 — Video Clips

1. LLM generates dense HuMo video prompts (Qwen2.5-VL caption style)
2. HuMo TIA mode generates clips using text + reference image + audio segment
3. Scenes longer than 3.88 s are automatically split into overlapping sub-clips
4. User reviews and approves; can override prompts

### Stage 4 — Assembly

1. Approved clips are concatenated in scene order
2. Original audio is muxed back precisely
3. Exports:
   - `output/rough_cut.mp4` — ready-to-view video
   - `output/timeline.edl` — CMX 3600 for DaVinci Resolve
   - `output/timeline.fcpxml` — FCPXML 1.10 for DaVinci Resolve / Final Cut Pro

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
├── segments/             # Per-scene audio slices
├── images/               # FLUX reference images (scene_001.png …)
├── clips/                # HuMo video clips (scene_001.mp4 …)
│   └── sub/              # Sub-clips for long scenes
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
      lora_path: assets/loras/protagonist.safetensors
  settings:
    - id: rooftop
      description: "City rooftop at dusk, neon signs in background, gravel surface"
```

Style sheet elements are automatically injected into every image and video prompt.

---

## AceStep Integration

If your song was generated with [AceStep](https://github.com/ace-step/ace-step), place the companion `.json` file alongside the audio when uploading. MusicVision auto-imports:

- BPM, key, duration
- Section-marked lyrics (`[verse]`, `[chorus]`, etc.)
- Genre/mood caption (used as context for segmentation and prompts)

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/
```

---

## Tech Stack

| Role | Library / Model |
|------|----------------|
| LLM (cloud) | Anthropic Claude (`claude-sonnet-4-20250514`) |
| LLM (local) | vLLM — OpenAI-compatible API |
| Transcription | OpenAI Whisper large-v3 |
| Image generation | FLUX dev / schnell (diffusers) |
| Video generation | HuMo TIA (ByteDance, Apache 2.0) |
| Audio analysis | librosa |
| AV processing | ffmpeg |
| API | FastAPI + uvicorn |
| Data models | Pydantic v2 |
| Package manager | uv |
