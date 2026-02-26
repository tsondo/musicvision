# MusicVision Pipeline Specification

## Overview

MusicVision is a music video production pipeline that combines open-source AI tools to turn a song (audio + lyrics) into a complete music video. It wraps HuMo (ByteDance) for video generation and FLUX for reference image generation into an iterative, user-controlled workflow.

## System Requirements

### Inference Workstation (image + video generation)
- **Primary GPU (GPU0)**: NVIDIA RTX 5090 (32GB VRAM) — runs DiT for both FLUX and HuMo
- **Secondary GPU (GPU1)**: NVIDIA RTX 4080 (16GB VRAM) — offloads text encoders (T5), VAE, Whisper, audio separator
- **Multi-GPU Strategy**: Proven in ComfyUI. DiT on GPU0 (5090), everything else on GPU1 (4080). T5 (~10 GB) + VAE (~0.4 GB) + Whisper (~1.5 GB) fit comfortably in 16 GB. This allows running HuMo-17B and FLUX-dev at full quality.
- **Model Swapping**: FLUX and HuMo run in different pipeline stages (not simultaneously). Within each stage, model components are split across both GPUs. Weights fully unloaded between stages.

### Local LLM Server (optional, for prompt generation)
- **GPU**: NVIDIA RTX 3090 Ti (24GB VRAM) — runs vLLM for local LLM inference
- **Use case**: Scene segmentation and prompt generation without Claude API dependency
- **Recommended models**: Qwen2.5-32B-AWQ (~18 GB) or Mistral-Small-3.1-24B-Instruct (~12 GB 4-bit)
- **Not required**: Claude API (default) or auto-template fallback work without this machine

### General
- **Storage**: ~50GB for model weights (FLUX + HuMo + Whisper + VAE + audio separator). Additional space for project assets.
- **Python**: 3.11+ (HuMo requirement)
- **CUDA**: 12.4+ with flash_attn 2.6.3
- **Key dependency pins**: `torch==2.5.1`, `flash_attn==2.6.3`

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        MusicVision                          │
├─────────────┬──────────────┬──────────────┬─────────────────┤
│  Stage 1    │  Stage 2     │  Stage 3     │  Stage 4        │
│  INTAKE &   │  IMAGE GEN & │  VIDEO GEN   │  ASSEMBLY &     │
│  SEGMENTATION│  STORYBOARD │  (HuMo)      │  EXPORT         │
├─────────────┼──────────────┼──────────────┼─────────────────┤
│ Whisper     │ FLUX         │ HuMo TIA     │ ffmpeg          │
│ ffmpeg      │ (LoRA)       │ (17B or 1.7B)│ FCPXML/EDL gen  │
│ LLM (Claude)│              │              │                 │
└─────────────┴──────────────┴──────────────┴─────────────────┘
```

### GPU Memory Map (Inference Workstation)

```
GPU0 — RTX 5090 (32 GB)          GPU1 — RTX 4080 (16 GB)
┌─────────────────────────┐       ┌─────────────────────────┐
│ Stage 2: FLUX DiT       │       │ T5 text encoder  ~10 GB │
│   bf16: ~24 GB          │       │ VAE              ~0.4 GB│
│   FP8:  ~12 GB          │       │ Whisper          ~1.5 GB│
├─────────────────────────┤       │ Audio separator  ~0.5 GB│
│ Stage 3: HuMo DiT       │       │                         │
│   fp16: ~34 GB (swap)   │       │ Total: ~12.4 GB         │
│   fp8:  ~18 GB          │       │ Headroom: ~3.6 GB       │
│   gguf: 11–18.5 GB      │       └─────────────────────────┘
│   preview (1.7B): ~3.4GB│
└─────────────────────────┘
(Only one stage loaded at a time)
```

## Data Model

### Project Config (`project.yaml`)

```yaml
project:
  name: "My Music Video"
  created: "2026-02-23T12:00:00Z"

song:
  audio_file: "input/song.wav"
  lyrics_file: "input/lyrics.txt"       # User-provided or Whisper-generated
  bpm: 120                               # Auto-detected or user-specified
  duration_seconds: 180.0
  keyscale: ""                           # Optional AceStep key/scale metadata
  acestep: null                          # AceStep JSON metadata if available

style_sheet:
  visual_style: "cinematic, moody lighting, shallow depth of field"
  color_palette: "dark blues, warm amber highlights"
  aspect_ratio: "16:9"
  resolution: "1280x720"

  # Persistent elements injected into every image/video prompt
  characters:
    - id: "singer"
      description: "Young woman with short black hair, angular features"
      reference_image: "assets/characters/singer_ref.png"
      lora_path: "assets/loras/singer_v1.safetensors"  # Optional
      lora_weight: 0.8

  props:
    - id: "vintage_mic"
      description: "Chrome vintage ribbon microphone on a black stand"

  settings:
    - id: "stage"
      description: "Dimly lit stage with red velvet curtains and haze"

humo:
  tier: "fp8_scaled"          # fp16 | fp8_scaled | gguf_q8 | gguf_q6 | gguf_q4 | preview
  resolution: "720p"          # "720p" (1280×720) or "480p" (832×480)
  scale_a: 2.0                # Audio guidance strength
  scale_t: 7.5                # Text guidance strength
  denoising_steps: 50
  block_swap_count: 0         # Blocks to swap CPU↔GPU; 0 = no swap
  sub_clip_continuity: true   # Last frame of sub-clip N → reference for N+1

image_gen:
  model: "flux-dev"           # flux-dev (gated) | flux-schnell (open)
  quant: "auto"               # auto | fp8 | int8 | none
  steps: null                 # null = auto (4 for schnell, 28 for dev)
  guidance_scale: 3.5
  lora_path: null             # Optional LoRA for character consistency
  lora_weight: 0.8

vocal_separation:
  method: "roformer"          # roformer | demucs
  roformer_model: "MelBandRoformer.ckpt"
  demucs_model: "htdemucs"
```

### Scene Definition (`scenes.json`)

```json
{
  "scenes": [
    {
      "id": "scene_001",
      "order": 1,
      "time_start": 0.0,
      "time_end": 3.88,
      "type": "vocal",
      "lyrics": "Standing in the rain tonight",
      "audio_segment": "segments/scene_001.wav",
      "audio_segment_vocal": "segments_vocal/scene_001_vocal.wav",

      "image_prompt": "A young woman with short black hair stands on a dimly lit stage...",
      "image_prompt_user_override": null,
      "reference_image": "images/scene_001.png",
      "image_status": "pending",

      "video_prompt": "A close-up of a young woman with short black hair singing...",
      "video_prompt_user_override": null,
      "video_clip": "clips/scene_001.mp4",
      "video_status": "pending",

      "sub_clips": [],

      "characters": ["singer"],
      "props": ["vintage_mic"],
      "settings": ["stage"],

      "notes": ""
    }
  ]
}
```

### Key Design Decision: Sub-clips for Long Scenes

HuMo generates max ~3.88 seconds (97 frames @ 25fps). Scenes longer than 3.88s are automatically split:

```
Scene: 8 seconds (time 10.0 → 18.0)
  ├── sub_clip_a: frames 0-96   (10.0 → 13.88s)
  ├── sub_clip_b: frames 0-96   (13.88 → 17.76s)
  └── sub_clip_c: frames 0-30   (17.76 → 18.0s)  ← partial
```

Sub-clip continuity: last frame of sub-clip N is extracted via ffmpeg and used as the reference image for sub-clip N+1. This prevents visual discontinuity within a long scene but requires sequential generation.

## Stage 1: Intake & Segmentation

### Inputs
- Audio file (WAV/MP3/FLAC)
- Lyrics text file (optional — Whisper can transcribe)
- AceStep JSON metadata (optional — provides BPM, lyrics, section markers natively)

### Process

1. **Audio Analysis**
   - Load audio, resolve duration (AceStep metadata or ffprobe)
   - BPM detection via librosa beat tracker
   - Optional vocal separation (MelBandRoFormer or Demucs) → cleaner Whisper input

2. **Transcription**
   - Whisper-large-v3 transcription with word-level timestamps
   - Skippable if lyrics are pre-provided

3. **Scene Segmentation** (LLM-assisted, with rule-based fallback)
   - Input: lyrics with timestamps, song structure (verse/chorus/bridge/instrumental)
   - Rules:
     - Minimum scene: 2 seconds
     - Maximum scene: 10 seconds (split into sub-clips by HuMo engine; 3.88s hard limit)
     - Prefer cuts on musical phrase boundaries (beat-snapped within 0.15s tolerance)
     - Instrumental sections get their own scenes (type: "instrumental")
   - Output: `scenes.json` with timestamps and types
   - LLM fallback: `segment_scenes_simple()` rule-based segmenter (no LLM required)

4. **Audio Slicing**
   - ffmpeg splits the full audio into per-scene WAV segments
   - Each segment saved as WAV (pcm_s16le) at original sample rate — sample-accurate
   - Vocal-separated versions also sliced (for Whisper only — see critical note below)

> **⚠️ Critical: HuMo receives the full audio mix, not isolated vocals.** The vocal stem is used *only* to improve Whisper transcription accuracy. HuMo was trained on mixed audio — feeding it isolated vocals degrades A/V sync. Audio segments in `segments/` are always full-mix; `segments_vocal/` is consumed only by the transcription step.

### Outputs
- `project.yaml` (initial)
- `scenes.json` (scene list with timestamps)
- `segments/` — per-scene WAV clips (full mix — fed to HuMo)
- `segments_vocal/` — per-scene WAV clips (vocal stem — Whisper only)

## Stage 2: Image Generation & Storyboard

### Process

1. **Prompt Generation** (LLM-assisted)
   - For each scene, generate a FLUX image prompt:
     - Describes the visual content matching the lyrics/mood
     - Injects style_sheet elements (characters, props, settings)
     - Maintains narrative coherence across scenes
     - Accounts for scene type (vocal scenes show singer; instrumental scenes show environment/abstract)
   - Fallback: interactive terminal prompt (TTY) or auto-template (non-TTY)
   - User can override any prompt via `image_prompt_user_override` in `scenes.json`

2. **Image Generation** (FLUX via `FluxEngine`)
   - VRAM-tiered: 4 tiers from bf16 (≥28 GB) down to quantized sequential offload (<8 GB)
   - FP8 quantization on Ada/Hopper/Blackwell (compute cap ≥ 8.9); auto-falls back to INT8
   - Optional LoRA for character consistency
   - FLUX.1-dev is gated (requires `HUGGINGFACE_TOKEN`); FLUX.1-schnell is open
   - User can override individual prompts and regenerate per scene

3. **Storyboard Review** *(planned — not built)*
   - API endpoints exist for scene CRUD and approval (`PATCH /api/scenes/{id}`, `POST /api/scenes/approve-all`)
   - Visual storyboard UI (grid layout with thumbnails) is not yet built

### Outputs
- Updated `scenes.json` with image prompts and status
- `images/` directory with reference images (`images/scene_001.png`, etc.)

## Stage 3: Video Generation (HuMo)

### Process

1. **Video Prompt Generation** (LLM-assisted)
   - Generate dense Qwen2.5-VL-style captions for each scene (HuMo's training style)
   - More detailed than image prompt — describes motion, camera movement, expressions
   - Same LLM-unavailable fallback as image prompts
   - Auto-template anchors on image prompt, style sheet, and scene type

2. **HuMo Inference** (TIA mode via `HumoEngine`)
   - Input per scene: text prompt + reference image + audio segment (full mix)
   - Config from `project.yaml` (`humo.tier`, `resolution`, `scale_a`, `scale_t`, `denoising_steps`)
   - For scenes > 3.88s: `generate_scene()` automatically splits into sub-clips with last-frame chaining
   - Seed: always resolved (random if not provided), recorded in `HumoOutput.seed_used`
   - Block swap: `block_swap_count > 0` enables CPU↔GPU block migration to reduce DiT VRAM

3. **TIA Denoising Loop** (dual CFG, 3 DiT passes per step)
   ```
   v_cond      = DiT(z + img_pos, t, pos_text, audio)
   v_audio_neg = DiT(z + img_pos, t, pos_text, audio_zeros)
   v_text_neg  = DiT(z + img_neg, t, neg_text, audio)

   v_pred = v_text_neg
          + scale_a × (v_cond - v_audio_neg)
          + (scale_t - 2.0) × (v_audio_neg - v_text_neg)
   ```

4. **Scene Review** *(planned — not built)*
   - API endpoints for per-scene video status exist
   - Visual clip review UI is not yet built

### Outputs
- Updated `scenes.json` with video prompts and status
- `clips/scene_001.mp4`, `clips/scene_002.mp4`, etc.
- `clips/sub/scene_003_a.mp4`, `clips/sub/scene_003_b.mp4` (sub-clips for long scenes)

## Stage 4: Assembly & Export

### Process

1. **Concatenation** (`assemble_rough_cut()`)
   - Sorts scenes by `order`, joins sub-clips first
   - ffmpeg concat demuxer (no re-encode)
   - Muxes original full-song audio back over the assembled video

2. **Export Formats**
   - **Rough Cut**: `output/rough_cut.mp4` — full song, all scenes assembled
   - **EDL**: `output/timeline.edl` — CMX 3600 format, DaVinci Resolve compatible
   - **FCPXML**: `output/timeline.fcpxml` — FCPXML 1.10, DaVinci Resolve 18+ / Final Cut Pro

### Outputs
- `output/rough_cut.mp4`
- `output/timeline.edl`
- `output/timeline.fcpxml`

## Directory Structure

```
my_music_video/
├── project.yaml
├── scenes.json
├── input/
│   ├── song.wav
│   └── lyrics.txt
├── assets/
│   ├── characters/
│   ├── props/
│   ├── settings/
│   └── loras/
├── segments/                  # Full mix — fed to HuMo
│   ├── scene_001.wav
│   └── ...
├── segments_vocal/            # Vocal stem — Whisper transcription only
│   ├── scene_001_vocal.wav
│   └── ...
├── images/
│   ├── scene_001.png
│   └── ...
├── clips/
│   ├── scene_001.mp4
│   ├── scene_002.mp4
│   └── sub/
│       ├── scene_003_a.mp4
│       ├── scene_003_a_lastframe.png   ← continuity frame for sub-clip chaining
│       └── scene_003_b.mp4
└── output/
    ├── rough_cut.mp4
    ├── timeline.edl
    └── timeline.fcpxml
```

## Module Structure (Python)

```
src/musicvision/
├── __init__.py
├── cli.py                    # CLI entry point
├── llm.py                    # Unified LLM client (Anthropic + OpenAI-compat/vLLM)
├── models.py                 # All Pydantic v2 data models
├── project.py                # ProjectService (lifecycle, persistence)
├── api/
│   ├── __init__.py
│   └── app.py                # FastAPI application (25+ endpoints, Swagger at /docs)
├── intake/
│   ├── __init__.py
│   ├── audio_analysis.py     # BPM detection, vocal separation (RoFormer + Demucs)
│   ├── pipeline.py           # run_intake() orchestrator
│   ├── segmentation.py       # LLM-assisted scene segmentation + rule-based fallback
│   └── transcription.py      # Whisper transcription + timestamp alignment
├── imaging/
│   ├── __init__.py
│   ├── base.py               # Abstract ImageEngine base
│   ├── factory.py            # create_image_engine()
│   ├── flux_engine.py        # FLUX inference, VRAM-tiered loading
│   ├── prompt_generator.py   # LLM-assisted FLUX prompt generation
│   └── zimage_engine.py      # Placeholder / alternative engine stub
├── video/
│   ├── __init__.py
│   ├── audio_encoder.py      # HumoAudioEncoder: Whisper → 5-band features → windows
│   ├── base.py               # Abstract VideoEngine base
│   ├── block_swap.py         # BlockSwapManager: CPU↔GPU block migration
│   ├── factory.py            # create_video_engine()
│   ├── humo_engine.py        # HumoEngine: TIA inference orchestration
│   ├── model_loader.py       # Tiered loaders (FP16, FP8Scaled, GGUF, Preview1.7B)
│   ├── prompt_generator.py   # LLM-assisted HuMo video prompt generation
│   ├── scheduler.py          # FlowMatchScheduler (Euler, shift=5.0)
│   ├── wan_model.py          # Self-contained WanModel DiT (no wan.modules dependency)
│   ├── wan_t5.py             # WanT5Encoder wrapping HuggingFace T5EncoderModel
│   ├── wan_vae.py            # WanVideoVAE with CausalConv3d
│   └── weight_registry.py    # Weight spec, locate/download/status functions
├── assembly/
│   ├── __init__.py
│   ├── concatenator.py       # assemble_rough_cut() via ffmpeg concat demuxer
│   ├── exporter.py           # export_edl() CMX 3600, export_fcpxml() FCPXML 1.10
│   └── timecode.py           # Timecode utilities
└── utils/
    ├── __init__.py
    ├── audio.py              # ffmpeg wrappers: slice_audio(), mux_video_audio()
    ├── gpu.py                # GPU detection, DeviceMap, recommend_tier(), vram_info()
    └── paths.py              # ProjectPaths: all canonical project directory/file paths
```

## API Architecture (Current)

The pipeline is driven by a FastAPI REST API. A frontend has not been built yet — all interaction is via the API (Swagger UI at `/docs`) or CLI.

### Key Endpoints

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

### CLI Commands

```bash
musicvision create <dir> --name "My Video"
musicvision serve <dir> [--port 8000]
musicvision info <dir>
musicvision detect-hardware
musicvision download-weights --tier {fp16|fp8_scaled|gguf_q8|gguf_q6|gguf_q4|preview}
musicvision generate-video --project <dir> [--tier X] [--block-swap N] [--scene-ids ...]
```

## Planned Frontend (Not Built)

The intended user workflow is a storyboard-style UI. The REST API is designed to support it; the UI itself is not started. Two candidate approaches:

### Option A: React (preferred)
CORS already open for Vite/CRA default ports. All data flows through the REST API.

### Option B: Gradio
Simpler to build. Would replace (or wrap) the CLI.

### Intended Tab Layout

**Tab 1: Project Setup**
- Song upload (WAV/MP3/FLAC) + lyrics upload or paste
- Style sheet editor (visual style, color palette, characters, props, settings)
- LoRA management (upload, assign to characters)
- HuMo/FLUX config (tier, guidance scales, resolution)

**Tab 2: Segmentation**
- Lyrics display with word-level timestamps (from Whisper)
- Editable scene boundary markers
- Scene list with type labels (vocal/instrumental)
- "Re-segment" button

**Tab 3: Storyboard (main workflow)**

Grid layout — one row per scene:

| Column | Content | Interactive |
|--------|---------|------------|
| 1. Scene Info | Lyrics text (or "Instrumental"), timestamp range | Read-only |
| 2. Reference Image | Generated image thumbnail | Click to enlarge |
| 3. Image Prompt | Editable textbox with generated prompt | User editable + "Regenerate" |
| 4. Video Prompt | Editable textbox + LoRA selector | User editable + "Regenerate" |
| 5. Video Clip | Generated video player | Click to play + "Regenerate" |
| 6. Status | Approve/reject toggles for image and video | Toggle |

- Each row operates independently — regenerating scene 5 doesn't touch scene 4
- Batch operations: "Generate All Images", "Generate All Videos", "Approve All"

**Tab 4: Assembly & Export**
- Full rough-cut video player with audio
- Export controls: rough cut MP4, EDL, FCPXML

## Open Questions / Future Work

- **Frontend**: React vs Gradio — not decided. REST API is UI-agnostic.
- **Progress feedback**: No SSE/WebSocket for long-running jobs. API endpoints are synchronous — a 50-scene video gen blocks for hours with no progress feedback.
- **Partial failure recovery**: Exception propagates on failure; already-generated clips survive. Workaround: `--scene-ids` CLI flag. Needs a proper job/resume model.
- **Transitions**: Currently hard cuts only. Future: AI-generated transitions, crossfades.
- **Batch rendering**: Scenes generate sequentially. Future: multi-GPU or cloud parallelism.
- **LoRA training**: Pipeline accepts LoRA paths but doesn't include training workflows.
- **Camera movement**: HuMo handles some implicit camera motion via prompting. Could add explicit camera control via prompt templates.
- **Character consistency**: LoRA is the current approach. Could explore IP-Adapter or other identity preservation methods.
- **Render time estimation**: No upfront time estimate for users. 3× DiT passes × 50 steps × N scenes can take many hours.
