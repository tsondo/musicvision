# CLAUDE.md — MusicVision

Context file for Claude Code sessions. Read this before making changes.

## What This Is

MusicVision is an AI music video production pipeline: song → scenes → storyboard → video. It wraps HuMo (ByteDance) for video generation and FLUX for reference image generation into an iterative, user-controlled workflow with LLM-assisted prompt generation.

## Repo Structure

```
src/musicvision/
├── cli.py              # CLI entry point (musicvision command)
├── api.py              # FastAPI REST API (musicvision serve)
├── llm.py              # Unified LLM client (Anthropic + OpenAI/vLLM)
├── models.py           # Pydantic v2 data models (ProjectConfig, Scene, SceneList, etc.)
├── engine_registry.py  # Engine constraints, frame math, sub-clip computation
├── intake/
│   ├── audio_analysis.py    # BPM detection, vocal separation
│   ├── transcription.py     # Whisper large-v3 transcription + alignment
│   └── segmentation.py      # LLM-assisted scene segmentation
├── imaging/
│   ├── prompt_generator.py  # LLM-assisted FLUX image prompts
│   ├── flux_engine.py       # FLUX inference wrapper
│   └── storyboard.py        # Storyboard management
├── video/
│   ├── prompt_generator.py  # LLM-assisted HuMo video prompts
│   ├── humo_engine.py       # HuMo TIA inference wrapper
│   └── scene_manager.py     # Scene review/regeneration
├── assembly/
│   ├── concatenator.py      # ffmpeg clip joining + audio sync
│   ├── exporter.py          # FCPXML/EDL generation
│   └── timecode.py          # Timecode utilities
├── vendor/                  # Vendored/patched model code (Wan2.1 DiT, VAE, T5, etc.)
└── utils/
    ├── gpu.py               # Multi-GPU device maps, VRAM tiers, recommend_tier()
    ├── audio.py             # ffmpeg wrappers
    └── paths.py             # Project directory management

frontend/                    # React + Vite scene review GUI
├── src/
│   ├── App.tsx              # Main app shell (state machine: no-project → loaded)
│   ├── api/client.ts        # Fetch wrapper for backend API
│   ├── api/types.ts         # TS interfaces matching Pydantic models
│   ├── components/          # ProjectOpener, ProjectHeader, Storyboard, SceneRow, PreviewPanel, AudioPlayer
│   └── hooks/               # useProject, useScenes
├── vite.config.ts           # Proxies /api and /files to localhost:8000
└── package.json
```

## Hardware

Two machines on the same LAN:

**Inference workstation** (where this repo runs):
- GPU0: RTX 5090 32GB — runs DiT (FLUX diffusion, HuMo generation)
- GPU1: RTX 4080 16GB — offloads T5, VAE, Whisper, audio separator
- OS: Windows + WSL (Fedora also used for dev)

**vLLM server** (192.168.1.137):
- GPU: RTX 3090 Ti 24GB
- Model: Qwen2.5-32B-Instruct-AWQ via vLLM
- Serves OpenAI-compatible API at http://192.168.1.137:8000/v1
- Running 24/7, no per-token cost

## Environment Setup

```bash
uv sync --extra ml
pip install "audio-separator[gpu]"
cp .env.example .env
# Fill in: HUGGINGFACE_TOKEN, LLM_BACKEND + related vars
```

Key env vars (see .env.example for full list):
- `LLM_BACKEND` — `anthropic` (default) or `openai` (for vLLM)
- `OPENAI_BASE_URL` — `http://192.168.1.137:8000/v1` (when using vLLM)
- `OPENAI_MODEL` — `qwen32b` (the --served-model-name on vLLM)
- `HUGGINGFACE_TOKEN` — required for HuMo DiT weights and FLUX.1-dev
- `ANTHROPIC_API_KEY` — only if using Claude API instead of vLLM

Weight location env vars (for team setups with shared storage):
- `MUSICVISION_WEIGHTS_DIR` — HuMo weights (default: `~/.cache/musicvision/weights`)
- `HF_HOME` — HuggingFace hub cache for FLUX/Z-Image (default: `~/.cache/huggingface`)
- `HVA_REPO_DIR` — HunyuanVideo-Avatar repo path (no default, must be set)
- `HVA_VENV_PYTHON` — HVA venv python (auto-derived from `HVA_REPO_DIR/.venv/bin/python`)

PyTorch: 2.10.0+cu128 (upgraded for RTX 5090 sm_120 support). Do not downgrade.

## Coding Conventions

- Python 3.11+, type hints everywhere
- Pydantic v2 for all data models — no raw dicts
- `from __future__ import annotations` in every module
- Ruff for linting: `ruff check src/ tests/`
- Line length: 120 chars
- Logging via `logging.getLogger(__name__)`, not print()
- All pipeline logic in core modules (intake/, imaging/, video/, assembly/). UI and API are thin layers that call into these — never put generation logic in api.py or UI code.

## Testing

Two-layer strategy. See docs/TESTING.md for full details.

### Unit tests (tests/)
```bash
python -m pytest tests/ -v --tb=short
```
No GPU, no network, fast (<10s). Run after every code change. Currently ~107 tests.

Test files:
- `test_core.py` — project config, scene models, style sheet, ProjectService lifecycle
- `test_intake.py` — segmentation logic, timestamp parsing, AceStep metadata
- `test_image_engine.py` — FLUX engine config, prompt generator, batch prompt parsing
- `test_video_engine.py` — HuMo engine config, video prompt construction, sub-clip splitting
- `test_engine_registry.py` — engine constraints, frame math, sub-clip computation
- `test_hunyuan_avatar_engine.py` — HVA config, factory dispatch, engine lifecycle, scene splitting

### Integration tests (scripts/)
```bash
python scripts/test_vllm_prompts.py          # LLM prompts against vLLM (needs server)
python scripts/test_gpu_pipeline.py [opts]    # Full GPU pipeline (needs GPU + weights)
python scripts/test_humo_inference.py         # HuMo model forward passes (CPU, no GPU)
```

Run manually. Each has different hardware requirements — see docs/TESTING.md and docs/MUSICVISION_GPU_TEST.md.

**After LLM prompt changes:** run both `pytest tests/` and `scripts/test_vllm_prompts.py`.
**After GPU/model changes:** run `scripts/test_gpu_pipeline.py`.

## LLM Integration

`llm.py` provides a unified `LLMClient` that routes to Anthropic or OpenAI-compatible backends. Three LLM-assisted tasks:

1. **Scene segmentation** (`intake/segmentation.py`) — lyrics + timestamps → JSON scene list
2. **Image prompt generation** (`imaging/prompt_generator.py`) — scene → FLUX prompt (single + batch)
3. **Video prompt generation** (`video/prompt_generator.py`) — scene → HuMo TIA prompt

All three degrade gracefully: LLM → interactive terminal prompt → auto-template from style sheet. User overrides in scenes.json always win.

System prompts are defined as module-level constants (`SEGMENTATION_SYSTEM_PROMPT`, `IMAGE_PROMPT_SYSTEM`, `VIDEO_PROMPT_SYSTEM`). When editing these, always re-run both unit tests and `test_vllm_prompts.py`.

## Key Constraints

- HuMo max output: 97 frames (~3.88s at 25fps). Scenes >4s are auto-split into sub-clips.
- Sub-clip continuity: last frame of sub-clip N becomes reference image for sub-clip N+1.
- FLUX and HuMo run sequentially (different stages), never simultaneously. Models fully unloaded between stages.
- GPU0 (5090) runs DiT/UNet. GPU1 (4080) runs encoders/VAE. This split is managed by `utils/gpu.py`.
- VRAM tier system: fp16, fp8_scaled, gguf_q8, gguf_q6, gguf_q4, preview. `recommend_tier()` auto-selects based on available VRAM.

## Project Data Flow

```
Song audio + lyrics
  → Stage 1: Intake (BPM, vocal sep, Whisper, LLM segmentation) → scenes.json
  → Stage 2: Imaging (LLM prompts, FLUX generation) → images/
  → Stage 3: Video (LLM prompts, HuMo TIA generation) → clips/
  → Stage 4: Assembly (ffmpeg concat, audio sync) → output/rough_cut.mp4 + FCPXML/EDL
```

Each stage produces artifacts on disk. Any stage can be re-run independently. User reviews and approves at each stage.

## Frontend (Scene Review GUI)

```bash
cd frontend && npm install    # first time only
cd frontend && npm run dev    # starts Vite dev server on :5173
```

Requires the backend running: `musicvision serve <project-dir>`. Vite proxies `/api` and `/files` to `localhost:8000`.

No UI framework — plain CSS + React 19. Dark theme. No state management library.

## Common Tasks

**Run the full pipeline from CLI:**
All test/project output goes in `test_output/<YYYY-MM-DD_HHMM_description>/` (gitignored). Never create test projects in the repo root.
```bash
musicvision create ./test_output/2026-03-01_1400_my_test --name "My Video"
musicvision import-audio --project ./test_output/2026-03-01_1400_my_test --audio song.wav --lyrics lyrics.txt
musicvision intake --project ./test_output/2026-03-01_1400_my_test --skip-transcription
musicvision generate-images --project ./test_output/2026-03-01_1400_my_test --model z-image-turbo
musicvision generate-video --project ./test_output/2026-03-01_1400_my_test --engine hunyuan_avatar
musicvision assemble --project ./test_output/2026-03-01_1400_my_test
```

**Add a new CLI command:** Edit `cli.py`, add a `cmd_*` function + argparse subparser + dispatch entry.

**Add a new API endpoint:** Edit `api/app.py`. Call into the appropriate core module — do not put logic in the endpoint handler.

**Change LLM prompts:** Edit the `*_SYSTEM_PROMPT` constant in the relevant module. Run `pytest tests/` then `scripts/test_vllm_prompts.py`.

**Add a new VRAM tier:** Edit `utils/gpu.py` — add to the tier enum, update `recommend_tier()`, and add loader logic.

**Debug checkpoint loading:** Use `scripts/dump_keys.py` to inspect state_dict keys. See docs/FIXLOG.md for past key-mapping issues.

## Documentation

- `docs/PIPELINE_SPEC.md` — Full pipeline specification
- `docs/HUMO_REFERENCE.md` — HuMo model details, prompt guidelines, TIA mode
- `docs/STATUS.md` — Current implementation status
- `docs/TESTING.md` — Two-layer test strategy + end-to-end CLI test
- `docs/MUSICVISION_GPU_TEST.md` — GPU integration test guide
- `docs/FIXLOG.md` — Record of checkpoint loading fixes and architecture issues
