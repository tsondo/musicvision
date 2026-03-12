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
├── upscaling/
│   ├── base.py              # UpscaleEngine ABC + dataclasses
│   ├── factory.py           # Dispatch on UpscalerType → engine
│   ├── pipeline.py          # Orchestrator: group by engine, upscale, update scenes
│   ├── realesrgan_engine.py # Real-ESRGAN frame-by-frame upscaler
│   ├── seedvr2_engine.py    # SeedVR2 subprocess bridge (like HVA)
│   └── ltx_spatial_engine.py # LTX Spatial Upsampler (diffusers in-process)
├── assembly/
│   ├── concatenator.py      # ffmpeg clip joining + audio sync (prefers upscaled clips)
│   ├── exporter.py          # FCPXML/EDL generation
│   └── timecode.py          # Timecode utilities
├── vendor/                  # Vendored/patched model code (Wan2.1 DiT, VAE, T5, etc.)
└── utils/
    ├── gpu.py               # Multi-GPU device maps, VRAM tiers, recommend_tier()
    ├── audio.py             # ffmpeg wrappers
    ├── video.py             # ffprobe resolution, ffmpeg scale
    └── paths.py             # Project directory management

frontend/                    # React + Vite scene review GUI
├── src/
│   ├── App.tsx              # Main app shell (state machine: no-project → loaded)
│   ├── api/client.ts        # Fetch wrapper for backend API
│   ├── api/types.ts         # TS interfaces matching Pydantic models
│   ├── components/          # ProjectOpener, ProjectHeader, Storyboard, SceneRow, PreviewPanel, AudioPlayer
│   └── hooks/               # useProject, useScenes, usePipeline
├── vite.config.ts           # Proxies /api and /files to localhost:8000
└── package.json
```

## Hardware

Two machines on the same LAN:

**Inference workstation** (where this repo runs):
- GPU0: RTX 5090 32GB — runs DiT (FLUX diffusion, HuMo generation)
- GPU1: RTX 4080 16GB — offloads T5, VAE, Whisper, audio separator
- OS: Windows + WSL (Fedora also used for dev)

**vLLM server** (separate LAN machine):
- GPU: RTX 3090 Ti 24GB
- Model: Qwen2.5-32B-Instruct-AWQ via vLLM
- Serves OpenAI-compatible API (see `OPENAI_BASE_URL` in `.env`)
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
- `OPENAI_BASE_URL` — vLLM server URL, e.g. `http://<lan-ip>:8000/v1`
- `OPENAI_MODEL` — `qwen32b` (the --served-model-name on vLLM)
- `HUGGINGFACE_TOKEN` — required for HuMo DiT weights and FLUX.1-dev
- `ANTHROPIC_API_KEY` — only if using Claude API instead of vLLM

Weight location env vars (for team setups with shared storage):
- `MUSICVISION_WEIGHTS_DIR` — HuMo weights (default: `~/.cache/musicvision/weights`)
- `HF_HOME` — HuggingFace hub cache for FLUX/Z-Image (default: `~/.cache/huggingface`)
- `HVA_REPO_DIR` — HunyuanVideo-Avatar repo path (no default, must be set)
- `HVA_VENV_PYTHON` — HVA venv python (auto-derived from `HVA_REPO_DIR/.venv/bin/python`)
- `SEEDVR2_REPO_DIR` — SeedVR2 repo path (no default, must be set for SeedVR2 upscaler)
- `SEEDVR2_VENV_PYTHON` — SeedVR2 venv python (auto-derived from `SEEDVR2_REPO_DIR/.venv/bin/python`)

PyTorch: 2.10.0+cu128 (upgraded for RTX 5090 sm_120 support). Do not downgrade.

## Coding Conventions

- Python 3.11+, type hints everywhere
- Pydantic v2 for all data models — no raw dicts
- `from __future__ import annotations` in every module (required — prevents circular import issues with Pydantic v2 `model_rebuild()`, not just style)
- Ruff for linting: `ruff check src/ tests/`
- Line length: 120 chars
- Logging via `logging.getLogger(__name__)`, not print()
- All pipeline logic in core modules (intake/, imaging/, video/, assembly/). UI and API are thin layers that call into these — never put generation logic in api.py or UI code.
- GPU-heavy and long-running API endpoints must run their sync work in a background thread via `asyncio.to_thread()` so the FastAPI event loop stays responsive for polling and other requests. Any endpoint that calls into GPU inference (intake, image generation, video generation) is a candidate for this pattern.
- Always guard `import torch` behind try/except or function scope — unit tests must not require torch
- Subprocess engines (HVA, SeedVR2) must capture and log both stdout and stderr
- Any new engine must register in `engine_registry.py` ENGINES dict — this is the single source of truth
- Model loading and unloading must be symmetric — if `load()` moves weights to GPU, `unload()` must free them completely and verifiably
- Never hardcode resolution, frame count, or FPS — always read from engine constraints

## Do NOT

- Put generation/inference logic in api.py or UI code
- Use raw dicts where a Pydantic model exists
- Create test projects in the repo root (use `test_output/`)
- Import torch at module top-level in non-ML modules
- Assume fixed max duration — always read from engine config
- Use seconds for duration math — use frame counts (see `engine_registry.py`)
- Modify `vendor/` without documenting the patch in `FIXLOG.md`
- Remove `from __future__ import annotations` from any module
- Hardcode GPU device indices — use `utils/gpu.py` device map

## Before Editing Any Module

1. Run `pytest tests/ -v --tb=short` — confirm green baseline
2. Read the existing tests for that module BEFORE changing code
3. If touching `engine_registry.py`, `models.py`, or `gpu.py` — these are high-fan-out modules. Grep for all importers and run `ruff check src/ tests/` after changes
4. If touching `vendor/` — these are patched upstream files. Check `FIXLOG.md` for prior patches that must be preserved
5. Never modify a Pydantic model field without checking `scenes.json` backward compatibility

## Inference Debugging Playbook

When video/image generation produces bad output (noise, black frames, artifacts):

1. **Check denoising health**: Log min/max/mean of latents after each denoising step. Exploding values (>1e3) or collapsing to zero indicate a math bug, not a prompt issue.
2. **Verify checkpoint keys**: `python scripts/dump_keys.py <weights_path>` — look for prefix mismatches, missing keys, unexpected extras
3. **Isolate the variable**: Change ONE thing at a time (tier, resolution, steps, prompt, reference image)
4. **Compare against ComfyUI when possible**: If a ComfyUI workflow produces good output with the same model/settings, diff the parameters systematically (sigma schedule, CFG values, sampler, input scaling)
5. **VRAM pressure**: OOM mid-inference can produce partial/corrupt output without raising an exception. Check `torch.cuda.memory_allocated()` after generation.
6. **Subprocess engines (HVA, SeedVR2)**: Check stderr first — the subprocess may fail silently from the caller's perspective. Parse return codes.
7. **Known gotchas** (see FIXLOG.md):
   - Timestep must be scaled ×1000 for HuMo
   - Sigma shift 8.0 (not 5.0) matches ComfyUI
   - FP8 needs dynamic per-tensor input scaling
   - Reference frame goes at LAST temporal position, not first

## Common Failure Modes

- **Silent dtype mismatch**: Model expects bf16 input but receives fp32 → runs but produces garbage. Always verify `tensor.dtype` at engine boundaries.
- **Stale model on GPU**: Previous stage's model wasn't fully unloaded → OOM on next stage. Verify with `torch.cuda.memory_allocated()` after `engine.unload()`.
- **Audio/video length mismatch in assembly**: Almost always caused by seconds-based math instead of frame-based. All duration math must go through `engine_registry.py`.
- **Subprocess engine env contamination**: HVA and SeedVR2 run in their own venvs. Don't let MusicVision's env vars (especially `CUDA_VISIBLE_DEVICES`) leak unintentionally.
- **LoRA swap without full unload**: Some engines cache LoRA weights. Call `engine.unload()` + `engine.load()` when switching LoRAs, not just swapping the path.

## Testing

Two-layer strategy. See docs/TESTING.md for full details.

### Unit tests (tests/)
```bash
python -m pytest tests/ -v --tb=short
```
No GPU, no network, fast (<10s). Run after every code change. Currently ~250 tests.

Test files:
- `test_core.py` — project config, scene models, style sheet, ProjectService lifecycle
- `test_intake.py` — segmentation logic, timestamp parsing, AceStep metadata
- `test_image_engine.py` — FLUX engine config, prompt generator, batch prompt parsing
- `test_video_engine.py` — HuMo engine config, video prompt construction, sub-clip splitting
- `test_engine_registry.py` — engine constraints, frame math, sub-clip computation
- `test_hunyuan_avatar_engine.py` — HVA config, factory dispatch, engine lifecycle, scene splitting
- `test_upscaler.py` — upscaler enums, config auto-selection, factory, pipeline orchestrator, Scene/SubClip fields

### Integration tests (scripts/)
```bash
python scripts/test_vllm_prompts.py          # LLM prompts against vLLM (needs server)
python scripts/test_gpu_pipeline.py [opts]    # Full GPU pipeline (needs GPU + weights)
python scripts/test_humo_inference.py         # HuMo model forward passes (CPU, no GPU)
```

Run manually. Each has different hardware requirements — see docs/TESTING.md and docs/MUSICVISION_GPU_TEST.md.

**After LLM prompt changes:** run both `pytest tests/` and `scripts/test_vllm_prompts.py`.
**After GPU/model changes:** run `scripts/test_gpu_pipeline.py`.

## Key Constraints

- HuMo max output: 97 frames (~3.88s at 25fps). Scenes >4s are auto-split into sub-clips.
- Sub-clip continuity: last frame of sub-clip N becomes reference image for sub-clip N+1.
- FLUX and HuMo run sequentially (different stages), never simultaneously. Models fully unloaded between stages.
- GPU0 (5090) runs DiT/UNet. GPU1 (4080) runs encoders/VAE. This split is managed by `utils/gpu.py`.
- VRAM tier system: fp16, fp8_scaled, gguf_q8, gguf_q6, gguf_q4, preview. `recommend_tier()` auto-selects based on available VRAM.
- Upscaling: per-engine strategy — LTX-2 → LTX Spatial (latent-space), HVA/HuMo → SeedVR2 (pixel-space), preview → Real-ESRGAN or NONE. Assembly prefers upscaled clips.

## Project Data Flow

```
Song audio + lyrics
  → Stage 1: Intake (BPM, vocal sep, Whisper, LLM segmentation) → scenes.json
  → Stage 2: Imaging (LLM prompts, FLUX generation) → images/
  → Stage 3: Video (LLM prompts, HuMo/HVA/LTX generation) → clips/
  → Stage 4: Upscale (per-engine upscaler selection) → clips_upscaled/
  → Stage 5: Assembly (ffmpeg concat, audio sync) → output/rough_cut.mp4 + FCPXML/EDL
```

Each stage produces artifacts on disk. Any stage can be re-run independently. User reviews and approves at each stage.

## LLM Integration

`llm.py` provides a unified `LLMClient` that routes to Anthropic or OpenAI-compatible backends. Three LLM-assisted tasks:

1. **Scene segmentation** (`intake/segmentation.py`) — lyrics + timestamps → JSON scene list
2. **Image prompt generation** (`imaging/prompt_generator.py`) — scene → FLUX prompt (single + batch)
3. **Video prompt generation** (`video/prompt_generator.py`) — scene → HuMo TIA prompt

All three degrade gracefully: LLM → interactive terminal prompt → auto-template from style sheet. User overrides in scenes.json always win.

System prompts are defined as module-level constants (`SEGMENTATION_SYSTEM_PROMPT`, `IMAGE_PROMPT_SYSTEM`, `VIDEO_PROMPT_SYSTEM`). When editing these, always re-run both unit tests and `test_vllm_prompts.py`.

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
musicvision upscale --project ./test_output/2026-03-01_1400_my_test --resolution 1080p
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
