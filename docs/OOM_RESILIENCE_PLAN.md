# OOM Resilience — Retrospective & Remaining Concerns

**Original:** 2026-03-03 | **Updated:** 2026-03-11

---

## Original Problem

First HVA preview run OOM'd on RTX 5090 (32GB) because preview mode sent default settings requiring 40GB+. With only one working video engine at the time and no fallback, the run crashed with no recovery. Resolution reduction below 320p wasn't viable, so we needed strategies beyond lowering resolution: per-scene engine selection, pre-flight VRAM checks, mid-run OOM recovery, fallback engines, and upscaling to decouple creative review from final quality.

---

## What Was Implemented

### Per-scene engine selection — DONE

Three working video engines with per-scene selection in both CLI and GUI:

- **HunyuanVideo-Avatar** — primary, audio-driven lip sync, 5.16s max clips
- **LTX-Video 2** — cinematic/instrumental scenes, 10.71s max clips (replaced CogVideoX in the plan)
- **HuMo** — audio-reactive, 24 bugs fixed, 3.88s max clips

`Scene.video_engine` field allows mixing engines within a single project. The API groups scenes by engine and processes each group sequentially. The UI has per-scene engine dropdowns.

### Preview/final render quality presets — DONE

Per-engine resolution and step-count presets. Preview mode (320p/10 steps) fits comfortably on 32GB. The original blocking bug (HVA receiving default settings instead of preview settings) was fixed.

### Mid-run OOM recovery — DONE

Both `api/app.py` and `cli.py` wrap generation in OOM-aware try/except:

- `is_oom_error()` detects CUDA OOM from direct exceptions and subprocess stderr
- `_oom_suggestion()` provides human-readable recovery advice per engine
- Consecutive OOM counter: after 2 consecutive OOMs in the same engine group, remaining scenes in that group are skipped (CUDA state is unreliable after OOM)
- Failed scenes are reported with suggested alternative settings
- CLI prints a re-run command for failed scene IDs
- HVA engine detects server OOM and skips futile wrapper fallback

### Pre-flight VRAM budget check — PARTIAL

`estimate_vram_gb()` exists in `utils/gpu.py` with empirical models for HVA and LTX-Video 2. Both CLI and API call it before render and warn if estimated VRAM exceeds available. However, the UI does not yet filter engine/resolution dropdowns to only show feasible options — it warns but doesn't prevent infeasible selections.

### PYTORCH_CUDA_ALLOC_CONF — DONE

Set to `max_split_size_mb:512` in:
- HVA engine startup (both direct and wrapper modes)
- HVA server script
- SeedVR2 engine subprocess
- `.env.example` documents `expandable_segments:True` as the recommended default

### Text encoder offload — DONE

HuMo engine has `_should_offload()` that checks free VRAM on the encoder GPU and offloads to CPU when < 2GB free. Encoders are offloaded after use to maximize VRAM headroom for the DiT. HVA uses `apply_group_offloading(block_level)` with CPU offload. LTX-Video 2 uses sequential CPU offload via diffusers.

### Upscaling pipeline — DONE

Originally listed as "future" — now fully implemented as Stage 4 with three engines:

- **SeedVR2** — subprocess bridge, for HVA/HuMo pixel-space upscaling
- **LTX Spatial** — in-process diffusers, for LTX-Video 2 latent-space upscaling
- **Real-ESRGAN** — frame-by-frame, lightweight fallback

Auto-selection per video engine. Pipeline orchestrator groups by engine, loads, processes, unloads. Assembly prefers upscaled clips. Full details in PIPELINE_SPEC.md.

### OOM resilience tests — DONE

`tests/test_oom_resilience.py` covers `is_oom_error()`, `_oom_suggestion()`, and `estimate_vram_gb()` with various exception types and engine configurations.

---

## Remaining Concerns

### Pre-flight VRAM filtering in UI — NOT DONE

The UI engine/resolution dropdowns show all options regardless of available VRAM. The plan called for filtering to only feasible options with auto-selection of the best available. Currently the pre-flight check only warns (CLI prints a warning, API logs it) but does not block.

### estimate_vram_gb() coverage gaps

Only HVA and LTX-Video 2 have empirical VRAM models. HuMo returns 0.0 (unknown). As more data points are collected, the estimates should be refined.

### Animated pan/zoom engine — NOT IMPLEMENTED

Pure ffmpeg Ken Burns effect over a still image. Zero VRAM, useful for establishing shots and transitions. Was listed as "implement immediately" but never built. Would be a `VideoEngineType.PAN_ZOOM` with no model loading — just ffmpeg filter graphs.

### LivePortrait integration — NOT IMPLEMENTED

Lightweight portrait animation (~4-6GB VRAM) for the "character singing but primary engine won't fit" fallback case. Has not been researched beyond initial notes. Lower priority now that three engines cover most scene types.

---

## Key Principles

These remain valid and should guide future OOM-related work:

- **No surprise degradation.** The user always knows what engine and resolution each scene will use before rendering starts.
- **Per-scene flexibility.** Different scenes can use different engines. A talking head and a desert landscape don't need the same pipeline.
- **Pre-flight over retry.** Catch problems before committing GPU time. Mid-run OOM recovery is a safety net, not the primary strategy.
- **320p is the floor, not a dead end.** With upscaling, a 320p render becomes the fast-iteration step in a two-stage pipeline.
- **Engines are pluggable.** New engines implement the `VideoEngine` interface and register as a `VideoEngineType`. The pipeline doesn't care what's inside — it just needs `generate()` to return frames.
