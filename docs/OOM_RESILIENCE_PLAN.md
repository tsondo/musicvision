# OOM Resilience & Multi-Engine Fallback Plan

**Date:** 2026-03-03
**Context:** First HVA preview run OOM'd on RTX 5090 (32GB) because preview mode wasn't sending low-quality settings — model ran at defaults requiring 40GB+. This plan addresses both the immediate bug and the broader architectural need for graceful degradation.

---

## Problem Statement

We have one working video engine (HunyuanVideo-Avatar) with one checkpoint (BF16, ~30GB). HuMo is deprioritized (noisy output, never properly denoised). If HVA can't fit in VRAM for a given scene, we currently have no fallback — the run crashes.

Resolution reduction below 320p is not viable (useless as preview). Resolution reduction in final-cut mode sacrifices quality. We need strategies beyond lowering resolution.

---

## Architecture: Per-Scene Engine Selection

**Core change:** Move from "one engine per run" to "best engine per scene based on capability budget."

Each scene already has a `Scene.video_engine` field and the pipeline already groups scenes by engine. Extend this so the UI presents only engines that can actually run on the current hardware, per scene.

### Scene capability matching

Different scenes have different requirements:

| Scene type | Needs lip sync? | Needs full-body motion? | Minimum engine |
|---|---|---|---|
| Character singing/talking | Yes | Yes | HVA (primary) or LivePortrait (fallback) |
| Character with body motion, no speech | No | Yes | HVA or lighter T2V model |
| Landscape / establishing shot | No | No | Lightweight T2V, or animated pan over FLUX image |
| Abstract / mood visuals | No | No | Any T2V model, lowest VRAM requirement |

The UI scene editor should:
1. Run a pre-flight VRAM check for each engine at each supported resolution
2. Populate the engine/resolution dropdown with only options that fit
3. Auto-select the best available option (highest quality that fits)
4. Allow user override within the feasible set

It is acceptable — and expected — for a single video to contain scenes rendered by different engines. A talking head close-up in one scene and a desert landscape in another is fine as long as the story flows. Post-production upscaling and color matching can harmonize visual differences.

---

## Pre-Flight VRAM Budget Check

Before any render begins, estimate memory requirements and validate against available VRAM. This runs once per scene configuration, not per frame.

### Implementation

```python
def estimate_vram_gb(engine: str, width: int, height: int, n_frames: int) -> float:
    """
    Empirical VRAM estimates based on GPU test results.
    
    Known data points (HVA on RTX 5090):
      - 320p / 129 frames / BF16 + block offload → 16.6 GB peak
      - 704p / 129 frames / BF16 + block offload → 31.9 GB peak
    
    Known data points (FLUX / Z-Image on RTX 5090):
      - Z-Image-Turbo 768x512 → 12.5 GB peak
      - FLUX-schnell 768x512 bf16 CPU offload → ~17 GB peak
    """
    # Per-engine empirical models — refine as we collect more data points
    ...
```

### Pre-flight flow

```
For each scene in project:
  1. Determine scene requirements (lip sync, motion, etc.)
  2. List candidate engines in quality order
  3. For each candidate:
     a. estimate_vram_gb(engine, resolution, frames)
     b. Compare against torch.cuda.mem_get_info() with 2GB headroom
     c. If fits → select this engine + settings, stop
  4. If nothing fits → flag scene as "cannot render" with explanation
  5. Lock settings for the entire run (no mid-run changes for THIS attempt)
```

### VRAM headroom

Always reserve ~2GB below the physical limit. CUDA fragmentation, kernel overhead, and display server (on workstations) consume memory that `mem_get_info` may not fully account for. The `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` environment variable should be set by default in engine startup to reduce fragmentation-induced OOMs.

---

## Graceful Mid-Run OOM Recovery

If a scene OOMs despite pre-flight checks (fragmentation, unexpected memory spike, etc.), recover without crashing the entire run.

### Recovery flow

```
try:
    result = engine.generate(scene_params)
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    gc.collect()
    
    log.warning("OOM on scene %s with engine=%s res=%s", 
                scene.id, scene.video_engine, scene.resolution)
    
    # Mark this scene as failed with the current settings
    scene.status = "oom_failed"
    scene.oom_context = {
        "engine": scene.video_engine,
        "resolution": scene.resolution,
        "vram_available": torch.cuda.mem_get_info()[0] / 1024**3,
    }
    
    # Continue to next scene — do NOT retry inline
    # Rationale: CUDA state after OOM can be unreliable.
    # Better to finish what we can, then offer re-run options.
    continue
```

### After the run completes with failures

1. Report which scenes failed and why
2. Suggest alternative settings for failed scenes (lower resolution, different engine, more aggressive offloading)
3. Let the user re-run only the failed scenes with adjusted settings
4. **Do not** automatically retry with different settings mid-run — CUDA state after OOM is unreliable, and mixing settings within an engine group can cause subtle device placement bugs

### Important: no mixed settings within a single attempt

When the user initiates a render, each scene's settings are locked. If scenes fail, the user gets a report and chooses how to re-render the failures. This avoids the complexity and fragility of mid-run parameter changes while still giving the user full control. Different scenes CAN use different engines/resolutions — that's chosen at planning time, not recovery time.

---

## Fallback Engine Candidates

We need at least one lightweight video engine that can produce acceptable results when HVA won't fit. These should be evaluated and integrated as additional `VideoEngineType` options.

### Tier 1: LivePortrait (recommended first integration)

**Why:** Shares the same input contract we already produce (reference image + audio segment). Portrait animation with lip sync at ~4-6GB VRAM. Fast inference.

- **Input:** Reference image (from Stage 2) + audio segment
- **Output:** Animated portrait video with lip sync
- **VRAM:** ~4-6 GB — fits on virtually any GPU
- **Limitation:** Portrait/upper-body only, no full-scene generation
- **Best for:** Character singing/talking scenes where HVA won't fit
- **License:** Check before integrating

**Integration path:** New `video/liveportrait_engine.py` implementing the same `VideoEngine` interface. Subprocess isolation like HVA if dependency conflicts exist.

### Tier 2: CogVideoX-5B

**Why:** Full scene generation (not just portraits), text+image conditioning, fits in 16GB with quantization.

- **Input:** Text prompt + reference image
- **Output:** Full video clip (no native audio conditioning)
- **VRAM:** ~10-16 GB depending on quantization
- **Limitation:** No audio-driven lip sync — suitable for non-speaking scenes
- **Best for:** Landscapes, establishing shots, abstract visuals
- **License:** Apache 2.0

### Tier 3: Animated image pan/zoom

**Why:** Zero additional VRAM beyond what FLUX/Z-Image already uses. Ken Burns effect over a still image.

- **Input:** Reference image from Stage 2
- **Output:** Slow pan/zoom video clip via ffmpeg
- **VRAM:** 0 GB (pure ffmpeg operation)
- **Limitation:** No motion, no lip sync — just camera movement over a still
- **Best for:** Establishing shots, transitions, mood scenes
- **Implementation:** Pure ffmpeg, no model needed. Could be done today.

### Priority order for integration

1. **Animated pan/zoom** — implement immediately, zero cost, useful for non-character scenes
2. **LivePortrait** — research and prototype, covers the critical "character singing but HVA won't fit" case
3. **CogVideoX** — evaluate if we need full-scene generation at lower VRAM than HVA

---

## Upscaling Pipeline (Post-Processing)

Generate everything at 320p for fast iteration, then batch-upscale approved scenes to 720p/1080p. This decouples creative review from final quality.

### Why this matters

- 320p preview at ~16.6 GB is safe on 32GB cards — this is our reliable floor
- Upscaling 320p → 720p requires far less VRAM than generating at 720p directly
- User reviews and approves at 320p (fast cycle), then kicks off upscale (slow, one-time)
- If 320p is the only resolution that fits, the result isn't a dead end

### Candidate upscalers

- **RealESRGAN** (4x) — well-tested, ~2-4 GB VRAM, fast. Good baseline.
- **TOPAZ Video AI** (commercial) — highest quality but proprietary, not suitable for pipeline integration
- **Real-ESRGAN + frame interpolation** — upscale + smooth motion in one pass

### Integration point

New Stage 4.5 between video generation and assembly, or as a post-assembly batch operation. Should be optional and per-scene (user may want some scenes at native resolution if they rendered at 704p).

---

## Immediate Action Items (for Claude Code)

### Bug fix (blocking — in progress)
1. Preview mode must send low-quality settings (320p, 10 steps, block offload ON) to HVA wrapper. Currently sending defaults that require 40GB+.

### OOM resilience (high priority)
2. Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` in engine startup (both HVA wrapper and HuMo engine)
3. Add pre-flight VRAM check before render — `estimate_vram_gb()` function with empirical data from our test results
4. Wrap engine.generate() in OOM try/except — log failure, mark scene, continue run
5. After run, report failed scenes with suggested alternative settings
6. Verify text encoders (LLaVA-LLaMA-3-8B + CLIP-L) are fully offloaded to CPU before HVA diffusion loop starts — add explicit `clear_vram()` + VRAM assertion between encoding and diffusion

### Engine dropdown (medium priority)
7. Per-scene engine selector in UI that only shows feasible options based on pre-flight check
8. `Scene.video_engine` already exists — extend to support new engine types as they're added

### Fallback engines (next phase)
9. Implement animated pan/zoom as `VideoEngineType.PAN_ZOOM` — pure ffmpeg, no model, works today
10. Research LivePortrait — VRAM requirements, license, output quality, integration complexity
11. Evaluate CogVideoX-5B as a full-scene fallback

### Upscaling (future)
12. Prototype RealESRGAN integration as optional post-processing step
13. Design UX: "approve at 320p → upscale to 720p" workflow

---

## Key Principles

- **No surprise degradation.** The user always knows what engine and resolution each scene will use before rendering starts.
- **Per-scene flexibility.** Different scenes can use different engines. A talking head and a desert landscape don't need the same pipeline.
- **Pre-flight over retry.** Catch problems before committing GPU time. Mid-run OOM recovery is a safety net, not the primary strategy.
- **320p is the floor, not a dead end.** With upscaling, a 320p render becomes the fast-iteration step in a two-stage pipeline.
- **Engines are pluggable.** New engines implement the `VideoEngine` interface and register as a `VideoEngineType`. The pipeline doesn't care what's inside — it just needs generate() to return frames.
