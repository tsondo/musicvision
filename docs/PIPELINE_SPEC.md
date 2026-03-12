# MusicVision Pipeline Specification

## Overview

MusicVision is a music video production pipeline that combines open-source AI tools to turn a song (audio + lyrics) into a complete music video. It wraps multiple video generation backends (HuMo, LTX-Video 2) and FLUX for reference image generation into an iterative, user-controlled workflow.

> **Note:** HunyuanVideo-Avatar (HVA) was previously supported as a third video engine but was deprecated and removed in commit 35cda2a (2026-03-11). References to HVA in historical context are preserved for documentation purposes.

### Core Principle: Segment for Video, Assemble with Original Audio

The pipeline segments audio for **video generation only** — each scene's sliced audio drives lip sync and motion in the video engine. During **final assembly**, the original uncut audio track is muxed over the concatenated video, ensuring seamless audio flow with no splice artifacts. This means:

- Scene audio segments are a **generation tool**, not a final output.
- The total video duration must match the original audio duration **exactly** (within one frame).
- Any drift in sub-clip boundaries accumulates and breaks the final sync. Frame-accurate math is mandatory.

## System Requirements

### Inference Workstation (image + video generation)
- **Primary GPU**: NVIDIA RTX 5090 (32 GB VRAM) — runs DiT for FLUX, HuMo, or LTX-Video 2
- **Secondary GPU**: NVIDIA RTX 4080 (16 GB VRAM) — offloads text encoders (T5-XXL, CLIP), VAE, Whisper, audio separator
- **Multi-GPU Strategy**: Proven in ComfyUI. DiT on the 5090, everything else on the 4080. The GPU with the most VRAM is automatically selected as primary by `gpu.py`, regardless of CUDA index (the 4080 is `cuda:0`, the 5090 is `cuda:1` — this does not affect pipeline behavior).
- **Model Swapping**: FLUX and video engines run in different pipeline stages (not simultaneously). Within each stage, model components are split across both GPUs. Weights fully unloaded between stages.

### Local LLM Server (optional, for prompt generation)
- **GPU**: NVIDIA RTX 3090 Ti (24GB VRAM) — runs vLLM for local LLM inference
- **Use case**: Scene segmentation and prompt generation without Claude API dependency
- **Recommended models**: Qwen2.5-32B-AWQ (~18 GB) or Mistral-Small-3.1-24B-Instruct (~12 GB 4-bit)
- **Not required**: Claude API (default) or auto-template fallback work without this machine

### General
- **Storage**: ~50 GB for model weights (FLUX + video engine + Whisper + VAE + audio separator). Additional space for project assets.
- **Python**: 3.11+ (HuMo requirement)
- **CUDA**: 12.8+ with PyTorch 2.10.0+cu128 (required for RTX 5090 sm_120/Blackwell support)
- **Key dependency pins**: `torch==2.10.0`; `flash_attn` may be dropped in favor of native SDPA

## Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                            MusicVision                               │
├──────────────┬───────────────┬────────────────────┬──────────────────┬──────────────────┤
│  Stage 1     │  Stage 2      │  Stage 3           │  Stage 4         │  Stage 5         │
│  INTAKE &    │  IMAGE GEN &  │  VIDEO GEN         │  UPSCALE         │  ASSEMBLY &      │
│  SEGMENTATION│  STORYBOARD   │  (selectable)      │  (per-engine)    │  EXPORT          │
├──────────────┼───────────────┼────────────────────┼──────────────────┼──────────────────┤
│ Whisper      │ FLUX          │ HuMo TIA           │ LTX Spatial      │ ffmpeg           │
│ ffmpeg       │ Z-Image       │ LTX-Video 2        │ SeedVR2          │ FCPXML/EDL gen   │
│ LLM (Claude) │ (LoRA)        │                    │ Real-ESRGAN      │                  │
└──────────────┴───────────────┴────────────────────┴──────────────────┴──────────────────┘
```

### Video Engine Registry

Each video engine has fixed frame constraints that govern the entire pipeline:

| Engine | Max Frames | FPS | Max Duration | Min Frames | Notes |
|--------|-----------|-----|-------------|-----------|-------|
| HuMo 17B | 97 | 25 | 3.88s | 25 (1.0s) | TIA mode (text+image+audio) |
| HuMo 1.7B | 97 | 25 | 3.88s | 25 (1.0s) | Preview/iteration |
| LTX-Video 2 | 257 | 24 | 10.71s | 9 (0.375s) | Joint audio+video generation |

**The active engine's constraints are injected into segmentation and sub-clip splitting.** The pipeline must never assume a fixed max duration — always read from the engine config.

```python
# Engine constraint constants — single source of truth
ENGINE_CONSTRAINTS = {
    "humo":              {"max_frames": 97,  "fps": 25, "min_frames": 25},
    "ltx_video":         {"max_frames": 257, "fps": 24, "min_frames": 9},
}
```

### GPU Memory Map (Inference Workstation)

```
GPU0 — RTX 5090 (32 GB)          GPU1 — RTX 4080 (16 GB)
┌─────────────────────────┐       ┌─────────────────────────┐
│ Stage 2: FLUX DiT       │       │ T5 text encoder  ~10 GB │
│   bf16: ~24 GB          │       │ VAE              ~0.4 GB│
│   FP8:  ~12 GB          │       │ Whisper          ~1.5 GB│
├─────────────────────────┤       │ Audio separator  ~0.5 GB│
│ Stage 3 (one at a time):│       │                         │
│  HuMo 17B               │       │ Total: ~12-13 GB        │
│                         │       │ Headroom: ~3-4 GB       │
│                         │       └─────────────────────────┘
│   fp8:  ~18 GB          │
│   gguf: 11–18.5 GB      │
│  HuMo 1.7B (preview)    │
│   fp16: ~3.4 GB         │
└─────────────────────────┘
(Only one stage loaded at a time)
```

## Frame-Accurate Alignment System

### The Alignment Problem

The segmentation stage produces scenes with arbitrary durations (2-10s). The video engine has a hard maximum clip length. Without careful alignment, three failure modes emerge:

1. **Audio/video length mismatch** — A 9s scene produces 2-3 sub-clips of video, but audio is one 9s chunk. Sub-clip audio must be sliced frame-accurately after sub-clip frame counts are computed.
2. **Tiny remainder sub-clips** — A 5.2s scene with a 3.88s engine max produces sub-clip A (3.88s) + sub-clip B (1.32s). The 1.32s clip may be too short for meaningful content or below the engine's minimum frame count.
3. **Accumulated drift** — Floating-point second-based math drifts across scenes. Over 30+ scenes, rounding errors can shift the final video by multiple frames relative to the audio.

### Rule: Frames First, Seconds Derived

**All duration math uses integer frame counts.** Seconds are derived only for display and audio slicing. This eliminates floating-point drift.

```python
FPS = 25  # or engine-specific

def scene_frames(time_start: float, time_end: float, fps: int) -> int:
    """Total frames for this scene. This is the authoritative duration."""
    return round((time_end - time_start) * fps)

def frames_to_seconds(frames: int, fps: int) -> float:
    """Derive seconds from frame count — never the other way around."""
    return frames / fps
```

### Sub-Clip Frame Splitting

When a scene exceeds the engine's max frames, it is split into sub-clips. The split algorithm guarantees:

1. `sum(sub_clip_frames) == total_scene_frames` (exact, no drift)
2. No sub-clip falls below `MIN_SUBCLIP_FRAMES`
3. Sub-clips are as equal-length as possible to avoid one dominant clip and one tiny clip

```python
MIN_SUBCLIP_FRAMES = 38  # ~1.5s at 25fps — below this, content is meaningless

def compute_subclip_frames(total_frames: int, max_frames: int, min_frames: int) -> list[int]:
    """
    Divide total_frames into sub-clips respecting engine constraints.
    
    Returns a list of frame counts. sum(result) == total_frames always.
    
    Strategy:
    - If total fits in one clip, return [total_frames]
    - Otherwise compute n = ceil(total / max). If remainder < min_frames,
      reduce n by 1 and redistribute evenly.
    - Equal distribution: each clip gets total // n, first (total % n) clips
      get one extra frame.
    """
    if total_frames <= max_frames:
        return [total_frames]
    
    n = math.ceil(total_frames / max_frames)
    remainder = total_frames - (n - 1) * max_frames
    
    # If remainder is too short, use fewer clips with equal distribution
    if remainder < min_frames:
        n -= 1
        if n == 0:
            return [total_frames]
    
    # Equal distribution across n clips
    base = total_frames // n
    extra = total_frames % n
    
    # First 'extra' clips get base+1 frames, rest get base
    counts = [base + 1] * extra + [base] * (n - extra)
    
    assert sum(counts) == total_frames, \
        f"Frame count mismatch: {sum(counts)} != {total_frames}"
    assert all(c >= min_frames for c in counts), \
        f"Sub-clip below minimum: {min(counts)} < {min_frames}"
    assert all(c <= max_frames for c in counts), \
        f"Sub-clip above maximum: {max(counts)} > {max_frames}"
    
    return counts
```

**Example (HuMo, max=97 frames, min=25 frames):**

```
Scene: 200 frames (8.0s)
  ceil(200/97) = 3 sub-clips
  remainder = 200 - 2*97 = 6 frames → below min (25)
  Reduce to n=2: 200/2 = 100 each → exceeds max (97)
  So use n=3 with equal distribution: 200//3 = 66, 200%3 = 2
  Result: [67, 67, 66] — all within [25, 97] ✓
```

**Example (LTX-Video 2, max=257 frames, min=9 frames):**

```
Scene: 500 frames (20.83s)
  ceil(500/257) = 2 sub-clips
  remainder = 500 - 257 = 243 frames → above min (9) ✓
  Result: [250, 250] — equal distribution, all within [9, 257] ✓
```

### Sub-Clip Audio Slicing

Audio segments for sub-clips are derived **from the sub-clip frame counts**, not from arbitrary time divisions:

```python
def slice_subclip_audio(
    scene_audio: Path,
    scene_id: str,
    subclip_frames: list[int],
    fps: int,
    output_dir: Path,
) -> list[Path]:
    """
    Slice a scene's audio segment into sub-clip audio files.
    
    Timing is derived from frame counts to maintain frame-accuracy.
    Called AFTER compute_subclip_frames(), BEFORE video generation.
    """
    paths = []
    cursor_frames = 0
    
    for i, n_frames in enumerate(subclip_frames):
        start_sec = cursor_frames / fps
        end_sec = (cursor_frames + n_frames) / fps
        
        sub_audio = output_dir / f"{scene_id}_sub_{i:02d}.wav"
        slice_audio(scene_audio, sub_audio, start_sec, end_sec)
        paths.append(sub_audio)
        
        cursor_frames += n_frames
    
    return paths
```

### Assembly Duration Assertion

The rough cut assembler **must** verify that total video duration matches the original audio:

```python
def assemble_rough_cut(scenes, paths, original_audio, approved_only=True):
    # ... concatenate video clips ...
    
    video_duration = get_video_duration(rough_cut_path)
    audio_duration = get_audio_duration(original_audio)
    fps = get_engine_fps()
    
    tolerance = 1.0 / fps  # one frame
    drift = abs(video_duration - audio_duration)
    
    if drift > tolerance:
        log.error(
            "SYNC FAILURE: video=%.4fs, audio=%.4fs, drift=%.4fs (%.1f frames)",
            video_duration, audio_duration, drift, drift * fps,
        )
        raise SyncError(
            f"Video/audio duration mismatch: {drift:.4f}s "
            f"({drift * fps:.1f} frames). Check sub-clip frame math."
        )
    
    # Mux original uncut audio over the assembled video
    mux_video_audio(rough_cut_path, original_audio, final_output_path)
```

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

   Input: lyrics with timestamps, song structure (verse/chorus/bridge/instrumental)

   **Engine-Aware Constraints** (injected into LLM prompt and rule-based splitter):
   - Minimum scene duration: `min_frames / fps` seconds (engine-dependent)
   - Maximum scene duration: 10 seconds (pipeline limit, not engine limit — sub-clips handle the rest)
   - Preferred scene durations: clean multiples of `max_frames / fps`, or shorter
   - Avoid scene durations that produce remainder sub-clips below `min_frames`
   - Prefer cuts on musical phrase boundaries (beat-snapped within 0.15s tolerance)
   - Instrumental sections get their own scenes (type: "instrumental")
   - Scenes must NOT cross section boundaries

   **Post-LLM Validation & Adjustment** (`_validate_and_adjust_scenes()`):

   After the LLM returns scene boundaries, a post-processing pass enforces hard constraints:

   ```python
   def _validate_and_adjust_scenes(
       scenes: list[Scene],
       song_duration: float,
       engine_config: EngineConstraints,
       beat_times: list[float] | None = None,
   ) -> list[Scene]:
       """
       Post-process LLM segmentation to enforce frame-accurate constraints.
       
       1. Convert all boundaries to frame numbers
       2. Snap to beat times if available (within tolerance)
       3. Check each scene's sub-clip remainder — if the last sub-clip would
          be below min_frames, adjust the scene boundary:
          a. Try shrinking (push remainder to next scene)
          b. Try growing (absorb from next scene)
          c. Pick whichever lands closer to a beat boundary
       4. Verify first scene starts at frame 0, last ends at total_frames
       5. Verify no gaps or overlaps between consecutive scenes
       """
   ```

   **Segmentation System Prompt** (updated excerpt):

   ```
   Video engine constraints:
   - Engine: {engine_name}
   - Max clip: {max_clip_seconds}s ({max_frames} frames @ {fps}fps)
   - Min clip: {min_clip_seconds}s ({min_frames} frames @ {fps}fps)
   
   Duration guidance:
   - Preferred scene durations: ≤{max_clip_seconds}s (single clip) or
     multiples of {max_clip_seconds}s (clean sub-clip split).
   - Avoid durations that leave a remainder under {min_clip_seconds}s
     when divided by {max_clip_seconds}s.
     Example: {bad_duration_example}s is problematic
     ({max_clip_seconds} + {bad_remainder}s). Prefer {good_duration_short}s
     or {good_duration_long}s instead.
   ```

   Output: `scenes.json` with timestamps and types.

   LLM fallback: `segment_scenes_simple()` rule-based segmenter (no LLM required).

4. **Audio Slicing**
   - ffmpeg splits the full audio into per-scene WAV segments
   - Each segment saved as WAV (pcm_s16le) at original sample rate — sample-accurate
   - Vocal-separated versions also sliced (for Whisper only — see critical note below)
   - **Sub-clip audio is NOT sliced at this stage** — it happens in Stage 3 after `compute_subclip_frames()` determines the exact frame counts

> **⚠️ Critical: The video engine receives the full audio mix, not isolated vocals.** The vocal stem is used *only* to improve Whisper transcription accuracy. Video engines were trained on mixed audio — feeding isolated vocals degrades A/V sync. Audio segments in `segments/` are always full-mix; `segments_vocal/` is consumed only by the transcription step.

### Outputs
- `project.yaml` (initial)
- `scenes.json` (scene list with timestamps + frame counts)
- `segments/` — per-scene WAV clips (full mix — fed to video engine)
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

3. **Storyboard Review**
   - React storyboard with scene grid, preview panel, per-scene approval and regeneration.
   - API endpoints for scene CRUD and approval (`PATCH /api/scenes/{id}`, `POST /api/scenes/approve-all`)

### Outputs
- Updated `scenes.json` with image prompts and status
- `images/` directory with reference images (`images/scene_001.png`, etc.)

## Stage 3: Video Generation

### Video Engine Selection

The active video engine is set in `project.yaml` → `video_engine`. All engines follow the same interface: text prompt + reference image + audio segment → video clip.

| Engine | Strengths | When to Use |
|--------|----------|-------------|
| HuMo TIA | Audio-conditioned with dual CFG, character preservation, lip sync | Default for most projects |
| LTX-Video 2 | Joint audio+video generation, cinematic | Cinematic/instrumental scenes, long clips (up to 10.7s) |

### Process

1. **Video Prompt Generation** (LLM-assisted)
   - Generate dense descriptive captions for each scene
   - More detailed than image prompt — describes motion, camera movement, expressions
   - Same LLM-unavailable fallback as image prompts
   - Auto-template anchors on image prompt, style sheet, and scene type

2. **Sub-Clip Planning** (frame-accurate)

   Before any video generation begins, compute the exact sub-clip frame counts for every scene:

   ```python
   for scene in scenes:
       total_frames = scene_frames(scene.time_start, scene.time_end, engine.fps)
       scene.subclip_frame_counts = compute_subclip_frames(
           total_frames, engine.max_frames, engine.min_frames
       )
       scene.generation_audio_segments = slice_subclip_audio(
           scene_audio=paths.segment_path(scene.id),
           scene_id=scene.id,
           subclip_frames=scene.subclip_frame_counts,
           fps=engine.fps,
           output_dir=paths.sub_segment_dir,
       )
   ```

   This is computed **once** and saved to `scenes.json`. Video generation reads it — never recomputes.

3. **Video Inference** (per sub-clip)
   - Input per sub-clip: text prompt + reference image + audio segment
   - For sub-clip 0: reference image is the scene's FLUX-generated image
   - For sub-clip N (N>0): reference image is the last frame of sub-clip N-1 (continuity chaining)
   - Seed: always resolved (random if not provided), recorded in output
   - Config from `project.yaml` (engine-specific section)

4. **Scene Review**
   - Scene review via React storyboard.
   - API endpoints for per-scene video status and approval.

### Outputs
- Updated `scenes.json` with video prompts, sub-clip frame counts, and status
- `clips/scene_001.mp4`, `clips/scene_002.mp4`, etc.
- `clips/sub/scene_003_a.mp4`, `clips/sub/scene_003_b.mp4` (sub-clips for long scenes)
- `clips/sub/scene_003_a_lastframe.png` (continuity frames)

## Stage 4: Video Upscaling

### Purpose

Video clips from different engines come out at different resolutions (LTX-2: 768×512, HuMo: 1280×720 — or lower in preview mode). The upscaling stage enhances quality and normalizes resolution before assembly.

### Per-Engine Upscaler Strategy

| Video Engine | Default Upscaler | Type | VRAM |
|---|---|---|---|
| LTX-Video 2 | LTX Spatial Upsampler | Latent-space, temporally aware | ~12 GB |
| HuMo | SeedVR2 | Pixel-space one-step diffusion | ~16 GB (FP8) |
| (preview mode) | NONE or Real-ESRGAN | Frame-by-frame / skip | ~2-4 GB |

### Process

1. **Filter** to scenes with video clips (`video_clip` or `sub_clips`)
2. **Group** scenes by video engine → select upscaler per group
3. For each group: create upscaler → `load()` → process clips → `unload()`
4. **Sub-clips**: upscale each sub-clip individually, then join
5. Store results in `clips_upscaled/`, update `scene.upscaled_clip`

### Output Artifacts

- `clips_upscaled/scene_001.mp4` — upscaled scene clips
- `clips_upscaled/sub/scene_003_a.mp4` — upscaled sub-clips
- `clips_upscaled/scene_003_joined.mp4` — joined upscaled sub-clips

### Configuration

`UpscalerConfig` in `project.yaml`:
- `target_resolution`: `720p` | `1080p` (default) | `1440p` | `4k`
- `upscaler_override`: force specific upscaler for all engines
- `preview_upscaler`: upscaler used in preview mode (default: `none`)
- Engine-specific settings: model IDs, FP8 flags, inference steps

---

## Stage 5: Assembly & Export

### Process

1. **Concatenation** (`assemble_rough_cut()`)
   - Sorts scenes by `order`, joins sub-clips within each scene first
   - **Prefers upscaled clips** — falls back to raw clips if upscaled version missing
   - ffmpeg concat demuxer (no re-encode)
   - **Duration assertion**: verifies total video duration matches audio within 1 frame (see Frame-Accurate Alignment System)
   - Muxes original full-song audio (uncut) back over the assembled silent video
   - The generation audio segments are discarded at this point — they served their purpose

2. **Export Formats**
   - **Rough Cut**: `output/rough_cut.mp4` — full song, all scenes assembled, original audio
   - **EDL**: `output/timeline.edl` — CMX 3600 format, DaVinci Resolve compatible
   - **FCPXML**: `output/timeline.fcpxml` — FCPXML 1.10, DaVinci Resolve 18+ / Final Cut Pro

3. **Individual Scene Exports**
   - Each scene clip retains its generation audio (for lip sync preview/QA)
   - These are NOT used in the final rough cut

### Outputs
- `output/rough_cut.mp4`
- `output/timeline.edl`
- `output/timeline.fcpxml`

## Data Model

### Project Config (`project.yaml`)

```yaml
song:
  audio_file: "input/song.wav"
  lyrics_file: "input/lyrics.txt"
  bpm: null                     # Auto-detected if null
  acestep:
    json_file: null             # AceStep companion JSON
    caption: null
    lyrics: null

video_engine: "humo"            # humo | ltx_video

humo:
  tier: "fp8_scaled"
  resolution: "544p"
  scale_a: 5.5
  scale_t: 5.0
  shift: 5.0
  sampler: "uni_pc"
  denoising_steps: 50
  block_swap_count: 0
  sub_clip_continuity: true
  lora: null
  seed: null

image_gen:
  model: "flux-dev"
  quant: "auto"
  steps: null
  guidance_scale: 3.5
  lora_path: null
  lora_weight: 0.8

vocal_separation:
  method: "demucs"
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
      "frame_start": 0,
      "frame_end": 97,
      "total_frames": 97,
      "type": "vocal",
      "section": "verse_1",
      "lyrics": "Standing in the rain tonight",

      "audio_segment": "segments/scene_001.wav",
      "audio_segment_vocal": "segments_vocal/scene_001_vocal.wav",

      "subclip_frame_counts": [97],
      "generation_audio_segments": ["segments/scene_001.wav"],

      "image_prompt": "A young woman with short black hair stands on a dimly lit stage...",
      "image_prompt_user_override": null,
      "reference_image": "images/scene_001.png",
      "image_status": "pending",

      "video_prompt": "A close-up of a young woman with short black hair singing...",
      "video_prompt_user_override": null,
      "video_clip": "clips/scene_001.mp4",
      "video_status": "pending",
      "video_engine": "humo",              // humo | ltx_video

      "sub_clips": [],

      "characters": ["singer"],
      "props": ["vintage_mic"],
      "settings": ["stage"],

      "notes": ""
    },
    {
      "id": "scene_005",
      "order": 5,
      "time_start": 15.0,
      "time_end": 23.0,
      "frame_start": 375,
      "frame_end": 575,
      "total_frames": 200,
      "type": "vocal",
      "section": "chorus",
      "lyrics": "We could be forever young, dancing under neon lights...",

      "audio_segment": "segments/scene_005.wav",
      "audio_segment_vocal": "segments_vocal/scene_005_vocal.wav",

      "subclip_frame_counts": [67, 67, 66],
      "generation_audio_segments": [
        "segments/sub/scene_005_sub_00.wav",
        "segments/sub/scene_005_sub_01.wav",
        "segments/sub/scene_005_sub_02.wav"
      ],

      "image_prompt": "...",
      "image_prompt_user_override": null,
      "reference_image": "images/scene_005.png",
      "image_status": "approved",

      "video_prompt": "...",
      "video_prompt_user_override": null,
      "video_clip": null,
      "video_status": "pending",
      "video_engine": "humo",              // humo | ltx_video

      "sub_clips": [
        "clips/sub/scene_005_a.mp4",
        "clips/sub/scene_005_b.mp4",
        "clips/sub/scene_005_c.mp4"
      ],

      "characters": ["singer", "dancer"],
      "props": [],
      "settings": ["neon_club"],

      "notes": "3 sub-clips, equal distribution (67+67+66=200 frames)"
    }
  ]
}
```

### Key Design Decision: Sub-clips for Long Scenes

Each video engine has a maximum clip duration. Scenes exceeding it are automatically split into sub-clips using frame-count-first math (see Frame-Accurate Alignment System above).

**Sub-clip continuity**: last frame of sub-clip N is extracted via ffmpeg and used as the reference image for sub-clip N+1. This prevents visual discontinuity within a long scene but requires sequential generation. Disable via engine config `sub_clip_continuity: false`.

**Sub-clip audio**: sliced from the scene's full-mix audio segment using frame-derived timestamps. Stored in `segments/sub/`. These are generation-only artifacts — assembly uses the original uncut audio.

## Audio Pipeline Clarification

Three distinct audio paths exist in the pipeline. Conflating them causes bugs.

```
Original Song Audio (input/song.wav)
  │
  ├─→ Vocal Separation → vocal stem → Whisper transcription (ONLY)
  │                       └─→ segments_vocal/scene_XXX_vocal.wav
  │
  ├─→ Per-Scene Slicing → segments/scene_XXX.wav (full mix)
  │                         │
  │                         └─→ Per-Sub-Clip Slicing (Stage 3)
  │                              └─→ segments/sub/scene_XXX_sub_NN.wav
  │                                   └─→ Fed to video engine for generation
  │
  └─→ Final Assembly → muxed directly over concatenated video (UNCUT)
       This is the ONLY audio in the final output.
```

**Rules:**
- Video engines always receive **full-mix audio**, never isolated vocals.
- Sub-clip audio boundaries are derived from **frame counts**, never arbitrary time splits.
- The final rough cut uses the **original uncut song file**, not reassembled segments.
- Individual scene clips keep their generation audio for QA preview only.

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
├── segments/                  # Full mix — fed to video engine
│   ├── scene_001.wav
│   ├── scene_005.wav
│   └── sub/                   # Sub-clip audio (generation only)
│       ├── scene_005_sub_00.wav
│       ├── scene_005_sub_01.wav
│       └── scene_005_sub_02.wav
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
│       ├── scene_005_a.mp4
│       ├── scene_005_a_lastframe.png   ← continuity frame for sub-clip chaining
│       ├── scene_005_b.mp4
│       ├── scene_005_b_lastframe.png
│       └── scene_005_c.mp4
└── output/
    ├── rough_cut.mp4           ← original uncut audio muxed here
    ├── timeline.edl
    └── timeline.fcpxml
```

## Module Structure (Python)

```
src/musicvision/
├── __init__.py
├── cli.py                    # CLI entry point
├── api/app.py                # FastAPI REST API
├── llm.py                    # Unified LLM client (Anthropic + OpenAI/vLLM)
├── models.py                 # Pydantic v2 data models
├── engine_registry.py        # Engine constraints, frame math, sub-clip computation
├── intake/
│   ├── audio_analysis.py     # BPM detection, vocal separation
│   ├── transcription.py      # Whisper large-v3 transcription + alignment
│   ├── segmentation.py       # LLM-assisted scene segmentation + post-processing
│   └── pipeline.py           # Intake orchestrator
├── imaging/
│   ├── prompt_generator.py   # LLM-assisted FLUX image prompts
│   ├── flux_engine.py        # FLUX inference wrapper
│   └── storyboard.py         # Storyboard management
├── video/
│   ├── prompt_generator.py        # LLM-assisted video prompts
│   ├── base.py                    # Abstract video engine interface
│   ├── factory.py                 # Engine factory dispatch
│   ├── humo_engine.py             # HuMo TIA inference wrapper
│   └── ltx_video_engine.py        # LTX-Video 2 inference wrapper
├── upscaling/
│   ├── base.py                    # UpscaleEngine ABC + dataclasses
│   ├── factory.py                 # Dispatch on UpscalerType → engine
│   ├── pipeline.py                # Orchestrator: group by engine, upscale, update scenes
│   ├── realesrgan_engine.py       # Real-ESRGAN frame-by-frame upscaler
│   ├── seedvr2_engine.py          # SeedVR2 subprocess bridge
│   └── ltx_spatial_engine.py      # LTX Spatial Upsampler (diffusers in-process)
├── assembly/
│   ├── concatenator.py       # ffmpeg clip joining + audio sync + duration assertion
│   ├── exporter.py           # FCPXML/EDL generation
│   └── timecode.py           # Timecode utilities
├── vendor/                   # Vendored/patched model code
└── utils/
    ├── audio.py              # ffmpeg wrappers, slice, mux, duration
    ├── gpu.py                # Device map, VRAM tier, recommend_tier()
    └── paths.py              # ProjectPaths helper
```

### Module: `engine_registry.py`

Single source of truth for engine constraints and frame math:

```python
"""
Engine constraints and frame-accurate sub-clip computation.

This module is the ONLY place engine max/min frames, FPS, and sub-clip
math are defined. All other modules import from here.
"""

import math
from dataclasses import dataclass

@dataclass(frozen=True)
class EngineConstraints:
    name: str
    max_frames: int
    min_frames: int
    fps: int
    
    @property
    def max_seconds(self) -> float:
        return self.max_frames / self.fps
    
    @property
    def min_seconds(self) -> float:
        return self.min_frames / self.fps

ENGINES = {
    "humo":             EngineConstraints("HuMo",                max_frames=97,  min_frames=25, fps=25),
    "ltx_video":        EngineConstraints("LTX-Video 2",        max_frames=257, min_frames=9,  fps=24),
}

def get_engine(name: str) -> EngineConstraints:
    if name not in ENGINES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(ENGINES.keys())}")
    return ENGINES[name]

def scene_frames(time_start: float, time_end: float, fps: int) -> int:
    return round((time_end - time_start) * fps)

def compute_subclip_frames(
    total_frames: int,
    max_frames: int,
    min_frames: int,
) -> list[int]:
    """See Frame-Accurate Alignment System section for algorithm."""
    if total_frames <= max_frames:
        return [total_frames]
    
    n = math.ceil(total_frames / max_frames)
    remainder = total_frames - (n - 1) * max_frames
    
    if remainder < min_frames:
        n -= 1
        if n == 0:
            return [total_frames]
    
    base = total_frames // n
    extra = total_frames % n
    counts = [base + 1] * extra + [base] * (n - extra)
    
    assert sum(counts) == total_frames
    assert all(c >= min_frames for c in counts)
    assert all(c <= max_frames for c in counts)
    
    return counts
```

## REST API Endpoints

*(No changes to existing API endpoints. All 25+ endpoints remain as-is.)*

## CLI Commands

*(No changes to existing CLI. The `--video-engine` flag selects the active engine.)*

## UI — React Storyboard

A React + Vite frontend is implemented for scene review and approval. It communicates with the FastAPI backend via proxied `/api` and `/files` routes.

**Implemented features:**
- Scene grid with thumbnails showing reference images and video clip status
- Preview panel for viewing generated images and video clips
- Per-scene prompt editing (image and video prompts)
- Per-scene approval and regeneration controls
- Project header with song info, BPM, duration, and engine selection
- Dark theme, plain CSS + React 19, no state management library

**Future UI work (not yet built):**
- Waveform display with scene boundary editing
- Drag-to-reorder scenes
- Assembly & export controls
- Progress feedback for long-running generation jobs (SSE/WebSocket)

## Open Questions / Future Work

- **Progress feedback**: No SSE/WebSocket for long-running jobs. API endpoints are synchronous.
- **Partial failure recovery**: Exception propagates on failure; already-generated clips survive. Needs a proper job/resume model.
- **Transitions**: Currently hard cuts only. Future: AI-generated transitions, crossfades.
- **Batch rendering**: Scenes generate sequentially. Future: multi-GPU or cloud parallelism.
- **LoRA training**: Pipeline accepts LoRA paths but doesn't include training workflows.
- **Future engine integrations**: HunyuanVideo 1.5 (step-distilled fast mode) is a candidate for future integration.
- **Engine hot-swap**: Per-scene engine selection is implemented. Future: UI for per-scene engine assignment.
- **Render time estimation**: No upfront time estimate for users.
