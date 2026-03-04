# MusicVision — Lip Sync Post-Processing Spec

**Date:** 2026-03-04
**Status:** Proposal (not yet implemented)

---

## Overview

Lip sync post-processing enables singing scenes in video engines that lack native audio conditioning (LTX-2, future engines). It runs as an optional step between video generation (Stage 3) and upscaling (Stage 4), processing only scenes where `lip_sync_mode == "post"`.

Engines with native lip sync (HunyuanVideo-Avatar, HuMo) skip this step entirely — their audio-driven generation already produces synced output.

### Per-Scene Lip Sync Modes

Each scene in `scenes.json` gets a `lip_sync_mode` field:

| Mode | Meaning | When to Use |
|------|---------|------------|
| `off` | No lip sync processing | Instrumental scenes, landscapes, abstract visuals, scenes with no visible face |
| `in_process` | Engine generates lip sync natively | HunyuanVideo-Avatar, HuMo (audio-conditioned engines) |
| `post` | Post-process with LatentSync after video generation | LTX-2 or any engine without audio conditioning, when face + singing is needed |

**Default assignment logic:**
- If `scene.type == "instrumental"` → `off`
- If engine has native audio-visual sync (HVA, HuMo) → `in_process`
- If engine lacks native sync AND `scene.type == "vocal"` → `post`
- User can override any scene's mode in the UI or `scenes.json`

---

## Engine: LatentSync 1.6 (ByteDance)

### Why LatentSync

- 512×512 face region (v1.6) — significantly sharper than alternatives (MuseTalk is 256×256)
- Temporal consistency via TREPA (Temporal REPresentation Alignment)
- Diffusion-based (SD 1.5 UNet) — higher quality than single-step inpainting approaches
- ~18 GB VRAM for inference (v1.6) — fits on the 5090 or 4080
- ComfyUI wrapper exists for validation before pipeline integration
- Active development (v1.5 → v1.6 within months)

### Key Constraints

- **Face must be visible and front-facing.** Side profiles and occluded faces degrade quality.
- **Input video should be 25 fps.** Matches MusicVision's pipeline FPS — no conversion needed.
- **Audio should be isolated vocals.** Instrumentals interfere with lip sync accuracy.
- **Does not support anime/cartoon faces.** Trained on real human video datasets (VoxCeleb2, HDTF).
- **VRAM grows with video length.** Reports of OOM around 2:30+ minutes on a 4090. Our sub-clips are 3–5 seconds — well within safe range.
- **Inference speed:** ~20 DDIM steps per clip. Slower than MuseTalk but quality justifies it for non-real-time pipeline.

### Parameters

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `inference_steps` | 20 | 10–50 | Higher = better quality, slower |
| `guidance_scale` | 1.5 | 1.0–3.0 | Higher = tighter lip sync, risk of distortion |
| `lips_expression` | 1.5 | 1.0–2.5 | Increase for singing (more open mouth); decrease for subtle speech |

**Recommendation for singing:** `lips_expression: 2.0–2.5`, `guidance_scale: 1.5`, `inference_steps: 20` (increase to 30–50 for final renders).

---

## Audio Path for Lip Sync

LatentSync needs **isolated vocals** — the opposite of what video engines receive (full mix). This creates a new audio path:

```
Original Song Audio (input/song.wav)
  │
  ├─→ Vocal Separation → vocal stem
  │     │
  │     ├─→ segments_vocal/scene_XXX_vocal.wav → Whisper transcription (existing)
  │     │
  │     └─→ segments_vocal/sub/scene_XXX_sub_NN_vocal.wav → LatentSync (NEW)
  │
  ├─→ Per-Scene Slicing → segments/scene_XXX.wav (full mix)
  │     └─→ segments/sub/scene_XXX_sub_NN.wav → Video engine (existing)
  │
  └─→ Final Assembly → muxed directly over concatenated video (UNCUT, unchanged)
```

### New: Per-Sub-Clip Vocal Slicing

When a scene has `lip_sync_mode == "post"`, the pipeline must also slice the vocal stem into sub-clip segments, using the same frame-derived boundaries as the full-mix sub-clip audio:

```python
# In Stage 3, after compute_subclip_frames():
if scene.lip_sync_mode == "post" and scene.audio_segment_vocal:
    scene.generation_vocal_segments = slice_subclip_audio(
        scene_audio=paths.vocal_segment_path(scene.id),
        scene_id=scene.id,
        subclip_frames=scene.subclip_frame_counts,
        fps=engine.fps,
        output_dir=paths.sub_vocal_segment_dir,  # segments_vocal/sub/
        suffix="_vocal",
    )
```

**If vocal separation wasn't run during intake** (e.g., user skipped it), and a scene is set to `lip_sync_mode == "post"`, the pipeline should:
1. Run vocal separation on-demand for that scene's audio segment
2. Use Kim_Vocal_2 (MelBandRoFormer) by default; fall back to Demucs if quality is insufficient
3. Cache the result in `segments_vocal/` for reuse

---

## Pipeline Integration

### Where It Runs

Lip sync post-processing runs as **Stage 3.5** — after video generation, before upscaling:

```
Stage 1       Stage 2        Stage 3          Stage 3.5        Stage 4       Stage 5
INTAKE   →   IMAGE GEN  →   VIDEO GEN   →   LIP SYNC POST  →  UPSCALE  →   ASSEMBLY
                                              (conditional)
```

**Why before upscaling:** LatentSync operates on the face region at 512×512. Running it on already-upscaled 1080p+ video would require downscaling the face crop anyway. Processing the raw Stage 3 output (typically 320p–704p) is more efficient and avoids double-processing artifacts.

**Why not during Stage 3:** Video engines and lip sync models have different VRAM profiles and dependencies. Keeping them as separate stages allows full model unload between them, consistent with the existing sequential stage architecture.

### Execution Flow

```python
def run_lip_sync_post(project: Project) -> dict:
    """Stage 3.5: Post-process lip sync for scenes that need it."""

    scenes_to_process = [
        s for s in project.scenes.scenes
        if s.lip_sync_mode == "post"
        and s.video_status == "approved"  # only process approved video
        and s.lip_sync_status != "complete"
    ]

    if not scenes_to_process:
        return {"status": "skipped", "reason": "no scenes need post lip sync"}

    # Ensure vocal segments exist
    ensure_vocal_segments(project, scenes_to_process)

    # Load LatentSync once, process all scenes, then unload
    engine = LatentSyncEngine(project.config.lip_sync)
    engine.load(device=device_map.primary)  # 5090 preferred

    try:
        for scene in scenes_to_process:
            clips = scene.effective_clips()  # sub_clips or [video_clip]
            vocal_segments = scene.effective_vocal_segments()

            for clip_path, vocal_path in zip(clips, vocal_segments):
                output_path = lip_sync_output_path(clip_path)
                engine.process(
                    video_path=project.resolve_path(clip_path),
                    audio_path=project.resolve_path(vocal_path),
                    output_path=output_path,
                )
                # Replace clip path with lip-synced version
                update_clip_path(scene, clip_path, output_path)

            scene.lip_sync_status = "complete"
            project.save_scenes()
    finally:
        engine.unload()
```

### File Layout

```
project/
├── clips/
│   ├── scene_001.mp4                    # Original (engine without lip sync)
│   ├── scene_001_lipsync.mp4            # Post-processed (LatentSync output)
│   ├── sub/
│   │   ├── scene_005_a.mp4              # Original sub-clip
│   │   ├── scene_005_a_lipsync.mp4      # Post-processed sub-clip
│   │   ├── scene_005_b.mp4
│   │   └── scene_005_b_lipsync.mp4
├── segments_vocal/
│   ├── scene_001_vocal.wav              # Per-scene vocal (existing)
│   └── sub/
│       ├── scene_005_sub_00_vocal.wav    # Per-sub-clip vocal (NEW)
│       ├── scene_005_sub_01_vocal.wav
│       └── scene_005_sub_02_vocal.wav
```

**Assembly preference chain** (Stage 5): upscaled_lipsync > lipsync > upscaled > raw clip.

---

## scenes.json Schema Changes

### New Fields Per Scene

```jsonc
{
  "id": "scene_005",
  // ... existing fields ...

  "lip_sync_mode": "post",           // "off" | "in_process" | "post"
  "lip_sync_status": "complete",     // null | "pending" | "complete" | "failed"
  "lip_sync_engine": "latentsync",   // null | "latentsync" (extensible for future engines)

  // Vocal segments for lip sync (only populated when lip_sync_mode == "post")
  "generation_vocal_segments": [
    "segments_vocal/sub/scene_005_sub_00_vocal.wav",
    "segments_vocal/sub/scene_005_sub_01_vocal.wav",
    "segments_vocal/sub/scene_005_sub_02_vocal.wav"
  ],

  // Sub-clips updated to point to lip-synced versions
  "sub_clips": [
    { "path": "clips/sub/scene_005_a_lipsync.mp4", "original": "clips/sub/scene_005_a.mp4", "frames": 67 },
    { "path": "clips/sub/scene_005_b_lipsync.mp4", "original": "clips/sub/scene_005_b.mp4", "frames": 67 },
    { "path": "clips/sub/scene_005_c_lipsync.mp4", "original": "clips/sub/scene_005_c.mp4", "frames": 66 }
  ]
}
```

### SubClip Model Update

Currently sub_clips is `list[str]`. This needs to become `list[SubClip]` where:

```python
@dataclass
class SubClip:
    path: str               # Active clip path (may be lipsync or original)
    original: str | None    # Original pre-lipsync path (None if no post-processing)
    frames: int             # Frame count for this sub-clip
```

This preserves the original clip so the user can compare or revert.

---

## project.yaml Config

```yaml
lip_sync:
  engine: "latentsync"          # Only option for now; extensible
  version: "1.6"                # v1.5 fallback for lower VRAM
  inference_steps: 20           # 10-50, default 20
  guidance_scale: 1.5           # 1.0-3.0
  lips_expression: 2.0          # 1.0-2.5, higher for singing
  device: "auto"                # "auto" | "cuda:0" | "cuda:1"

  # Vocal separation for lip sync (may differ from intake separation)
  vocal_separator: "auto"       # "auto" | "kim_vocal_2" | "demucs"
  demucs_model: "htdemucs_ft"   # Only used if vocal_separator is demucs
```

---

## API & CLI

### CLI

```bash
# Run lip sync post-processing on all eligible scenes
musicvision lip-sync --project ./my-project

# Run on specific scenes
musicvision lip-sync --project ./my-project --scenes scene_003 scene_005

# Override lip_sync_mode for a scene
musicvision set-scene --project ./my-project --scene scene_003 --lip-sync-mode post

# Override engine params for a run
musicvision lip-sync --project ./my-project --inference-steps 30 --lips-expression 2.5
```

### API

```
POST /api/pipeline/lip-sync
{
  "scene_ids": ["scene_003", "scene_005"],  // null = all eligible
  "inference_steps": 30,                     // optional override
  "lips_expression": 2.5                     // optional override
}

PATCH /api/scenes/{id}
{
  "lip_sync_mode": "post"  // or "off" or "in_process"
}
```

---

## GPU / VRAM Strategy

LatentSync 1.6 needs ~18 GB VRAM for inference. Two viable options:

| Config | GPU | Notes |
|--------|-----|-------|
| **Primary (recommended)** | RTX 5090 (32 GB) | Plenty of headroom. Load LatentSync after unloading video engine. |
| **Secondary** | RTX 4080 (16 GB) | Tight but possible with v1.5 (8 GB). v1.6 at 18 GB won't fit. |

**Recommended approach:** Run on the 5090 between Stage 3 (video gen unloaded) and Stage 4 (upscaler not yet loaded). Sequential stage execution already guarantees no VRAM contention.

For cloud (A100/H100 80 GB): trivially fits alongside everything else.

---

## Integration with LatentSync

### Dependency Approach

LatentSync has its own dependency tree (diffusers, mediapipe, face-alignment, etc.). Two options:

| Approach | Pros | Cons |
|----------|------|------|
| **Subprocess/venv (like HVA)** | Isolated deps, no conflicts | JSON IPC overhead, separate install step |
| **Direct import** | Simpler code path, faster | Risk of dep conflicts with main pipeline |

**Recommendation: Subprocess isolation**, matching the HVA pattern. LatentSync's SD 1.5 UNet + diffusers version may conflict with the main pipeline's dependencies. A wrapper script (`scripts/latentsync_wrapper.py`) accepts a JSON request and writes a JSON response, just like `hva_wrapper.py`.

### Wrapper Interface

```python
# scripts/latentsync_wrapper.py
# Reads JSON from stdin or file, processes, writes result

{
  "video_path": "/path/to/clip.mp4",
  "audio_path": "/path/to/vocal_segment.wav",
  "output_path": "/path/to/clip_lipsync.mp4",
  "inference_steps": 20,
  "guidance_scale": 1.5,
  "lips_expression": 2.0
}
# → Writes output video to output_path
# → Returns JSON: {"status": "ok", "output_path": "...", "elapsed_seconds": 12.3}
```

---

## Interaction with Upscaling (Stage 4)

Upscaling runs after lip sync post-processing. The upscaler receives whichever clip is "active":
- If lip-synced: upscales `scene_005_a_lipsync.mp4`
- If no lip sync: upscales `scene_005_a.mp4`

The existing assembly preference chain extends:

```
upscaled_lipsync.mp4 > lipsync.mp4 > upscaled.mp4 > raw.mp4
```

This means Stage 4 doesn't need to know about lip sync — it just processes whatever clips are current.

---

## Face Detection Consideration

LatentSync requires a visible face. Some "vocal" scenes may not show a face (e.g., camera on hands playing guitar while singing off-screen). The pipeline should:

1. Before processing, run a lightweight face detection pass on frame 0 of each clip
2. If no face detected, skip lip sync for that clip and log a warning
3. Set `lip_sync_status = "skipped_no_face"` rather than failing

This can use the same face detection LatentSync uses internally (mediapipe or InsightFace), invoked cheaply on a single frame before committing to full inference.

---

## Open Questions

1. **LatentSync on AI-generated faces:** All benchmarks are on real human video. Quality on FLUX/Z-Image-generated characters is unknown. Needs testing.
2. **Style consistency:** LatentSync modifies the face region. Does this create a visible seam or style shift on stylized/artistic video output? Needs testing with LTX-2 output.
3. **Sub-clip boundary artifacts:** LatentSync processes each sub-clip independently. Mouth state at the boundary between sub-clips may not match. May need 1–2 frame overlap or crossfade.
4. **Singing vs. speaking:** LatentSync was trained primarily on speech. Singing involves wider mouth openings, sustained vowels, and different temporal patterns. The `lips_expression` parameter helps but may not fully cover this. Needs qualitative testing.
5. **MuseTalk as fast preview:** Consider supporting MuseTalk (256×256, single-step, ~4 GB) as a "draft" lip sync engine for quick iteration, with LatentSync for final renders. Mirrors the existing draft/production preset pattern.
