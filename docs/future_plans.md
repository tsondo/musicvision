# MusicVision — Future Plans

**Last updated:** 2026-03-20

---

## Vision

MusicVision is a proof of concept for a larger creative tool that handles the full pipeline from **story writing → visual novel / manga panels → animated video** (cartoon, anime, music video, etc.). The music video generator validates the core architecture; the long-term goal is a complete verbal, visual, and storytelling platform.

---

## Near-Term: Integrated Creation Pipeline

The local vLLM server (Qwen2.5-32B-AWQ on the 3090 Ti) opens up a fully local, end-to-end creative loop with no external API dependencies:

```
Write (LLM) → Compose (AceStep) → Visualize (MusicVision) → Export
```

### Unified App Concept

A single application with four panels, where each panel's output feeds the next:

| Panel | Engine | Input | Output |
|-------|--------|-------|--------|
| **Write** | vLLM (Qwen2.5-32B) | Genre, mood, topic, structure | Lyrics with section markers |
| **Compose** | AceStep | Lyrics, genre tags, BPM target | Song audio + JSON metadata |
| **Visualize** | MusicVision pipeline | Audio + lyrics + metadata | Storyboard → video clips |
| **Export** | ffmpeg + FCPXML | Approved clips + original audio | Rough cut + DaVinci project |

Users can enter at any panel with their own content — write lyrics by hand, bring an existing song, or supply pre-made reference images.

### Why This Works Now

- AceStep already has a Gradio interface; wrapping it as a tab is straightforward
- The vLLM server is running 24/7 on the LAN with no per-token cost
- Lyric generation is a natural fit for Qwen2.5-32B's capabilities
- The shared data between panels is minimal: audio file, JSON metadata, lyrics text
- MusicVision's pipeline/UI separation means adding upstream panels doesn't touch core video logic

### Implementation Notes

- AceStep integration: import its generation function as a module, or wrap its existing Gradio app as a sub-block
- Lyric generation: system prompt defines song structure conventions (section markers, verse/chorus patterns, syllable density for singability); user provides genre + mood + topic
- The LLM can also generate AceStep's `tags` field (genre/instrumentation description) from the same creative brief
- Keep each panel independently usable — the app is four tools that happen to chain together, not a monolith

---

## Current Pipeline: What's Not Built Yet

### Lip Sync Post-Processing (Stage 3.5)

**Status:** Spec complete ([LIP_SYNC_POST.md](LIP_SYNC_POST.md)), needs ComfyUI validation then implementation.

Per-scene `lip_sync_mode` field (`off | in_process | post`) enables singing in video engines that lack native audio conditioning (LTX-2, future engines). LatentSync 1.6 (ByteDance) is the primary post-processing engine — 512×512 face region, diffusion-based, ~18 GB VRAM inference. Runs as Stage 3.5 between video generation and upscaling.

Pipeline flow: Video engine generates scene → isolated vocals via Kim_Vocal_2 or Demucs → LatentSync applies lip sync → upscaler processes the result.

**Before implementation:**
- Validate LatentSync 1.6 in ComfyUI on an LTX-2-generated clip with vocal audio
- Test quality on AI-generated faces (all benchmarks are on real human video)
- Assess singing vs speech quality (LatentSync trained primarily on speech)
- Subprocess isolation matching the HVA pattern (avoid dependency conflicts)

### Frontend Refinements

React storyboard is implemented with scene grid, preview panel, per-scene approval/regeneration, engine selection, waveform editor with scene boundary visualization, lyrics line editor/mapper, video concept field, and per-scene sigma shift slider. Remaining work:

- **Drag-to-reorder scenes** and renumber
- **Assembly & export controls** in the UI
- **Per-scene lip sync mode toggle** (off / in_process / post)

### Progress & Reliability

- **Progress/status tracking** — no WebSocket or SSE for long-running generation jobs; API endpoints are synchronous. A 50-scene video gen blocks for hours with no progress feedback.
- **Partial failure recovery** — if video generation fails at scene 22 of 50, the exception propagates and scenes 23–50 are never attempted. Already-generated clips survive on disk. Workaround: `--scene-ids` CLI flag. Needs a proper job/resume model with per-scene status tracking.
- **Render time estimation** — no upfront time estimate for users before committing to a full render.

### Transitions & Effects

- **Scene transitions** — hard cuts only today. Future: AI-generated transitions, crossfades, dissolves between scenes.
- **Batch parallelism** — scenes generate sequentially. Future: concurrent generation across multiple GPUs or cloud workers.

### SD.Next as Alternative Image Backend

**Status:** Evaluated (2026-03-20). Not a drop-in replacement, but a viable alternative backend.

The SD.Next project (`~/projects/sdnext`) exposes a full REST API (`/sdapi/v1/txt2img`, model management, LoRA, ControlNet, IP-Adapter) with FLUX support. It could be wrapped as an `SDNextEngine` implementing the existing `ImageEngine` ABC, adding it alongside `FluxEngine` and `ZImageEngine` via the factory.

**Where it adds value over the current in-process FLUX engine:**
- **ControlNet / IP-Adapter** — reference-image-guided generation for character consistency without LoRA training
- **Platform breadth** — MPS, ROCm, Intel Arc, ONNX handled by SD.Next, not us
- **Decoupled VRAM** — runs as a separate server process; image generation doesn't compete with MusicVision's video model loading

**Trade-offs:**
- Service dependency (SD.Next server must be running)
- Coarser VRAM control (SD.Next manages its own memory; less deterministic for the FLUX→HuMo unload handoff)
- LoRA mapping needs adaptation (SD.Next uses `extra_networks`, not MusicVision's two-level project+scene LoRA system)

**Implementation effort:** ~150–200 lines for an `SDNextEngine` adapter. Add `ImageModel.SDNEXT` to the enum, dispatch in factory.py. Not a priority until ControlNet/IP-Adapter consistency features are needed.

---

## Platform Expansion

### Cloud CUDA (A100 / H100 / H200)

**Status:** ✅ Implemented. Single ≥48 GB GPU → FP16 tier. No-offload FLUX path for high-VRAM cards.

Remaining: Dockerfile + weight caching strategy for cold starts. FSDP multi-GPU sharding for FP16Loader (~8–12 hours, deferred).

### Apple Silicon MPS (M-series Mac)

**Status:** ✅ Implemented (PLATFORM_SUPPORT branch, merged to main). Awaiting Mac contributor smoke test.

All blocking issues resolved: RoPE float32 downcast, FP8 blocked on MPS, device-aware autocast/seeding/cache clearing, `psutil` RAM detection for FLUX, T5 FP16 on MPS. Preview tier (1.7B HuMo) is the target for M-series.

Remaining: GGUF tiers on ≥32 GB RAM (Phase 2, after preview smoke test). MLX rewrite (~3–5 weeks) out of scope.

### GPU Power Profiling

**Status:** ✅ Implemented. Per-engine power limit management in `humo_engine.py` and `ltx_video_engine.py`. PowerShell and bash launchers with GPU power cap support. WSL `nvidia-smi -pl` requires Windows UAC elevation (documented).

### Dependency Simplification

- **Drop `flash_attn`** — Native SDPA on PyTorch 2.6+ (currently 2.10.x) is equivalent on Ampere/Hopper/Blackwell. Eliminates build friction, simplifies containers, one fewer binary dependency. SDPA fallback already works in all vendored code.

---

## What Transfers from MusicVision

- **Style sheet system** — characters, props, settings with LoRA paths are the embryo of a full asset consistency system
- **Five-stage pipeline pattern** — intake → image gen → video gen → upscale → assembly generalizes directly to other media types
- **Pipeline/UI separation** — core modules are UI-agnostic, enabling future frontends without rewriting logic
- **LLM integration with graceful degradation** — Claude API / local vLLM / manual fallback pattern works for any creative generation step
- **Config-driven projects** — YAML/JSON project files, Pydantic models, ProjectService lifecycle
- **Per-scene engine selection** — different engines per scene, extensible to different generation backends per panel/shot
- **Frame-accurate alignment system** — integer frame counts as authoritative duration, frame-first math eliminates drift

---

## The Big Gap: A Persistent Story Model

MusicVision's data model is flat — `scenes.json` is a linear sequence tied to a song's timeline. The larger project needs a **hierarchical narrative structure**:

```
Story
├── Arc / Act
│   ├── Chapter / Sequence
│   │   ├── Scene
│   │   │   ├── Panel / Shot
│   │   │   │   ├── Characters present (with emotional state, pose)
│   │   │   │   ├── Dialogue / narration
│   │   │   │   ├── Setting / environment
│   │   │   │   └── Camera / framing
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

### Story Bible

A structured document (YAML or lightweight SQLite) that every downstream generator queries:

- **Characters**: appearance, personality, relationships, arc progression
- **Settings**: locations with visual descriptions, time-of-day variants, mood associations
- **Props**: recurring objects with narrative significance
- **Timeline**: chronological event ordering, cause-and-effect chains
- **Themes & motifs**: visual and narrative recurring elements

The LLM helps populate the story bible, but the user owns it. All generation modules query it for context.

---

## Character / Asset Consistency at Scale

LoRA per character works for a music video with 1–2 characters. A visual novel or anime with a full cast needs a layered approach:

| Layer | Technique | Use Case |
|-------|-----------|----------|
| **Style LoRA** | Single LoRA for overall visual language | Applied to every generation — defines the "look" |
| **Primary character LoRA** | Per-character LoRA | Main cast (2–4 characters trained individually) |
| **Secondary characters** | IP-Adapter / reference-image conditioning | Supporting cast — no training, reference images only |
| **Expression / pose control** | ControlNet or prompt-driven | Emotional states, action poses |

### Consistency Module Interface

Abstract the consistency system behind a clean interface:

```python
class ConsistencyEngine:
    def get_character_conditioning(
        self, character_id: str, expression: str, pose: str
    ) -> CharacterConditioning:
        """Returns LoRA config, reference images, and prompt fragments."""
        ...
```

This allows swapping underlying tech (LoRA → IP-Adapter → future methods) without changing pipeline code.

### Asset Library

Beyond character LoRAs, the consistency system needs a broader **asset library** that organizes all reusable generation assets by type. Each asset is a named entry with metadata, a file path, and conditions for when it should be applied.

#### Camera Motion LoRAs

Video engines (HuMo, LTX-2, and future models) can use motion-specific LoRAs to produce controlled camera movements that are difficult to achieve through prompting alone. These apply at the video generation stage, not the image stage.

| Motion Type | Description | Use Case |
|-------------|-------------|----------|
| **Dolly zoom** | Push-in with FOV change (vertigo effect) | Dramatic reveals, tension escalation |
| **Orbit** | Camera circles the subject | Character introductions, 360° views |
| **Crane up/down** | Vertical camera movement | Scene transitions, establishing shots |
| **Tracking shot** | Camera follows subject laterally | Walking/running sequences, performances |
| **Zoom in/out** | Focal length change, camera stationary | Emotional emphasis, wide-to-close transitions |
| **Static** | Locked camera | Dialogue, still moments, lyric emphasis |
| **Handheld** | Subtle camera shake | Intimacy, documentary feel, live performance |
| **Pan** | Horizontal rotation on axis | Landscape reveals, scanning environments |
| **Tilt** | Vertical rotation on axis | Tall subjects, building reveals |

**Integration with the pipeline:**
- Camera motion is a per-scene property (like `video_engine` or `sigma_shift`)
- The LLM prompt generator can suggest camera motion based on lyrical content and scene type (performance vs. narrative)
- The scene model gains a `camera_motion: Optional[str]` field referencing an asset library entry
- At video generation time, the engine loads the appropriate motion LoRA alongside any character/style LoRA

**Asset library entry format (conceptual):**
```yaml
assets:
  camera_motions:
    - id: dolly_zoom
      name: "Dolly Zoom"
      type: camera_motion
      lora_path: "assets/loras/camera/dolly_zoom.safetensors"
      lora_weight: 0.7
      compatible_engines: [humo, ltx_video]
      tags: [dramatic, tension, reveal]
    - id: orbit
      name: "Orbit"
      type: camera_motion
      lora_path: "assets/loras/camera/orbit.safetensors"
      lora_weight: 0.6
      compatible_engines: [humo, ltx_video]
      tags: [introduction, 360, character]
  style_loras:
    - id: anime_style
      name: "Anime Style"
      type: style
      lora_path: "assets/loras/style/anime.safetensors"
      lora_weight: 0.8
      compatible_engines: [flux, z_image, humo, ltx_video]
  characters:
    # ... existing CharacterDef entries migrate here
```

**Open questions:**
- LoRA stacking: can camera motion + character + style LoRAs compose without quality degradation? Needs testing per engine.
- Engine-specific LoRAs: HuMo and LTX-2 use different architectures, so camera motion LoRAs are likely engine-specific. The asset library must track `compatible_engines`.
- Training pipeline: camera motion LoRAs need to be trained or sourced. CivitAI has some for SD/SDXL; FLUX and HuMo motion LoRAs are emerging but less mature.
- SD.Next integration: if using SD.Next as an image/video backend, its LoRA management (`extra_networks`) can serve these assets without custom loading code.

---

## The Manga / Panel Intermediate

Panels are the natural bridge between story and animation:

```
Story Bible → Panel Layout → Panel Images → Animation → Assembled Video
```

### Why Panels Matter

- **Composition constraints**: framing, character placement, speech bubbles, panel borders
- **User review checkpoint**: cheap to generate, easy to iterate before expensive video rendering
- **Animation input**: each panel is essentially a storyboard frame — "bring this panel to life" is exactly what video engines do with reference images
- **Standalone output**: manga / visual novel is a valid end product, not just an intermediate step

### Panel Generation Requirements

- Layout engine: grid-based panel arrangements (1–6 panels per page)
- Speech bubble placement and text rendering
- Consistent character rendering across panels (via consistency module)
- Style presets: manga, comic, webtoon, storyboard, etc.

---

## Target Output Formats

| Format | Description | Pipeline Depth |
|--------|-------------|----------------|
| **Script / screenplay** | Text-only story output | Story model only |
| **Visual novel** | Static panels + dialogue + choices | Story model + panel generation |
| **Manga / comic** | Laid-out pages with panels and speech bubbles | Story model + panel generation + layout |
| **Animated slideshow** | Panels with Ken Burns / parallax motion + audio | + simple animation |
| **Music video** | Full AI video generation synced to music | + video generation (current MusicVision) |
| **Anime / cartoon** | Scene-by-scene animated video with dialogue | + video generation + TTS / voice acting |

Each format is a progressively deeper pass through the pipeline. Users can stop at any stage and get a usable output.

---

## Development Sequencing

### Phase 1: Validate MusicVision ✅ (mostly complete)
- ✅ All five pipeline stages code-complete and GPU-tested
- ✅ Two video engines: HuMo (audio-driven, 24 bugs fixed, working) and LTX-Video 2 (cinematic, audio-conditioned). HunyuanVideo-Avatar removed (deprecated 2026-03-11).
- ✅ Three upscalers: SeedVR2 (faces), LTX Spatial (latent), Real-ESRGAN (fast)
- ✅ Two image engines: Z-Image (ungated, fast) and FLUX (LoRA support)
- ✅ React storyboard with scene review, approval, regeneration, waveform editor, lyrics mapper
- ✅ CLI and REST API for all stages
- ✅ Frame-accurate alignment system, per-scene engine selection
- ✅ End-to-end storyboard test passed (2026-03-01)
- ✅ Platform support: cloud CUDA (single high-VRAM GPU), Apple Silicon MPS (preview tier)
- ✅ GPU power limit management with PowerShell/bash launchers
- ✅ Per-scene sigma shift control for HuMo tuning
- 🔲 Lip sync post-processing (Stage 3.5) — spec complete, needs implementation
- 🔲 Progress feedback (SSE/WebSocket)

### Phase 1.5: Integrated Creation App
- Wrap lyric generation (vLLM) + AceStep + MusicVision into a single multi-panel app
- Lyric generation panel: genre/mood/topic → structured lyrics with section markers
- AceStep panel: lyrics → song audio + metadata JSON
- Keep each panel independently usable with manual input
- Validate the full prompt-to-video loop end-to-end locally

### Phase 2: Story Bible Module
- Extract style sheet into a standalone story bible with richer character/relationship modeling
- Hierarchical scene structure (acts → scenes → shots)
- LLM-assisted story bible population from text descriptions or existing scripts
- Character relationship graph and arc tracking

### Phase 3: Panel / Manga Generator
- Panel layout engine (grid templates + AI-assisted composition)
- Speech bubble and text overlay system
- Share image generation modules with MusicVision (FLUX/Z-Image + LoRA)
- Visual novel export (static panels + dialogue trees)
- Manga page export (PDF / image sequence)

### Phase 4: Animation from Panels
- Panel → video using current video engines (or successors)
- Camera motion inference from panel composition
- Transition generation between scenes (not just hard cuts)
- TTS integration for dialogue (optional)
- Full animated video assembly with audio sync

### Phase 5: Unified Creative Tool
- Single project can produce any output format from the same story bible
- Branching narratives (visual novel choice trees → multiple video paths)
- Collaborative editing (multiple users on one story bible)
- Plugin architecture for new generation backends as models improve

---

## Models to Watch

- **Wan 2.2** — MoE architecture splits denoising across timesteps into specialized experts. No audio conditioning, but the efficiency approach is relevant for consumer hardware.
- **LTX-2 evolution** — Already integrated. Native audio+video generation in a single pass is unique. Quality and controllability will improve with newer checkpoints.
- **LatentSync** — Lip sync post-processing. v1.6 is current (512×512, diffusion-based). Watch for singing-specific improvements and higher resolution support.
- **MuseTalk** — Real-time lip sync (single-step, 256×256). Useful as a fast preview engine if quality gap with LatentSync closes.
- **VBVR / Reasoning-Oriented Training Data** — [VBVR-Wan2.2](https://huggingface.co/Video-Reason/VBVR-Wan2.2) ([paper](https://arxiv.org/abs/2602.20159)) fine-tunes Wan2.2-I2V-A14B on 1M+ video clips spanning 200 reasoning tasks (spatial, causal, temporal, perceptual) with **zero architecture changes**. Result: 84.6% improvement in video reasoning benchmarks, beating Sora 2 and Veo 3.1 on physical plausibility and cause-effect consistency. Key insight: reasoning quality scales with training data volume under fixed architectures — in-domain scores rose from 0.412 → 0.771 at 400K samples before saturating. Not directly usable (Wan2.2-A14B weights, incompatible with HVA/LTX-2; 14B active params needs 24GB+), but signals that the next wave of quality improvements across all video engines will come from better training data rather than new architectures. Watch for: HVA or LTX releases citing reasoning-oriented data; community fine-tunes of engines we use; smaller VBVR-tuned checkpoints (<16GB) that could serve as a cinematic-scene engine.


---

## Key Design Principles

1. **Modular pipeline stages** — every stage produces a usable intermediate artifact
2. **User owns the creative decisions** — LLM assists, human approves; never fully automated
3. **Backend-agnostic generation** — abstract interfaces for image, video, and text generation so models can be swapped
4. **Config-driven projects** — everything reproducible from project files; no hidden state
5. **Progressive depth** — users can stop at script, panels, or full video; each level adds value
6. **Fully local option** — every stage can run without external APIs using vLLM + local models
7. **Frame-first math** — integer frame counts as the authoritative duration unit; derive seconds from frames, never the reverse

---

## Open Research Questions

- **Long-form consistency**: How to maintain character appearance across 50+ scenes without per-scene LoRA tuning?
- **Narrative-aware prompting**: Can the LLM generate prompts that account for story progression (character mood shifts, time-of-day changes, escalating tension)?
- **Panel-to-animation mapping**: What's the best way to encode composition and camera intent from a static panel into video generation parameters?
- **Style transfer at scale**: One style LoRA per project, or dynamic style conditioning that adapts per scene?
- **Interactive narratives**: How does branching (visual novel choices) interact with the linear video pipeline?
- **Lyric-melody alignment**: Can the LLM learn to write lyrics with syllable counts and stress patterns that work well with AceStep's melody generation?
- **Lip sync on AI faces**: LatentSync is benchmarked on real human video. Quality on FLUX/Z-Image-generated characters is unknown and needs testing.
- **Singing vs speech**: Lip sync models are trained primarily on speech. Singing involves wider mouth openings, sustained vowels, and different temporal patterns. The `lips_expression` parameter helps but may not fully cover this.
- **Automated quality checks via vision LLM**: Run a Qwen2.5-VL model (on the vLLM server or swapped in) to evaluate generated video output programmatically. Use cases: scene-prompt coherence scoring, artifact detection (checkerboard, banding, temporal flicker), lip sync quality assessment, frame-to-frame consistency across sub-clips. Could feed back into the pipeline as an auto-reject/retry gate or surface quality scores in the review GUI.
