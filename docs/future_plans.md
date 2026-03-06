# MusicVision — Future Plans

**Last updated:** 2026-03-04

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

**Status:** Spec complete ([LIP_SYNC_SPEC.md](LIP_SYNC_SPEC.md)), needs ComfyUI validation then implementation.

Per-scene `lip_sync_mode` field (`off | in_process | post`) enables singing in video engines that lack native audio conditioning (LTX-2, future engines). LatentSync 1.6 (ByteDance) is the primary post-processing engine — 512×512 face region, diffusion-based, ~18 GB VRAM inference. Runs as Stage 3.5 between video generation and upscaling.

Pipeline flow: Video engine generates scene → isolated vocals via Kim_Vocal_2 or Demucs → LatentSync applies lip sync → upscaler processes the result.

**Before implementation:**
- Validate LatentSync 1.6 in ComfyUI on an LTX-2-generated clip with vocal audio
- Test quality on AI-generated faces (all benchmarks are on real human video)
- Assess singing vs speech quality (LatentSync trained primarily on speech)
- Subprocess isolation matching the HVA pattern (avoid dependency conflicts)

### Frontend Refinements

React storyboard is implemented with scene grid, preview panel, per-scene approval/regeneration, and engine selection. Remaining work:

- **Waveform display** with scene boundary visualization and editing
- **Drag-to-reorder scenes** and renumber
- **Assembly & export controls** in the UI
- **Per-scene lip sync mode toggle** (off / in_process / post)
- **Per-scene engine assignment** in the UI (backend supports it, UI doesn't expose it yet)

### Progress & Reliability

- **Progress/status tracking** — no WebSocket or SSE for long-running generation jobs; API endpoints are synchronous. A 50-scene video gen blocks for hours with no progress feedback.
- **Partial failure recovery** — if video generation fails at scene 22 of 50, the exception propagates and scenes 23–50 are never attempted. Already-generated clips survive on disk. Workaround: `--scene-ids` CLI flag. Needs a proper job/resume model with per-scene status tracking.
- **Render time estimation** — no upfront time estimate for users before committing to a full render.

### Transitions & Effects

- **Scene transitions** — hard cuts only today. Future: AI-generated transitions, crossfades, dissolves between scenes.
- **Batch parallelism** — scenes generate sequentially. Future: concurrent generation across multiple GPUs or cloud workers.

---

## Platform Expansion

### Cloud CUDA (A100 / H100 / H200)

**Status:** Planned, ~2h effort. See [PLATFORM_SUPPORT_PLAN.md](PLATFORM_SUPPORT_PLAN.md).

Mostly works today. Two gaps: `recommend_tier()` doesn't offer FP16 for single high-VRAM GPUs (≥48 GB), and FLUX applies unnecessary CPU offload on 80 GB+ cards. Single-GPU configs are simpler than the consumer two-GPU split — no model splitting needed.

Future: Dockerfile + weight caching strategy for cold starts. FSDP multi-GPU sharding for FP16Loader (~8–12 hours, deferred).

### Apple Silicon MPS (M-series Mac)

**Status:** Planned, ~9–12h effort. See [PLATFORM_SUPPORT_PLAN.md](PLATFORM_SUPPORT_PLAN.md).

Does not work today. Blocking issues: RoPE float64/complex128 ops not supported on MPS, FP8/quanto not available. Preview tier (1.7B model) is the initial target. GGUF tiers on ≥32 GB RAM deferred to Phase 2.

MLX would offer 1.5–2× better throughput than MPS+PyTorch but requires a full inference stack rewrite (~3–5 weeks). Out of scope for now.

### GPU Power Profiling

Benchmark different power limits on RTX 5090 to find the optimal power/performance/thermal tradeoff for sustained batch rendering. Early data: 450W / 80C gives ~6% speed gain over 400W with acceptable thermals. Goal: recommended power profile in docs and optionally auto-set via `nvidia-smi -pl` at engine startup.

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
- ✅ Three video engines: HunyuanVideo-Avatar (audio-driven), LTX-Video 2 (cinematic), HuMo (audio-driven, FP8)
- ✅ Three upscalers: SeedVR2 (faces), LTX Spatial (latent), Real-ESRGAN (fast)
- ✅ Two image engines: Z-Image (ungated, fast) and FLUX (LoRA support)
- ✅ React storyboard with scene review, approval, regeneration
- ✅ CLI and REST API for all stages
- ✅ Frame-accurate alignment system, per-scene engine selection
- ✅ End-to-end storyboard test passed (2026-03-01)
- 🔲 Lip sync post-processing (Stage 3.5) — spec complete, needs implementation
- 🔲 Progress feedback (SSE/WebSocket)
- 🔲 Platform expansion (cloud + MPS)

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
