# MusicVision — Future Plans

**Last updated:** 2026-02-26

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

## What Transfers from MusicVision

- **Style sheet system** — characters, props, settings with LoRA paths are the embryo of a full asset consistency system
- **Four-stage pipeline pattern** — intake → asset generation → animation → assembly generalizes directly to other media types
- **Pipeline/UI separation** — core modules are UI-agnostic, enabling future frontends without rewriting logic
- **LLM integration with graceful degradation** — Claude API / local vLLM / manual fallback pattern works for any creative generation step
- **Config-driven projects** — YAML/JSON project files, Pydantic models, ProjectService lifecycle

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
- **Animation input**: each panel is essentially a storyboard frame — "bring this panel to life" is exactly what HuMo TIA mode does
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
| **Music video** | Full AI video generation synced to music | + HuMo video generation (current MusicVision) |
| **Anime / cartoon** | Scene-by-scene animated video with dialogue | + video generation + TTS / voice acting |

Each format is a progressively deeper pass through the pipeline. Users can stop at any stage and get a usable output.

---

## Development Sequencing

### Phase 1: Validate MusicVision (current)
- Complete first GPU integration test
- Generate real music videos end-to-end
- Build the frontend (storyboard UI)
- Observe pain points around consistency and narrative flow

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
- Share image generation modules with MusicVision (FLUX + LoRA)
- Visual novel export (static panels + dialogue trees)
- Manga page export (PDF / image sequence)

### Phase 4: Animation from Panels
- Panel → video using HuMo TIA (or successor models)
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

## Key Design Principles

1. **Modular pipeline stages** — every stage produces a usable intermediate artifact
2. **User owns the creative decisions** — LLM assists, human approves; never fully automated
3. **Backend-agnostic generation** — abstract interfaces for image, video, and text generation so models can be swapped
4. **Config-driven projects** — everything reproducible from project files; no hidden state
5. **Progressive depth** — users can stop at script, panels, or full video; each level adds value
6. **Fully local option** — every stage can run without external APIs using vLLM + local models

---

## Open Research Questions

- **Long-form consistency**: How to maintain character appearance across 50+ scenes without per-scene LoRA tuning?
- **Narrative-aware prompting**: Can the LLM generate prompts that account for story progression (character mood shifts, time-of-day changes, escalating tension)?
- **Panel-to-animation mapping**: What's the best way to encode composition and camera intent from a static panel into video generation parameters?
- **Style transfer at scale**: One style LoRA per project, or dynamic style conditioning that adapts per scene?
- **Interactive narratives**: How does branching (visual novel choices) interact with the linear video pipeline?
- **Lyric-melody alignment**: Can the LLM learn to write lyrics with syllable counts and stress patterns that work well with AceStep's melody generation?
