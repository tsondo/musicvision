# Claude Code Task: MusicVision Spec Update & Sub-clip Conditioning

## Context

You are working on the MusicVision codebase — a music video production pipeline. The project has 3 specification/reference documents that need updating, plus one feature implementation. Read all three docs first before making any changes.

## Files to Read First

- `PIPELINE_SPEC.md` — Architecture and design spec
- `HUMO_REFERENCE.md` — HuMo technical reference
- `STATUS.md` — Current implementation status

## Task 1: Update Hardware References (all three docs)

The hardware has changed. Replace ALL references to the old setup with the new setup:

**OLD:** RTX 5080 (32GB) primary + RTX 3080 Ti (12GB) secondary
**NEW:** RTX 5090 (32GB) primary + RTX 4080 (16GB) secondary

This affects:
- PIPELINE_SPEC.md: System Requirements section, GPU references throughout, module docstrings mentioning GPU split
- HUMO_REFERENCE.md: Model Variants table notes, multi-GPU section
- STATUS.md: Target hardware description, any 3080 Ti references

Also update the GPU split strategy notes: the 4080 at 16GB can handle heavier encoder workloads than the old 3080 Ti at 12GB. The `fp8_scaled` tier (~18GB DiT) can now fit on the secondary GPU as a fallback — note this in the VRAM tier table commentary.

Also check and update `gpu.py` if it has any hardcoded 3080 Ti thresholds (search for "3080", "12GB", "12288" in the codebase).

## Task 2: Resolve HuMo Source Status (HUMO_REFERENCE.md, STATUS.md)

**Finding: HuMo is NOT gated.** The full source code and weights have been publicly available since Sep 10, 2025:
- GitHub: https://github.com/Phantom-video/HuMo (Apache 2.0)
- HuggingFace: https://huggingface.co/bytedance-research/HuMo
- The repo contains: `humo/` package, `main.py` entry point, `scripts/` for inference, `examples/test_case.json`
- Inference scripts: `scripts/infer_ta.sh`, `scripts/infer_tia.sh` (and `_1_7B` variants)
- Config: `humo/configs/inference/generate.yaml`

Additionally, kijai's ComfyUI-WanVideoWrapper (https://github.com/kijai/ComfyUI-WanVideoWrapper) provides a battle-tested reference implementation with:
- FP8 scaled weights at `Kijai/WanVideo_comfy_fp8_scaled`
- GGUF support
- Block swapping implementation
- Full denoising loop in `nodes_sampler.py`
- HuMoEmbeds (image + audio conditioning) in `HuMo/nodes.py`

**Updates needed:**
1. In STATUS.md: Remove the "gap" / "blocked" / "gated" language from Stage 3. The four stubs (`_encode_text`, `_encode_image`, `_denoise`, `_decode_latent`) are NOT blocked — source is available. Reframe as "implementation pending — reference code available from both the official HuMo repo and kijai's ComfyUI wrapper."
2. In STATUS.md "Design Decisions" section: Update the "Risk" note about HuMo source being partially gated — it's fully open.
3. In HUMO_REFERENCE.md: Update any language suggesting the source is unavailable. Add a note about the official repo structure (`humo/` package with `main.py` entry point).
4. In HUMO_REFERENCE.md: Add kijai's FP8 scaled weights as a resource: `https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled`

## Task 3: Add Z-Image as Alternative Image Engine (PIPELINE_SPEC.md)

The codebase already has a Z-Image engine (`ZImageEngine`) alongside FLUX, selected via a `create_engine()` factory. Update PIPELINE_SPEC.md to document this:

1. In Stage 2 description, mention both FLUX and Z-Image as options
2. Add a brief comparison:
   - **FLUX.1-dev**: Highest quality, gated (requires HuggingFace token), VRAM-heavy
   - **Z-Image** (Tongyi-MAI, 6B params): Open-weight, ~12GB fp16, no token required, faster iteration
3. Note the factory pattern: `ImageGenConfig.model` selects engine, `create_engine()` instantiates
4. Update the `project.yaml` config example to show `model: "flux-dev"` with a comment noting `"z-image"` as alternative
5. Update the image_gen section of the data model if needed

## Task 4: Remove Gradio, Confirm React + FastAPI (PIPELINE_SPEC.md, STATUS.md)

The project already has a comprehensive FastAPI backend with 25+ endpoints and CORS configured for React dev servers. Gradio was planned as an interim UI but was never built and is no longer needed.

1. In PIPELINE_SPEC.md:
   - Replace "UI Architecture (Gradio)" section with "UI Architecture (React + FastAPI)"
   - Keep the same tab structure and storyboard grid layout description, but frame it as React components calling the existing REST API
   - Remove all Gradio-specific references
   - Update the module structure: remove `app.py` (Gradio entry point), keep `cli.py`
   - The `ui/` directory should be described as the planned React frontend (not yet built)
2. In STATUS.md:
   - Update "What's Not Built Yet" to say "React frontend" instead of referencing Gradio
   - Remove any Gradio mentions

## Task 5: Fix Vocal Separation Audio Input Guidance (PIPELINE_SPEC.md, HUMO_REFERENCE.md)

There's a contradictory guidance issue. HuMo TIA mode should receive the **full mix** (vocals + instrumental), not vocal-only audio. The vocal stem is only for improving Whisper transcription quality.

1. In PIPELINE_SPEC.md Stage 3, clarify: HuMo TIA input uses the full-mix audio segment from `segments/`, NOT the vocal-separated segment from `segments_vocal/`
2. In HUMO_REFERENCE.md Audio Segment Guidelines, ensure the guidance is clear: "For TIA mode, use the full mix (vocals + instrumental) as audio input"
3. In PIPELINE_SPEC.md Stage 1, clarify: vocal separation output (`segments_vocal/`) is used for Whisper transcription quality, not for HuMo input

## Task 6: Implement Sub-clip Last-Frame Conditioning (code change)

Currently `generate_scene()` in the HuMo engine splits scenes >3.88s into sub-clips that all share the same reference image. For visual continuity, sub-clip N+1 should use the **last frame of sub-clip N** as its reference image.

Find the `generate_scene()` method (likely in `video/humo_engine.py` or similar) and modify it:

1. After generating sub-clip N, extract the last frame from the output video
2. Save it as a temporary reference image (e.g., `clips/sub/scene_003_a_lastframe.png`)
3. Pass that last frame as the `reference_image` for sub-clip N+1's `HumoInput`
4. The FIRST sub-clip of a scene still uses the scene's original reference image
5. Add a config option `sub_clip_continuity: bool = True` (default True) to `HumoConfig` to allow disabling this behavior
6. Update the docstring to explain the continuity strategy

Also update PIPELINE_SPEC.md's "Sub-clips for Long Scenes" section to document this last-frame conditioning approach.

Also update STATUS.md to reflect this feature.

## Task 7: Update DaVinci Resolve Export Format (PIPELINE_SPEC.md, STATUS.md, code)

Research finding: DaVinci Resolve supports FCPXML 1.9 natively (via File > Import > Timeline). It also supports OTIO since v18.5. EDL (CMX 3600) is the most limited format (single video track, limited metadata).

**Recommendation:** Make FCPXML 1.9 the primary export format (not 1.10 — Resolve's support for 1.10 is inconsistent). Keep EDL as a legacy fallback. Optionally add OTIO if the `opentimelineio` Python package is acceptable as a dependency.

1. In PIPELINE_SPEC.md Stage 4:
   - List FCPXML 1.9 as the primary/recommended format
   - EDL (CMX 3600) as fallback for maximum compatibility
   - Mention OTIO (.otio/.otioz) as optional if opentimelineio is installed
   - Note: FCPXML 1.9 carries more metadata (clip names, markers, frame rate) than EDL
2. In STATUS.md: Update the assembly section to reflect FCPXML 1.9 as primary
3. If the codebase has an FCPXML exporter, verify it outputs version="1.9" not "1.10". If it says 1.10, change to 1.9.
4. Search for any hardcoded FCPXML version strings in the assembly code and update accordingly.

## Task 8: Add AceStep Metadata to PIPELINE_SPEC.md

AceStep metadata auto-detection is implemented in the intake pipeline but not documented in PIPELINE_SPEC.md. Add it:

1. In Stage 1 (Intake & Segmentation), add AceStep JSON companion file detection as an input source
2. Note that AceStep provides: BPM, key signature, lyrics with section markers (verse/chorus/bridge), genre descriptions
3. This metadata enriches the segmentation step and can be used for style sheet suggestions

## Execution Order

1. Read all three docs completely
2. Search the codebase for hardware references, Gradio references, FCPXML version strings, and the generate_scene() method
3. Apply Tasks 1-5, 7-8 (doc updates) in a single pass per file to minimize edits
4. Apply Task 6 (code change)
5. Run existing tests to verify nothing breaks: `uv run pytest tests/ -v`
