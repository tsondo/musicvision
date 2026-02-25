# Claude CLI Task: Engineering Hardening Pass

## Context

STATUS.md review identified several engineering gaps. Address all items below in a single pass. Read the full STATUS.md, PIPELINE_SPEC.md, and HUMO_REFERENCE.md from the project knowledge base before starting.

## Task 1: FP8 on Blackwell GPUs

The RTX 5090 is Blackwell architecture (compute capability 12.0). The current FP8 quantization check in `FluxEngine` and/or `HumoEngine` gates on compute cap ≥ 8.9 (Ada/Hopper). Blackwell fully supports FP8 e4m3fn.

**Action:**
- Find all compute capability checks related to FP8 (search for `8.9`, `compute_capability`, `major >= 9`, etc. across the codebase).
- Ensure the check is `>= 8.9` (not `== 8.9` or `in [8, 9]`), which naturally includes Blackwell (12.0). If it already works, add a comment noting Blackwell compatibility. If it doesn't cover 12.0, fix it.
- If there's a separate code path that assumes Ada-specific behavior (e.g., specific scaled FP8 format), verify it also applies to Blackwell. PyTorch's `torch.float8_e4m3fn` works on both.

## Task 2: CUDA Stream Management for Two-GPU Pipeline

The video generation stage runs T5 encoding, audio encoding, DiT denoising, and VAE decoding across two GPUs but there's no explicit CUDA stream overlap.

**Action:**
- In `video/humo_engine.py` (or wherever the TIA generation orchestration lives), evaluate whether T5 text encoding on GPU1 and noise initialization on GPU0 can overlap using separate CUDA streams.
- **If the benefit is real** (T5 encoding takes >1s and doesn't depend on noise): implement stream overlap with `torch.cuda.Stream()` for the encoding phase. The denoising loop itself is sequential by nature (each step depends on the previous), so don't try to parallelize that.
- **If the benefit is marginal** (T5 encoding is fast or already cached): add a code comment in the generation method explaining why streams aren't used, e.g., `# T5 encoding is ~0.3s per scene — stream overlap not worth the complexity. Re-evaluate if batching scenes.`
- Also consider: VAE decoding on GPU1 can potentially overlap with the start of the next scene's T5 encoding. Note this as a future optimization if not implementing now.

## Task 3: Sub-clip Audio Slicing Precision

Audio sub-clips for scenes >3.88s are sliced via ffmpeg. WAV is sample-accurate, but this assumption should be explicit.

**Action:**
- Find the audio slicing code (likely in `utils/audio.py` or `video/humo_engine.py`'s `generate_scene()`).
- Confirm the slicing uses WAV format with explicit sample-accurate seeking (ffmpeg's `-ss` before `-i` for input seeking, or `-af atrim`).
- Add a comment or assertion: `# WAV slicing is sample-accurate. If format changes to MP3/AAC, switch to -af atrim to avoid codec frame boundary drift.`
- If the code currently allows non-WAV input to the slicer, either convert to WAV first or add the `atrim` filter as a safety measure.

## Task 4: Seed Reproducibility

`HumoInput` accepts a `seed` field, and STATUS.md lists "Seed propagation" under "What's Not Built Yet." Investigate whether the seed is actually used.

**Action:**
- Search for where initial noise is generated in the denoising loop (look for `torch.randn`, `torch.randn_like`, or `torch.Generator`).
- Check if `HumoInput.seed` (or equivalent) is passed to a `torch.Generator` that seeds the noise tensor.
- **If seed is NOT wired to noise generation:**
  1. Add a `torch.Generator` seeded with `HumoInput.seed` to the noise initialization.
  2. Set `torch.manual_seed(seed)` and `torch.cuda.manual_seed(seed)` at the start of `generate()`.
  3. Store the used seed in `HumoOutput` (or `scenes.json` sub-clip entries) so regeneration with the same seed is reproducible.
  4. If no seed is provided (None), generate a random one and still record it.
  5. Update the "What's Not Built Yet" section in STATUS.md to remove seed propagation or mark it as resolved.
- **If seed IS already wired:** update STATUS.md to clarify that the plumbing exists and the remaining issue is just verification of deterministic output (which requires a GPU test).

## Task 5: Update "What's Not Built Yet" in STATUS.md

The current list is too short and omits known gaps. Replace the section with a more comprehensive version.

**Action:**
Replace the "What's Not Built Yet" section in STATUS.md with:

```markdown
## What's Not Built Yet

### Frontend & UI
- **React frontend** — pipeline driven entirely via REST API and CLI today; Swagger UI at `/docs` for manual testing
- **Gradio UI** — originally planned as the storyboard interface; not started. Core pipeline modules are UI-agnostic by design.
- **Scene approval UI** — API endpoints for scene CRUD and approval exist, but no visual side-by-side review interface

### Pipeline Features
- **Progress/status tracking** — no WebSocket or SSE for long-running generation jobs; API endpoints are synchronous. A 50-scene video gen blocks for hours with no progress feedback.
- **Partial failure recovery** — if video generation fails at scene 22 of 50, there's no automatic resume. Workaround: use `--scene-ids` CLI flag to regenerate specific scenes. Needs a proper job/resume model.
- **Scene reordering** — scenes are ordered by `order` field in scenes.json, but no API endpoint or UI to drag-reorder and renumber.
- **Transitions** — hard cuts only between scenes. No crossfades, AI-generated transitions, or blending.
- **Batch parallelism** — scenes generate sequentially. No concurrent generation across multiple GPUs or cloud workers.

### Model & Generation
- **LoRA training** — pipeline accepts LoRA paths for character consistency but doesn't include training workflows
- **Seed reproducibility** — `HumoInput` accepts a seed field; wiring to noise initialization [needs verification / has been implemented — update per Task 4 findings]
- **Render time estimation** — no estimated time display for users. 3× DiT passes per step × 50 steps × N scenes can take many hours; users need to know upfront.

### Export
- **Audio-only export** — no way to export just the segmented audio clips without running image/video gen
- **Rough cut preview without full render** — storyboard slideshow (images + audio, no video) would let users validate before committing to HuMo render time
```

Adjust the seed line based on your findings from Task 4.

## General Instructions

- Make minimal, targeted changes. Don't refactor unrelated code.
- Add comments explaining *why*, not just *what*.
- Run `uv run pytest tests/ -v` after changes to confirm nothing breaks.
- If any task requires information you can't determine from the code alone (e.g., actual T5 encoding time), add a `# TODO: measure on GPU` comment rather than guessing.
