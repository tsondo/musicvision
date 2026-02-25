# STATUS.md Update Instructions

These are directed changes to apply to `STATUS.md`. Read the full file first, then apply each change.

---

## 1. Fix mystery RTX 5070 reference (line ~175)

In the "LLM Availability" section, the vLLM candidates parenthetical says:

> **vLLM candidates for local inference (28 GB VRAM: RTX 4080 16 GB + RTX 5070 12 GB):**

Replace with:

> **vLLM candidates for local inference (RTX 3090 Ti, 24 GB VRAM):**

And update the model list to reflect 24 GB budget:
- `Qwen/Qwen2.5-32B-Instruct-AWQ` (~18 GB, best quality) ← keep
- `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (~12 GB 4-bit, fast) ← keep
- `Qwen/Qwen2.5-14B-Instruct` (~28 GB bf16) ← **remove** (exceeds 24 GB)

---

## 2. Restructure the status summary table

Replace the current summary table with this version. Separate pipeline stages from non-pipeline items:

```markdown
| Pipeline Stage | Status |
|-------|--------|
| Intake (audio, BPM, Whisper, segmentation) | ✅ Complete |
| Reference image generation (FLUX) | ✅ Complete |
| Video generation (HuMo TIA) | ✅ Complete — **pending first GPU test** |
| Assembly (rough cut, EDL, FCPXML) | ✅ Complete |

| Infrastructure | Status |
|-------|--------|
| API + CLI | ✅ Complete |
| Frontend (React or Gradio) | ❌ Not started |
| Progress/status feedback (SSE/WebSocket) | ❌ Not started |
```

---

## 3. Add current failure behavior note

In "What's Not Built Yet → Pipeline Features", expand the **Partial failure recovery** bullet. Replace:

> - **Partial failure recovery** — if video generation fails at scene 22 of 50, there's no automatic resume. Workaround: use `--scene-ids` CLI flag to regenerate specific scenes. Needs a proper job/resume model.

With:

> - **Partial failure recovery** — if video generation fails at scene 22 of 50, there's no automatic resume. The exception propagates to the API caller or CLI, and scenes 23–50 are never attempted. Already-generated clips (scenes 1–21) are saved to disk and remain valid. Workaround: use `--scene-ids` CLI flag to regenerate specific scenes. Needs a proper job/resume model with per-scene status tracking.

---

## 4. Remove approximate test counts

In the Tests table, drop `~` counts that haven't been verified:

| Before | After |
|--------|-------|
| `tests/test_intake.py` (~15 tests) | `tests/test_intake.py` |
| `tests/test_image_engine.py` (~20 tests) | `tests/test_image_engine.py` |
| `tests/test_video_engine.py` (~20 tests) | `tests/test_video_engine.py` |

Keep exact counts for `test_core.py` (26 tests) and `test_humo_inference.py` (11 tests) — those appear verified.

---

## 5. Promote critical architectural decisions

Add a new section **## Critical Design Decisions** immediately after "## Current Status Summary" and before "## Pipeline Stages". Move and promote these from the "Design Notes" appendix:

```markdown
## Critical Design Decisions

> **HuMo receives the full audio mix, not isolated vocals.** The vocal stem from Kim_Vocal_2 / Demucs is used *only* to improve Whisper transcription accuracy. HuMo was trained on mixed audio — feeding it isolated vocals degrades A/V sync. Audio segments in `segments/` are always full-mix; `segments_vocal/` is consumed only by the transcription step.

> **Sub-clip continuity uses last-frame chaining.** For scenes longer than 3.88s, after generating sub-clip N, ffmpeg extracts the last frame and uses it as the reference image for sub-clip N+1. This prevents visual discontinuity within a scene but means sub-clips must generate sequentially.

> **FLUX and HuMo never run simultaneously.** They occupy the same VRAM (DiT on GPU0). The pipeline enforces sequential stage execution. Model weights are fully unloaded between stages.
```

Keep the "Design Notes" section at the bottom but add a cross-reference: "See also **Critical Design Decisions** above for the most important architectural constraints."

---

## 6. Add dependency version constraints

In the "Environment Setup" section, after the `uv sync` commands, add:

```markdown
**Pinned dependency constraints** (from HuMo requirements):
- `torch==2.5.1` with CUDA 12.4 (`--index-url https://download.pytorch.org/whl/cu124`)
- `flash_attn==2.6.3`
- Python 3.11+
- See `pyproject.toml` for the full lockfile.
```

---

## 7. Fix minor formatting and stale stats

- Line 334: add a blank line before the `---` separator (between the last bullet and the horizontal rule).
- Remove the line-count stat ("~9,200 lines across 41 Python modules + scripts") from the header. Keep only commit hash and date.
- Replace CORS note `CORS open for localhost:5173 / localhost:3000.` with `CORS open for local development (Vite and CRA default ports).`

---

## 8. Add vLLM machine to hardware description

In the "What This Project Is" section, replace:

> **Target hardware:** Dual-GPU workstation (RTX 5090 32 GB + RTX 4080 16 GB), with VRAM-tiered fallbacks for single-GPU and lighter configs.

With:

> **Target hardware:** Dual-GPU inference workstation (RTX 5090 32 GB + RTX 4080 16 GB) for image/video generation, with VRAM-tiered fallbacks for lighter configs. Separate RTX 3090 Ti (24 GB) available for local vLLM inference.
