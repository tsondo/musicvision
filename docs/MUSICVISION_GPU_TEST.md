# MusicVision — GPU Integration Test Guide

Run `scripts/test_gpu_pipeline.py` to verify the full HuMo video generation pipeline on your GPU hardware. This script tests model loading, single-clip inference, sub-clip splitting for long scenes, and final assembly — all without needing a real song or reference photo.

---

## Prerequisites

1. **Environment installed** — you should already have run:
   ```bash
   uv sync --extra ml
   pip install "audio-separator[gpu]"
   ```

2. **Environment variables** — copy `.env.example` to `.env` and fill in:
   - `HUGGINGFACE_TOKEN` — required to download HuMo DiT weights (shared weights like T5/VAE/Whisper download automatically without a token)
   - `ANTHROPIC_API_KEY` — *not* needed for these tests

3. **ffmpeg installed** — the test uses ffmpeg/ffprobe for audio slicing and video validation:
   ```bash
   ffmpeg -version   # should print version info
   ```

4. **Model weights downloaded** — either let the test download them on first run, or pre-download:
   ```bash
   musicvision download-weights --tier fp8_scaled
   ```
   If you plan to test a different tier, replace `fp8_scaled` with your tier of choice.

---

## Quick Start (Simplest Run)

From the repo root:

```bash
python scripts/test_gpu_pipeline.py
```

This will:
- Auto-detect your GPUs and pick the best tier for your VRAM
- Synthesize a 10-second test tone and a placeholder reference image
- Run all four test phases
- Save outputs to `test_output/<timestamp>/`

**Expect it to take 10–40 minutes** depending on your GPU and the auto-selected tier. The bulk of the time is phases 2 and 3 (actual HuMo inference).

---

## What Each Phase Tests

| Phase | Name | What It Does | Pass Criteria |
|-------|------|--------------|---------------|
| 1 | Hardware detection | Finds GPUs, picks a tier, checks if weights are present | CUDA available, all weight files found |
| 2 | Single clip smoke test | Loads HuMo engine, generates one ~3.8s video clip | Output MP4 exists, has video track, duration > 0.5s |
| 3 | Scene splitting | Generates a 7.5s scene (auto-splits into 2 sub-clips), checks last-frame continuity | 2 sub-clips produced, both valid, `_lastframe.png` exists |
| 4 | Assembly | Concatenates all clips, muxes audio back on top | Rough cut MP4 valid, duration within 1.5s of expected |

---

## Common Options

**Force a specific tier** (skip auto-detection):
```bash
python scripts/test_gpu_pipeline.py --tier fp8_scaled
```
Available tiers: `fp16`, `fp8_scaled`, `gguf_q8`, `gguf_q6`, `gguf_q4`, `preview`

**Reduce denoising steps** (faster but lower quality — fine for testing):
```bash
python scripts/test_gpu_pipeline.py --steps 20
```

**Use block swap** (if you're tight on VRAM — moves N DiT blocks to CPU):
```bash
python scripts/test_gpu_pipeline.py --block-swap 20
```

**Use your own song and reference image** instead of synthetics:
```bash
python scripts/test_gpu_pipeline.py --audio ~/music/song.wav --image ~/photos/ref.png
```
Audio should be at least 8 seconds. Image should be a clear face/figure shot.

**Run only one phase** (useful for debugging a specific failure):
```bash
python scripts/test_gpu_pipeline.py --phase 1   # hardware only
python scripts/test_gpu_pipeline.py --phase 2   # single clip only
python scripts/test_gpu_pipeline.py --phase 3   # scene splitting only
python scripts/test_gpu_pipeline.py --phase 4   # assembly only (needs clips from phase 2/3)
```

**Custom output directory:**
```bash
python scripts/test_gpu_pipeline.py --out-dir ./my_test_run
```

---

## Reading the Output

The script prints color-coded results as it goes:

- **PASS** (green) — that check succeeded
- **FAIL** (red) — something broke; read the note next to it
- **WARN** (yellow) — skipped (e.g., no clips available for assembly)

At the end you get a summary table. Exit code is 0 if everything passed, 1 if anything failed.

### Output directory structure

```
test_output/<timestamp>/
├── assets/
│   ├── test_audio.wav          # synthesized test tone (if no --audio provided)
│   └── test_ref.png            # generated placeholder image (if no --image provided)
├── segments/
│   ├── smoke_test.wav          # 3.8s audio slice for phase 2
│   ├── scene_test.wav          # 7.5s audio slice for phase 3
│   ├── scene_test_sub_00.wav   # first sub-clip audio
│   └── scene_test_sub_01.wav   # second sub-clip audio
├── clips/
│   ├── smoke_test.mp4          # phase 2 output
│   ├── scene_test_a.mp4        # phase 3 sub-clip 0
│   ├── scene_test_a_lastframe.png  # continuity frame
│   └── scene_test_b.mp4        # phase 3 sub-clip 1
└── output/
    └── rough_cut.mp4           # phase 4 final assembly
```

---

## Troubleshooting

**"Missing weights and no HUGGINGFACE_TOKEN found"**
Set `HUGGINGFACE_TOKEN` in your `.env` file or export it in your shell, then either re-run the test (it will download automatically) or run `musicvision download-weights --tier <your-tier>` first.

**Phase 2 fails with CUDA OOM**
Try a lighter tier (`--tier gguf_q4` or `--tier preview`) or enable block swap (`--block-swap 20`).

**Phase 2 fails with a key mismatch / missing keys**
This likely means the checkpoint key names don't match our model code. Check the logged missing/unexpected keys. See FIXLOG.md for context on past key-mapping issues.

**Phase 3 produces 1 sub-clip instead of 2**
The sub-clip splitting logic may have a boundary calculation issue. Check that the 7.5s scene audio was correctly sliced by looking at `segments/scene_test_sub_00.wav` and `_sub_01.wav`.

**Phase 4 skipped ("No clips available")**
Earlier phases failed to produce any clips. Fix the phase 2/3 failures first.

**ffmpeg/ffprobe not found**
Install ffmpeg. On Fedora: `sudo dnf install ffmpeg`. On Ubuntu: `sudo apt install ffmpeg`.

---

## After the Test Passes

Once all four phases pass, you're ready to run the full pipeline on a real song:

```bash
musicvision create ./my-video --name "My Video"
musicvision serve ./my-video
# Open http://localhost:8000/docs to interact via the API
```

Or generate video for specific scenes via CLI:
```bash
musicvision generate-video --project ./my-video --tier fp8_scaled --scene-ids scene_001 scene_002
```
