# MusicVision Testing

MusicVision uses a two-layer test strategy: fast unit tests that run anywhere, and integration tests that require real hardware or services.

---

## Unit Tests (`tests/`)

- **Runner:** `python -m pytest tests/ -v --tb=short`
- **Requirements:** No GPU, no network, no model weights. Just the Python environment.
- **Purpose:** Test data models, config parsing, segmentation logic, prompt construction, JSON serialization, project scaffolding.
- **When to run:** After any code change. These should always pass.

### Test files

| File | Covers |
|------|--------|
| `test_core.py` | Project config, scene models, style sheet serialization, project service lifecycle |
| `test_intake.py` | Segmentation logic, timestamp parsing, AceStep metadata integration |
| `test_image_engine.py` | FLUX + Z-Image engine config, factory dispatch, prompt generator, batch prompt parsing |
| `test_video_engine.py` | HuMo engine config, video prompt construction, sub-clip splitting logic |
| `test_hunyuan_avatar_engine.py` | HVA config, VideoEngineType enum, factory dispatch, engine lifecycle, scene splitting |

---

## Integration Tests (`scripts/`)

- **Runner:** `python scripts/<test_name>.py [options]`
- **Requirements:** Vary per script — GPU, model weights, running vLLM server, etc.
- **Purpose:** Test actual inference, checkpoint loading, end-to-end pipeline with real hardware.
- **When to run:** Before merging significant changes, after hardware/driver updates, or when validating a new model tier.

### Scripts

| Script | What it tests | Requirements |
|--------|--------------|--------------|
| `test_image_gen.py` | Z-Image-Turbo + FLUX-schnell GPU image generation (2 images each, man + woman) | GPU + `HUGGINGFACE_TOKEN` for FLUX |
| `test_gpu_pipeline.py` | Full 4-phase HuMo GPU integration test (model loading, single clip, sub-clip splitting, assembly) | GPU + HuMo weights. See `MUSICVISION_GPU_TEST.md`. |
| `test_humo_inference.py` | Isolated HuMo checkpoint loading and inference smoke test | GPU + HuMo weights |
| `test_vllm_prompts.py` | All three LLM prompt paths (segmentation, image prompts, video prompts, batch consistency) against a running vLLM server | vLLM server running on LAN with Qwen2.5-32B-AWQ |
| `dump_keys.py` | Utility to dump checkpoint `state_dict` keys for debugging weight loading issues | Model weights on disk |

---

## End-to-End CLI Test

Full pipeline from terminal (requires GPU + weights + audio file):

```bash
musicvision create ./test_storyboard --name "Test Storyboard"
musicvision import-audio --project ./test_storyboard --audio /path/to/song.wav --lyrics /path/to/lyrics.txt
musicvision intake --project ./test_storyboard --skip-transcription
musicvision info ./test_storyboard
musicvision generate-images --project ./test_storyboard --model z-image-turbo
musicvision generate-video --project ./test_storyboard --engine hunyuan_avatar
musicvision assemble --project ./test_storyboard
# → test_storyboard/output/rough_cut.mp4
```

---

## Recommended Test Workflow

1. **After code changes:** `python -m pytest tests/ -v --tb=short` (123 tests, <10 seconds)
2. **After LLM prompt changes:** run pytest, then `python scripts/test_vllm_prompts.py`
3. **After image engine changes:** `python scripts/test_image_gen.py`
4. **After HuMo/video changes:** `python scripts/test_gpu_pipeline.py`
5. **Before any major milestone:** run all of the above
