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
| `test_image_engine.py` | FLUX engine config, prompt generator construction, batch prompt parsing |
| `test_video_engine.py` | HuMo engine config, video prompt construction, sub-clip splitting logic |

---

## Integration Tests (`scripts/`)

- **Runner:** `python scripts/<test_name>.py [options]`
- **Requirements:** Vary per script — GPU, model weights, running vLLM server, etc.
- **Purpose:** Test actual inference, checkpoint loading, end-to-end pipeline with real hardware.
- **When to run:** Before merging significant changes, after hardware/driver updates, or when validating a new model tier.

### Scripts

| Script | What it tests | Requirements |
|--------|--------------|--------------|
| `test_gpu_pipeline.py` | Full 4-phase GPU integration test (model loading, single clip, sub-clip splitting, assembly) | GPU + model weights. See `MUSICVISION_GPU_TEST.md` for detailed instructions. |
| `test_humo_inference.py` | Isolated HuMo checkpoint loading and inference smoke test | GPU + HuMo weights |
| `test_vllm_prompts.py` | All three LLM prompt paths (segmentation, image prompts, video prompts, batch consistency) against a running vLLM server | vLLM server running on LAN with Qwen2.5-32B-AWQ |
| `dump_keys.py` | Utility to dump checkpoint `state_dict` keys for debugging weight loading issues | Model weights on disk |

---

## Recommended Test Workflow

1. **After code changes:** `python -m pytest tests/ -v --tb=short` (should take <10 seconds)
2. **After LLM prompt changes:** run pytest, then `python scripts/test_vllm_prompts.py`
3. **After GPU/model/driver changes:** `python scripts/test_gpu_pipeline.py`
4. **Before any major milestone:** run all of the above
