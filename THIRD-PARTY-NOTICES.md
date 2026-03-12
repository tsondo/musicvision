# Third-Party Notices

MusicVision integrates third-party AI models, vendored source code, and open-source libraries. This document lists their licenses and attribution. MusicVision's own license ([PolyForm Noncommercial 1.0.0](LICENSE) / [Commercial](LICENSE-COMMERCIAL)) covers only the MusicVision source code and tooling — it does not grant any rights to the components listed below.

Users are responsible for complying with all applicable upstream licenses independently.

---

## Vendored Source Code

These files are included directly in the MusicVision repository under `src/musicvision/video/vendor/`. They have been adapted for self-contained use; modifications are documented in [FIXLOG.md](docs/FIXLOG.md).

| File | Source | License | Copyright |
|------|--------|---------|-----------|
| `wan_dit_arch.py` | [Phantom-video/HuMo](https://github.com/Phantom-video/HuMo) | Apache 2.0 | 2024-2025 The Alibaba Wan Team Authors |
| `wan_t5_arch.py` | [Wan-AI/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache 2.0 | 2024-2025 The Alibaba Wan Team Authors |
| `wan_vae_arch.py` | [Wan-AI/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache 2.0 | 2024-2025 The Alibaba Wan Team Authors |
| `wan_tokenizers.py` | [Wan-AI/Wan2.1](https://github.com/Wan-Video/Wan2.1) | Apache 2.0 | 2024-2025 The Alibaba Wan Team Authors |

Full text of the Apache 2.0 license: https://www.apache.org/licenses/LICENSE-2.0

---

## AI Models (Downloaded at Runtime)

These models are not distributed with MusicVision. They are downloaded by the user at runtime from their respective sources. Users must comply with each model's license independently.

### Video Generation

| Model | Creator | License | Commercial Use | Source |
|-------|---------|---------|---------------|--------|
| HuMo (14B / 1.7B) | ByteDance | Apache 2.0 | Yes | [GitHub](https://github.com/Phantom-video/HuMo) |
| HunyuanVideo-Avatar | Tencent | Tencent Hunyuan Community License | Check license | [GitHub](https://github.com/tencent/HunyuanVideo) |
| LTX-Video 2 | Lightricks | OpenRail-M | Yes (with behavioral restrictions) | [GitHub](https://github.com/Lightricks/LTX-Video) |
| Wan2.1 (base weights) | Wan-AI / Alibaba | Apache 2.0 | Yes | [GitHub](https://github.com/Wan-Video/Wan2.1) |

### Image Generation

| Model | Creator | License | Commercial Use | Source |
|-------|---------|---------|---------------|--------|
| FLUX.1-dev | Black Forest Labs | FLUX.1-dev Non-Commercial License | **No** — requires separate commercial license from BFL | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| FLUX.1-schnell | Black Forest Labs | Apache 2.0 | Yes | [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |
| Z-Image / Z-Image-Turbo | Tongyi (Alibaba) | Apache 2.0 | Yes | [HuggingFace](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) |

### Audio & Speech

| Model | Creator | License | Commercial Use | Source |
|-------|---------|---------|---------------|--------|
| Whisper large-v3 | OpenAI | MIT | Yes | [GitHub](https://github.com/openai/whisper) |
| Demucs v4 | Meta AI | MIT | Yes | [GitHub](https://github.com/facebookresearch/demucs) |

### Upscaling

| Model | Creator | License | Commercial Use | Source |
|-------|---------|---------|---------------|--------|
| SeedVR2 | ByteDance | Apache 2.0 | Yes | [GitHub](https://github.com/ByteDance-Seed/SeedVR) |
| Real-ESRGAN | xinntao | BSD 3-Clause | Yes | [GitHub](https://github.com/xinntao/Real-ESRGAN) |

### Text Encoding

| Model | Creator | License | Commercial Use | Source |
|-------|---------|---------|---------------|--------|
| T5 / UMT5-XXL | Google | Apache 2.0 | Yes | [HuggingFace](https://huggingface.co/google/umt5-xxl) |

---

## Key License Implications

**Non-commercial users:** All models listed above are available for non-commercial use.

**Commercial users:** Most models permit commercial use. The notable exception is **FLUX.1-dev**, which has a non-commercial license. Commercial deployments should use **FLUX.1-schnell** (Apache 2.0) or **Z-Image** (Apache 2.0) instead, or obtain a separate commercial license from Black Forest Labs.

---

## Python Dependencies

MusicVision's Python dependencies are listed in `pyproject.toml`. Key licenses include:

- PyTorch, torchvision, torchaudio — BSD (Meta)
- transformers, diffusers, accelerate, safetensors — Apache 2.0 (Hugging Face)
- FastAPI — MIT
- Pydantic — MIT
- librosa — ISC
- Pillow — HPND
- NumPy, SciPy — BSD

For a complete list, run `uv pip list --format=columns` in the project virtualenv.
