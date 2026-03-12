# Third-Party Notices

MusicVision integrates third-party AI models, libraries, and tools that are **not owned by or licensed from Todd Green** and are **not covered by the MusicVision license** (PolyForm Noncommercial 1.0.0 or any MusicVision commercial license). Each component is governed solely by its own license, listed below.

**Users are responsible for complying with all applicable upstream licenses**, including any restrictions on commercial use. A MusicVision commercial license does not override upstream restrictions.

---

## Vendored Code

The following source files are vendored (copied and adapted) into the MusicVision repository. They retain their original licenses and are **not** covered by the PolyForm Noncommercial License 1.0.0.

| File | Origin | License | Modifications |
|------|--------|---------|---------------|
| `src/musicvision/video/vendor/wan_dit_arch.py` | [Phantom-video/HuMo](https://github.com/Phantom-video/HuMo) | Apache 2.0 | SDPA fallback (no hard `flash_attn` dep), removed `einops`/`diffusers`/Ulysses SP deps, manual reshape/permute |
| `src/musicvision/video/vendor/wan_t5_arch.py` | [Wan-AI/Wan2.1](https://github.com/Wan-AI/Wan2.1) | Apache 2.0 | Standalone extraction, no upstream module deps |
| `src/musicvision/video/vendor/wan_tokenizers.py` | [Wan-AI/Wan2.1](https://github.com/Wan-AI/Wan2.1) | Apache 2.0 | Standalone extraction |
| `src/musicvision/video/vendor/wan_vae_arch.py` | [Wan-AI/Wan2.1](https://github.com/Wan-AI/Wan2.1) | Apache 2.0 | Standalone extraction |

See `docs/FIXLOG.md` for detailed modification history.

---

## AI Model Weights (Downloaded at Runtime)

MusicVision does not distribute model weights. Weights are downloaded by the user at runtime from their respective sources. Each model's license governs the user's rights to those weights independently of MusicVision's license.

### Video Generation

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [HunyuanVideo-Avatar](https://github.com/tencent/HunyuanVideo) | Tencent | [Tencent Hunyuan Community License](https://github.com/Tencent/HunyuanVideo/blob/main/LICENSE.txt) | Restricted — see license | Audio-driven video with lip sync. License permits non-commercial and limited commercial use; review terms carefully. |
| [HuMo](https://github.com/Phantom-video/HuMo) | ByteDance / Phantom-video | Apache 2.0 | **Yes** | Audio-conditioned video (TIA mode). Built on Wan2.1-T2V-1.3B. |
| [LTX-Video 2](https://github.com/Lightricks/LTX-Video) | Lightricks | [LTXV License](https://github.com/Lightricks/LTX-Video/blob/main/LICENSE) | Restricted — see license | Joint audio+video DiT. Review license for commercial terms. |

### Image Generation

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [FLUX.1-dev](https://github.com/black-forest-labs/flux) | Black Forest Labs | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) | **No** | Gated model. Non-commercial only. |
| [FLUX.1-schnell](https://github.com/black-forest-labs/flux) | Black Forest Labs | Apache 2.0 | **Yes** | Commercially permissive alternative to FLUX.1-dev. |
| [Z-Image / Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI (Alibaba) | Apache 2.0 | **Yes** | Ungated, fast inference. |

### Upscaling

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [SeedVR2](https://github.com/ByteDance/SeedVR2) | ByteDance | Apache 2.0 | **Yes** | Face-aware video upscaling. |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Xintao Wang et al. | BSD 3-Clause | **Yes** | Frame-by-frame super-resolution. |
| [LTX Spatial Upsampler](https://github.com/Lightricks/LTX-Video) | Lightricks | See LTX-Video license | Restricted — see license | Latent-space upsampler for LTX-2 output. |

### Audio & Text

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [Whisper large-v3](https://github.com/openai/whisper) | OpenAI | MIT | **Yes** | Speech transcription and alignment. |
| [Kim_Vocal_2 / Demucs](https://github.com/facebookresearch/demucs) | Meta Research | MIT | **Yes** | Vocal separation. |
| [UMT5-XXL](https://huggingface.co/google/umt5-xxl) | Google | Apache 2.0 | **Yes** | Text encoder for Wan2.1/HuMo. Weights used via vendored T5 architecture. |

### LLM (Optional)

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [Qwen2.5-32B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ) | Alibaba Qwen | Apache 2.0 | **Yes** | Local LLM for scene segmentation / prompt generation (via vLLM). Optional — Claude API is the default backend. |

---

## Python Libraries (Installed via pip)

These are standard open-source dependencies installed into the user's environment. This is not an exhaustive list — see `pyproject.toml` for the full dependency specification.

| Library | License | Notes |
|---------|---------|-------|
| PyTorch | BSD 3-Clause | Core ML framework |
| Transformers (Hugging Face) | Apache 2.0 | Model loading, Whisper |
| Diffusers (Hugging Face) | Apache 2.0 | FLUX, LTX-Video pipelines |
| Accelerate (Hugging Face) | Apache 2.0 | Model offloading |
| Safetensors | Apache 2.0 | Weight loading |
| FastAPI | MIT | REST API server |
| Pydantic | MIT | Data models |
| Pillow | HPND | Image processing |
| NumPy | BSD 3-Clause | Numerical computing |
| LibROSA | ISC | Audio analysis |
| einops | MIT | Tensor operations (used by vendored VAE) |

---

## Tools

| Tool | License | Notes |
|------|---------|-------|
| [ffmpeg](https://ffmpeg.org/) | LGPL 2.1+ / GPL 2+ | Audio slicing, video concatenation, muxing. License depends on build configuration. |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 | Local LLM serving (optional). |

---

## Commercial Use Summary

If you hold a MusicVision commercial license, you must **independently** ensure that every upstream model and library you use permits commercial use. The following models have known non-commercial or restricted licenses:

- **FLUX.1-dev** — non-commercial only. Use FLUX.1-schnell (Apache 2.0) instead.
- **HunyuanVideo-Avatar** — Tencent Hunyuan Community License. Review terms before commercial deployment.
- **LTX-Video 2** — Lightricks license. Review terms before commercial deployment.

All other models and libraries listed above are under permissive licenses (Apache 2.0, MIT, BSD) that generally allow commercial use, but users should verify current license terms at the upstream repositories before commercial deployment.

---

*Last updated: 2026-03-12*
