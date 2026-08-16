# Third-Party Notices

MusicVision's own source code is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). That grant covers the **code only**.

MusicVision also integrates third-party AI models, libraries, and tools that are **not owned by or licensed from Todd Green** and are **not covered by MusicVision's Apache 2.0 license**. Each component is governed solely by its own license, listed below. In particular, the AI **model weights** the pipeline downloads at runtime keep their own terms — and several restrict commercial use regardless of MusicVision's license.

**Users are responsible for complying with all applicable upstream licenses**, including any restrictions on commercial use. MusicVision's Apache 2.0 license does not lift or override any upstream restriction on model weights.

---

## Vendored Code

The following source files are vendored (copied and adapted) into the MusicVision repository. They retain their original upstream licenses (Apache 2.0) and copyright attribution, independent of MusicVision's own Apache 2.0 license.

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
| [HuMo](https://github.com/Phantom-video/HuMo) | ByteDance / Phantom-video | Apache 2.0 | **Yes** | Audio-conditioned video (TIA mode). Built on Wan2.1-T2V-1.3B. |
| [LTX-Video 2](https://github.com/Lightricks/LTX-2) | Lightricks | [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) | Conditional — free under $10M ARR | Joint audio+video DiT. Free for academic research and commercial use by companies with less than $10M annual recurring revenue. Organizations above that threshold must obtain a separate commercial license from Lightricks. |

### Image Generation

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [FLUX.1-dev](https://github.com/black-forest-labs/flux) | Black Forest Labs | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) | **No** | Gated model. Non-commercial only. |
| [FLUX.1-schnell](https://github.com/black-forest-labs/flux) | Black Forest Labs | Apache 2.0 | **Yes** | Commercially permissive alternative to FLUX.1-dev. |
| [Z-Image / Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | Tongyi-MAI (Alibaba) | Apache 2.0 | **Yes** | Ungated, fast inference. |

### Image Consistency — IP-Adapters (FLUX)

Optional zero-shot character/style consistency adapters for FLUX. Each is a derivative of FLUX.1-dev and inherits the **FLUX.1-dev Non-Commercial License**. `IPAdapterConfig` (asset library) can select among them; the default is pending an external bake-off.

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [FLUX IP-Adapter v1](https://huggingface.co/XLabs-AI/flux-ip-adapter) | XLabs-AI | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) | **No** | CLIP-ViT-L encoder. Mature, well-tested in ComfyUI. |
| [FLUX IP-Adapter v2](https://huggingface.co/XLabs-AI/flux-ip-adapter-v2) | XLabs-AI | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) | **No** | CLIP-ViT-L encoder. More training than v1. |
| [InstantX FLUX IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) | InstantX | [FLUX.1-dev Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md) | **No** | SigLIP-so400m encoder. Better face fidelity reported. |

### Upscaling

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [SeedVR2](https://github.com/ByteDance/SeedVR2) | ByteDance | Apache 2.0 | **Yes** | Face-aware video upscaling. |
| [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | Xintao Wang et al. | BSD 3-Clause | **Yes** | Frame-by-frame super-resolution. |
| [LTX Spatial Upsampler](https://github.com/Lightricks/LTX-2) | Lightricks | [LTX-2 Community License](https://github.com/Lightricks/LTX-2/blob/main/LICENSE) | Conditional — free under $10M ARR | Latent-space upsampler for LTX-2 output. Same license terms as LTX-Video 2. |

### Audio & Text

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [Whisper large-v3](https://github.com/openai/whisper) | OpenAI | MIT | **Yes** | Speech transcription and alignment. |
| [Kim_Vocal_2 / Demucs](https://github.com/facebookresearch/demucs) | Meta Research | MIT | **Yes** | Vocal separation. |
| [UMT5-XXL](https://huggingface.co/google/umt5-xxl) | Google | Apache 2.0 | **Yes** | Text encoder for Wan2.1/HuMo. Weights used via vendored T5 architecture. |

### LLM (Optional)

| Model | Author | License | Commercial Use | Notes |
|-------|--------|---------|----------------|-------|
| [Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) | Alibaba Qwen (AWQ quant by cyankiwi) | Apache 2.0 | **Yes** | Local LLM for scene segmentation / prompt generation (via vLLM). Optional — Claude API is the default backend. |

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
| [ffmpeg](https://ffmpeg.org/) | LGPL 2.1+ / GPL 2+ | Audio slicing, video concatenation, muxing. License depends on build configuration — most Linux/Homebrew builds include GPL-licensed codecs (x264, x265). Statically linking or bundling ffmpeg with these codecs triggers GPL copyleft obligations. Commercial users should verify their ffmpeg build's license with `ffmpeg -L`. |
| [vLLM](https://github.com/vllm-project/vllm) | Apache 2.0 | Local LLM serving (optional). |

---

## Commercial Use Summary

MusicVision's own code is Apache 2.0 and may be used commercially. That says **nothing** about the model weights: each model's weights carry their own license, and using MusicVision commercially does not grant any right to a weight whose license forbids it. You must **independently** ensure that every upstream model you run permits your intended use. The following components have known non-commercial or restricted licenses:

- **FLUX.1-dev** — non-commercial only. For commercial use, route image generation through **FLUX.1-schnell** (Apache 2.0) or **Z-Image / Z-Image-Turbo** (Apache 2.0) instead.
- **FLUX IP-Adapters (XLabs v1/v2, InstantX)** — inherit the FLUX.1-dev Non-Commercial License. Do not use for commercial output.
- **LTX-Video 2 / LTX Spatial Upsampler** — LTX-2 Community License. Free for academic research and commercial use by companies under $10M ARR. Companies above $10M ARR must obtain a commercial license from Lightricks.

All other models and libraries listed above are under permissive licenses (Apache 2.0, MIT, BSD) that generally allow commercial use, but users should verify current license terms at the upstream repositories before commercial deployment.

---

*Last updated: 2026-07-05*
