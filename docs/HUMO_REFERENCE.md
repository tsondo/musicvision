# HuMo Technical Reference

## Overview

HuMo (Human-Centric Video Generation via Collaborative Multi-Modal Conditioning) is an open-source video generation framework by ByteDance Research. Released under Apache 2.0 license.

- **GitHub**: https://github.com/Phantom-video/HuMo
- **HuggingFace**: https://huggingface.co/bytedance-research/HuMo
- **Paper**: https://arxiv.org/abs/2509.08519

## Model Variants

| Tier | Model | Format | DiT VRAM | Speed (480p) | Speed (720p) | Quality |
|------|-------|--------|----------|-------------|-------------|---------|
| `fp16` | HuMo-17B | FP16 safetensors | ~34 GB | ~20 min | ~40 min | Best (2× GPU FSDP) |
| `fp8_scaled` | HuMo-17B | FP8 e4m3fn scaled | ~18 GB | ~25 min | ~50 min | Excellent (fits single RTX 4080 16 GB as fallback) |
| `gguf_q8` | HuMo-17B | GGUF Q8_0 | ~18.5 GB | ~25 min | ~50 min | Excellent |
| `gguf_q6` | HuMo-17B | GGUF Q6_K | ~14.4 GB | ~30 min | ~60 min | Very good |
| `gguf_q4` | HuMo-17B | GGUF Q4_K_M | ~11.5 GB | ~35 min | ~75 min | Good |
| `preview` | HuMo-1.7B | FP16 safetensors | ~3.4 GB | ~8 min | ~15 min | Good (sync quality similar to 17B) |

The 1.7B model has lower visual quality but nearly identical audio-visual sync accuracy. Good for iteration/preview; use 17B for final render.

**Note on `fp8_scaled`:** The FP8 scaled weights (`Kijai/WanVideo_comfy_fp8_scaled`) bring the 17B DiT down to ~18 GB — fitting on a single RTX 4080 (16 GB) as a secondary-GPU fallback, with model components paged via block swap.

## Installation

```bash
conda create -n humo python=3.11
conda activate humo
pip install torch==2.10.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install flash_attn==2.6.3
pip install -r requirements.txt
conda install -c conda-forge ffmpeg
```

> **Note:** MusicVision uses `torch==2.10.0` with CUDA 12.8 (required for RTX 5090 sm_120 support). The upstream HuMo repo pins `torch==2.5.1` — either version works for inference.

## Model Weights

```bash
# Base model (required — includes wan.modules package)
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir ./weights/Wan2.1-T2V-1.3B

# HuMo weights (official repo — Apache 2.0, fully public since Sep 10 2025)
huggingface-cli download bytedance-research/HuMo --local-dir ./weights/HuMo

# FP8 scaled weights — kijai's optimized version (~18 GB, fits single 16 GB GPU)
huggingface-cli download Kijai/WanVideo_comfy_fp8_scaled --local-dir ./weights/HuMo_fp8

# Audio processing
huggingface-cli download openai/whisper-large-v3 --local-dir ./weights/whisper-large-v3
huggingface-cli download huangjackson/Kim_Vocal_2 --local-dir ./weights/audio_separator
```

### Official Repository Structure

The `bytedance-research/HuMo` repo contains:
- `humo/` — Python package with model architecture
- `main.py` — entry point
- `scripts/infer_ta.sh`, `scripts/infer_tia.sh` (and `_1_7B` variants) — inference scripts
- `examples/test_case.json` — input format example
- `humo/configs/inference/generate.yaml` — inference configuration

## Input Modes

| Mode | Inputs | Use Case |
|------|--------|----------|
| **TI** (Text+Image) | text prompt + reference image | Appearance control, no audio sync |
| **TA** (Text+Audio) | text prompt + audio track | Audio-driven motion, no identity lock |
| **TIA** (Text+Image+Audio) | text + image + audio | **Full control** — identity + audio sync + scene direction |

**MusicVision uses TIA mode** for all vocal scenes and TA mode as a fallback for instrumental/abstract scenes.

## Configuration (`generate.yaml`)

```yaml
generation:
  frames: 97              # Max frames (97 = ~3.88s at 25fps). Do NOT exceed.
  scale_a: 2.0            # Audio guidance strength. Higher = stronger lip sync.
  scale_t: 7.5            # Text guidance strength. Higher = more prompt adherence.
  mode: "TIA"             # "TA" for text+audio; "TIA" for text+image+audio
  height: 720             # 720 or 480
  width: 1280             # 1280 or 832

dit:
  sp_size: 1              # Sequence parallelism. Set equal to number of GPUs.

diffusion:
  timesteps:
    sampling:
      steps: 50           # Denoising steps. 30-40 faster, 50 best quality.
```

### Guidance Scale Tips

- `scale_t` (text): 5.0-10.0 range. Upstream default 7.5; MusicVision default 5.0. Higher values = stronger prompt following but can reduce naturalness.
- `scale_a` (audio): 1.0-5.5 range. Upstream default 2.0; MusicVision default 5.5. Higher = tighter lip sync but can cause artifacts.
- For **instrumental scenes** (TA mode): reduce `scale_a` to 1.0-1.5 since there are no vocals to sync.
- HuMo uses a **time-adaptive CFG strategy** — guidance strength varies across denoising steps automatically.

## Input Format (`test_case.json`)

```json
[
  {
    "text": "A close-up of a young woman with short black hair singing into a chrome vintage microphone on a dimly lit stage. She sways gently, rain streaks visible on a window behind her. The lighting shifts subtly with the music.",
    "image": "path/to/reference_image.png",
    "audio": "path/to/audio_segment.wav",
    "output": "path/to/output.mp4"
  }
]
```

### Text Prompt Best Practices

HuMo responds best to **dense, descriptive prompts** (similar to the training captions generated by Qwen2.5-VL):

- **DO**: Describe appearance in detail (hair, clothing, accessories, expression)
- **DO**: Describe the environment and lighting
- **DO**: Describe the action/motion ("singing into a microphone", "walking forward")
- **DO**: Include camera framing ("close-up", "medium shot", "full body")
- **DON'T**: Use abstract/artistic language ("ethereal vibes", "haunting beauty")
- **DON'T**: Include temporal instructions ("then she turns around") — the clip is only 4s
- **DON'T**: Describe audio/music in the text prompt — audio input handles this

**Example prompt style** (from HuMo training data):
> "A close-up of a flight attendant with short blonde hair, seated upright in a tan leather airplane seat, speaking on a silver corded phone. She wears a dark uniform with a matching neck scarf, maintaining a composed, professional demeanor. Closed window blinds and a softly lit table lamp set the cabin background as she continues speaking, facing forward."

### Reference Image Guidelines

- High resolution, clear face visible
- HuMo preserves **identity** from the reference but can change clothing/setting per the text prompt
- The reference image does NOT need to match the scene — it's for identity, not composition
- Diverse reference images (different angle, lighting, clothing from the video scene) actually produce better results due to how the model was trained

### Audio Segment Guidelines

- WAV format recommended
- Must be the exact duration you want the video to cover
- For vocal scenes: include vocals (HuMo syncs lip movement)
- For instrumental scenes: the audio still influences motion rhythm/timing
- Kim_Vocal_2 can separate vocals if needed, but for TIA mode, use the **full mix** (vocals + instrumental) as audio input

## Running Inference

```bash
# Text+Audio mode (17B)
bash scripts/infer_ta.sh

# Text+Audio mode (1.7B)
bash scripts/infer_ta_1_7B.sh

# Text+Image+Audio mode (17B)
bash scripts/infer_tia.sh

# Text+Image+Audio mode (1.7B)
bash scripts/infer_tia_1_7B.sh
```

### Multi-GPU (FSDP + Sequence Parallel)

Set `dit.sp_size` in `generate.yaml` equal to the number of GPUs:

```yaml
dit:
  sp_size: 2    # For 2 GPUs (e.g. RTX 5090 32 GB + RTX 4080 16 GB)
```

**MusicVision two-GPU split (RTX 5090 + RTX 4080):**
- GPU 0 (RTX 5090, 32 GB): DiT/UNet computation
- GPU 1 (RTX 4080, 16 GB): T5 text encoder, VAE, Whisper encoder — all smaller models fit comfortably at 16 GB

For single-GPU setups, use `fp8_scaled` or a GGUF tier with `block_swap_count > 0` to stay within VRAM.

## Output

- Video: MP4, 25fps, 720p (1280x720) or 480p (832x480)
- 97 frames = 3.88 seconds
- No audio track in output (audio must be muxed back via ffmpeg)

## Key Limitations for MusicVision

1. **4-second max**: Each generation is ~3.88s. Longer scenes require multiple generations stitched together.
2. **No audio in output**: HuMo outputs video-only. Audio must be re-attached via ffmpeg.
3. **VRAM hungry**: 17B model needs 24GB+ even with optimizations. Cannot run alongside FLUX.
4. **Consistency between clips**: Different generations of the same character may vary slightly. Using the same reference image + consistent prompting helps. LoRA on the image generation side adds another layer of consistency.
5. **Motion complexity**: Works best for talking/singing head, upper body movement. Full-body complex choreography is hit-or-miss.
6. **No explicit camera control**: Camera movement is implicit via prompting ("tracking shot", "slow zoom") but not guaranteed.

## ComfyUI Integration

HuMo-17B is integrated into kijai's [ComfyUI-WanVideoWrapper](https://github.com/kijai/ComfyUI-WanVideoWrapper) (as of Sep 2025). This provides a battle-tested reference implementation including:
- Full denoising loop in `nodes_sampler.py` (WanVideoSampler)
- HuMoEmbeds (image + audio conditioning) in `HuMo/nodes.py`
- GGUF support and block swapping implementation

This could be an alternative execution backend for MusicVision if a node-based workflow is preferred.

## HuMoSet Dataset

670K video samples with diverse reference images, dense captions, and strict A/V sync. Released Dec 2025. Useful for:
- Understanding HuMo's expected prompt style
- Fine-tuning or training LoRAs
- Benchmarking

## Useful Links

- Official source: https://github.com/Phantom-video/HuMo (Apache 2.0)
- HuggingFace weights: https://huggingface.co/bytedance-research/HuMo
- FP8 scaled weights: https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled
- ComfyUI wrapper: https://github.com/kijai/ComfyUI-WanVideoWrapper
- HuggingFace Space: Available for quick testing without local setup
- Stage-1 dataset (subject preservation training): Released Sep 2025
- HuMoSet dataset (670K A/V training samples): Released Dec 2025
