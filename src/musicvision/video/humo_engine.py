"""
HuMo inference engine for TIA (Text-Image-Audio) video generation.

The engine is tier-agnostic: it delegates model loading to a loader class
(FP16Loader / FP8ScaledLoader / GGUFLoader / Preview1_7BLoader) and then
runs the *same* denoising loop regardless of precision.  This follows the
principle stated in the implementation plan:

  "The inference loop does not change between tiers.
   Only model loading and the linear layer forward pass differ."

Lifecycle
---------
    engine = HumoEngine(config, device_map)
    engine.load()                          # download + place weights
    output = engine.generate(humo_input)   # one clip (≤ 97 frames)
    # or for multi-clip scenes:
    outputs = engine.generate_scene(...)
    engine.unload()                        # free VRAM before FLUX / next stage

TIA mode conditioning
---------------------
  1. Encode text prompt via T5-XXL → text_embeds
  2. Encode reference image → image condition latents (HuMoEmbeds)
  3. Encode audio segment via Whisper encoder → audio_embeds
  4. Initialize noise latent (shape determined by frame count + resolution)
  5. Denoising loop (Flow Matching scheduler):
     - CFG with scale_t (text), scale_a (audio)
     - Negative conditioning: zero audio, uncond text
     - Block swap if enabled: execute blocks sequentially via BlockSwapManager
  6. VAE decode → video frames
  7. Save as MP4 (silent — audio muxed separately by assembly stage)

References
----------
- kijai/ComfyUI-WanVideoWrapper nodes_sampler.py — WanVideoSampler
- kijai/ComfyUI-WanVideoWrapper nodes.py — HuMoEmbeds
- bytedance-research/HuMo scripts/infer_tia.sh — TIA inference pipeline
- Wan-AI/Wan2.1-T2V-1.3B wan/modules/ — model architecture
"""

from __future__ import annotations

import logging
import math
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from musicvision.engine_registry import (
    ENGINES,
    get_constraints,
    sub_clip_suffixes as _registry_sub_clip_suffixes,
)
from musicvision.models import HumoConfig, HumoTier
from musicvision.video.model_loader import HumoModelBundle, get_loader

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)

# HuMo hard limits — canonical values now live in engine_registry.ENGINES["humo"].
# These are kept as deprecated aliases for any external code that imports them.
_HUMO_CONSTRAINTS = get_constraints("humo")
MAX_FRAMES   = _HUMO_CONSTRAINTS.max_frames   # 97
FPS          = _HUMO_CONSTRAINTS.fps            # 25
MAX_DURATION = _HUMO_CONSTRAINTS.max_seconds    # 3.88

# Negative prompt from original HuMo inference config (Chinese).
# Describes common generation artifacts to steer CFG away from.
_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


# ---------------------------------------------------------------------------
# Input / Output types
# ---------------------------------------------------------------------------

@dataclass
class HumoInput:
    """Input for a single HuMo TIA clip generation."""
    text_prompt: str
    reference_image: Path   # PNG — clear frontal view preferred
    audio_segment: Path     # WAV — exact clip duration
    output_path: Path       # destination MP4 (silent)
    seed: int | None = None


@dataclass
class HumoOutput:
    """Result of a single clip generation."""
    video_path: Path
    frames_generated: int
    duration_seconds: float
    seed_used: int = 0  # seed used for noise init; 0 means unset


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class HumoEngine:
    """
    Tier-agnostic HuMo video generation engine.

    The loader selected by *config.tier* handles weight format, device
    placement, and quantized forward passes.  This class only orchestrates
    the conditioning pipeline and denoising loop.
    """

    def __init__(self, config: HumoConfig, device_map: "DeviceMap") -> None:
        self.config = config
        self.device_map = device_map
        self._bundle: HumoModelBundle | None = None
        self._zero_vae: "Any" = None  # cached VAE encoding of black frame, on CPU

    @property
    def _is_dual_gpu(self) -> bool:
        """True when encoders and DiT are on separate GPUs."""
        return self.device_map.encoder_device != self.device_map.dit_device

    def _should_offload(self) -> bool:
        """Decide whether to offload idle encoder models to CPU.

        Single-GPU: always offload (everything competes for the same VRAM).
        Dual-GPU: only offload if encoder GPU has < 2 GB free (the VAE image
        encode and Whisper audio encode need working memory for activations).
        """
        if not self._is_dual_gpu:
            return True
        import torch
        enc = self.device_map.encoder_device
        if enc.type != "cuda":
            return False
        free = (torch.cuda.get_device_properties(enc).total_memory
                - torch.cuda.memory_allocated(enc)) / 1024**3
        if free < 2.0:
            log.info("Encoder GPU free VRAM %.1f GB < 2 GB — offloading to CPU", free)
            return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load all model components for the configured tier.

        Shared weights (T5, VAE, Whisper) are open models and auto-download
        on first use — no token required.  The DiT checkpoint may require
        HUGGINGFACE_TOKEN depending on the repo; if missing the engine raises
        with a clear message pointing at `musicvision download-weights`.

        Applies block swap if config.block_swap_count > 0.
        """
        import os

        from musicvision.video.weight_registry import (
            weight_status, download_dit, download_shared,
        )

        tier = self.config.tier
        status = weight_status(tier)
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

        # Shared weights are open — auto-download without requiring a token.
        # The individual loaders (_load_t5 / _load_vae / _load_whisper) also
        # handle this on FileNotFoundError, so this is an optional fast-path.
        for key in ("t5", "vae", "whisper"):
            if not status.get(key, False):
                log.info("Auto-downloading shared weight '%s' (open model)…", key)
                try:
                    download_shared(key, hf_token=hf_token)
                except Exception as exc:
                    log.warning(
                        "Pre-download of '%s' failed (%s) — loader will retry on demand.", key, exc
                    )

        # DiT checkpoint — download if missing (token required for some repos).
        if not status.get("dit", False):
            if hf_token:
                log.info("Downloading DiT weights for tier %s…", tier.value)
                download_dit(tier, hf_token=hf_token)
            else:
                raise RuntimeError(
                    f"HuMo DiT weights for tier '{tier.value}' are not present locally. "
                    f"Set HUGGINGFACE_TOKEN in .env and re-run, "
                    f"or run: musicvision download-weights --tier {tier.value}"
                )

        loader = get_loader(tier)
        log.info(
            "Loading HuMo engine (%s, block_swap=%d, %s)…",
            tier.value, self.config.block_swap_count, self.config.resolution,
        )
        self._bundle = loader.load(self.config, self.device_map)

        # Pre-compute zero_vae: VAE encoding of a black frame.
        # The original HuMo uses pre-computed zero_vae_129frame.pt for this.
        # We compute it once here and cache on CPU so _encode_image() can tile it.
        self._compute_zero_vae()

        log.info("HuMo engine ready — tier: %s", tier.value)

    def unload(self) -> None:
        """Release all model components from VRAM."""
        if self._bundle is None:
            return
        if self._bundle.block_swap is not None:
            self._bundle.block_swap.teardown()
        for attr in ("dit", "t5", "vae", "whisper"):
            obj = getattr(self._bundle, attr, None)
            if obj is not None:
                del obj
        del self._bundle
        self._bundle = None
        from musicvision.utils.gpu import clear_vram
        clear_vram()
        log.info("HuMo engine unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(self, inp: HumoInput) -> HumoOutput:
        """
        Generate a single video clip (≤ 97 frames) in TIA mode.

        Args:
            inp: HumoInput with prompt, reference image, audio, and output path.

        Returns:
            HumoOutput with path and metadata.
        """
        if self._bundle is None:
            raise RuntimeError("Call load() before generate()")

        audio_dur = _audio_duration(inp.audio_segment)
        n_frames  = min(MAX_FRAMES, math.ceil(audio_dur * FPS))
        log.info(
            "Generating %d frames (%.2fs) for %s",
            n_frames, n_frames / FPS, inp.output_path.name,
        )

        # Step 1-3: encode conditioning signals sequentially.
        # Offload each encoder after use to maximize VRAM headroom on the
        # encoder GPU (16 GB 4080). T5 ~9 GB, VAE ~1 GB, Whisper ~3 GB —
        # keeping all three loaded simultaneously is tight. Sequential
        # load/encode/offload matches ComfyUI's approach.
        text_embeds = self._encode_text(inp.text_prompt)
        self._offload("t5")

        image_cond = self._encode_image(inp.reference_image, n_frames)
        self._offload("vae")

        audio_embeds = self._encode_audio(inp.audio_segment, n_frames)
        self._offload("whisper")

        # Resolve seed: use provided seed or generate a random one so the result
        # is always reproducible.  The used seed is recorded in HumoOutput.
        seed = inp.seed if inp.seed is not None else (
            self.config.seed if self.config.seed is not None else random.randint(0, 2**32 - 1)
        )
        log.info("Using seed %d for %s", seed, inp.output_path.name)

        # Step 4-5: denoising loop
        video_latent = self._denoise(
            n_frames=n_frames,
            text_embeds=text_embeds,
            image_cond=image_cond,
            audio_embeds=audio_embeds,
            seed=seed,
        )

        # Step 6-7: decode and save — reload VAE on encoder GPU for decode
        self._reload("vae")
        frames = self._decode_latent(video_latent)
        self._offload("vae")
        _save_mp4(frames, inp.output_path, fps=FPS)

        # Step 8: Mux source audio into clip for lip sync preview
        _mux_clip_audio(inp.output_path, inp.audio_segment)

        duration = n_frames / FPS
        log.info("Clip saved → %s (%.2fs, seed=%d)", inp.output_path.name, duration, seed)
        return HumoOutput(
            video_path=inp.output_path,
            frames_generated=n_frames,
            duration_seconds=duration,
            seed_used=seed,
        )

    def generate_scene(
        self,
        text_prompt: str,
        reference_image: Path,
        audio_segment: Path,
        output_dir: Path,
        scene_id: str,
        duration: float,
        subclip_frame_counts: list[int] | None = None,
        subclip_audio_paths: list[Path] | None = None,
    ) -> list[HumoOutput]:
        """
        Generate video for a full scene, splitting into sub-clips when duration > 3.88s.

        When *subclip_frame_counts* and *subclip_audio_paths* are provided
        (from ``engine_registry.plan_subclips``), those pre-computed values
        are used directly.  Otherwise falls back to the old float-based path.

        Sub-clip continuity (config.sub_clip_continuity, default True):
            Sub-clip N+1 uses the last frame of sub-clip N as its reference image.

        Returns:
            List of HumoOutput (one per sub-clip, or a single item for short scenes).
        """
        if self._bundle is None:
            raise RuntimeError("Call load() before generate_scene()")

        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Frame-plan path (preferred) ---
        if subclip_frame_counts is not None:
            if len(subclip_frame_counts) == 1:
                output_path = output_dir / f"{scene_id}.mp4"
                result = self.generate(HumoInput(
                    text_prompt=text_prompt,
                    reference_image=reference_image,
                    audio_segment=audio_segment,
                    output_path=output_path,
                ))
                return [result]

            # Multiple sub-clips with pre-computed frame counts
            n_sub = len(subclip_frame_counts)
            suffixes = _sub_clip_suffixes(n_sub)
            outputs: list[HumoOutput] = []
            current_reference = reference_image

            for i in range(n_sub):
                sub_audio = (
                    subclip_audio_paths[i]
                    if subclip_audio_paths and i < len(subclip_audio_paths)
                    else audio_segment.parent / f"{scene_id}_sub_{i:02d}.wav"
                )
                if not sub_audio.exists():
                    log.warning("Sub-clip audio not found: %s — skipping sub-clip %d", sub_audio.name, i)
                    continue

                suffix = suffixes[i]
                sub_id = f"{scene_id}_{suffix}"
                sub_out = output_dir / f"{sub_id}.mp4"

                result = self.generate(HumoInput(
                    text_prompt=text_prompt,
                    reference_image=current_reference,
                    audio_segment=sub_audio,
                    output_path=sub_out,
                ))
                outputs.append(result)

                if self.config.sub_clip_continuity and i < n_sub - 1:
                    lastframe_path = output_dir / f"{sub_id}_lastframe.png"
                    try:
                        _extract_last_frame(result.video_path, lastframe_path)
                        current_reference = lastframe_path
                    except Exception as exc:
                        log.warning("Failed to extract last frame from %s: %s", result.video_path.name, exc)
                        current_reference = reference_image

            return outputs

        # --- Legacy float-based path ---
        if duration <= MAX_DURATION:
            output_path = output_dir / f"{scene_id}.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = self.generate(HumoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=audio_segment,
                output_path=output_path,
            ))
            return [result]

        # Scene is longer than one clip — split into sub-clips
        outputs_legacy: list[HumoOutput] = []
        n_sub = math.ceil(duration / MAX_DURATION)
        suffixes = _sub_clip_suffixes(n_sub)
        log.info(
            "Scene %s is %.2fs — splitting into %d sub-clips",
            scene_id, duration, n_sub,
        )

        current_reference = reference_image

        for i in range(n_sub):
            sub_audio = audio_segment.parent / f"{scene_id}_sub_{i:02d}.wav"
            if not sub_audio.exists():
                log.warning(
                    "Sub-clip audio not found: %s — skipping sub-clip %d",
                    sub_audio.name, i,
                )
                continue
            suffix = suffixes[i]
            sub_id = f"{scene_id}_{suffix}"
            sub_out = output_dir / f"{sub_id}.mp4"
            sub_out.parent.mkdir(parents=True, exist_ok=True)

            result = self.generate(HumoInput(
                text_prompt=text_prompt,
                reference_image=current_reference,
                audio_segment=sub_audio,
                output_path=sub_out,
            ))
            outputs_legacy.append(result)

            if self.config.sub_clip_continuity and i < n_sub - 1:
                lastframe_path = output_dir / f"{sub_id}_lastframe.png"
                try:
                    _extract_last_frame(result.video_path, lastframe_path)
                    current_reference = lastframe_path
                except Exception as exc:
                    log.warning(
                        "Failed to extract last frame from %s: %s — "
                        "falling back to original reference for sub-clip %d",
                        result.video_path.name, exc, i + 1,
                    )
                    current_reference = reference_image

        return outputs_legacy

    # ------------------------------------------------------------------
    # VRAM management: offload / reload models on encoder GPU
    # ------------------------------------------------------------------

    def _get_nn_module(self, name: str):
        """Get the underlying nn.Module for a bundle component."""
        model = getattr(self._bundle, name, None)
        if model is None:
            return None
        # WanT5Encoder: ._model.model is the nn.Module (T5Encoder)
        if hasattr(model, '_model') and hasattr(model._model, 'model'):
            return model._model.model
        # WanVideoVAE._vae is WanVAE (plain class), WanVAE.model is the nn.Module (WanVAE_)
        if hasattr(model, '_vae') and model._vae is not None:
            return model._vae.model
        # WanVideoVAE (alternate): .model is the nn.Module
        if hasattr(model, 'model'):
            return model.model
        # Whisper encoder: is directly an nn.Module
        return model

    def _offload(self, name: str) -> None:
        """Move a model component from GPU to CPU to free VRAM."""
        import torch, gc

        if self._bundle is None:
            return
        nn_mod = self._get_nn_module(name)
        if nn_mod is None:
            return
        nn_mod.to("cpu")
        # Update VAE wrapper chain: WanVideoVAE.device + WanVAE mean/std/scale tensors
        if name == "vae" and self._bundle.vae is not None:
            self._bundle.vae.device = torch.device("cpu")
            vae_inner = self._bundle.vae._vae
            if vae_inner is not None:
                vae_inner.mean = vae_inner.mean.to("cpu")
                vae_inner.std = vae_inner.std.to("cpu")
                vae_inner.scale = [vae_inner.mean, 1.0 / vae_inner.std]
        torch.cuda.empty_cache()
        gc.collect()
        log.debug("Offloaded %s to CPU", name)

    def _reload(self, name: str) -> None:
        """Move a model component back from CPU to encoder GPU (no-op if already there)."""
        import torch

        if self._bundle is None:
            return
        device = self._bundle.encoder_device
        nn_mod = self._get_nn_module(name)
        if nn_mod is None:
            return
        # Check if already on the right device
        try:
            first_param = next(nn_mod.parameters())
            if first_param.device == torch.device(device):
                return  # already on GPU
        except StopIteration:
            return
        nn_mod.to(device)
        # Update VAE wrapper chain: WanVideoVAE.device + WanVAE mean/std/scale tensors
        if name == "vae" and self._bundle.vae is not None:
            self._bundle.vae.device = torch.device(device)
            vae_inner = self._bundle.vae._vae
            if vae_inner is not None:
                vae_inner.mean = vae_inner.mean.to(device)
                vae_inner.std = vae_inner.std.to(device)
                vae_inner.scale = [vae_inner.mean, 1.0 / vae_inner.std]
                vae_inner.device = str(device)
        log.debug("Reloaded %s to %s", name, device)

    # ------------------------------------------------------------------
    # Internal: zero_vae pre-computation
    # ------------------------------------------------------------------

    def _compute_zero_vae(self) -> None:
        """
        Compute VAE encoding of a multi-frame all-black video and cache on CPU.

        The original HuMo loads pre-computed zero_vae_129frame.pt / zero_vae_720p_161frame.pt.
        The causal 3D convolution in the VAE produces DIFFERENT latent values at each
        temporal position (earlier frames have less temporal context), so encoding a
        single frame and tiling is WRONG — it gives identical values at every position
        whereas the model was trained with position-dependent zero_vae values.

        We encode a 97-frame black video (max HuMo clip length) → 25 latent frames,
        then slice to the needed length in _encode_image().
        """
        import torch

        if self._bundle is None or self._bundle.vae is None:
            log.warning("VAE not loaded — cannot compute zero_vae")
            return

        vae = self._bundle.vae
        H, W = self.config.height, self.config.width

        # Use the primary (DiT) GPU for this heavy one-time computation.
        # Encoding 97 frames through the causal 3D VAE accumulates a large
        # feature cache that can OOM on the 16GB encoder GPU.  The 32GB DiT
        # GPU has plenty of headroom, and the result is cached on CPU anyway.
        dit_device = self._bundle.dit_device
        enc_device = self._bundle.encoder_device

        # We need total_lat_f = lat_f + 1 = 26 latent frames (25 noise + 1 ref slot).
        # 129 frames → 33 latent frames (matching original's zero_vae_129frame.pt),
        # giving plenty of headroom for any clip length.
        n_black_frames = 129
        log.info("Computing zero_vae (129-frame black video at %dx%d on %s)...", W, H, dit_device)

        # Temporarily move VAE to DiT GPU for this computation
        vae_nn = self._get_nn_module("vae")
        original_vae_device = vae.device
        if vae_nn is not None:
            vae_nn.to(dit_device)
        vae.device = dit_device
        # Also move the normalization tensors
        if vae._vae is not None:
            vae._vae.mean = vae._vae.mean.to(dit_device)
            vae._vae.std = vae._vae.std.to(dit_device)
            vae._vae.scale = [vae._vae.mean, 1.0 / vae._vae.std]

        # [1, 3, 97, H, W] all-black video in [0,1] range
        black_video = torch.zeros(1, 3, n_black_frames, H, W, device=dit_device)

        with torch.no_grad():
            zero_latent = vae.encode(black_video)  # [1, 16, 25, lat_h, lat_w]

        # Cache on CPU
        self._zero_vae = zero_latent.cpu()
        del black_video, zero_latent
        torch.cuda.empty_cache()

        # Move VAE back to encoder device
        if vae_nn is not None:
            vae_nn.to(enc_device)
        vae.device = enc_device
        if vae._vae is not None:
            vae._vae.mean = vae._vae.mean.to(enc_device)
            vae._vae.std = vae._vae.std.to(enc_device)
            vae._vae.scale = [vae._vae.mean, 1.0 / vae._vae.std]

        log.info(
            "zero_vae cached: shape %s, mean=%.6f, std=%.6f",
            list(self._zero_vae.shape),
            self._zero_vae.float().mean().item(),
            self._zero_vae.float().std().item(),
        )

    # ------------------------------------------------------------------
    # Internal: conditioning encoders
    # ------------------------------------------------------------------

    def _encode_text(self, prompt: str) -> "Any":
        """
        Encode prompt via the UMT5-XXL text encoder.
        Returns (pos_embeds, neg_embeds) tuple, each [1, 512, 4096].
        """
        if self._bundle is None or self._bundle.t5 is None:
            raise RuntimeError("T5 encoder not loaded — call load() first")
        self._reload("t5")  # no-op if already on GPU
        pos_embeds, neg_embeds = self._bundle.t5.encode_pair(prompt, _NEGATIVE_PROMPT)
        return pos_embeds, neg_embeds

    def _encode_image(self, image_path: Path, n_frames: int) -> "Any":
        """
        Process reference image into image conditioning tensors.

        Returns (image_cond_pos, image_cond_neg) both [1, 20, total_lat_f, lat_h, lat_w].
        The 20 channels = 4 mask + 16 latent.

        Positive: zero_vae for noise frames, ref image latent at last position.
        Negative: zero_vae for ALL frames (including ref position).

        zero_vae is the VAE encoding of a black frame — NOT torch.zeros().
        The model was trained with these non-zero baseline values; using literal
        zeros shifts conditioning and causes noise artifacts.
        """
        if self._bundle is None or self._bundle.vae is None:
            raise RuntimeError("VAE not loaded — call load() first")

        import torch
        from PIL import Image
        import numpy as np

        vae = self._bundle.vae
        enc_device = self._bundle.encoder_device

        # Determine resolution from config
        H, W = self.config.height, self.config.width

        # Latent dimensions
        lat_f = (n_frames - 1) // 4 + 1   # temporal latent frames for noise
        total_lat_f = lat_f + 1            # +1 for reference frame
        lat_h, lat_w = H // 8, W // 8

        # Load and resize reference image — aspect-ratio-preserving + white padding
        # to match original HuMo's load_image_latent_ref_id().
        from PIL import ImageOps
        img = Image.open(image_path).convert("RGB")
        img_ratio = img.width / img.height
        target_ratio = W / H
        if img_ratio > target_ratio:
            new_w = W
            new_h = max(1, int(new_w / img_ratio))
        else:
            new_h = H
            new_w = max(1, int(new_h * img_ratio))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        delta_w = W - img.size[0]
        delta_h = H - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - delta_w // 2, delta_h - delta_h // 2)
        img = ImageOps.expand(img, padding, fill=(255, 255, 255))  # white padding

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_t = img_t.to(enc_device)

        # VAE encode the reference image
        self._reload("vae")
        with torch.no_grad():
            img_latent = vae.encode_image(img_t)  # [1, 16, 1, lat_h, lat_w]

        # Get zero_vae: pre-computed VAE encoding of multi-frame black video.
        # The causal 3D VAE produces position-dependent latent values — each temporal
        # position has different statistics.  We slice to the needed length.
        need_zero_frames = total_lat_f - 1  # non-reference positions
        if self._zero_vae is not None:
            # zero_vae is [1, 16, 25, lat_h, lat_w] on CPU — slice to needed frames
            zv_pos = self._zero_vae[:, :, :need_zero_frames, :, :].to(
                device=enc_device, dtype=img_latent.dtype
            )
            zv_neg = self._zero_vae[:, :, :total_lat_f, :, :].to(
                device=enc_device, dtype=img_latent.dtype
            )
        else:
            # Fallback: compute single-frame tiled (shouldn't happen if load() succeeded)
            log.warning("zero_vae not cached — using single-frame fallback (suboptimal)")
            black = torch.zeros(1, 3, H, W, device=enc_device)
            with torch.no_grad():
                zv_single = vae.encode_image(black)
            zv_pos = zv_single.expand(-1, -1, need_zero_frames, -1, -1)
            zv_neg = zv_single.expand(-1, -1, total_lat_f, -1, -1)

        # Build positive image condition: [1, 20, total_lat_f, lat_h, lat_w]
        # Channel layout: [4 mask | 16 latent]
        # Mask: 1 at the LAST temporal position (reference frame), 0 elsewhere.
        # The model was trained with ref-at-end; _decode_latent strips [:, :, :-1].
        mask_pos = torch.zeros(1, 4, total_lat_f, lat_h, lat_w, device=enc_device)
        mask_pos[:, :, -1, :, :] = 1.0  # ref frame at last position

        # Latent: zero_vae for noise frames, ref image at last position
        # Original HuMo: y_c = cat([zero_vae[:, :(total-1)], ref_latent], dim=1)
        latent_frames = torch.cat([zv_pos, img_latent], dim=2)  # [1, 16, total_lat_f, lat_h, lat_w]

        image_cond_pos = torch.cat([mask_pos, latent_frames], dim=1)  # [1, 20, total_lat_f, lat_h, lat_w]

        # Negative: same mask, ALL positions = zero_vae (including ref slot)
        # Original HuMo: y_null = zero_vae[:, :total], then cat with mask
        mask_neg = torch.zeros(1, 4, total_lat_f, lat_h, lat_w, device=enc_device)
        mask_neg[:, :, -1, :, :] = 1.0  # same mask position as positive
        image_cond_neg = torch.cat([mask_neg, zv_neg], dim=1)

        return image_cond_pos, image_cond_neg

    def _encode_audio(self, audio_path: Path, n_frames: int) -> "Any":
        """
        Encode audio segment via Whisper encoder → windowed audio features.
        Returns [1, total_lat_f, 8, 5, 1280] tensor.
        """
        if self._bundle is None or self._bundle.whisper is None:
            raise RuntimeError("Whisper encoder not loaded — call load() first")
        self._reload("whisper")  # no-op if already on GPU

        from musicvision.video.audio_encoder import HumoAudioEncoder

        lat_f = (n_frames - 1) // 4 + 1
        total_lat_f = lat_f + 1  # +1 for ref frame

        encoder = HumoAudioEncoder(
            whisper_model=self._bundle.whisper,
            device=self._bundle.encoder_device,
        )
        audio_features = encoder.encode(
            audio_path=audio_path,
            num_latent_frames=lat_f,
            include_ref_frame=True,
        )
        return audio_features  # [1, total_lat_f, 8, 5, 1280]

    # ------------------------------------------------------------------
    # Internal: denoising loop
    # ------------------------------------------------------------------

    def _denoise(
        self,
        n_frames: int,
        text_embeds: "Any",
        image_cond: "Any",
        audio_embeds: "Any",
        seed: int | None = None,
    ) -> "Any":
        """
        Flow-Matching denoising loop with TIA dual CFG guidance.

        Dual CFG formula (from HuMo paper):
            v_pred = v_text_neg + scale_a * (v_cond - v_audio_neg)
                     + (scale_t - 2.0) * (v_audio_neg - v_text_neg)

        Three DiT forward passes per step:
            v_cond      = dit(z + img_pos, t, pos_text, audio)
            v_audio_neg = dit(z + img_pos, t, pos_text, audio_zeros)
            v_text_neg  = dit(z + img_neg, t, neg_text, audio)
        """
        import torch
        from musicvision.video.scheduler import FlowMatchScheduler, FlowMatchUniPCScheduler

        if self._bundle is None or self._bundle.dit is None:
            raise RuntimeError("DiT not loaded — call load() first")

        dit = self._bundle.dit
        dit_device = self._bundle.dit_device
        enc_device = self._bundle.encoder_device

        # Unpack conditioning
        pos_text, neg_text = text_embeds        # each [1, 512, 4096]
        image_cond_pos, image_cond_neg = image_cond  # each [1, 20, total_lat_f, lat_h, lat_w]

        # Move all conditioning to dit device in bfloat16 (denoising dtype)
        _bf16 = torch.bfloat16
        pos_text = pos_text.to(dit_device, dtype=_bf16)
        neg_text = neg_text.to(dit_device, dtype=_bf16)
        image_cond_pos = image_cond_pos.to(dit_device, dtype=_bf16)
        image_cond_neg = image_cond_neg.to(dit_device, dtype=_bf16)

        # Audio features — move to dit device, create zero version
        if audio_embeds is not None:
            audio_embeds = audio_embeds.to(dit_device, dtype=_bf16)
            audio_zeros = torch.zeros_like(audio_embeds)
        else:
            audio_zeros = None

        # Determine latent shape
        H, W = self.config.height, self.config.width
        lat_f = (n_frames - 1) // 4 + 1
        total_lat_f = lat_f + 1
        lat_h, lat_w = H // 8, W // 8

        # Initialize noise — seed is always set by generate() before this call,
        # but guard here too for direct _denoise() callers (e.g. tests).
        # Both CPU and CUDA RNGs must be seeded: torch.randn on a CUDA device
        # uses the CUDA RNG exclusively, so manual_seed alone is not sufficient.
        if seed is not None:
            torch.manual_seed(seed)
            if dit_device.type == "cuda":
                torch.cuda.manual_seed(seed)
            elif dit_device.type == "mps":
                torch.mps.manual_seed(seed)
        # Original HuMo generates noise in float32 and relies on amp.autocast
        # for bfloat16 conversion.  bfloat16 noise has only 7 mantissa bits,
        # producing a coarser initial distribution that can accumulate errors.
        noise = torch.randn(
            1, 16, total_lat_f, lat_h, lat_w,
            device=dit_device, dtype=torch.float32,
        )
        z = noise.to(dtype=_bf16)  # convert to working precision for denoising

        # Create scheduler (shift from config, default 8.0 matches ComfyUI workflow)
        if self.config.sampler == "uni_pc":
            scheduler = FlowMatchUniPCScheduler(
                num_inference_steps=self.config.denoising_steps,
                shift=self.config.shift,
            )
        else:
            scheduler = FlowMatchScheduler(
                num_inference_steps=self.config.denoising_steps,
                shift=self.config.shift,
            )

        # Use bfloat16 tensors for CFG scales to avoid promoting v_pred to float32
        # (Python float * bfloat16 tensor → float32 promotion)
        scale_a = torch.tensor(self.config.scale_a, dtype=_bf16, device=dit_device)
        scale_t = torch.tensor(self.config.scale_t, dtype=_bf16, device=dit_device)
        _two = torch.tensor(2.0, dtype=_bf16, device=dit_device)

        use_unipc = isinstance(scheduler, FlowMatchUniPCScheduler)
        log.info(
            "Denoising: %d steps (%s), lat shape [1,16,%d,%d,%d], scale_a=%.1f, scale_t=%.1f",
            self.config.denoising_steps, "UniPC" if use_unipc else "Euler",
            total_lat_f, lat_h, lat_w,
            scale_a.item(), scale_t.item(),
        )

        use_swap = self._bundle.block_swap is not None
        total_steps = self.config.denoising_steps
        denoise_t0 = time.monotonic()
        peak_vram = 0.0

        # Lightx2V LoRA fast path: single forward pass, no CFG guidance needed
        has_lora = getattr(self._bundle, "has_lora", False)
        use_cfg1 = has_lora and self.config.scale_t == 1.0

        with torch.no_grad():
            for step_idx in range(total_steps):
                step_t0 = time.monotonic()
                # Timestep for the model: UniPC provides pre-computed timesteps
                # (already ×1000); Euler computes from sigmas.
                if use_unipc:
                    t = scheduler.timesteps[step_idx].to(device=dit_device, dtype=torch.bfloat16)
                else:
                    t = (scheduler.sigmas[step_idx] * 1000).to(device=dit_device, dtype=torch.bfloat16)
                timestep = t.expand(1)

                # Build DiT inputs: cat noise latent + image conditioning
                x_pos = torch.cat([z, image_cond_pos], dim=1)  # [1, 36, total_lat_f, lat_h, lat_w]

                if use_cfg1:
                    # Lightx2V distilled: single forward pass, guidance baked into weights
                    if use_swap:
                        v_pred = self._forward_with_swap(x_pos, timestep, pos_text, audio_embeds)
                    else:
                        v_pred = dit(x_pos, timestep, pos_text, audio_embeds)
                else:
                    x_neg = torch.cat([z, image_cond_neg], dim=1)
                    sigma = scheduler.sigmas[step_idx].item()

                    # v_cond: fully conditioned (positive text, positive image, audio)
                    # v_audio_neg: text+image but NO audio
                    # v_text_neg: time-adaptive (see below)
                    if use_swap:
                        v_cond = self._forward_with_swap(x_pos, timestep, pos_text, audio_embeds)
                        v_audio_neg = self._forward_with_swap(x_pos, timestep, pos_text, audio_zeros)
                    else:
                        v_cond = dit(x_pos, timestep, pos_text, audio_embeds)
                        v_audio_neg = dit(x_pos, timestep, pos_text, audio_zeros)

                    if sigma > 0.98:
                        # Early steps (high noise): keep positive image in negative
                        # to preserve identity. Use audio_zeros for clean baseline.
                        if use_swap:
                            v_text_neg = self._forward_with_swap(x_pos, timestep, neg_text, audio_zeros)
                        else:
                            v_text_neg = dit(x_pos, timestep, neg_text, audio_zeros)
                        # Early CFG formula (identity-preserving)
                        v_pred = (
                            v_text_neg
                            + scale_a * (v_cond - v_audio_neg)
                            + scale_t * (v_audio_neg - v_text_neg)
                        )
                    else:
                        # Later steps: full unconditional negative (null image, null text, null audio)
                        if use_swap:
                            v_text_neg = self._forward_with_swap(x_neg, timestep, neg_text, audio_zeros)
                        else:
                            v_text_neg = dit(x_neg, timestep, neg_text, audio_zeros)
                        # Standard dual CFG formula
                        v_pred = (
                            v_text_neg
                            + scale_a * (v_cond - v_audio_neg)
                            + (scale_t - _two) * (v_audio_neg - v_text_neg)
                        )

                # Diagnostic: check v_pred for NaN/Inf and log magnitude
                v_abs_mean = v_pred.abs().mean().item()
                v_has_nan = torch.isnan(v_pred).any().item()
                v_has_inf = torch.isinf(v_pred).any().item()
                if v_has_nan or v_has_inf:
                    log.error(
                        "Step %d: v_pred has NaN=%s Inf=%s — denoising will fail!",
                        step_idx, v_has_nan, v_has_inf,
                    )

                if use_unipc:
                    z = scheduler.step(v_pred, scheduler.timesteps[step_idx], z)
                else:
                    z = scheduler.step(v_pred, z, step_idx)
                # Guard: ensure z stays bfloat16 (norms and float32 sigma
                # arithmetic can promote tensors through the CFG/step chain)
                if z.dtype != _bf16:
                    z = z.to(_bf16)

                # Per-step progress at INFO level with diagnostic info
                step_dur = time.monotonic() - step_t0
                elapsed = time.monotonic() - denoise_t0
                alloc_gb = torch.cuda.memory_allocated(dit_device) / 1024**3
                total_gb = torch.cuda.get_device_properties(dit_device).total_memory / 1024**3
                peak_vram = max(peak_vram, alloc_gb)
                z_abs_mean = z.abs().mean().item()
                log.info(
                    "Step %d/%d [%.1fs] sigma=%.4f |v|=%.4f |z|=%.4f (elapsed: %.1fs) GPU%s %.1f/%.1f GB",
                    step_idx + 1, total_steps, step_dur,
                    scheduler.sigmas[step_idx].item(),
                    v_abs_mean, z_abs_mean, elapsed,
                    dit_device.index if dit_device.index is not None else 0,
                    alloc_gb, total_gb,
                )

        total_denoise = time.monotonic() - denoise_t0
        log.info(
            "Denoising complete: %d steps in %.1fs (%.2fs/step), peak VRAM %.1f GB",
            total_steps, total_denoise, total_denoise / total_steps, peak_vram,
        )
        return z  # [1, 16, total_lat_f, lat_h, lat_w]

    def _forward_with_swap(
        self,
        x: "Any",
        timestep: "Any",
        text_embeds: "Any",
        audio_features: "Any",
    ) -> "Any":
        """
        DiT forward pass with block swap: executes blocks one at a time,
        moving each from CPU to GPU and back.

        Calls dit.pre_blocks() → BlockSwapManager.execute_block() × N → dit.post_blocks()
        """
        import torch

        dit = self._bundle.dit
        swap = self._bundle.block_swap

        # Run pre-block processing (patch embed, time/text/audio conditioning)
        x_seq, block_kwargs, time_emb_raw, F_frames, h, w = dit.pre_blocks(
            x, timestep, text_embeds, audio_features
        )

        # Execute each block via block swap manager
        for block_idx in range(len(dit.blocks)):
            x_seq = swap.execute_block(
                block_idx,
                x_seq,
                **block_kwargs,
            )

        # Run post-block processing (head AdaLN + unpatchify)
        out = dit.post_blocks(x_seq, time_emb_raw, F_frames, h, w)
        return out

    # ------------------------------------------------------------------
    # Internal: VAE decode
    # ------------------------------------------------------------------

    def _decode_latent(self, latent: "Any") -> "Any":
        """
        Decode video latent → pixel-space frames tensor (T, H, W, 3) uint8.

        Strips the reference frame (last position), decodes remaining frames.
        Returns (T, H, W, 3) uint8 numpy array or torch tensor.
        """
        if self._bundle is None or self._bundle.vae is None:
            raise RuntimeError("VAE not loaded — call load() first")

        import torch

        vae = self._bundle.vae
        enc_device = self._bundle.encoder_device

        # Strip reference frame (it's appended at the end, position total_lat_f-1)
        # latent: [1, 16, total_lat_f, lat_h, lat_w]
        # Drop the last latent frame (reference frame used for conditioning)
        noise_latent = latent[:, :, :-1, :, :]  # [1, 16, lat_f, lat_h, lat_w]

        # Move to encoder device for VAE decode
        noise_latent = noise_latent.to(enc_device).to(torch.float16)

        with torch.no_grad():
            pixels = vae.decode(noise_latent)  # [1, 3, T, H, W] in [-1, 1]

        # Convert to (T, H, W, 3) uint8 on CPU (ready for ffmpeg pipe or torchvision)
        # VAE output is [-1, 1] — map to [0, 1] before scaling to [0, 255].
        pixels = pixels.squeeze(0)           # [3, T, H, W]
        pixels = pixels.permute(1, 2, 3, 0)  # [T, H, W, 3]
        pixels = (pixels.clamp(-1, 1).mul(0.5).add(0.5).mul(255)).to(torch.uint8).cpu()

        return pixels  # (T, H, W, 3) uint8, CPU


# ---------------------------------------------------------------------------
# Tier recommendation — canonical implementation is in musicvision.utils.gpu
# ---------------------------------------------------------------------------

# Re-export so existing callers that imported from here continue to work.
from musicvision.utils.gpu import recommend_tier  # noqa: E402, F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_frame(video_path: Path, output_path: Path) -> None:
    """
    Extract the last frame of a video file and save it as a PNG image.

    Used by generate_scene() for sub-clip continuity: the last frame of
    sub-clip N becomes the reference image for sub-clip N+1.

    Args:
        video_path: Source MP4 file.
        output_path: Destination PNG file.

    Raises:
        RuntimeError: If ffmpeg fails or the video cannot be read.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-sseof", "-0.1",          # seek to 0.1s before end
        "-i", str(video_path),
        "-frames:v", "1",          # grab exactly one frame
        "-q:v", "1",               # lossless-quality JPEG; PNG overrides via extension
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg last-frame extraction failed: {result.stderr.decode()}"
        )
    if not output_path.exists():
        raise RuntimeError(f"ffmpeg ran successfully but {output_path} was not created")


def _sub_clip_suffixes(n: int) -> list[str]:
    """Generate sub-clip suffixes: a, b, c, ... aa, ab, ...

    Deprecated: use ``engine_registry.sub_clip_suffixes()`` instead.
    """
    return _registry_sub_clip_suffixes(n)


def _audio_duration(path: Path) -> float:
    """Return audio duration in seconds via soundfile (no ffmpeg needed)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        from musicvision.utils.audio import get_duration
        return get_duration(path)


def _mux_clip_audio(video_path: Path, audio_path: Path) -> None:
    """Mux source audio into a video clip for lip sync preview.

    Writes to a temp file first, then replaces the original to avoid ffmpeg
    reading and writing the same file. If muxing fails, the silent video
    is preserved unchanged.
    """
    tmp_path = video_path.with_name(video_path.stem + "_tmp.mp4")
    try:
        from musicvision.utils.audio import mux_video_audio
        mux_video_audio(video_path, audio_path, tmp_path)
        tmp_path.replace(video_path)
        log.info("Audio muxed into %s", video_path.name)
    except Exception as exc:
        log.warning("Audio mux failed for %s: %s — keeping silent video", video_path.name, exc)
        if tmp_path.exists():
            tmp_path.unlink()


def _save_mp4(frames: "Any", path: Path, fps: int = 25) -> None:
    """
    Save frames tensor to a silent MP4 file.

    frames: (T, H, W, 3) uint8 numpy array or torch tensor

    Uses torchvision.io.write_video when available; falls back to ffmpeg pipe.

    TODO: implement once frame tensor format is known from VAE decode output.
    """
    if frames is None:
        log.warning("_save_mp4: no frames to save (generation stubs not yet implemented)")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        import torchvision.io as tio
        if not isinstance(frames, torch.Tensor):
            frames = torch.from_numpy(frames)
        tio.write_video(str(path), frames.cpu(), fps=fps)
    except Exception as exc:
        log.error("Failed to save MP4 via torchvision: %s — trying ffmpeg pipe", exc)
        _save_frames_as_mp4_ffmpeg(frames, path, fps=fps)


def _save_frames_as_mp4_ffmpeg(frames, output_path: Path, fps: int = 25) -> None:
    """
    Save a (T, H, W, C) uint8 tensor as MP4 using ffmpeg raw pipe.

    Uses raw video pipe to ffmpeg to avoid torchvision.io dependency issues.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torch
    if not isinstance(frames, torch.Tensor):
        import numpy as np
        frames = torch.from_numpy(np.asarray(frames))

    t, h, w, c = frames.shape

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an",  # no audio
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(frames.cpu().numpy().tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed saving video: {stderr}")


# Type alias for annotations inside this module (avoids circular imports)
Any = object
