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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from musicvision.models import HumoConfig, HumoTier
from musicvision.video.model_loader import HumoModelBundle, get_loader

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)

# HuMo hard limits
MAX_FRAMES   = 97           # 97 frames @ 25 fps = 3.88 s
FPS          = 25
MAX_DURATION = MAX_FRAMES / FPS   # 3.88 s


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

        # Step 1-3: encode conditioning signals (sequential — no CUDA stream overlap)
        # T5 encoding runs on GPU1 (~0.3–1s/scene) and noise init on GPU0 are
        # independent, but stream overlap adds complexity for minimal gain at
        # single-scene throughput.  Re-evaluate if batching multiple scenes.
        # Future: VAE decode (GPU1) could overlap the next scene's T5 encode.
        # TODO: measure T5 encode time on GPU to quantify the opportunity.
        text_embeds  = self._encode_text(inp.text_prompt)
        image_cond   = self._encode_image(inp.reference_image, n_frames)
        audio_embeds = self._encode_audio(inp.audio_segment, n_frames)

        # Resolve seed: use provided seed or generate a random one so the result
        # is always reproducible.  The used seed is recorded in HumoOutput.
        seed = inp.seed if inp.seed is not None else random.randint(0, 2**32 - 1)
        log.info("Using seed %d for %s", seed, inp.output_path.name)

        # Step 4-5: denoising loop
        video_latent = self._denoise(
            n_frames=n_frames,
            text_embeds=text_embeds,
            image_cond=image_cond,
            audio_embeds=audio_embeds,
            seed=seed,
        )

        # Step 6-7: decode and save
        frames = self._decode_latent(video_latent)
        _save_mp4(frames, inp.output_path, fps=FPS)

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
    ) -> list[HumoOutput]:
        """
        Generate video for a full scene, splitting into sub-clips when duration > 3.88s.

        Sub-clip continuity (config.sub_clip_continuity, default True):
            Sub-clip N+1 uses the last frame of sub-clip N as its reference image.
            This produces smooth visual continuity across sub-clips without requiring
            the model to re-anchor to a static reference image for each segment.
            The first sub-clip always uses the scene's original reference image.
            Disable via HumoConfig.sub_clip_continuity = False to revert to static
            reference behaviour (all sub-clips share the original reference image).

        Sub-clip audio segments must be pre-sliced and stored as
        <segment_dir>/scene_XXX_sub_NN.wav by the intake pipeline.

        Returns:
            List of HumoOutput (one per sub-clip, or a single item for short scenes).
        """
        if self._bundle is None:
            raise RuntimeError("Call load() before generate_scene()")

        output_dir.mkdir(parents=True, exist_ok=True)

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
        outputs: list[HumoOutput] = []
        n_sub = math.ceil(duration / MAX_DURATION)
        suffixes = _sub_clip_suffixes(n_sub)
        log.info(
            "Scene %s is %.2fs — splitting into %d sub-clips",
            scene_id, duration, n_sub,
        )

        current_reference = reference_image  # updated per sub-clip when continuity is on

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
            outputs.append(result)

            # Extract last frame for next sub-clip (continuity mode)
            if self.config.sub_clip_continuity and i < n_sub - 1:
                lastframe_path = output_dir / f"{sub_id}_lastframe.png"
                try:
                    _extract_last_frame(result.video_path, lastframe_path)
                    current_reference = lastframe_path
                    log.debug(
                        "Sub-clip continuity: using %s as reference for sub-clip %d",
                        lastframe_path.name, i + 1,
                    )
                except Exception as exc:
                    log.warning(
                        "Failed to extract last frame from %s: %s — "
                        "falling back to original reference for sub-clip %d",
                        result.video_path.name, exc, i + 1,
                    )
                    current_reference = reference_image

        return outputs

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
        pos_embeds, neg_embeds = self._bundle.t5.encode_pair(prompt, "")
        return pos_embeds, neg_embeds

    def _encode_image(self, image_path: Path, n_frames: int) -> "Any":
        """
        Process reference image into image conditioning tensors.

        Returns (image_cond_pos, image_cond_neg) both [1, 20, total_lat_f, lat_h, lat_w].
        The 20 channels = 4 mask + 16 latent.

        Positive: ref image latent at position 0, zeros for noise frames.
        Negative: all zeros (uncond).
        """
        if self._bundle is None or self._bundle.vae is None:
            raise RuntimeError("VAE not loaded — call load() first")

        import torch
        from PIL import Image
        import numpy as np

        vae = self._bundle.vae
        enc_device = self._bundle.encoder_device

        # Determine resolution from config
        if self.config.resolution == "720p":
            H, W = 720, 1280
        elif self.config.resolution == "480p":
            H, W = 480, 832
        else:
            H, W = 720, 1280

        # Latent dimensions
        lat_f = (n_frames - 1) // 4 + 1   # temporal latent frames for noise
        total_lat_f = lat_f + 1            # +1 for reference frame
        lat_h, lat_w = H // 8, W // 8

        # Load and resize reference image
        img = Image.open(image_path).convert("RGB").resize((W, H), Image.LANCZOS)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img_t = img_t.to(enc_device)

        # VAE encode the reference image
        with torch.no_grad():
            img_latent = vae.encode_image(img_t)  # [1, 16, 1, lat_h, lat_w]

        # Build positive image condition: [1, 20, total_lat_f, lat_h, lat_w]
        # Channel layout: [4 mask | 16 latent]
        # Mask: 1 where we have a reference frame (frame 0), 0 elsewhere
        mask_pos = torch.zeros(1, 4, total_lat_f, lat_h, lat_w, device=enc_device)
        mask_pos[:, :, 0, :, :] = 1.0  # ref frame mask

        # Latent: reference image at frame 0, zeros for noise frames
        latent_frames = torch.zeros(1, 16, total_lat_f, lat_h, lat_w, device=enc_device)
        latent_frames[:, :, 0, :, :] = img_latent[:, :, 0, :, :]

        image_cond_pos = torch.cat([mask_pos, latent_frames], dim=1)  # [1, 20, total_lat_f, lat_h, lat_w]

        # Negative: all zeros
        image_cond_neg = torch.zeros_like(image_cond_pos)

        return image_cond_pos, image_cond_neg

    def _encode_audio(self, audio_path: Path, n_frames: int) -> "Any":
        """
        Encode audio segment via Whisper encoder → windowed audio features.
        Returns [1, total_lat_f, 8, 5, 1280] tensor.
        """
        if self._bundle is None or self._bundle.whisper is None:
            raise RuntimeError("Whisper encoder not loaded — call load() first")

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
        from musicvision.video.scheduler import FlowMatchScheduler

        if self._bundle is None or self._bundle.dit is None:
            raise RuntimeError("DiT not loaded — call load() first")

        dit = self._bundle.dit
        dit_device = self._bundle.dit_device
        enc_device = self._bundle.encoder_device

        # Unpack conditioning
        pos_text, neg_text = text_embeds        # each [1, 512, 4096]
        image_cond_pos, image_cond_neg = image_cond  # each [1, 20, total_lat_f, lat_h, lat_w]

        # Move text embeds to dit device
        pos_text = pos_text.to(dit_device)
        neg_text = neg_text.to(dit_device)
        image_cond_pos = image_cond_pos.to(dit_device)
        image_cond_neg = image_cond_neg.to(dit_device)

        # Audio features — move to dit device, create zero version
        if audio_embeds is not None:
            audio_embeds = audio_embeds.to(dit_device)
            audio_zeros = torch.zeros_like(audio_embeds)
        else:
            audio_zeros = None

        # Determine latent shape
        if self.config.resolution == "720p":
            H, W = 720, 1280
        elif self.config.resolution == "480p":
            H, W = 480, 832
        else:
            H, W = 720, 1280
        lat_f = (n_frames - 1) // 4 + 1
        total_lat_f = lat_f + 1
        lat_h, lat_w = H // 8, W // 8

        # Initialize noise — seed is always set by generate() before this call,
        # but guard here too for direct _denoise() callers (e.g. tests).
        # Both CPU and CUDA RNGs must be seeded: torch.randn on a CUDA device
        # uses the CUDA RNG exclusively, so manual_seed alone is not sufficient.
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        noise = torch.randn(
            1, 16, total_lat_f, lat_h, lat_w,
            device=dit_device, dtype=torch.bfloat16,
        )
        z = noise.clone()

        # Create scheduler
        scheduler = FlowMatchScheduler(
            num_inference_steps=self.config.denoising_steps,
            shift=5.0,
        )

        scale_a = float(self.config.scale_a)
        scale_t = float(self.config.scale_t)

        log.info(
            "Denoising: %d steps, lat shape [1,16,%d,%d,%d], scale_a=%.1f, scale_t=%.1f",
            self.config.denoising_steps, total_lat_f, lat_h, lat_w, scale_a, scale_t,
        )

        use_swap = self._bundle.block_swap is not None

        with torch.no_grad():
            for step_idx in range(self.config.denoising_steps):
                t = scheduler.sigmas[step_idx].to(dit_device)
                timestep = t.expand(1)

                # Build DiT inputs: cat noise latent + image conditioning
                x_pos = torch.cat([z, image_cond_pos], dim=1)  # [1, 36, total_lat_f, lat_h, lat_w]
                x_neg = torch.cat([z, image_cond_neg], dim=1)

                if use_swap:
                    v_cond = self._forward_with_swap(x_pos, timestep, pos_text, audio_embeds)
                    v_audio_neg = self._forward_with_swap(x_pos, timestep, pos_text, audio_zeros)
                    v_text_neg = self._forward_with_swap(x_neg, timestep, neg_text, audio_embeds)
                else:
                    v_cond = dit(x_pos, timestep, pos_text, audio_embeds)
                    v_audio_neg = dit(x_pos, timestep, pos_text, audio_zeros)
                    v_text_neg = dit(x_neg, timestep, neg_text, audio_embeds)

                # TIA dual CFG combination
                v_pred = (
                    v_text_neg
                    + scale_a * (v_cond - v_audio_neg)
                    + (scale_t - 2.0) * (v_audio_neg - v_text_neg)
                )

                z = scheduler.step(v_pred, z, step_idx)

                if (step_idx + 1) % 10 == 0:
                    log.debug("Denoising step %d/%d", step_idx + 1, self.config.denoising_steps)

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
            pixels = vae.decode(noise_latent)  # [1, 3, T, H, W] in [0,1]

        # Convert to (T, H, W, 3) uint8
        pixels = pixels.squeeze(0)           # [3, T, H, W]
        pixels = pixels.permute(1, 2, 3, 0)  # [T, H, W, 3]
        pixels = (pixels.clamp(0, 1) * 255).to(torch.uint8)

        return pixels  # (T, H, W, 3) uint8


# ---------------------------------------------------------------------------
# Tier recommendation
# ---------------------------------------------------------------------------

def recommend_tier(device_map: "DeviceMap") -> HumoTier:
    """
    Suggest the best HumoTier for the detected hardware.

    The recommendation is conservative: it selects the highest-quality tier
    that fits comfortably (not the absolute maximum), leaving headroom for
    text encoder, VAE, and Whisper on the secondary device.
    """
    try:
        import torch
        primary_gb = torch.cuda.get_device_properties(device_map.dit_device).total_memory / 1024**3
        n_gpus = torch.cuda.device_count()
    except Exception:
        log.warning("CUDA not available — recommending preview tier (CPU-only not supported)")
        return HumoTier.PREVIEW

    if n_gpus >= 2 and primary_gb >= 24:
        return HumoTier.FP16
    if primary_gb >= 20:
        return HumoTier.FP8_SCALED
    if primary_gb >= 16:
        return HumoTier.GGUF_Q6
    if primary_gb >= 12:
        return HumoTier.GGUF_Q4
    return HumoTier.PREVIEW


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
    """Generate sub-clip suffixes: a, b, c, ... aa, ab, ..."""
    suffixes = []
    for i in range(n):
        if i < 26:
            suffixes.append(chr(ord("a") + i))
        else:
            suffixes.append(chr(ord("a") + i // 26 - 1) + chr(ord("a") + i % 26))
    return suffixes


def _audio_duration(path: Path) -> float:
    """Return audio duration in seconds via soundfile (no ffmpeg needed)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.duration
    except Exception:
        from musicvision.utils.audio import get_duration
        return get_duration(path)


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
    proc.stdin.write(frames.numpy().tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed saving video: {stderr}")


# Type alias for annotations inside this module (avoids circular imports)
Any = object
