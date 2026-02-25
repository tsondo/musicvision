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

        Downloads missing weights automatically if HUGGINGFACE_TOKEN is set.
        Applies block swap if config.block_swap_count > 0.
        """
        import os

        from musicvision.video.weight_registry import weight_status, download_all_for_tier

        tier = self.config.tier
        status = weight_status(tier)
        missing = [k for k, present in status.items() if not present]

        if missing:
            hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
            if hf_token:
                log.info(
                    "Missing weights for tier %s: %s — downloading…",
                    tier.value, missing,
                )
                download_all_for_tier(tier, hf_token=hf_token)
            else:
                raise RuntimeError(
                    f"HuMo weights for tier '{tier.value}' are not present locally "
                    f"(missing: {missing}). Set HUGGINGFACE_TOKEN in .env and re-run, "
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

        # Step 1-3: encode conditioning signals
        text_embeds  = self._encode_text(inp.text_prompt)
        image_latent = self._encode_image(inp.reference_image)
        audio_embeds = self._encode_audio(inp.audio_segment)

        # Step 4-5: denoising loop
        video_latent = self._denoise(
            n_frames=n_frames,
            text_embeds=text_embeds,
            image_latent=image_latent,
            audio_embeds=audio_embeds,
            seed=inp.seed,
        )

        # Step 6-7: decode and save
        frames = self._decode_latent(video_latent)
        _save_mp4(frames, inp.output_path, fps=FPS)

        duration = n_frames / FPS
        log.info("Clip saved → %s (%.2fs)", inp.output_path.name, duration)
        return HumoOutput(
            video_path=inp.output_path,
            frames_generated=n_frames,
            duration_seconds=duration,
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
        Encode *prompt* via the UMT5-XXL text encoder.

        Returns text_embeds tensor on encoder_device.

        TODO: implement using wan.modules.t5.WanT5Encoder once HuMo source is available.
        Reference: kijai nodes_sampler.py — encode_prompt()
        """
        log.warning("_encode_text is a stub — requires wan.modules.t5")
        return None

    def _encode_image(self, image_path: Path) -> "Any":
        """
        Process reference image through the HuMoEmbeds pipeline.

        Steps (from kijai nodes.py — HuMoEmbeds):
          1. Load image → normalize to [-1, 1]
          2. VAE encode → image latent
          3. Apply positional embeddings for image conditioning
          4. Return image_cond tensor for the DiT cross-attention

        TODO: implement using wan.modules.vae.WanVideoVAE
        """
        log.warning("_encode_image is a stub — requires wan.modules.vae + HuMoEmbeds logic")
        return None

    def _encode_audio(self, audio_path: Path) -> "Any":
        """
        Encode audio segment via Whisper encoder → audio_embeds tensor.

        Steps (from kijai nodes.py — HuMoEmbeds audio branch):
          1. Load WAV, resample to 16 kHz
          2. Whisper feature extraction (log-mel spectrogram, 80 bins)
          3. Pass through Whisper encoder (no decoder)
          4. Return last_hidden_state on encoder_device

        The full-mix audio (not isolated vocals) is the correct input here
        because HuMo was trained with mixed audio for A/V synchronisation.

        TODO: complete once Whisper encoder is loaded in _bundle.whisper
        """
        if self._bundle is None or self._bundle.whisper is None:
            log.warning("_encode_audio is a stub — Whisper encoder not loaded")
            return None

        try:
            import torch
            from transformers import WhisperFeatureExtractor
            import soundfile as sf

            wav, sr = sf.read(str(audio_path), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)  # mono

            extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
            features = extractor(
                wav, sampling_rate=sr, return_tensors="pt"
            ).input_features.to(self._bundle.encoder_device)

            with torch.no_grad():
                audio_embeds = self._bundle.whisper(features).last_hidden_state

            return audio_embeds
        except Exception as exc:
            log.warning("Audio encoding failed: %s — returning None", exc)
            return None

    # ------------------------------------------------------------------
    # Internal: denoising loop
    # ------------------------------------------------------------------

    def _denoise(
        self,
        n_frames: int,
        text_embeds: "Any",
        image_latent: "Any",
        audio_embeds: "Any",
        seed: int | None = None,
    ) -> "Any":
        """
        Flow-Matching denoising loop producing the video latent.

        Scheduler: UniPC / DPM++ or the Flow-Matching scheduler from HuMo
        (see bytedance-research/HuMo generate.yaml — solver type).

        Guidance: dual CFG with scale_t (text) and scale_a (audio).
        Negative conditioning: zeros for audio_embeds, uncond text token for text.

        Block swap: if self._bundle.block_swap is not None, each transformer
        block is executed via swap.execute_block(idx, hidden, ...) which
        handles CPU↔GPU migration transparently.

        TODO: implement once WanModel forward signature is known from HuMo source.
        Reference: kijai nodes_sampler.py — WanVideoSampler.sample()
        """
        log.warning(
            "_denoise is a stub — requires WanModel forward() signature from HuMo source. "
            "Reference: kijai/ComfyUI-WanVideoWrapper/nodes_sampler.py"
        )
        return None

    # ------------------------------------------------------------------
    # Internal: VAE decode
    # ------------------------------------------------------------------

    def _decode_latent(self, latent: "Any") -> "Any":
        """
        Decode video latent → pixel-space frames tensor (T, H, W, 3) uint8.

        TODO: implement using wan.modules.vae.WanVideoVAE.decode()
        """
        log.warning("_decode_latent is a stub — requires wan.modules.vae")
        return None


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
