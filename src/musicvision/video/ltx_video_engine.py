"""
LTX-Video 2 engine — in-process via diffusers.

19B param joint audio-video DiT.  Generates synchronized video + audio
from a reference image, text prompt, and optional audio conditioning.

For MusicVision we:
  - Use LTX2ImageToVideoPipeline (image + text → video + audio)
  - Optionally encode scene audio via audio_vae for motion sync
  - Discard generated audio — assembly muxes the original song
  - Save video frames as silent MP4 via ffmpeg raw pipe

Lifecycle:
    engine = LtxVideoEngine(config, device_map)
    engine.load()
    results = engine.generate_scene(...)
    engine.unload()
"""

from __future__ import annotations

import gc
import logging
import math
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from musicvision.video.base import VideoEngine, VideoInput, VideoResult

if TYPE_CHECKING:
    from musicvision.models import LtxVideoConfig
    from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)


def _sub_clip_suffixes(n: int) -> list[str]:
    """Return suffix letters: a, b, c, ... z, aa, ab, ..."""
    if n <= 26:
        return [chr(ord("a") + i) for i in range(n)]
    return [f"{chr(ord('a') + i // 26)}{chr(ord('a') + i % 26)}" for i in range(n)]


def _save_video_ffmpeg(frames, output_path: Path, fps: int = 24) -> None:
    """Save (T, H, W, C) uint8 numpy array as silent MP4 via ffmpeg raw pipe."""
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(frames, np.ndarray):
        frames = np.asarray(frames)

    # Ensure uint8
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)

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
        "-an",
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    proc.stdin.write(frames.tobytes())
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"ffmpeg failed saving video: {stderr}")


class LtxVideoEngine(VideoEngine):
    """LTX-Video 2 engine using diffusers LTX2ImageToVideoPipeline."""

    def __init__(self, config: LtxVideoConfig, device_map: DeviceMap) -> None:
        self.config = config
        self.device_map = device_map
        self._pipe = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load LTX-2 pipeline from HuggingFace and apply offloading."""
        import torch

        try:
            from diffusers import LTX2ImageToVideoPipeline
        except ImportError:
            raise ImportError(
                "LTX2ImageToVideoPipeline not found in diffusers. "
                "Install from git main: uv pip install git+https://github.com/huggingface/diffusers.git"
            )

        log.info("Loading LTX-Video 2 from %s ...", self.config.model_id)

        pipe = LTX2ImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch.bfloat16,
        )

        # Apply CPU offloading strategy
        primary = str(self.device_map.dit_device)
        if self.config.cpu_offload == "sequential":
            pipe.enable_sequential_cpu_offload(device=primary)
        elif self.config.cpu_offload == "model":
            pipe.enable_model_cpu_offload(device=primary)
        else:
            pipe = pipe.to(primary)

        # VAE tiling to avoid OOM during decode
        if self.config.vae_tiling:
            pipe.vae.enable_tiling()

        self._pipe = pipe
        self._loaded = True
        log.info("LTX-Video 2 loaded (offload=%s, device=%s)", self.config.cpu_offload, primary)

    def generate(self, input: VideoInput) -> VideoResult:
        """Generate a single video clip."""
        if not self._loaded or self._pipe is None:
            raise RuntimeError("Call load() before generate()")

        import numpy as np
        import torch
        from PIL import Image

        # Load reference image
        image = Image.open(input.reference_image).convert("RGB")

        # Determine frame count from audio duration
        # Audio duration drives the clip length — snap to valid (N*8)+1
        from musicvision.utils.audio import get_duration
        audio_dur = get_duration(input.audio_segment)
        target_frames = int(round(audio_dur * self.config.fps))
        num_frames = self.config.snap_frames(min(target_frames, self.config.max_frames))
        num_frames = max(num_frames, 9)  # absolute minimum

        # Seed
        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)

        log.info(
            "LTX-2 generating: %s → %d frames @ %dfps (%dx%d, %d steps)",
            input.output_path.name, num_frames, self.config.fps,
            self.config.width, self.config.height, self.config.num_inference_steps,
        )

        # Audio conditioning: encode scene audio through audio_vae
        audio_latents = None
        if self.config.use_audio_conditioning and hasattr(self._pipe, "audio_vae"):
            try:
                audio_latents = self._encode_audio(input.audio_segment, num_frames)
            except Exception:
                log.warning("Audio conditioning failed, generating without audio sync", exc_info=True)

        # Run pipeline
        result = self._pipe(
            image=image,
            prompt=input.text_prompt,
            negative_prompt=self.config.negative_prompt,
            width=self.config.width,
            height=self.config.height,
            num_frames=num_frames,
            frame_rate=self.config.fps,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            audio_latents=audio_latents,
            generator=generator,
            output_type="np",
            return_dict=False,
        )

        # Pipeline returns (video, audio) tuple
        video = result[0] if isinstance(result, (tuple, list)) else result
        # video shape: (batch, T, H, W, C) or (T, H, W, C)
        if isinstance(video, np.ndarray):
            if video.ndim == 5:
                video = video[0]  # remove batch dim
        elif hasattr(video, "numpy"):
            video = video.cpu().numpy()
            if video.ndim == 5:
                video = video[0]

        # Save as silent MP4
        _save_video_ffmpeg(video, input.output_path, fps=self.config.fps)

        duration = num_frames / self.config.fps
        log.info("LTX-2 clip saved: %s (%.2fs, %d frames)", input.output_path.name, duration, num_frames)

        return VideoResult(
            video_path=input.output_path,
            frames_generated=num_frames,
            duration_seconds=duration,
            metadata={"engine": "ltx_video", "seed": self.config.seed},
        )

    def _encode_audio(self, audio_path: Path, num_frames: int):
        """Encode audio through the pipeline's audio_vae for conditioning."""
        import torch
        import torchaudio

        waveform, sr = torchaudio.load(str(audio_path))

        # Resample to 16kHz if needed (LTX-2 audio VAE expects 16kHz)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Compute mel spectrogram for audio VAE
        # The audio_vae expects mel-spectrogram input
        audio_vae = self._pipe.audio_vae
        device = next(audio_vae.parameters()).device
        dtype = next(audio_vae.parameters()).dtype

        waveform = waveform.to(device=device, dtype=dtype)

        # Use pipeline's built-in audio encoding if available
        if hasattr(self._pipe, "_encode_audio"):
            return self._pipe._encode_audio(waveform)

        # Fallback: encode through VAE directly
        with torch.no_grad():
            latents = audio_vae.encode(waveform.unsqueeze(0)).latent_dist.sample()

        return latents

    def unload(self) -> None:
        """Free pipeline and VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        self._loaded = False

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        log.info("LTX-Video 2 unloaded")

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
    ) -> list[VideoResult]:
        """Generate video for a full scene, splitting into sub-clips if needed."""
        if not self._loaded:
            raise RuntimeError("Call load() before generate_scene()")

        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Frame-plan path (preferred) ---
        if subclip_frame_counts is not None:
            if len(subclip_frame_counts) == 1:
                output_path = output_dir / f"{scene_id}.mp4"
                result = self.generate(VideoInput(
                    text_prompt=text_prompt,
                    reference_image=reference_image,
                    audio_segment=audio_segment,
                    output_path=output_path,
                ))
                return [result]

            n_sub = len(subclip_frame_counts)
            suffixes = _sub_clip_suffixes(n_sub)
            outputs: list[VideoResult] = []
            current_ref = reference_image

            for i, suffix in enumerate(suffixes):
                sub_audio = (
                    subclip_audio_paths[i]
                    if subclip_audio_paths and i < len(subclip_audio_paths)
                    else audio_segment.parent / f"{scene_id}_sub_{i:02d}.wav"
                )
                if not sub_audio.exists():
                    log.warning("Sub-clip audio not found: %s, using full segment", sub_audio)
                    sub_audio = audio_segment

                output_path = output_dir / f"{scene_id}_{suffix}.mp4"
                result = self.generate(VideoInput(
                    text_prompt=text_prompt,
                    reference_image=current_ref,
                    audio_segment=sub_audio,
                    output_path=output_path,
                ))
                outputs.append(result)

                # Sub-clip continuity: extract last frame for next clip's reference
                if i < n_sub - 1:
                    last_frame = self._extract_last_frame(result.video_path)
                    if last_frame is not None:
                        current_ref = last_frame

            return outputs

        # --- Legacy float-based path ---
        max_dur = self.config.max_duration

        if duration <= max_dur:
            output_path = output_dir / f"{scene_id}.mp4"
            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=audio_segment,
                output_path=output_path,
            ))
            return [result]

        # Split into sub-clips
        n_sub = math.ceil(duration / max_dur)
        suffixes = _sub_clip_suffixes(n_sub)
        outputs_legacy: list[VideoResult] = []

        for i, suffix in enumerate(suffixes):
            sub_audio = audio_segment.parent / f"{scene_id}_sub_{i:02d}.wav"
            if not sub_audio.exists():
                log.warning("Sub-clip audio not found: %s, using full segment", sub_audio)
                sub_audio = audio_segment

            output_path = output_dir / f"{scene_id}_{suffix}.mp4"
            result = self.generate(VideoInput(
                text_prompt=text_prompt,
                reference_image=reference_image,
                audio_segment=sub_audio,
                output_path=output_path,
            ))
            outputs_legacy.append(result)

        return outputs_legacy

    @staticmethod
    def _extract_last_frame(video_path: Path) -> Path | None:
        """Extract the last frame of a video as a PNG for sub-clip continuity."""
        frame_path = video_path.with_suffix(".last_frame.png")
        cmd = [
            "ffmpeg", "-y",
            "-sseof", "-0.1",
            "-i", str(video_path),
            "-frames:v", "1",
            "-update", "1",
            str(frame_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=30)
            if proc.returncode == 0 and frame_path.exists():
                return frame_path
        except Exception:
            log.warning("Failed to extract last frame from %s", video_path, exc_info=True)
        return None
