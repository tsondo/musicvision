"""
Real-ESRGAN video upscaling engine.

Frame-by-frame super-resolution via realesrgan-ncnn-vulkan binary
or the Python realesrgan package. Fast (~2-4GB VRAM) but no temporal
consistency.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.utils.video import get_video_resolution

log = logging.getLogger(__name__)


class RealEsrganEngine(UpscaleEngine):
    """Real-ESRGAN upscaler using realesrgan-ncnn-vulkan or Python package."""

    def __init__(self, model_name: str = "realesrgan-x4plus-anime"):
        self._model_name = model_name
        self._loaded = False
        self._binary: str | None = None

    def load(self) -> None:
        # Check for realesrgan-ncnn-vulkan binary
        self._binary = shutil.which("realesrgan-ncnn-vulkan")
        if not self._binary:
            log.info("realesrgan-ncnn-vulkan not found, will use Python fallback")
        self._loaded = True

    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        source_res = get_video_resolution(input.video_path)
        input.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._binary:
            self._upscale_ncnn(input)
        else:
            self._upscale_ffmpeg_fallback(input)

        output_res = (input.target_width, input.target_height)
        return UpscaleResult(
            video_path=input.output_path,
            source_resolution=source_res,
            output_resolution=output_res,
            metadata={"model": self._model_name},
        )

    def _upscale_ncnn(self, input: UpscaleInput) -> None:
        """Upscale using realesrgan-ncnn-vulkan (frame extraction → upscale → reassemble)."""
        with tempfile.TemporaryDirectory(prefix="realesrgan_") as tmpdir:
            tmp = Path(tmpdir)
            frames_in = tmp / "input"
            frames_out = tmp / "output"
            frames_in.mkdir()
            frames_out.mkdir()

            # Extract frames
            ffmpeg = shutil.which("ffmpeg")
            if not ffmpeg:
                raise RuntimeError("ffmpeg not found")
            subprocess.run(
                [ffmpeg, "-y", "-i", str(input.video_path), str(frames_in / "frame_%06d.png")],
                capture_output=True, check=True, timeout=300,
            )

            # Get FPS from source
            import json
            ffprobe = shutil.which("ffprobe")
            probe_result = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=r_frame_rate", "-of", "json",
                 str(input.video_path)],
                capture_output=True, text=True, timeout=30,
            )
            probe_data = json.loads(probe_result.stdout)
            fps_str = probe_data["streams"][0]["r_frame_rate"]
            num, den = fps_str.split("/")
            fps = float(num) / float(den)

            # Run realesrgan-ncnn-vulkan
            subprocess.run(
                [self._binary, "-i", str(frames_in), "-o", str(frames_out),
                 "-n", self._model_name, "-s", "4", "-f", "png"],
                capture_output=True, check=True, timeout=600,
            )

            # Reassemble with target resolution
            subprocess.run(
                [ffmpeg, "-y", "-framerate", str(fps),
                 "-i", str(frames_out / "frame_%06d.png"),
                 "-vf", f"scale={input.target_width}:{input.target_height}:flags=lanczos",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                 str(input.output_path)],
                capture_output=True, check=True, timeout=300,
            )

    def _upscale_ffmpeg_fallback(self, input: UpscaleInput) -> None:
        """Fallback: just use ffmpeg lanczos scaling (no AI upscaling)."""
        from musicvision.utils.video import scale_video

        log.warning("No Real-ESRGAN binary found — using ffmpeg lanczos scaling as fallback")
        scale_video(input.video_path, input.output_path, input.target_width, input.target_height)

    def unload(self) -> None:
        self._loaded = False
        self._binary = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded
