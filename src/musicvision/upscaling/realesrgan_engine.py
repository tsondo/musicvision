"""
Real-ESRGAN video upscaling engine.

Frame-by-frame super-resolution. Priority order:
  1. Python ``realesrgan`` package (CUDA) — best option, no Vulkan needed
  2. ``realesrgan-ncnn-vulkan`` binary (Vulkan) — if available on PATH
  3. ffmpeg lanczos scaling — last-resort fallback (no AI upscaling)

Fast (~2-4 GB VRAM) but no temporal consistency between frames.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.utils.video import get_video_resolution

log = logging.getLogger(__name__)

# Model name → (netscale, RRDBNet params)
_MODEL_PARAMS: dict[str, dict] = {
    "realesrgan-x4plus-anime": {"netscale": 4, "num_block": 6, "num_in_ch": 3, "num_out_ch": 3, "num_feat": 64},
    "realesrgan-x4plus": {"netscale": 4, "num_block": 23, "num_in_ch": 3, "num_out_ch": 3, "num_feat": 64},
    "realesr-animevideov3": {"netscale": 4, "num_block": 6, "num_in_ch": 3, "num_out_ch": 3, "num_feat": 64},
}

_MODEL_URLS: dict[str, str] = {
    "realesrgan-x4plus-anime": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    ),
    "realesrgan-x4plus": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    ),
    "realesr-animevideov3": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    ),
}


class RealEsrganEngine(UpscaleEngine):
    """Real-ESRGAN upscaler — Python (CUDA), ncnn-vulkan binary, or ffmpeg fallback."""

    def __init__(self, model_name: str = "realesrgan-x4plus-anime"):
        self._model_name = model_name
        self._loaded = False
        self._binary: str | None = None
        self._upscaler = None  # RealESRGANer instance
        self._mode: str = "none"  # "python", "ncnn", "ffmpeg"

    def load(self) -> None:
        # Try Python realesrgan package first (CUDA)
        if self._try_load_python():
            self._mode = "python"
            log.info("Real-ESRGAN loaded via Python package (CUDA)")
        else:
            # Try ncnn-vulkan binary
            self._binary = shutil.which("realesrgan-ncnn-vulkan")
            if self._binary:
                self._mode = "ncnn"
                log.info("Real-ESRGAN using ncnn-vulkan binary: %s", self._binary)
            else:
                self._mode = "ffmpeg"
                log.warning("No Real-ESRGAN backend found — will use ffmpeg lanczos scaling")
        self._loaded = True

    def _try_load_python(self) -> bool:
        """Try loading the Python realesrgan package with CUDA."""
        try:
            import sys

            import torch
            import torchvision.transforms.functional as tv_functional

            # Patch for basicsr compatibility with newer torchvision
            sys.modules["torchvision.transforms.functional_tensor"] = tv_functional

            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            params = _MODEL_PARAMS.get(self._model_name, _MODEL_PARAMS["realesrgan-x4plus-anime"])
            netscale = params.pop("netscale", 4)
            model = RRDBNet(**params)
            params["netscale"] = netscale  # restore for next time

            model_url = _MODEL_URLS.get(self._model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            self._upscaler = RealESRGANer(
                scale=netscale,
                model_path=model_url,
                model=model,
                tile=0,  # 0 = no tiling (faster); increase if OOM
                tile_pad=10,
                pre_pad=0,
                half=torch.cuda.is_available(),
                device=device,
            )
            return True
        except Exception:
            log.debug("Python realesrgan not available", exc_info=True)
            return False

    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        if not self._loaded:
            raise RuntimeError("Engine not loaded. Call load() first.")

        source_res = get_video_resolution(input.video_path)
        input.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self._mode == "python":
            self._upscale_python(input)
        elif self._mode == "ncnn":
            self._upscale_ncnn(input)
        else:
            self._upscale_ffmpeg_fallback(input)

        output_res = get_video_resolution(input.output_path)
        return UpscaleResult(
            video_path=input.output_path,
            source_resolution=source_res,
            output_resolution=output_res,
            metadata={"model": self._model_name, "mode": self._mode},
        )

    def _upscale_python(self, input: UpscaleInput) -> None:
        """Upscale using the Python realesrgan package (CUDA)."""
        import cv2

        with tempfile.TemporaryDirectory(prefix="realesrgan_") as tmpdir:
            tmp = Path(tmpdir)

            # Extract frames + get FPS
            fps = _extract_frames(input.video_path, tmp / "input")

            # Upscale each frame
            out_dir = tmp / "output"
            out_dir.mkdir()
            frame_files = sorted((tmp / "input").glob("frame_*.png"))
            for i, frame_path in enumerate(frame_files):
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                output, _ = self._upscaler.enhance(img, outscale=4)
                cv2.imwrite(str(out_dir / frame_path.name), output)
                if (i + 1) % 20 == 0:
                    log.debug("Upscaled frame %d/%d", i + 1, len(frame_files))

            # Reassemble
            _reassemble_frames(out_dir, input.output_path, fps, input.target_width, input.target_height)

    def _upscale_ncnn(self, input: UpscaleInput) -> None:
        """Upscale using realesrgan-ncnn-vulkan binary."""
        with tempfile.TemporaryDirectory(prefix="realesrgan_") as tmpdir:
            tmp = Path(tmpdir)

            fps = _extract_frames(input.video_path, tmp / "input")

            frames_out = tmp / "output"
            frames_out.mkdir()

            subprocess.run(
                [self._binary, "-i", str(tmp / "input"), "-o", str(frames_out),
                 "-n", self._model_name, "-s", "4", "-f", "png"],
                capture_output=True, check=True, timeout=600,
            )

            _reassemble_frames(frames_out, input.output_path, fps, input.target_width, input.target_height)

    def _upscale_ffmpeg_fallback(self, input: UpscaleInput) -> None:
        """Fallback: ffmpeg lanczos scaling (no AI upscaling)."""
        from musicvision.utils.video import scale_video

        log.warning("No Real-ESRGAN backend found — using ffmpeg lanczos scaling as fallback")
        scale_video(input.video_path, input.output_path, input.target_width, input.target_height)

    def unload(self) -> None:
        if self._upscaler is not None:
            del self._upscaler
            self._upscaler = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._loaded = False
        self._binary = None
        self._mode = "none"

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _extract_frames(video_path: Path, out_dir: Path) -> float:
    """Extract all frames from a video and return the FPS."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(video_path), str(out_dir / "frame_%06d.png")],
        capture_output=True, check=True, timeout=300,
    )

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe not found")
    probe_result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of", "json",
         str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    probe_data = json.loads(probe_result.stdout)
    fps_str = probe_data["streams"][0]["r_frame_rate"]
    num, den = fps_str.split("/")
    return float(num) / float(den)


def _reassemble_frames(frames_dir: Path, output_path: Path, fps: float, target_w: int, target_h: int) -> None:
    """Reassemble PNG frames into an MP4 at the target resolution."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found")
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps),
         "-i", str(frames_dir / "frame_%06d.png"),
         "-vf", f"scale={target_w}:{target_h}:flags=lanczos",
         "-c:v", "libx264", "-preset", "fast", "-crf", "18",
         str(output_path)],
        capture_output=True, check=True, timeout=300,
    )
