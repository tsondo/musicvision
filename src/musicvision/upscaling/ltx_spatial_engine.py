"""
LTX Spatial Upsampler engine (diffusers in-process).

Deterministic latent-space upscaler for LTX-Video 2 output.
Encodes video to LTX latent space, spatially upsamples, then decodes.
Temporally aware — preserves motion consistency across frames.
~12GB VRAM with sequential offload.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.utils.video import get_video_resolution

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)


class LtxSpatialEngine(UpscaleEngine):
    """LTX Spatial Upsampler via diffusers LTXLatentUpsamplePipeline."""

    def __init__(
        self,
        model_id: str = "Lightricks/ltxv-spatial-upscaler-0.9.7",
        num_inference_steps: int = 10,
        device_map: DeviceMap | None = None,
    ):
        self._model_id = model_id
        self._steps = num_inference_steps  # kept for metadata, pipeline is deterministic
        self._device_map = device_map
        self._pipe = None
        self._loaded = False

    def load(self) -> None:
        try:
            import torch
            from diffusers import LTXLatentUpsamplePipeline
        except ImportError as e:
            raise RuntimeError(
                "LTX Spatial Upsampler requires diffusers with LTXLatentUpsamplePipeline. "
                "Install from git main: pip install git+https://github.com/huggingface/diffusers"
            ) from e

        log.info("Loading LTX Spatial Upsampler: %s", self._model_id)
        self._pipe = LTXLatentUpsamplePipeline.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
        )
        self._pipe.enable_sequential_cpu_offload()
        self._loaded = True
        log.info("LTX Spatial Upsampler loaded")

    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        if not self._loaded or self._pipe is None:
            raise RuntimeError("Engine not loaded. Call load() first.")

        source_res = get_video_resolution(input.video_path)
        input.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load video frames
        from diffusers.utils import export_to_video, load_video

        video_frames = load_video(str(input.video_path))

        # Detect source FPS for output
        fps = _get_fps(input.video_path)

        # Pipeline outputs 2× the given height/width, so pass half the target (snapped to 32).
        h = (input.target_height // 2 // 32) * 32
        w = (input.target_width // 2 // 32) * 32
        log.info("LTX Spatial: %dx%d → encode at %dx%d → output %dx%d", *source_res, w, h, w * 2, h * 2)
        result = self._pipe(
            video=video_frames,
            height=h,
            width=w,
            output_type="pil",
        )

        # Export upscaled frames to a temp video, then ffmpeg scale to exact target
        frames = result.frames[0] if hasattr(result, "frames") else result[0]
        raw_w, raw_h = w * 2, h * 2
        if raw_w == input.target_width and raw_h == input.target_height:
            export_to_video(frames, str(input.output_path), fps=fps)
        else:
            # Export to temp, then scale
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            export_to_video(frames, str(tmp_path), fps=fps)
            from musicvision.utils.video import scale_video
            scale_video(tmp_path, input.output_path, input.target_width, input.target_height)
            tmp_path.unlink(missing_ok=True)

        output_res = get_video_resolution(input.output_path)
        log.info(
            "LTX Spatial upscaled %s: %s → %s",
            input.video_path.name, source_res, output_res,
        )
        return UpscaleResult(
            video_path=input.output_path,
            source_resolution=source_res,
            output_resolution=output_res,
            metadata={"model": self._model_id},
        )

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
        self._loaded = False
        log.info("LTX Spatial Upsampler unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _get_fps(video_path: Path) -> int:
    """Get FPS from video file via ffprobe."""
    import json
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 24  # fallback
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "json",
             str(video_path)],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(result.stdout)
        num, den = data["streams"][0]["r_frame_rate"].split("/")
        return round(int(num) / int(den))
    except Exception:
        return 24
