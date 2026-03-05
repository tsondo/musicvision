"""
Video utilities — ffprobe resolution queries and ffmpeg scale operations.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from musicvision.models import Scene

log = logging.getLogger(__name__)


def _check_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise RuntimeError("ffprobe not found in PATH. Install with: sudo apt install ffmpeg")
    return path


def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")
    return path


def get_video_resolution(video_path: Path) -> tuple[int, int]:
    """Get video resolution as (width, height) using ffprobe.

    Raises:
        RuntimeError: If ffprobe fails or output cannot be parsed.
    """
    ffprobe = _check_ffprobe()
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {video_path}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    w = streams[0]["width"]
    h = streams[0]["height"]
    return (int(w), int(h))


def scale_video(
    input_path: Path,
    output_path: Path,
    target_width: int,
    target_height: int,
) -> Path:
    """Scale a video to target resolution using ffmpeg.

    Uses lanczos scaling and preserves codec. Falls back to libx264 encoding.
    """
    ffmpeg = _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        ffmpeg, "-y", "-i", str(input_path),
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg scale failed: {result.stderr.strip()}")

    log.debug("Scaled %s → %s (%dx%d)", input_path.name, output_path.name, target_width, target_height)
    return output_path


def update_scene_resolution(scene: Scene, project_root: Path) -> None:
    """Set scene.video_width/height from the best available clip (upscaled > original).

    Silently skips if no clip exists or ffprobe fails.
    """
    clip = scene.upscaled_clip or scene.video_clip
    if not clip:
        return
    clip_path = Path(clip)
    if not clip_path.is_absolute():
        clip_path = project_root / clip_path
    if not clip_path.exists():
        return
    try:
        w, h = get_video_resolution(clip_path)
        scene.video_width = w
        scene.video_height = h
    except Exception:
        log.debug("Could not read resolution for %s", clip, exc_info=True)
