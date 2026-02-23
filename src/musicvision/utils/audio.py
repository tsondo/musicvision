"""
Audio utilities — ffmpeg wrappers for slicing, muxing, and conversion.

All audio manipulation goes through ffmpeg subprocess calls.
No ffmpeg Python bindings — they add complexity and break on WSL more often than the CLI.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")
    return path


def get_duration(audio_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def slice_audio(
    source: Path,
    output: Path,
    start_seconds: float,
    end_seconds: float,
    sample_rate: int | None = None,
) -> Path:
    """
    Extract a segment from an audio file.

    Args:
        source: Input audio file
        output: Output WAV path
        start_seconds: Start time
        end_seconds: End time
        sample_rate: Optional resample rate (None = keep original)

    Returns:
        Path to the output file
    """
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    duration = end_seconds - start_seconds
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_seconds:.6f}",
        "-i", str(source),
        "-t", f"{duration:.6f}",
        "-vn",  # no video
    ]

    if sample_rate:
        cmd.extend(["-ar", str(sample_rate)])

    cmd.extend(["-c:a", "pcm_s16le", str(output)])

    log.debug("Slicing audio: %s → %s (%.2fs – %.2fs)", source.name, output.name, start_seconds, end_seconds)
    subprocess.run(cmd, capture_output=True, check=True)
    return output


def mux_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_offset: float = 0.0,
) -> Path:
    """
    Combine a video file (no audio) with an audio track.

    HuMo outputs video-only MP4. This muxes the original audio back.
    """
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
    ]

    if audio_offset > 0:
        cmd.extend(["-itsoffset", f"{audio_offset:.6f}"])

    cmd.extend([
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_path),
    ])

    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def concat_videos(clip_paths: list[Path], output_path: Path) -> Path:
    """
    Concatenate multiple video clips using ffmpeg's concat demuxer.

    All clips must have the same resolution, codec, and framerate.
    """
    _check_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write concat list file
    list_file = output_path.parent / "_concat_list.txt"
    with open(list_file, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve()}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_file),
        "-c", "copy",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    list_file.unlink(missing_ok=True)
    return output_path


def convert_to_wav(source: Path, output: Path, sample_rate: int = 16000) -> Path:
    """Convert any audio format to WAV (mono, 16-bit PCM). Used for Whisper input."""
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(source),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(output),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    return output
