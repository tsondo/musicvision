"""
Audio utilities — ffmpeg wrappers for slicing, muxing, and conversion.

All audio manipulation goes through ffmpeg subprocess calls.
No ffmpeg Python bindings — they add complexity and break on WSL more often than the CLI.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _check_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not found in PATH. Install with: sudo apt install ffmpeg")
    return path


def detect_silences(
    audio_path: Path,
    min_silence_duration: float = 0.15,
    silence_threshold_db: float = -35.0,
) -> list[tuple[float, float]]:
    """
    Detect silence intervals in an audio file using ffmpeg silencedetect.

    Args:
        audio_path: Path to audio file (any format ffmpeg supports).
        min_silence_duration: Minimum silence length in seconds to report.
        silence_threshold_db: Volume threshold below which audio is considered silent.

    Returns:
        List of (start, end) tuples in seconds. Sorted by start time.
    """
    _check_ffmpeg()

    cmd = [
        "ffmpeg", "-i", str(audio_path),
        "-af", f"silencedetect=noise={silence_threshold_db}dB:d={min_silence_duration}",
        "-f", "null", "-",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # silencedetect writes to stderr
    stderr = result.stderr

    starts: list[float] = []
    ends: list[float] = []
    for line in stderr.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        if m_start:
            starts.append(float(m_start.group(1)))
            continue
        m_end = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_end:
            ends.append(float(m_end.group(1)))

    # Pair starts and ends. If a silence is still active at EOF, ffmpeg may
    # emit a start without a matching end — cap it at the file duration.
    silences: list[tuple[float, float]] = []
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else s + min_silence_duration
        silences.append((s, e))

    log.debug("Detected %d silence intervals in %s", len(silences), audio_path.name)
    return silences


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
    # -ss before -i uses fast input seek; output codec pcm_s16le (WAV) is
    # sample-accurate because PCM has no codec frame boundaries.
    # WARNING: if source is ever MP3/AAC/Opus, switch to -af atrim to avoid
    # sub-frame drift at the cut point.  WAV → WAV is always exact.
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


def slice_subclip_audio(
    scene_audio: Path,
    scene_id: str,
    subclip_frames: list[int],
    fps: int,
    output_dir: Path,
) -> list[Path]:
    """
    Slice a scene's audio segment into sub-clip audio files.

    Timing is derived from frame counts to maintain frame-accuracy.
    Called AFTER compute_subclip_frames(), BEFORE video generation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cursor_frames = 0

    for i, n_frames in enumerate(subclip_frames):
        start_sec = cursor_frames / fps
        end_sec = (cursor_frames + n_frames) / fps

        sub_audio = output_dir / f"{scene_id}_sub_{i:02d}.wav"
        slice_audio(scene_audio, sub_audio, start_sec, end_sec)
        paths.append(sub_audio)

        cursor_frames += n_frames

    return paths


def generate_silence(output: Path, duration: float, sample_rate: int = 16000) -> Path:
    """Generate a silent WAV file of the given duration."""
    _check_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sample_rate}:cl=mono",
        "-t", f"{duration:.6f}",
        "-c:a", "pcm_s16le",
        str(output),
    ]

    subprocess.run(cmd, capture_output=True, check=True)
    log.debug("Generated silence: %s (%.2fs)", output.name, duration)
    return output


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
