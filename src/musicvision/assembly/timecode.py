"""
Timecode utilities for scene filenames and export formats.
"""

from __future__ import annotations


def seconds_to_timecode(seconds: float, fps: int = 25) -> str:
    """Convert seconds to HH:MM:SS:FF timecode string."""
    total_frames = int(seconds * fps)
    h = total_frames // (fps * 3600)
    remainder = total_frames % (fps * 3600)
    m = remainder // (fps * 60)
    remainder = remainder % (fps * 60)
    s = remainder // fps
    f = remainder % fps
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def seconds_to_filename_stamp(seconds: float) -> str:
    """
    Convert seconds to filename-safe timestamp: MMmSSsNNN

    Example: 63.88 → "01m03s880"
    """
    total_ms = int(seconds * 1000)
    m = total_ms // 60000
    remainder = total_ms % 60000
    s = remainder // 1000
    ms = remainder % 1000
    return f"{m:02d}m{s:02d}s{ms:03d}"


def scene_clip_filename(scene_id: str, time_start: float, time_end: float) -> str:
    """
    Generate a timecoded filename for a scene clip.

    Example: scene_001_00m00s000_00m03s880.mp4
    """
    start_str = seconds_to_filename_stamp(time_start)
    end_str = seconds_to_filename_stamp(time_end)
    return f"{scene_id}_{start_str}_{end_str}.mp4"
