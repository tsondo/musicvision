"""
Engine constraints and frame-accurate sub-clip computation.

This module is the ONLY place engine max/min frames, FPS, and sub-clip
math are defined. All other modules import from here.

No dependencies on other musicvision modules (except models for type hints
in plan_subclips).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConstraints:
    """Hard frame limits for a video generation engine."""

    name: str
    max_frames: int
    min_frames: int
    fps: int

    @property
    def max_seconds(self) -> float:
        return self.max_frames / self.fps

    @property
    def min_seconds(self) -> float:
        return self.min_frames / self.fps


# Registry — single source of truth.
# Keys match VideoEngineType enum values.
ENGINES: dict[str, EngineConstraints] = {
    "humo": EngineConstraints(
        name="HuMo",
        max_frames=97,
        min_frames=25,
        fps=25,
    ),
    "hunyuan_avatar": EngineConstraints(
        name="HunyuanVideo Avatar",
        max_frames=129,
        min_frames=33,
        fps=25,
    ),
}


def get_constraints(engine_key: str) -> EngineConstraints:
    """Look up constraints by engine key (e.g. 'humo', 'hunyuan_avatar')."""
    if engine_key not in ENGINES:
        raise ValueError(f"Unknown engine: {engine_key!r}. Available: {list(ENGINES.keys())}")
    return ENGINES[engine_key]


def scene_frames(time_start: float, time_end: float, fps: int) -> int:
    """Total frames for a scene. This is the authoritative duration."""
    return round((time_end - time_start) * fps)


def frames_to_seconds(frames: int, fps: int) -> float:
    """Derive seconds from frame count — never the other way around."""
    return frames / fps


def compute_subclip_frames(
    total_frames: int,
    max_frames: int,
    min_frames: int,
) -> list[int]:
    """
    Divide total_frames into sub-clips respecting engine constraints.

    Returns a list of frame counts. ``sum(result) == total_frames`` always.

    Strategy:
    - If total fits in one clip, return [total_frames].
    - Otherwise compute n = ceil(total / max). If the remainder
      (what the last clip would get) is below min_frames, reduce n by 1
      and redistribute evenly.
    - Equal distribution: each clip gets total // n, first (total % n) clips
      get one extra frame.
    """
    if total_frames <= 0:
        return []

    if total_frames <= max_frames:
        return [total_frames]

    n = math.ceil(total_frames / max_frames)
    remainder = total_frames - (n - 1) * max_frames

    # If the remainder (what the last clip would get in a naive split) is
    # below min_frames, try using fewer clips.  But only if the reduced
    # count still keeps every clip within max_frames.
    if remainder < min_frames and n > 1:
        candidate = n - 1
        if candidate > 0 and math.ceil(total_frames / candidate) <= max_frames:
            n = candidate

    # Equal distribution across n clips
    base = total_frames // n
    extra = total_frames % n

    # First 'extra' clips get base+1 frames, rest get base
    counts = [base + 1] * extra + [base] * (n - extra)

    assert sum(counts) == total_frames, f"Frame count mismatch: {sum(counts)} != {total_frames}"
    assert all(c >= min_frames for c in counts), f"Sub-clip below minimum: {min(counts)} < {min_frames}"
    assert all(c <= max_frames for c in counts), f"Sub-clip above maximum: {max(counts)} > {max_frames}"

    return counts


def compute_subclip_frames_at_silences(
    total_frames: int,
    max_frames: int,
    min_frames: int,
    fps: int,
    silences: list[tuple[float, float]],
) -> list[int]:
    """
    Divide total_frames into sub-clips, splitting at silence midpoints.

    Greedy walk: accumulate frames toward max_frames, split at the last
    valid silence midpoint before exceeding max. If no silence is available
    in the valid range, hard-cut at max_frames (same as equal distribution).

    Falls back to compute_subclip_frames() if the result violates constraints.

    Args:
        total_frames: Total frames in the scene.
        max_frames: Engine max frames per sub-clip.
        min_frames: Engine min frames per sub-clip.
        fps: Frames per second.
        silences: List of (start, end) silence intervals in seconds.

    Returns:
        List of frame counts summing to total_frames.
    """
    if total_frames <= 0:
        return []

    if total_frames <= max_frames:
        return [total_frames]

    # Convert silence midpoints to frame indices
    silence_frames = sorted(
        round(((s + e) / 2) * fps)
        for s, e in silences
    )

    counts: list[int] = []
    cursor = 0  # frames consumed so far

    while cursor < total_frames:
        remaining = total_frames - cursor

        # Last chunk — take everything
        if remaining <= max_frames:
            counts.append(remaining)
            break

        # Find the last silence midpoint frame within [cursor + min_frames, cursor + max_frames]
        window_lo = cursor + min_frames
        window_hi = cursor + max_frames

        # Also ensure the remainder after this split is >= min_frames
        # (unless the remainder would be the very last chunk, i.e. <= max_frames)
        best_split = None
        for sf in reversed(silence_frames):
            if sf < window_lo:
                break  # sorted, so no point continuing
            if sf > window_hi:
                continue
            chunk = sf - cursor
            leftover = total_frames - sf
            # Accept if leftover fits in one clip, or leftover is large enough to split further
            if leftover <= max_frames or leftover >= min_frames:
                best_split = chunk
                break

        if best_split is not None:
            counts.append(best_split)
            cursor += best_split
        else:
            # No silence in range — hard cut at max_frames
            counts.append(max_frames)
            cursor += max_frames

    # Validate
    valid = (
        sum(counts) == total_frames
        and all(c >= min_frames for c in counts)
        and all(c <= max_frames for c in counts)
    )

    if not valid:
        # Fall back to equal distribution
        return compute_subclip_frames(total_frames, max_frames, min_frames)

    return counts


def sub_clip_suffixes(n: int) -> list[str]:
    """Generate sub-clip suffixes: a, b, c, ... z, aa, ab, ..."""
    result: list[str] = []
    for i in range(n):
        if i < 26:
            result.append(chr(ord("a") + i))
        else:
            result.append(chr(ord("a") + (i // 26) - 1) + chr(ord("a") + (i % 26)))
    return result


def plan_subclips(
    scenes: list,
    constraints: EngineConstraints,
    segments_dir: "Path",
    sub_segments_dir: "Path",
) -> None:
    """
    Pre-compute sub-clip frame counts and slice audio for all scenes.

    Mutates each scene in-place, populating:
      - frame_start, frame_end, total_frames
      - subclip_frame_counts
      - generation_audio_segments (paths to sub-clip WAVs)

    Args:
        scenes: List of Scene objects.
        constraints: Engine frame constraints.
        segments_dir: Directory containing per-scene audio WAVs.
        sub_segments_dir: Directory for sub-clip audio output.
    """
    import logging
    from pathlib import Path

    from musicvision.utils.audio import detect_silences, slice_subclip_audio

    log = logging.getLogger(__name__)
    fps = constraints.fps

    for scene in scenes:
        total = scene_frames(scene.time_start, scene.time_end, fps)
        if total <= 0:
            continue

        frame_start = scene_frames(0.0, scene.time_start, fps)
        scene.frame_start = frame_start
        scene.frame_end = frame_start + total
        scene.total_frames = total

        # Try silence-aware splitting if scene needs sub-clips
        seg_path = segments_dir / f"{scene.id}.wav"
        if total > constraints.max_frames and seg_path.exists():
            try:
                silences = detect_silences(seg_path)
                counts = compute_subclip_frames_at_silences(
                    total, constraints.max_frames, constraints.min_frames, fps, silences,
                )
                if silences:
                    log.debug(
                        "Scene %s: silence-aware split (%d silences detected)",
                        scene.id, len(silences),
                    )
            except Exception:
                log.warning("Silence detection failed for scene %s, using equal split", scene.id, exc_info=True)
                counts = compute_subclip_frames(total, constraints.max_frames, constraints.min_frames)
        else:
            counts = compute_subclip_frames(total, constraints.max_frames, constraints.min_frames)

        scene.subclip_frame_counts = counts

        if len(counts) <= 1:
            # Single clip — use the scene's existing audio segment
            scene.generation_audio_segments = [str(seg_path)]
            continue

        # Multiple sub-clips — slice audio
        if not seg_path.exists():
            log.warning("Scene audio not found for sub-clip slicing: %s", seg_path)
            continue

        sub_paths = slice_subclip_audio(
            scene_audio=seg_path,
            scene_id=scene.id,
            subclip_frames=counts,
            fps=fps,
            output_dir=sub_segments_dir,
        )
        scene.generation_audio_segments = [str(p) for p in sub_paths]
        log.info(
            "Scene %s: %d frames → %d sub-clips (%s)",
            scene.id, total, len(counts), "+".join(str(c) for c in counts),
        )
