"""
Clip concatenation and audio sync for final assembly.

Assembles approved scene clips (and sub-clips) into a rough cut MP4
with the original song audio muxed back.
"""

from __future__ import annotations

import logging
from pathlib import Path

from musicvision.models import ApprovalStatus, Scene, SceneList
from musicvision.utils.audio import build_mixed_audio, concat_videos, get_duration, mux_video_audio
from musicvision.utils.paths import ProjectPaths

log = logging.getLogger(__name__)


class SyncError(RuntimeError):
    """Raised when assembled video duration does not match audio duration."""


def assemble_rough_cut(
    scenes: SceneList,
    paths: ProjectPaths,
    original_audio: Path,
    approved_only: bool = False,
) -> Path:
    """
    Assemble scene clips into a rough cut MP4.

    Steps:
      1. Sort scenes by order; optionally filter to approved only
      2. For scenes with sub-clips, join sub-clips into one clip first
      3. Concatenate all scene clips with ffmpeg concat demuxer
      4. Mux the original uncut song audio back over the video
      5. Write output/rough_cut.mp4

    Args:
        scenes: SceneList with scene metadata
        paths: Project paths resolver
        original_audio: Absolute path to the original song file
        approved_only: If True, only include scenes where video_status == APPROVED

    Returns:
        Path to output/rough_cut.mp4

    Raises:
        RuntimeError: If no clips are available to assemble
    """
    ordered = sorted(scenes.scenes, key=lambda s: s.order)

    clip_paths: list[Path] = []
    skipped = 0

    for scene in ordered:
        if approved_only and scene.video_status != ApprovalStatus.APPROVED:
            log.debug("Scene %s skipped (not approved)", scene.id)
            skipped += 1
            continue

        clip = _resolve_scene_clip(scene, paths)
        if clip is None:
            log.warning("Scene %s has no clip — skipping", scene.id)
            skipped += 1
            continue

        clip_paths.append(clip)

    if not clip_paths:
        raise RuntimeError(
            "No clips available to assemble. "
            "Run video generation first, or disable approved_only."
        )

    if skipped:
        log.warning("%d scene(s) skipped (no clip or not approved)", skipped)

    log.info("Assembling rough cut from %d clips...", len(clip_paths))

    paths.output_dir.mkdir(parents=True, exist_ok=True)

    if len(clip_paths) == 1:
        # Single clip — skip the concat step
        silent_cut = clip_paths[0]
    else:
        silent_cut = paths.output_dir / "_rough_cut_silent.mp4"
        concat_videos(clip_paths, silent_cut)

    # Duration sync check: verify video matches audio within one frame
    try:
        video_dur = get_duration(silent_cut)
        audio_dur = get_duration(original_audio)
        fps = 25  # default; could be made configurable via engine constraints
        tolerance = 1.0 / fps
        drift = abs(video_dur - audio_dur)

        if drift > tolerance:
            log.warning(
                "SYNC WARNING: video=%.4fs, audio=%.4fs, drift=%.4fs (%.1f frames)",
                video_dur, audio_dur, drift, drift * fps,
            )
            # Log but don't raise — assembly can still proceed, the mux uses -shortest
    except Exception as exc:
        log.debug("Could not verify sync: %s", exc)

    # Build mixed audio if any scene uses generated audio overlay
    mixed_audio = build_mixed_audio(
        original_audio,
        ordered,
        paths.root,
        paths.output_dir / "_mixed_audio.wav",
    )
    final_audio = mixed_audio if mixed_audio else original_audio

    output = paths.output_dir / "rough_cut.mp4"
    mux_video_audio(silent_cut, final_audio, output)

    # Clean up intermediate files
    if silent_cut != clip_paths[0] and silent_cut.exists():
        silent_cut.unlink(missing_ok=True)
    if mixed_audio and mixed_audio.exists():
        mixed_audio.unlink(missing_ok=True)

    log.info("Rough cut saved: %s (%.1fs, %d clips)", output, scenes.total_duration, len(clip_paths))
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_scene_clip(scene: Scene, paths: ProjectPaths) -> Path | None:
    """
    Return the final single-file clip path for a scene.

    Prefers upscaled clips over raw clips. For scenes with sub-clips,
    joins them first (preferring upscaled sub-clips).
    """
    if scene.sub_clips:
        return _join_sub_clips(scene, paths)

    # Prefer upscaled clip if available
    if scene.upscaled_clip:
        p = _abs(scene.upscaled_clip, paths)
        if p.exists():
            return p
        log.warning("Upscaled clip file missing for %s: %s — falling back to raw clip", scene.id, p)

    if scene.video_clip:
        p = _abs(scene.video_clip, paths)
        if p.exists():
            return p
        log.warning("Clip file missing for %s: %s", scene.id, p)

    return None


def _join_sub_clips(scene: Scene, paths: ProjectPaths) -> Path | None:
    """Join a scene's sub-clips into a single clip, returning its path.

    Prefers upscaled sub-clips over raw sub-clips.
    """
    sub_paths: list[Path] = []

    for sc in scene.sub_clips:
        # Prefer upscaled clip
        clip_path = sc.upscaled_clip or sc.video_clip
        if clip_path:
            p = _abs(clip_path, paths)
            if p.exists():
                sub_paths.append(p)
            else:
                log.warning("Sub-clip file missing for %s/%s: %s", scene.id, sc.id, p)

    if not sub_paths:
        return None

    if len(sub_paths) < len(scene.sub_clips):
        log.warning(
            "Scene %s: only %d/%d sub-clips ready — joining what's available",
            scene.id, len(sub_paths), len(scene.sub_clips),
        )

    if len(sub_paths) == 1:
        return sub_paths[0]

    joined = paths.clips_dir / f"{scene.id}_joined.mp4"
    log.debug("Joining %d sub-clips → %s", len(sub_paths), joined.name)
    concat_videos(sub_paths, joined)
    return joined


def _abs(relative_or_abs: str, paths: ProjectPaths) -> Path:
    """Resolve a path that may be project-relative or absolute."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else paths.root / p
