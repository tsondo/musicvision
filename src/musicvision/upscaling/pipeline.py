"""
Upscale pipeline orchestrator.

Groups scenes by video engine, selects the appropriate upscaler for each group,
and processes all clips. Handles sub-clips (upscale each, then join).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from musicvision.models import JobProgress, Scene, SceneList, UpscalerConfig, UpscalerType, VideoEngineType
from musicvision.upscaling.base import UpscaleInput
from musicvision.upscaling.factory import create_upscale_engine
from musicvision.utils.paths import ProjectPaths

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)


def upscale_clips(
    scenes: SceneList,
    paths: ProjectPaths,
    upscaler_config: UpscalerConfig,
    default_engine: VideoEngineType,
    render_mode: str = "final",
    scene_ids: list[str] | None = None,
    device_map: DeviceMap | None = None,
    on_progress: Callable[[JobProgress], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    skip_current: bool = False,
) -> dict:
    """Upscale video clips for scenes.

    Args:
        scenes: Scene list with video_clip / sub_clips populated.
        paths: Project paths resolver.
        upscaler_config: Upscaler configuration.
        default_engine: Project-level default video engine.
        render_mode: "preview" or "final" — affects upscaler selection.
        scene_ids: If provided, only upscale these scenes.
        device_map: GPU device map (needed for LTX Spatial).
        on_progress: Optional per-scene progress callback (job model). Absent
            = behavior unchanged (CLI).
        should_cancel: Optional cooperative-cancel check, polled between
            scenes (never mid-upscale). When it returns True the remaining
            scenes are skipped and the result reports ``cancelled: True``.
        skip_current: Skip scenes whose upscaled clip exists on disk at the
            target resolution (listed under ``skipped``). Video jobs clear
            ``upscaled_clip`` when they record a new clip, so a skipped
            upscale is never stale.

    Returns:
        Dict with "upscaled" and "failed" scene ID lists (plus
        ``"cancelled": True`` if stopped early via ``should_cancel``).
    """
    target_w, target_h = upscaler_config.target_width_height()

    # Gate high resolutions on hardware — 1440p+ needs ≥48GB VRAM
    if target_h > 1080:
        try:
            import torch
            if torch.cuda.is_available():
                primary_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                max_res = UpscalerConfig.max_resolution_for_vram(primary_vram)
                max_w, max_h = UpscalerConfig(target_resolution=max_res).target_width_height()
                if target_h > max_h:
                    log.warning(
                        "Target %dp exceeds hardware capability (%.0fGB VRAM, max %dp). Clamping to %dp.",
                        target_h, primary_vram, max_h, max_h,
                    )
                    target_w, target_h = max_w, max_h
        except Exception:
            pass

    # Filter to scenes with video clips
    targets = [s for s in scenes.scenes if _has_video(s)]
    if scene_ids:
        targets = [s for s in targets if s.id in scene_ids]

    skipped: list[str] = []
    if skip_current:
        keep = []
        for s in targets:
            if _upscale_is_current(s, paths, target_w, target_h):
                skipped.append(s.id)
            else:
                keep.append(s)
        targets = keep
        if skipped:
            log.info("Skipping %d scene(s) already upscaled to %dx%d", len(skipped), target_w, target_h)

    if not targets:
        log.info("No scenes with video clips to upscale")
        return {"upscaled": [], "failed": [], "skipped": skipped}

    # Group by video engine → select upscaler per group
    engine_groups: dict[UpscalerType, list[Scene]] = defaultdict(list)
    for scene in sorted(targets, key=lambda s: s.order):
        engine = scene.video_engine or default_engine
        upscaler_type = upscaler_config.get_upscaler_for_engine(engine, render_mode)
        if upscaler_type == UpscalerType.NONE:
            log.debug("Skipping upscale for %s (upscaler=NONE)", scene.id)
            continue
        engine_groups[upscaler_type].append(scene)

    if not engine_groups:
        log.info("All scenes mapped to NONE upscaler — skipping")
        return {"upscaled": [], "failed": [], "skipped": skipped}

    upscaled: list[str] = []
    failed: list[str] = []
    total = sum(len(g) for g in engine_groups.values())
    done = 0

    def _report(scene_id: str | None, message: str) -> None:
        if on_progress is not None:
            on_progress(JobProgress(current=done, total=total, unit="scenes", message=message, scene_id=scene_id))

    for upscaler_type, group_scenes in engine_groups.items():
        if should_cancel is not None and should_cancel():
            break
        log.info("Upscaling %d scene(s) with %s", len(group_scenes), upscaler_type.value)

        engine = create_upscale_engine(upscaler_type, upscaler_config, device_map=device_map)
        engine.load()
        try:
            for scene in group_scenes:
                if should_cancel is not None and should_cancel():
                    break
                _report(scene.id, f"Upscaling {scene.id} with {getattr(engine, 'backend_label', upscaler_type.value)}")
                try:
                    _upscale_scene(scene, engine, paths, target_w, target_h)
                    # Update resolution metadata from the upscaled clip
                    from musicvision.utils.video import update_scene_resolution
                    update_scene_resolution(scene, paths.root)
                    upscaled.append(scene.id)
                    log.info("Upscaled %s", scene.id)
                except Exception:
                    log.exception("Failed to upscale %s", scene.id)
                    failed.append(scene.id)
                done += 1
                _report(scene.id, f"Upscaled {done}/{total}")
        finally:
            engine.unload()

    result: dict = {"upscaled": upscaled, "failed": failed, "skipped": skipped}
    # Cancelled only if scenes were actually left undone; a cancel that arrives
    # after the last scene finished is a normal completion.
    if done < total:
        result["cancelled"] = True
    return result


def _upscale_is_current(scene: Scene, paths: ProjectPaths, target_w: int, target_h: int) -> bool:
    """True if the scene's upscaled clip exists at exactly the target resolution."""
    if not scene.upscaled_clip:
        return False
    p = _abs(scene.upscaled_clip, paths)
    if not p.exists():
        return False
    try:
        from musicvision.utils.video import get_video_resolution

        return get_video_resolution(p) == (target_w, target_h)
    except Exception:
        return False


def _has_video(scene: Scene) -> bool:
    """Check if a scene has any video clips (direct or sub-clips)."""
    if scene.video_clip:
        return True
    return any(sc.video_clip for sc in scene.sub_clips)


def _upscale_scene(
    scene: Scene,
    engine,
    paths: ProjectPaths,
    target_w: int,
    target_h: int,
) -> None:
    """Upscale a single scene's clip(s) and update scene model."""
    paths.clips_upscaled_dir.mkdir(parents=True, exist_ok=True)
    paths.sub_clips_upscaled_dir.mkdir(parents=True, exist_ok=True)

    if scene.sub_clips:
        # Upscale each sub-clip individually
        for sc in scene.sub_clips:
            if not sc.video_clip:
                continue
            src = _abs(sc.video_clip, paths)
            if not src.exists():
                log.warning("Sub-clip missing: %s", src)
                continue

            out = paths.sub_clips_upscaled_dir / f"{sc.id}.mp4"
            inp = UpscaleInput(
                video_path=src,
                output_path=out,
                target_width=target_w,
                target_height=target_h,
            )
            engine.upscale(inp)
            sc.upscaled_clip = str(out.relative_to(paths.root))

        # Join upscaled sub-clips into one scene clip
        upscaled_subs = [
            _abs(sc.upscaled_clip, paths)
            for sc in scene.sub_clips
            if sc.upscaled_clip
        ]
        if upscaled_subs:
            if len(upscaled_subs) == 1:
                scene.upscaled_clip = str(upscaled_subs[0].relative_to(paths.root))
            else:
                from musicvision.utils.audio import concat_videos

                joined = paths.clips_upscaled_dir / f"{scene.id}_joined.mp4"
                concat_videos(upscaled_subs, joined)
                scene.upscaled_clip = str(joined.relative_to(paths.root))
    else:
        # Single clip
        if not scene.video_clip:
            return
        src = _abs(scene.video_clip, paths)
        if not src.exists():
            log.warning("Clip missing: %s", src)
            return

        out = paths.clips_upscaled_dir / f"{scene.id}.mp4"
        inp = UpscaleInput(
            video_path=src,
            output_path=out,
            target_width=target_w,
            target_height=target_h,
        )
        engine.upscale(inp)
        scene.upscaled_clip = str(out.relative_to(paths.root))


def _abs(relative_or_abs: str, paths: ProjectPaths) -> Path:
    """Resolve a path that may be project-relative or absolute."""
    p = Path(relative_or_abs)
    return p if p.is_absolute() else paths.root / p
