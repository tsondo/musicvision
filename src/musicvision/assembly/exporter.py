"""
FCPXML and EDL export for DaVinci Resolve / Final Cut Pro.

Both formats describe a timeline of scene clips at their correct positions.
Import into DaVinci Resolve via File → Import → Timeline (EDL or FCPXML).

EDL  — CMX 3600, universally supported, simple cut list
FCPXML — Apple format v1.10, richer metadata, supported by DaVinci Resolve 18+
"""

from __future__ import annotations

import hashlib
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from musicvision.assembly.timecode import seconds_to_timecode
from musicvision.models import SceneList
from musicvision.utils.paths import ProjectPaths

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EDL — CMX 3600
# ---------------------------------------------------------------------------

def export_edl(
    scenes: SceneList,
    paths: ProjectPaths,
    fps: int = 25,
    output_path: Path | None = None,
) -> Path:
    """
    Generate a CMX 3600 EDL file for DaVinci Resolve import.

    Each scene clip becomes one cut event on the video track.
    Timecodes are non-drop-frame at the given fps.

    Args:
        scenes: SceneList with scene metadata
        paths: Project paths resolver
        fps: Frames per second (default 25, matches HuMo output)
        output_path: Override output location (default: output/timeline.edl)

    Returns:
        Path to the written EDL file
    """
    output = output_path or (paths.output_dir / "timeline.edl")
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered = [s for s in sorted(scenes.scenes, key=lambda s: s.order) if s.video_clip]

    if not ordered:
        raise ValueError("No scenes with clips found — run video generation first.")

    lines: list[str] = [
        f"TITLE: {paths.root.name}",
        "FCM: NON-DROP FRAME",
        "",
    ]

    timeline_pos = 0.0  # running record position

    for event_num, scene in enumerate(ordered, 1):
        duration = scene.duration

        # Source timecodes — clip-relative (always starts at 00:00:00:00)
        src_in  = seconds_to_timecode(0.0,      fps)
        src_out = seconds_to_timecode(duration,  fps)

        # Record timecodes — timeline position
        rec_in  = seconds_to_timecode(timeline_pos,           fps)
        rec_out = seconds_to_timecode(timeline_pos + duration, fps)

        clip_path = Path(scene.video_clip)
        if not clip_path.is_absolute():
            clip_path = paths.root / clip_path
        clip_name = clip_path.name

        # EDL event line
        lines.append(
            f"{event_num:03d}  AX       V     C        "
            f"{src_in} {src_out} {rec_in} {rec_out}"
        )
        # Comment lines (recognised by DaVinci Resolve for media relinking)
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        lines.append(f"* SCENE: {scene.id}  TYPE: {scene.type.value}  BPM_ALIGNED: true")
        if scene.lyrics:
            # Truncate long lyrics so the comment stays on one line
            lyrics_preview = scene.lyrics[:60] + ("..." if len(scene.lyrics) > 60 else "")
            lines.append(f"* LYRICS: {lyrics_preview}")
        lines.append("")  # blank line between events

        timeline_pos += duration

    output.write_text("\n".join(lines), encoding="utf-8")
    log.info("EDL saved: %s (%d events)", output, len(ordered))
    return output


# ---------------------------------------------------------------------------
# FCPXML — v1.10
# ---------------------------------------------------------------------------

def export_fcpxml(
    scenes: SceneList,
    paths: ProjectPaths,
    fps: int = 25,
    width: int = 1280,
    height: int = 720,
    output_path: Path | None = None,
) -> Path:
    """
    Generate an FCPXML 1.10 file for DaVinci Resolve / Final Cut Pro import.

    Creates:
      - One <format> resource (fps + resolution)
      - One <asset> resource per scene clip
      - A <sequence> with a <spine> placing clips at their timeline positions

    Args:
        scenes: SceneList with scene metadata
        paths: Project paths resolver
        fps: Frames per second (default 25, matches HuMo output)
        width: Frame width in pixels
        height: Frame height in pixels
        output_path: Override output location (default: output/timeline.fcpxml)

    Returns:
        Path to the written FCPXML file
    """
    output = output_path or (paths.output_dir / "timeline.fcpxml")
    output.parent.mkdir(parents=True, exist_ok=True)

    ordered = [s for s in sorted(scenes.scenes, key=lambda s: s.order) if s.video_clip]

    if not ordered:
        raise ValueError("No scenes with clips found — run video generation first.")

    total_duration = sum(s.duration for s in ordered)

    # --- Build XML tree ---
    root = ET.Element("fcpxml", {"version": "1.10"})

    # Resources block
    resources = ET.SubElement(root, "resources")

    ET.SubElement(resources, "format", {
        "id": "r1",
        "name": f"FFVideoFormat{height}p{fps}",
        "frameDuration": f"1/{fps}s",
        "width": str(width),
        "height": str(height),
    })

    # One asset per scene
    asset_id_map: dict[str, str] = {}  # scene_id → asset element id

    for i, scene in enumerate(ordered, 2):  # r1 is the format, clips start at r2
        clip_path = Path(scene.video_clip)
        if not clip_path.is_absolute():
            clip_path = paths.root / clip_path

        asset_ref = f"r{i}"
        asset_id_map[scene.id] = asset_ref

        ET.SubElement(resources, "asset", {
            "id": asset_ref,
            "name": scene.id,
            "uid": _uid(clip_path),
            "src": clip_path.as_uri(),
            "start": "0s",
            "duration": _rational(scene.duration, fps),
            "hasVideo": "1",
            "hasAudio": "0",        # HuMo clips are video-only before muxing
            "audioSources": "0",
            "audioChannels": "0",
            "audioRate": "48000",
        })

    # Library → Event → Project → Sequence → Spine
    library  = ET.SubElement(root, "library", {"location": paths.root.as_uri()})
    event    = ET.SubElement(library, "event", {"name": "MusicVision"})
    project  = ET.SubElement(event, "project", {"name": paths.root.name})
    sequence = ET.SubElement(project, "sequence", {
        "duration":    _rational(total_duration, fps),
        "format":      "r1",
        "tcStart":     "0s",
        "tcFormat":    "NDF",
        "audioLayout": "stereo",
        "audioRate":   "48000",
    })
    spine = ET.SubElement(sequence, "spine")

    # Place each clip on the spine
    timeline_pos = 0.0
    for scene in ordered:
        ET.SubElement(spine, "clip", {
            "name":     scene.id,
            "ref":      asset_id_map[scene.id],
            "offset":   _rational(timeline_pos, fps),
            "duration": _rational(scene.duration, fps),
            "start":    "0s",
        })
        timeline_pos += scene.duration

    # --- Serialize ---
    ET.indent(root, space="    ")

    with open(output, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<!DOCTYPE fcpxml>\n')
        ET.ElementTree(root).write(f, encoding="unicode", xml_declaration=False)

    log.info("FCPXML saved: %s (%d clips)", output, len(ordered))
    return output


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rational(seconds: float, fps: int) -> str:
    """
    Convert seconds to FCPXML rational time notation.

    Examples (25 fps):
        0.0   → "0s"
        1.0   → "1s"
        3.88  → "97/25s"
        4.0   → "4s"
    """
    if seconds == 0.0:
        return "0s"
    frames = round(seconds * fps)
    if frames % fps == 0:
        return f"{frames // fps}s"
    return f"{frames}/{fps}s"


def _uid(path: Path) -> str:
    """Deterministic 32-char uppercase hex UID from a file path."""
    return hashlib.md5(str(path).encode()).hexdigest().upper()
