"""
Project directory management.

Every file MusicVision creates lives under a project root with a fixed structure.
This module creates the structure and resolves paths so nothing else hardcodes paths.
"""

from __future__ import annotations

from pathlib import Path


class ProjectPaths:
    """Resolves all paths within a MusicVision project directory."""

    def __init__(self, root: Path):
        self.root = root

    # --- Top-level files ---

    @property
    def config_file(self) -> Path:
        return self.root / "project.yaml"

    @property
    def scenes_file(self) -> Path:
        return self.root / "scenes.json"

    # --- Input ---

    @property
    def input_dir(self) -> Path:
        return self.root / "input"

    # --- Assets (user-provided references, LoRAs) ---

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def characters_dir(self) -> Path:
        return self.assets_dir / "characters"

    @property
    def props_dir(self) -> Path:
        return self.assets_dir / "props"

    @property
    def settings_dir(self) -> Path:
        return self.assets_dir / "settings"

    @property
    def loras_dir(self) -> Path:
        return self.assets_dir / "loras"

    # --- Generated segments ---

    @property
    def segments_dir(self) -> Path:
        return self.root / "segments"

    @property
    def segments_vocal_dir(self) -> Path:
        return self.root / "segments_vocal"

    @property
    def sub_segments_dir(self) -> Path:
        return self.segments_dir / "sub"

    # --- Generated images ---

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    # --- Generated video clips ---

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def sub_clips_dir(self) -> Path:
        return self.clips_dir / "sub"

    # --- Output ---

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def output_scenes_dir(self) -> Path:
        return self.output_dir / "scenes"

    # --- Convenience ---

    def segment_path(self, scene_id: str) -> Path:
        return self.segments_dir / f"{scene_id}.wav"

    def vocal_segment_path(self, scene_id: str) -> Path:
        return self.segments_vocal_dir / f"{scene_id}_vocal.wav"

    def image_path(self, scene_id: str) -> Path:
        return self.images_dir / f"{scene_id}.png"

    def clip_path(self, scene_id: str) -> Path:
        return self.clips_dir / f"{scene_id}.mp4"

    def sub_clip_path(self, scene_id: str, suffix: str) -> Path:
        return self.sub_clips_dir / f"{scene_id}_{suffix}.mp4"

    def scaffold(self) -> None:
        """Create the full directory tree. Safe to call multiple times."""
        for d in [
            self.input_dir,
            self.characters_dir,
            self.props_dir,
            self.settings_dir,
            self.loras_dir,
            self.segments_dir,
            self.sub_segments_dir,
            self.segments_vocal_dir,
            self.images_dir,
            self.clips_dir,
            self.sub_clips_dir,
            self.output_dir,
            self.output_scenes_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
