"""
Project lifecycle management.

This is the service layer between the API/CLI and the filesystem.
Creates projects, loads/saves config, manages scene state transitions.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from musicvision.models import ProjectConfig, SceneList, AceStepMeta
from musicvision.utils.paths import ProjectPaths

log = logging.getLogger(__name__)


class ProjectService:
    """Manages a single MusicVision project."""

    def __init__(self, project_dir: Path):
        self.paths = ProjectPaths(project_dir)
        self._config: ProjectConfig | None = None
        self._scenes: SceneList | None = None

    @property
    def config(self) -> ProjectConfig:
        if self._config is None:
            if self.paths.config_file.exists():
                self._config = ProjectConfig.load(self.paths.config_file)
            else:
                self._config = ProjectConfig()
        return self._config

    @config.setter
    def config(self, value: ProjectConfig) -> None:
        self._config = value

    @property
    def scenes(self) -> SceneList:
        if self._scenes is None:
            if self.paths.scenes_file.exists():
                self._scenes = SceneList.load(self.paths.scenes_file)
            else:
                self._scenes = SceneList()
        return self._scenes

    @scenes.setter
    def scenes(self, value: SceneList) -> None:
        self._scenes = value

    # --- Lifecycle ---

    @classmethod
    def create(cls, project_dir: Path, name: str = "Untitled Project") -> ProjectService:
        """Create a new project with scaffolded directories."""
        svc = cls(project_dir)
        svc.paths.scaffold()
        svc.config = ProjectConfig(name=name)
        svc.save_config()
        svc.scenes = SceneList()
        svc.save_scenes()
        log.info("Created project '%s' at %s", name, project_dir)
        return svc

    @classmethod
    def open(cls, project_dir: Path) -> ProjectService:
        """Open an existing project."""
        svc = cls(project_dir)
        if not svc.paths.config_file.exists():
            raise FileNotFoundError(f"No project.yaml found in {project_dir}")
        _ = svc.config  # force load to validate
        log.info("Opened project '%s' from %s", svc.config.name, project_dir)
        return svc

    # --- Persistence ---

    def save_config(self) -> None:
        self.config.save(self.paths.config_file)

    def save_scenes(self) -> None:
        self.scenes.save(self.paths.scenes_file)

    def save_all(self) -> None:
        self.save_config()
        self.save_scenes()

    # --- Audio import ---

    def import_audio(self, source_path: Path) -> Path:
        """
        Copy audio file into project input/ directory.

        Auto-detects AceStep JSON: if a .json file with the same stem exists
        next to the audio file, it's imported as AceStep metadata.
        """
        dest = self.paths.input_dir / source_path.name
        shutil.copy2(source_path, dest)
        self.config.song.audio_file = f"input/{source_path.name}"

        # Check for companion AceStep JSON (same name, .json extension)
        json_path = source_path.with_suffix(".json")
        if json_path.exists():
            log.info("Found AceStep JSON: %s", json_path.name)
            self._import_acestep_json(json_path)

        self.save_config()
        log.info("Imported audio: %s", dest)
        return dest

    def import_acestep_json(self, json_path: Path) -> None:
        """Explicitly import an AceStep JSON file."""
        self._import_acestep_json(json_path)
        self.save_config()

    def _import_acestep_json(self, json_path: Path) -> None:
        """Parse AceStep JSON and populate song metadata.

        Handles both flat format (old) and nested params/meta format (current):
          - params: user-specified values (bpm, key, duration, lyrics) — preferred
          - meta: auto-detected values (bpm can be wrong, e.g. half-time)
          - top-level: legacy flat format
        """
        import json

        with open(json_path) as f:
            data = json.load(f)

        params = data.get("params", {})
        meta_block = data.get("meta", {})

        # Prefer params (user-specified) > meta (auto-detected) > top-level (legacy)
        bpm = params.get("bpm") or meta_block.get("bpm") or data.get("bpm")
        keyscale = params.get("key") or meta_block.get("keyscale") or data.get("keyscale", "")
        duration = params.get("duration") or meta_block.get("duration") or data.get("duration")
        lyrics = params.get("lyrics") or meta_block.get("lyrics") or data.get("lyrics", "")
        caption = meta_block.get("prompt") or data.get("caption", "")
        seed = params.get("seed") or data.get("seed")
        instrumental = data.get("instrumental", False)

        meta = AceStepMeta(
            caption=caption,
            lyrics=lyrics,
            instrumental=instrumental,
            bpm=bpm,
            keyscale=keyscale,
            duration=duration,
            seed=seed,
            raw=data,
        )

        self.config.song.acestep = meta

        # Populate song-level fields from AceStep data
        if meta.bpm and not self.config.song.bpm:
            self.config.song.bpm = meta.bpm
        if meta.duration and not self.config.song.duration_seconds:
            self.config.song.duration_seconds = meta.duration
        if meta.keyscale:
            self.config.song.keyscale = meta.keyscale

        # If AceStep has lyrics, save them as the lyrics file
        if meta.lyrics and not self.config.song.lyrics_file:
            lyrics_dest = self.paths.input_dir / "lyrics_acestep.txt"
            lyrics_dest.write_text(meta.lyrics, encoding="utf-8")
            self.config.song.lyrics_file = "input/lyrics_acestep.txt"
            log.info("Extracted lyrics from AceStep JSON → %s", lyrics_dest.name)

        # The caption is extremely useful for the style sheet
        if meta.caption:
            log.info("AceStep caption available for style sheet: %s", meta.caption[:80])

        log.info(
            "AceStep metadata: BPM=%s, key=%s, duration=%ss, instrumental=%s",
            meta.bpm, meta.keyscale, meta.duration, meta.instrumental,
        )

    def import_lyrics(self, source_path: Path) -> Path:
        """Copy lyrics file into project input/ directory."""
        dest = self.paths.input_dir / source_path.name
        shutil.copy2(source_path, dest)
        self.config.song.lyrics_file = f"input/{source_path.name}"
        self.save_config()
        log.info("Imported lyrics: %s", dest)
        return dest

    def resolve_path(self, relative: str) -> Path:
        """Resolve a project-relative path to absolute."""
        return self.paths.root / relative
