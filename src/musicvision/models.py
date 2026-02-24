"""
Data models for MusicVision projects.

All persistent state flows through these models:
  project.yaml  →  ProjectConfig
  scenes.json   →  SceneList

Every module reads/writes via these types. No raw dict manipulation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SceneType(str, Enum):
    VOCAL = "vocal"
    INSTRUMENTAL = "instrumental"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class HumoModelSize(str, Enum):
    LARGE = "17B"
    SMALL = "1.7B"


class HumoResolution(str, Enum):
    HD = "720p"
    SD = "480p"


class ImageModel(str, Enum):
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    ZIMAGE = "z-image"
    ZIMAGE_TURBO = "z-image-turbo"


# Backward-compat alias
FluxModel = ImageModel


# ---------------------------------------------------------------------------
# Style Sheet — persistent visual identity for the project
# ---------------------------------------------------------------------------

class CharacterDef(BaseModel):
    id: str
    description: str
    reference_image: Optional[str] = None
    lora_path: Optional[str] = None
    lora_weight: float = 0.8


class PropDef(BaseModel):
    id: str
    description: str
    reference_image: Optional[str] = None


class SettingDef(BaseModel):
    id: str
    description: str
    reference_image: Optional[str] = None


class StyleSheet(BaseModel):
    visual_style: str = ""
    color_palette: str = ""
    aspect_ratio: str = "16:9"
    resolution: str = "1280x720"
    characters: list[CharacterDef] = Field(default_factory=list)
    props: list[PropDef] = Field(default_factory=list)
    settings: list[SettingDef] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine configs
# ---------------------------------------------------------------------------

class HumoConfig(BaseModel):
    model_size: HumoModelSize = HumoModelSize.LARGE
    resolution: HumoResolution = HumoResolution.HD
    scale_a: float = 2.0        # audio guidance strength (1.0–3.0)
    scale_t: float = 7.5        # text guidance strength (5.0–10.0)
    denoising_steps: int = 50   # 30–40 faster, 50 best quality

    @property
    def height(self) -> int:
        return 720 if self.resolution == HumoResolution.HD else 480

    @property
    def width(self) -> int:
        return 1280 if self.resolution == HumoResolution.HD else 832


class ImageGenConfig(BaseModel):
    model: ImageModel = ImageModel.FLUX_DEV
    steps: int = 28
    guidance_scale: float = 3.5


# Backward-compat alias
FluxConfig = ImageGenConfig


# ---------------------------------------------------------------------------
# Song metadata
# ---------------------------------------------------------------------------

class AceStepMeta(BaseModel):
    """Metadata from AceStep music generation JSON."""

    caption: str = ""                     # genre/mood/instrumentation description
    lyrics: str = ""                      # full lyrics with section markers
    instrumental: bool = False
    bpm: Optional[float] = None
    keyscale: str = ""                    # e.g. "A minor"
    duration: Optional[float] = None      # seconds
    seed: Optional[int] = None
    # Preserve the full original JSON for reference
    raw: dict = Field(default_factory=dict)


class SongInfo(BaseModel):
    audio_file: str = ""
    lyrics_file: str = ""
    bpm: Optional[float] = None
    duration_seconds: Optional[float] = None
    keyscale: str = ""
    acestep: Optional[AceStepMeta] = None  # populated if AceStep JSON was found


# ---------------------------------------------------------------------------
# Project config (project.yaml)
# ---------------------------------------------------------------------------

class ProjectConfig(BaseModel):
    name: str = "Untitled Project"
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    song: SongInfo = Field(default_factory=SongInfo)
    style_sheet: StyleSheet = Field(default_factory=StyleSheet)
    humo: HumoConfig = Field(default_factory=HumoConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)

    @model_validator(mode="before")
    @classmethod
    def _migrate_flux_key(cls, data: dict) -> dict:
        """Accept 'flux' as a deprecated alias for 'image_gen'."""
        if isinstance(data, dict) and "flux" in data and "image_gen" not in data:
            data["image_gen"] = data.pop("flux")
        return data

    @property
    def flux(self) -> ImageGenConfig:
        """Deprecated alias for image_gen. Use image_gen instead."""
        return self.image_gen

    @classmethod
    def load(cls, path: Path) -> ProjectConfig:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Sub-clip (for scenes > ~3.88s)
# ---------------------------------------------------------------------------

class SubClip(BaseModel):
    id: str                             # e.g. "scene_003_a"
    time_start: float
    time_end: float
    audio_segment: Optional[str] = None
    video_prompt: Optional[str] = None
    video_clip: Optional[str] = None    # path to generated clip
    status: ApprovalStatus = ApprovalStatus.PENDING


# ---------------------------------------------------------------------------
# Scene definition
# ---------------------------------------------------------------------------

class Scene(BaseModel):
    id: str                             # e.g. "scene_001"
    order: int
    time_start: float
    time_end: float
    type: SceneType = SceneType.VOCAL
    lyrics: str = ""

    # Audio
    audio_segment: Optional[str] = None         # path: segments/scene_001.wav
    audio_segment_vocal: Optional[str] = None   # path: segments_vocal/scene_001_vocal.wav

    # Image generation
    image_prompt: Optional[str] = None
    image_prompt_user_override: Optional[str] = None
    reference_image: Optional[str] = None       # path: images/scene_001.png
    image_status: ApprovalStatus = ApprovalStatus.PENDING

    # Video generation
    video_prompt: Optional[str] = None
    video_prompt_user_override: Optional[str] = None
    video_clip: Optional[str] = None            # path: clips/scene_001.mp4
    video_status: ApprovalStatus = ApprovalStatus.PENDING

    # Sub-clips for long scenes
    sub_clips: list[SubClip] = Field(default_factory=list)

    # Style sheet references
    characters: list[str] = Field(default_factory=list)     # character IDs
    props: list[str] = Field(default_factory=list)           # prop IDs
    settings: list[str] = Field(default_factory=list)        # setting IDs

    notes: str = ""

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    @property
    def needs_sub_clips(self) -> bool:
        """HuMo max is 97 frames @ 25fps ≈ 3.88s."""
        return self.duration > 3.88

    @property
    def effective_image_prompt(self) -> Optional[str]:
        return self.image_prompt_user_override or self.image_prompt

    @property
    def effective_video_prompt(self) -> Optional[str]:
        return self.video_prompt_user_override or self.video_prompt


# ---------------------------------------------------------------------------
# Scene list (scenes.json)
# ---------------------------------------------------------------------------

class SceneList(BaseModel):
    scenes: list[Scene] = Field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> SceneList:
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    def get_scene(self, scene_id: str) -> Optional[Scene]:
        return next((s for s in self.scenes if s.id == scene_id), None)

    @property
    def total_duration(self) -> float:
        if not self.scenes:
            return 0.0
        return max(s.time_end for s in self.scenes)
