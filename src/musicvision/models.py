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


class SeparationMethod(str, Enum):
    ROFORMER = "roformer"   # MelBandRoFormer via audio-separator (best for live recordings)
    DEMUCS   = "demucs"     # Demucs htdemucs/mdx via demucs library (often superior on AI-generated music)


class DemucsModel(str, Enum):
    HTDEMUCS    = "htdemucs"       # Hybrid Transformer — best overall, recommended default
    HTDEMUCS_FT = "htdemucs_ft"    # Fine-tuned on pop/rock — tighter transients
    MDX_EXTRA   = "mdx_extra"      # MDX STFT-based — excellent vocal isolation (may need CPU fallback)
    MDX_EXTRA_Q = "mdx_extra_q"    # MDX quantised — cleanest output; requires diffq


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class HumoTier(str, Enum):
    """
    HuMo precision tier.  Ordered from highest quality → lowest VRAM requirement.

    Tier          Model   Format               DiT VRAM   Min GPU
    ─────────────────────────────────────────────────────────────
    fp16          17B     FP16 safetensors     ~34 GB     2× GPU (FSDP)
    fp8_scaled    17B     FP8 e4m3fn scaled    ~18 GB     16 GB (single GPU)
    gguf_q8       17B     GGUF Q8_0            ~18.5 GB   20 GB
    gguf_q6       17B     GGUF Q6_K            ~14.4 GB   16 GB
    gguf_q4       17B     GGUF Q4_K_M          ~11.5 GB   12 GB
    preview       1.7B    FP16 safetensors     ~3.4 GB    8 GB (fast iteration)
    """
    FP16       = "fp16"
    FP8_SCALED = "fp8_scaled"
    GGUF_Q8    = "gguf_q8"
    GGUF_Q6    = "gguf_q6"
    GGUF_Q4    = "gguf_q4"
    PREVIEW    = "preview"


TIER_MODEL_SIZE: dict[str, str] = {
    "fp16":       "17B",
    "fp8_scaled": "17B",
    "gguf_q8":    "17B",
    "gguf_q6":    "17B",
    "gguf_q4":    "17B",
    "preview":    "1.7B",
}

TIER_VRAM_GB: dict[str, float] = {
    "fp16":       34.0,
    "fp8_scaled": 18.0,
    "gguf_q8":    18.5,
    "gguf_q6":    14.4,
    "gguf_q4":    11.5,
    "preview":     3.4,
}


class HumoQuality(str, Enum):
    """
    Quality presets for HuMo video generation.

    PREVIEW — 1.7B model + 480p + 10 steps: seconds per clip, for layout checks.
    DRAFT   — FP8 14B + 480p + 15 steps: quick iteration with real model quality.
    FAST    — FP8 14B + Lightx2V LoRA + 480p + 6 steps + CFG=1: ~5 min/clip.
    FINAL   — FP8 14B + 720p + 50 steps: full quality for final render.
    """
    PREVIEW = "preview"
    DRAFT   = "draft"
    FAST    = "fast"
    FINAL   = "final"


class ImageModel(str, Enum):
    FLUX_DEV    = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    ZIMAGE      = "z-image"
    ZIMAGE_TURBO = "z-image-turbo"

    # Backward-compat short aliases (so FluxModel.DEV / FluxModel.SCHNELL still work)
    DEV     = "flux-dev"
    SCHNELL = "flux-schnell"


# Backward-compat alias
FluxModel = ImageModel


class FluxQuant(str, Enum):
    AUTO = "auto"   # pick based on available VRAM at load time
    BF16 = "bf16"   # full precision — needs ≥24 GB per device
    FP8  = "fp8"    # fp8 quantized transformer — needs ≥12 GB (Ada/Hopper only)
    INT8 = "int8"   # int8 quantized transformer — needs ≥10 GB, any GPU


class VideoEngineType(str, Enum):
    """Selectable video generation backend."""
    HUMO            = "humo"
    HUNYUAN_AVATAR  = "hunyuan_avatar"


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
    tier: HumoTier = HumoTier.FP8_SCALED
    resolution: str = "720p"          # "720p", "480p", or "384p" (688×384, Lightx2V distilled)
    scale_a: float = 2.0              # audio guidance strength (1.0–3.0)
    scale_t: float = 7.5              # text guidance strength (5.0–10.0)
    denoising_steps: int = 50         # 30–40 faster, 50 best quality
    shift: float = 8.0                # sigma schedule shift (higher → more high-noise steps)
    block_swap_count: int = 0         # DiT blocks to keep on CPU (0 = all on GPU)
    sub_clip_continuity: bool = True  # pass last frame of sub-clip N as reference for sub-clip N+1
    sampler: str = "uni_pc"           # "uni_pc" or "euler"
    lora: str | None = None           # LoRA key from weight_registry (e.g. "lightx2v_i2v_480p")

    @classmethod
    def from_quality(cls, quality: HumoQuality | str, **overrides) -> "HumoConfig":
        """Create a HumoConfig from a quality preset.

        Any keyword argument overrides the preset defaults (e.g.
        ``HumoConfig.from_quality("draft", block_swap_count=20)``).
        """
        if isinstance(quality, str):
            quality = HumoQuality(quality)
        presets: dict[HumoQuality, dict] = {
            HumoQuality.PREVIEW: dict(
                tier=HumoTier.PREVIEW, resolution="480p", denoising_steps=10,
            ),
            HumoQuality.DRAFT: dict(
                tier=HumoTier.FP8_SCALED, resolution="480p", denoising_steps=15,
            ),
            HumoQuality.FAST: dict(
                tier=HumoTier.FP8_SCALED, resolution="384p", denoising_steps=6,
                shift=8.0, scale_t=1.0, scale_a=1.0, lora="lightx2v_i2v_480p",
            ),
            HumoQuality.FINAL: dict(
                tier=HumoTier.FP8_SCALED, resolution="720p", denoising_steps=50,
            ),
        }
        params = presets[quality]
        params.update(overrides)
        return cls(**params)

    @property
    def model_size(self) -> str:
        return TIER_MODEL_SIZE[self.tier.value]

    @property
    def height(self) -> int:
        return {"720p": 720, "480p": 480, "384p": 384}.get(self.resolution, 720)

    @property
    def width(self) -> int:
        return {"720p": 1280, "480p": 832, "384p": 688}.get(self.resolution, 1280)


class ImageGenConfig(BaseModel):
    model: ImageModel = ImageModel.FLUX_DEV
    quant: FluxQuant = FluxQuant.AUTO
    steps: Optional[int] = None         # None → auto: 4 for schnell, 28 for dev
    guidance_scale: float = 3.5
    lora_path: Optional[str] = None     # project-level style LoRA (relative to project root)
    lora_weight: float = 0.8

    @property
    def effective_steps(self) -> int:
        if self.steps is not None:
            return self.steps
        return 4 if self.model == ImageModel.FLUX_SCHNELL else 28


class HunyuanAvatarConfig(BaseModel):
    """Configuration for HunyuanVideo-Avatar subprocess engine."""

    hva_repo_dir: str = ""              # path to cloned HunyuanVideoAvatar repo
    hva_venv_python: str = ""           # path to python in HVA venv (e.g. ~/HunyuanVideoAvatar/.venv/bin/python)
    checkpoint: str = "bf16"            # "bf16" (recommended) or "fp8" — bf16 works with block offloading
    image_size: int = 704               # width (height auto-calculated for portrait)
    sample_n_frames: int = 129          # 129 frames @ 25fps ≈ 5.16s (fixed by HVA, cannot be reduced)
    cfg_scale: float = 7.5
    infer_steps: int = 30               # 30 good balance; 50 highest quality
    flow_shift: float = 5.0
    seed: int | None = None             # None → random
    use_deepcache: bool = True
    use_fp8: bool = False               # FP8 breaks block-level offloading; use bf16 instead
    cpu_offload: bool = True            # Required for ≤32GB VRAM; enables block-level transformer offloading
    fps: int = 25

    @property
    def max_duration(self) -> float:
        """Maximum clip duration in seconds."""
        return self.sample_n_frames / self.fps


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

class VocalSeparationConfig(BaseModel):
    method: SeparationMethod = SeparationMethod.ROFORMER
    demucs_model: DemucsModel = DemucsModel.HTDEMUCS
    roformer_model: str = "MelBandRoformer.ckpt"


class ProjectConfig(BaseModel):
    name: str = "Untitled Project"
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    song: SongInfo = Field(default_factory=SongInfo)
    style_sheet: StyleSheet = Field(default_factory=StyleSheet)
    video_engine: VideoEngineType = VideoEngineType.HUMO
    humo: HumoConfig = Field(default_factory=HumoConfig)
    hunyuan_avatar: HunyuanAvatarConfig = Field(default_factory=HunyuanAvatarConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    vocal_separation: VocalSeparationConfig = Field(default_factory=VocalSeparationConfig)

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
    video_engine: Optional[VideoEngineType] = None  # None → use project default

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

    def needs_sub_clips_for_engine(
        self, engine: VideoEngineType, hva_config: HunyuanAvatarConfig | None = None,
    ) -> bool:
        """Check if this scene needs sub-clips for a specific engine."""
        if engine == VideoEngineType.HUNYUAN_AVATAR and hva_config:
            return self.duration > hva_config.max_duration
        return self.duration > 3.88  # HuMo default

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
