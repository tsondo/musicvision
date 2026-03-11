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


class VideoType(str, Enum):
    """Overall video concept type — affects defaults and prompt injection."""
    PERFORMANCE = "performance"  # staged performance, lip sync on by default
    STORY       = "story"        # narrative scenes, no lip sync
    HYBRID      = "hybrid"       # mix of performance + narrative, lip sync opt-in


class SceneTreatment(str, Enum):
    """Per-scene visual treatment (hybrid mode)."""
    PERFORMANCE = "performance"  # performer on stage/venue
    NARRATIVE   = "narrative"    # story/cinematic scene


class SeparationMethod(str, Enum):
    DEMUCS = "demucs"  # Demucs htdemucs/mdx via demucs library


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

    PREVIEW — FP8 14B + LoRA + 384p + 6 steps + CFG=1: ~48s/clip, fast iteration.
    DRAFT   — FP8 14B + 480p + 15 steps: quick iteration with dual CFG.
    FAST    — FP8 14B + LoRA + 384p + 6 steps + CFG=1: alias for PREVIEW.
    FINAL   — FP8 14B + 544p + 30 steps: best quality (2x upscale → 1080p).
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
    LTX_VIDEO       = "ltx_video"


class SceneAudioMode(str, Enum):
    """Per-scene audio mixing mode for LTX-2 generated audio."""
    SONG_ONLY      = "song_only"        # Default: original song, no generated audio
    GENERATED_ONLY = "generated_only"   # Only LTX-2 audio (song silent for this scene)
    MIX            = "mix"              # Generated audio layered over ducked song


class UpscalerType(str, Enum):
    """Selectable video upscaler backend."""
    LTX_SPATIAL  = "ltx_spatial"     # Latent-space upsampler, LTX-2 output only
    SEEDVR2      = "seedvr2"         # Pixel-space one-step diffusion (ByteDance)
    REAL_ESRGAN  = "real_esrgan"     # Frame-by-frame super-resolution
    NONE         = "none"            # Skip upscaling


class TargetResolution(str, Enum):
    """Output resolution target for upscaling."""
    HD_720P   = "720p"     # 1280×720
    FHD_1080P = "1080p"    # 1920×1080
    QHD_1440P = "1440p"    # 2560×1440
    UHD_4K    = "4k"       # 3840×2160


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
    concept: str = ""  # overall video concept: "what kind of video are we making"
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
    resolution: str = "544p"          # "720p", "544p", "480p", or "384p"
    scale_a: float = 5.5              # audio guidance strength (original HuMo default 5.5)
    scale_t: float = 5.0              # text guidance strength (original HuMo default 5.0)
    denoising_steps: int = 50         # 6 with LoRA, 15–30 standard, 50 max quality
    shift: float = 5.0                # sigma schedule shift (original HuMo default 5.0)
    block_swap_count: int = 0         # DiT blocks to keep on CPU (0 = all on GPU)
    sub_clip_continuity: bool = True  # pass last frame of sub-clip N as reference for sub-clip N+1
    sampler: str = "uni_pc"           # "uni_pc" or "euler"
    lora: str | None = None           # LoRA key from weight_registry (e.g. "lightx2v_i2v_480p")
    seed: int | None = None           # global seed (overridden by per-clip HumoInput.seed)

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
                tier=HumoTier.FP8_SCALED, resolution="384p", denoising_steps=6,
                shift=5.0, scale_t=1.0, scale_a=1.0, lora="lightx2v_i2v_480p",
            ),
            HumoQuality.DRAFT: dict(
                tier=HumoTier.FP8_SCALED, resolution="480p", denoising_steps=20,
            ),
            HumoQuality.FAST: dict(
                tier=HumoTier.FP8_SCALED, resolution="384p", denoising_steps=6,
                shift=5.0, scale_t=1.0, scale_a=1.0, lora="lightx2v_i2v_480p",
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
        return {"720p": 720, "544p": 544, "480p": 480, "384p": 384}.get(self.resolution, 544)

    @property
    def width(self) -> int:
        return {"720p": 1280, "544p": 960, "480p": 832, "384p": 688}.get(self.resolution, 960)


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


class LtxVideoConfig(BaseModel):
    """Configuration for LTX-Video 2 in-process engine (diffusers)."""

    model_id: str = "Lightricks/LTX-2"
    use_fp8: bool = True                    # FP8 transformer (27GB vs 43GB BF16)
    gguf_repo: str = "gguf-org/ltx2-gguf"   # Pre-quantized GGUF transformer
    gguf_file: str = "ltx2-19b-dev-iq4_nl.gguf"  # IQ4_NL: ~11GB, comfortable on 5090
    width: int = 768                        # must be divisible by 32
    height: int = 512                       # must be divisible by 32
    num_frames: int = 121                   # must be (N*8)+1; max 257
    fps: int = 24
    num_inference_steps: int = 40
    guidance_scale: float = 4.0
    negative_prompt: str = (
        "shaky, glitchy, low quality, worst quality, deformed, distorted, "
        "disfigured, motion smear, motion artifacts, fused fingers, bad anatomy, "
        "weird hand, ugly, transition, static"
    )
    seed: int | None = None
    use_audio_conditioning: bool = False    # TODO: needs mel spectrogram encoding, not raw waveform
    vae_tiling: bool = True                 # required to avoid OOM at higher res
    cpu_offload: str = "model"              # "none" | "model" | "sequential"

    @property
    def max_frames(self) -> int:
        return 257

    @property
    def max_duration(self) -> float:
        return self.max_frames / self.fps

    @staticmethod
    def snap_frames(target: int) -> int:
        """Snap a target frame count to nearest valid (N*8)+1 value."""
        n = round((target - 1) / 8)
        return max(n * 8 + 1, 9)


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


class SongSection(BaseModel):
    """A labeled section of the song (e.g. 'Verse 1', 'Chorus')."""
    name: str
    time: float  # estimated start time in seconds


class SongInfo(BaseModel):
    audio_file: str = ""
    lyrics_file: str = ""
    bpm: Optional[float] = None
    duration_seconds: Optional[float] = None
    keyscale: str = ""
    acestep: Optional[AceStepMeta] = None  # populated if AceStep JSON was found
    beat_times: list[float] = Field(default_factory=list)
    sections: list[SongSection] = Field(default_factory=list)
    sections_source: str = ""  # "acestep" or "auto" — empty means not yet set
    analyzed: bool = False  # True after Phase 1 (BPM, Whisper, demucs) completes


# ---------------------------------------------------------------------------
# Project config (project.yaml)
# ---------------------------------------------------------------------------

class VocalSeparationConfig(BaseModel):
    method: SeparationMethod = SeparationMethod.DEMUCS
    demucs_model: DemucsModel = DemucsModel.HTDEMUCS_FT


# Resolution lookup: enum value → (width, height)
_RESOLUTION_WH: dict[str, tuple[int, int]] = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k":    (3840, 2160),
}

# Default upscaler per video engine (pixel-space engines use SeedVR2, LTX uses latent upsampler)
_ENGINE_UPSCALER: dict[str, UpscalerType] = {
    "humo":            UpscalerType.SEEDVR2,
    "ltx_video":       UpscalerType.LTX_SPATIAL,
}


class UpscalerConfig(BaseModel):
    """Configuration for the video upscaling stage."""

    enabled: bool = True
    target_resolution: TargetResolution = TargetResolution.FHD_1080P
    upscaler_override: UpscalerType | None = None  # force specific upscaler for all engines

    # LTX Spatial Upsampler
    ltx_spatial_model_id: str = "Lightricks/ltxv-spatial-upscaler-0.9.7"
    ltx_spatial_steps: int = 10

    # SeedVR2
    seedvr2_repo_dir: str = ""          # env: SEEDVR2_REPO_DIR
    seedvr2_venv_python: str = ""       # env: SEEDVR2_VENV_PYTHON
    seedvr2_use_fp8: bool = True
    seedvr2_model_id: str = "ByteDance-Seed/SeedVR2-3B"

    # Real-ESRGAN
    realesrgan_model: str = "realesrgan-x4plus-anime"

    # Preview mode skips upscaling
    preview_upscaler: UpscalerType = UpscalerType.NONE

    def get_upscaler_for_engine(
        self,
        engine: str | VideoEngineType,
        render_mode: str = "final",
    ) -> UpscalerType:
        """Select the appropriate upscaler for a video engine.

        Returns NONE if upscaling is disabled or in preview mode.
        """
        if not self.enabled:
            return UpscalerType.NONE
        if render_mode == "preview":
            return self.preview_upscaler
        if self.upscaler_override is not None:
            return self.upscaler_override
        key = engine.value if isinstance(engine, VideoEngineType) else engine
        preferred = _ENGINE_UPSCALER.get(key, UpscalerType.SEEDVR2)
        # Fall back to Real-ESRGAN if SeedVR2 isn't configured
        if preferred == UpscalerType.SEEDVR2 and not self.seedvr2_repo_dir:
            return UpscalerType.REAL_ESRGAN
        return preferred

    def target_width_height(self) -> tuple[int, int]:
        """Return (width, height) for the target resolution."""
        return _RESOLUTION_WH[self.target_resolution.value]

    @staticmethod
    def max_resolution_for_vram(primary_vram_gb: float) -> TargetResolution:
        """Highest upscale resolution supported by the primary GPU.

        - 720p / 1080p: ≤32 GB (RTX 4080/5090 class)
        - 1440p:        ≥48 GB (A6000 / dual-GPU offload)
        - 4K:           ≥48 GB
        """
        if primary_vram_gb >= 48:
            return TargetResolution.UHD_4K
        return TargetResolution.FHD_1080P


class ProjectConfig(BaseModel):
    name: str = "Untitled Project"
    created: datetime = Field(default_factory=lambda: datetime.now(UTC))
    song: SongInfo = Field(default_factory=SongInfo)
    style_sheet: StyleSheet = Field(default_factory=StyleSheet)
    video_type: VideoType = VideoType.HYBRID
    video_engine: VideoEngineType = VideoEngineType.HUMO
    humo: HumoConfig = Field(default_factory=HumoConfig)
    ltx_video: LtxVideoConfig = Field(default_factory=LtxVideoConfig)
    image_gen: ImageGenConfig = Field(default_factory=ImageGenConfig)
    vocal_separation: VocalSeparationConfig = Field(default_factory=VocalSeparationConfig)
    upscaler: UpscalerConfig = Field(default_factory=UpscalerConfig)

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
    upscaled_clip: Optional[str] = None # path to upscaled clip
    status: ApprovalStatus = ApprovalStatus.PENDING
    frame_count: Optional[int] = None   # authoritative duration in frames


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
    section: str = ""                   # e.g. "verse_1", "chorus" — from LLM or AceStep markers

    # Frame-accurate fields (populated by engine_registry.plan_subclips)
    frame_start: Optional[int] = None
    frame_end: Optional[int] = None
    total_frames: Optional[int] = None
    subclip_frame_counts: Optional[list[int]] = None
    generation_audio_segments: Optional[list[str]] = None

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
    video_width: Optional[int] = None            # resolution of current best clip
    video_height: Optional[int] = None
    upscaled_clip: Optional[str] = None          # path: clips_upscaled/scene_001.mp4
    video_status: ApprovalStatus = ApprovalStatus.PENDING
    video_engine: Optional[VideoEngineType] = None  # None → use project default
    video_seed: Optional[int] = None                # seed used for last generation (locked when approved)
    lip_sync: Optional[bool] = None  # None → auto (depends on video_type + treatment)
    treatment: Optional[SceneTreatment] = None  # None → auto from video_type
    # TODO: per-scene face mask for multi-person lip sync targeting

    # LTX-2 generated audio mixing
    generated_audio: Optional[str] = None                          # path to .gen_audio.wav
    audio_mode: SceneAudioMode = SceneAudioMode.SONG_ONLY
    generated_audio_volume: float = 0.8       # 0.0–1.0, gen audio loudness
    song_duck_volume: float = 0.3             # 0.0–1.0, song volume when ducking
    audio_fade_in: float = 0.15              # seconds, gen audio fade in
    audio_fade_out: float = 0.15             # seconds, gen audio fade out
    song_duck_fade_in: float = 0.3           # seconds, song duck ramp down
    song_duck_fade_out: float = 0.3          # seconds, song duck ramp up

    # Sub-clips for long scenes
    sub_clips: list[SubClip] = Field(default_factory=list)

    # Style sheet references
    characters: list[str] = Field(default_factory=list)     # character IDs
    props: list[str] = Field(default_factory=list)           # prop IDs
    settings: list[str] = Field(default_factory=list)        # setting IDs

    notes: str = ""

    @property
    def effective_lip_sync(self) -> bool:
        """Resolve lip_sync: explicit override wins, otherwise vocal=True, instrumental=False."""
        if self.lip_sync is not None:
            return self.lip_sync
        return self.type == SceneType.VOCAL

    @property
    def duration(self) -> float:
        return self.time_end - self.time_start

    @property
    def needs_sub_clips(self) -> bool:
        """HuMo max is 97 frames @ 25fps ≈ 3.88s.

        Deprecated: use ``needs_sub_clips_for_engine()`` with engine constraints
        from ``engine_registry`` for accurate frame-based checks.
        """
        return self.duration > 3.88

    def needs_sub_clips_for_engine(
        self, engine: VideoEngineType,
    ) -> bool:
        """Check if this scene needs sub-clips for a specific engine."""
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


# ---------------------------------------------------------------------------
# Analysis result + manual scene boundary (Phase 1 & 2 split)
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """Returned by run_analyze() — everything the waveform editor needs."""
    duration: float
    bpm: float | None = None
    beat_times: list[float] = Field(default_factory=list)
    word_timestamps: list[dict] = Field(default_factory=list)  # [{word, start, end}]
    vocal_path: str | None = None
    sections: list[SongSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SceneBoundary(BaseModel):
    """A manual scene boundary from the waveform editor."""
    time_start: float
    time_end: float
    section: str = ""
    type: SceneType = SceneType.VOCAL
    lyrics: str = ""
