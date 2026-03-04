"""Tests for the video upscaler models, factory, and pipeline logic."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from musicvision.models import (
    ProjectConfig,
    Scene,
    SceneList,
    SubClip,
    TargetResolution,
    UpscalerConfig,
    UpscalerType,
    VideoEngineType,
)
from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.upscaling.factory import create_upscale_engine
from musicvision.utils.paths import ProjectPaths

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------


class TestUpscalerTypeEnum:
    def test_values(self):
        assert UpscalerType.LTX_SPATIAL.value == "ltx_spatial"
        assert UpscalerType.SEEDVR2.value == "seedvr2"
        assert UpscalerType.REAL_ESRGAN.value == "real_esrgan"
        assert UpscalerType.NONE.value == "none"

    def test_all_values(self):
        assert len(UpscalerType) == 4


class TestTargetResolutionEnum:
    def test_values(self):
        assert TargetResolution.HD_720P.value == "720p"
        assert TargetResolution.FHD_1080P.value == "1080p"
        assert TargetResolution.QHD_1440P.value == "1440p"
        assert TargetResolution.UHD_4K.value == "4k"

    def test_all_values(self):
        assert len(TargetResolution) == 4


# ---------------------------------------------------------------------------
# UpscalerConfig tests
# ---------------------------------------------------------------------------


class TestUpscalerConfig:
    def test_defaults(self):
        cfg = UpscalerConfig()
        assert cfg.enabled is True
        assert cfg.target_resolution == TargetResolution.FHD_1080P
        assert cfg.upscaler_override is None
        assert cfg.preview_upscaler == UpscalerType.NONE

    def test_target_width_height_1080p(self):
        cfg = UpscalerConfig(target_resolution=TargetResolution.FHD_1080P)
        assert cfg.target_width_height() == (1920, 1080)

    def test_target_width_height_720p(self):
        cfg = UpscalerConfig(target_resolution=TargetResolution.HD_720P)
        assert cfg.target_width_height() == (1280, 720)

    def test_target_width_height_4k(self):
        cfg = UpscalerConfig(target_resolution=TargetResolution.UHD_4K)
        assert cfg.target_width_height() == (3840, 2160)

    def test_auto_select_humo(self):
        cfg = UpscalerConfig()
        result = cfg.get_upscaler_for_engine("humo", "final")
        assert result == UpscalerType.SEEDVR2

    def test_auto_select_hva(self):
        cfg = UpscalerConfig()
        result = cfg.get_upscaler_for_engine(VideoEngineType.HUNYUAN_AVATAR, "final")
        assert result == UpscalerType.SEEDVR2

    def test_auto_select_ltx(self):
        cfg = UpscalerConfig()
        result = cfg.get_upscaler_for_engine("ltx_video", "final")
        assert result == UpscalerType.LTX_SPATIAL

    def test_preview_mode_returns_none(self):
        cfg = UpscalerConfig()
        result = cfg.get_upscaler_for_engine("humo", "preview")
        assert result == UpscalerType.NONE

    def test_preview_mode_custom(self):
        cfg = UpscalerConfig(preview_upscaler=UpscalerType.REAL_ESRGAN)
        result = cfg.get_upscaler_for_engine("humo", "preview")
        assert result == UpscalerType.REAL_ESRGAN

    def test_disabled_returns_none(self):
        cfg = UpscalerConfig(enabled=False)
        result = cfg.get_upscaler_for_engine("humo", "final")
        assert result == UpscalerType.NONE

    def test_override_wins(self):
        cfg = UpscalerConfig(upscaler_override=UpscalerType.REAL_ESRGAN)
        result = cfg.get_upscaler_for_engine("ltx_video", "final")
        assert result == UpscalerType.REAL_ESRGAN


# ---------------------------------------------------------------------------
# Scene/SubClip upscaled_clip field
# ---------------------------------------------------------------------------


class TestUpscaledClipField:
    def test_scene_default_none(self):
        scene = Scene(id="s1", order=0, time_start=0, time_end=5)
        assert scene.upscaled_clip is None

    def test_scene_set(self):
        scene = Scene(id="s1", order=0, time_start=0, time_end=5, upscaled_clip="clips_upscaled/s1.mp4")
        assert scene.upscaled_clip == "clips_upscaled/s1.mp4"

    def test_subclip_default_none(self):
        sc = SubClip(id="s1_a", time_start=0, time_end=3)
        assert sc.upscaled_clip is None

    def test_subclip_set(self):
        sc = SubClip(id="s1_a", time_start=0, time_end=3, upscaled_clip="clips_upscaled/sub/s1_a.mp4")
        assert sc.upscaled_clip == "clips_upscaled/sub/s1_a.mp4"


# ---------------------------------------------------------------------------
# ProjectConfig YAML roundtrip
# ---------------------------------------------------------------------------


class TestProjectConfigUpscaler:
    def test_default_has_upscaler(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.upscaler, UpscalerConfig)
        assert cfg.upscaler.enabled is True

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ProjectConfig(name="test")
        cfg.upscaler.target_resolution = TargetResolution.UHD_4K
        cfg.upscaler.realesrgan_model = "custom-model"

        yaml_path = tmp_path / "project.yaml"
        cfg.save(yaml_path)

        loaded = ProjectConfig.load(yaml_path)
        assert loaded.upscaler.target_resolution == TargetResolution.UHD_4K
        assert loaded.upscaler.realesrgan_model == "custom-model"
        assert loaded.upscaler.enabled is True


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    def test_none_raises(self):
        with pytest.raises(ValueError, match="Cannot create engine"):
            create_upscale_engine(UpscalerType.NONE, UpscalerConfig())

    def test_real_esrgan_creates(self):
        engine = create_upscale_engine(UpscalerType.REAL_ESRGAN, UpscalerConfig())
        from musicvision.upscaling.realesrgan_engine import RealEsrganEngine
        assert isinstance(engine, RealEsrganEngine)

    def test_seedvr2_creates(self):
        engine = create_upscale_engine(UpscalerType.SEEDVR2, UpscalerConfig())
        from musicvision.upscaling.seedvr2_engine import SeedVR2Engine
        assert isinstance(engine, SeedVR2Engine)

    def test_ltx_spatial_creates(self):
        engine = create_upscale_engine(UpscalerType.LTX_SPATIAL, UpscalerConfig())
        from musicvision.upscaling.ltx_spatial_engine import LtxSpatialEngine
        assert isinstance(engine, LtxSpatialEngine)


# ---------------------------------------------------------------------------
# ProjectPaths tests
# ---------------------------------------------------------------------------


class TestProjectPaths:
    def test_clips_upscaled_dir(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.clips_upscaled_dir == tmp_path / "clips_upscaled"

    def test_sub_clips_upscaled_dir(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.sub_clips_upscaled_dir == tmp_path / "clips_upscaled" / "sub"

    def test_upscaled_clip_path(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        assert paths.upscaled_clip_path("scene_001") == tmp_path / "clips_upscaled" / "scene_001.mp4"

    def test_scaffold_creates_upscaled_dirs(self, tmp_path):
        paths = ProjectPaths(tmp_path)
        paths.scaffold()
        assert paths.clips_upscaled_dir.is_dir()
        assert paths.sub_clips_upscaled_dir.is_dir()


# ---------------------------------------------------------------------------
# Pipeline tests (with mock engine)
# ---------------------------------------------------------------------------


class MockUpscaleEngine(UpscaleEngine):
    """Test double for upscale engines."""

    def __init__(self):
        self._loaded = False
        self.calls: list[UpscaleInput] = []

    def load(self) -> None:
        self._loaded = True

    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        self.calls.append(input)
        # Create the output file (empty) so pipeline can find it
        input.output_path.parent.mkdir(parents=True, exist_ok=True)
        input.output_path.touch()
        return UpscaleResult(
            video_path=input.output_path,
            source_resolution=(480, 320),
            output_resolution=(input.target_width, input.target_height),
        )

    def unload(self) -> None:
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded


class TestPipeline:
    def _make_scene(self, sid: str, video_clip: str | None = None, engine: VideoEngineType | None = None) -> Scene:
        return Scene(
            id=sid, order=int(sid.split("_")[1]),
            time_start=0, time_end=5,
            video_clip=video_clip,
            video_engine=engine,
        )

    def test_skips_scenes_without_video(self, tmp_path):
        from musicvision.upscaling.pipeline import upscale_clips

        scenes = SceneList(scenes=[self._make_scene("scene_001")])
        paths = ProjectPaths(tmp_path)
        paths.scaffold()
        cfg = UpscalerConfig()

        result = upscale_clips(scenes, paths, cfg, VideoEngineType.HUMO)
        assert result["upscaled"] == []
        assert result["failed"] == []

    def test_groups_by_engine(self, tmp_path):
        """Scenes with different engines should get different upscalers."""

        s1 = self._make_scene("scene_001", video_clip="clips/scene_001.mp4", engine=VideoEngineType.HUMO)
        s2 = self._make_scene("scene_002", video_clip="clips/scene_002.mp4", engine=VideoEngineType.LTX_VIDEO)

        cfg = UpscalerConfig()
        # HuMo → SeedVR2
        assert cfg.get_upscaler_for_engine(s1.video_engine, "final") == UpscalerType.SEEDVR2
        # LTX → LTX Spatial
        assert cfg.get_upscaler_for_engine(s2.video_engine, "final") == UpscalerType.LTX_SPATIAL

    @patch("musicvision.upscaling.pipeline.create_upscale_engine")
    def test_upscale_single_clip(self, mock_factory, tmp_path):
        from musicvision.upscaling.pipeline import upscale_clips

        paths = ProjectPaths(tmp_path)
        paths.scaffold()

        # Create a fake video clip
        clip = paths.clips_dir / "scene_001.mp4"
        clip.touch()

        scene = self._make_scene("scene_001", video_clip="clips/scene_001.mp4")
        scenes = SceneList(scenes=[scene])

        mock_engine = MockUpscaleEngine()
        mock_factory.return_value = mock_engine

        cfg = UpscalerConfig()
        result = upscale_clips(scenes, paths, cfg, VideoEngineType.HUMO)

        assert result["upscaled"] == ["scene_001"]
        assert len(mock_engine.calls) == 1
        assert mock_engine.calls[0].target_width == 1920
        assert mock_engine.calls[0].target_height == 1080
        assert scene.upscaled_clip is not None

    @patch("musicvision.upscaling.pipeline.create_upscale_engine")
    def test_upscale_with_sub_clips(self, mock_factory, tmp_path):
        from musicvision.upscaling.pipeline import upscale_clips

        paths = ProjectPaths(tmp_path)
        paths.scaffold()

        # Create fake sub-clip files
        (paths.sub_clips_dir / "scene_001_a.mp4").touch()
        (paths.sub_clips_dir / "scene_001_b.mp4").touch()

        scene = Scene(
            id="scene_001", order=1, time_start=0, time_end=10,
            sub_clips=[
                SubClip(id="scene_001_a", time_start=0, time_end=5, video_clip="clips/sub/scene_001_a.mp4"),
                SubClip(id="scene_001_b", time_start=5, time_end=10, video_clip="clips/sub/scene_001_b.mp4"),
            ],
        )
        scenes = SceneList(scenes=[scene])

        mock_engine = MockUpscaleEngine()
        mock_factory.return_value = mock_engine

        cfg = UpscalerConfig()

        with patch("musicvision.utils.audio.concat_videos"):
            result = upscale_clips(scenes, paths, cfg, VideoEngineType.HUMO)

        assert result["upscaled"] == ["scene_001"]
        assert len(mock_engine.calls) == 2
        assert scene.sub_clips[0].upscaled_clip is not None
        assert scene.sub_clips[1].upscaled_clip is not None
        assert scene.upscaled_clip is not None

    def test_disabled_skips_all(self, tmp_path):
        from musicvision.upscaling.pipeline import upscale_clips

        paths = ProjectPaths(tmp_path)
        paths.scaffold()

        clip = paths.clips_dir / "scene_001.mp4"
        clip.touch()

        scene = self._make_scene("scene_001", video_clip="clips/scene_001.mp4")
        scenes = SceneList(scenes=[scene])

        cfg = UpscalerConfig(enabled=False)
        result = upscale_clips(scenes, paths, cfg, VideoEngineType.HUMO)

        assert result["upscaled"] == []
        assert scene.upscaled_clip is None

    def test_scene_ids_filter(self, tmp_path):
        from musicvision.upscaling.pipeline import upscale_clips

        paths = ProjectPaths(tmp_path)
        paths.scaffold()

        (paths.clips_dir / "scene_001.mp4").touch()
        (paths.clips_dir / "scene_002.mp4").touch()

        s1 = self._make_scene("scene_001", video_clip="clips/scene_001.mp4")
        s2 = self._make_scene("scene_002", video_clip="clips/scene_002.mp4")
        scenes = SceneList(scenes=[s1, s2])

        cfg = UpscalerConfig(enabled=False)  # disabled, so won't actually upscale
        result = upscale_clips(scenes, paths, cfg, VideoEngineType.HUMO, scene_ids=["scene_001"])

        # Both filtered by disabled config
        assert result["upscaled"] == []
