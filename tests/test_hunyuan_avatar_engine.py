"""Tests for the HunyuanVideo-Avatar engine integration."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from musicvision.models import (
    HumoConfig,
    HunyuanAvatarConfig,
    ProjectConfig,
    Scene,
    VideoEngineType,
)
from musicvision.video.base import VideoInput, VideoResult
from musicvision.video.factory import create_video_engine
from musicvision.video.hunyuan_avatar_engine import (
    HunyuanAvatarEngine,
    _sub_clip_suffixes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hva_config(tmp_path):
    """HVA config pointing at a fake repo dir."""
    repo = tmp_path / "HunyuanVideoAvatar"
    repo.mkdir()
    (repo / "hymm_sp").mkdir()
    (repo / "hymm_sp" / "sample_gpu_poor.py").write_text("# fake")
    venv_python = tmp_path / "python3"
    venv_python.write_text("#!/bin/sh\n")
    venv_python.chmod(0o755)
    return HunyuanAvatarConfig(
        hva_repo_dir=str(repo),
        hva_venv_python=str(venv_python),
    )


@pytest.fixture
def bad_config():
    """HVA config with non-existent paths."""
    return HunyuanAvatarConfig(
        hva_repo_dir="/nonexistent/repo",
        hva_venv_python="/nonexistent/python",
    )


# ---------------------------------------------------------------------------
# VideoEngineType enum
# ---------------------------------------------------------------------------


class TestVideoEngineType:
    def test_humo_value(self):
        assert VideoEngineType.HUMO.value == "humo"

    def test_hunyuan_avatar_value(self):
        assert VideoEngineType.HUNYUAN_AVATAR.value == "hunyuan_avatar"

    def test_from_string(self):
        assert VideoEngineType("hunyuan_avatar") == VideoEngineType.HUNYUAN_AVATAR


# ---------------------------------------------------------------------------
# HunyuanAvatarConfig
# ---------------------------------------------------------------------------


class TestHunyuanAvatarConfig:
    def test_defaults(self):
        cfg = HunyuanAvatarConfig()
        assert cfg.image_size == 512
        assert cfg.sample_n_frames == 129
        assert cfg.cfg_scale == 7.5
        assert cfg.infer_steps == 30
        assert cfg.flow_shift == 5.0
        assert cfg.use_deepcache is True
        assert cfg.use_fp8 is False
        assert cfg.cpu_offload is True
        assert cfg.fps == 25
        assert cfg.checkpoint == "bf16"

    def test_max_duration(self):
        cfg = HunyuanAvatarConfig()
        assert cfg.max_duration == pytest.approx(5.16)

    def test_max_duration_custom_frames(self):
        cfg = HunyuanAvatarConfig(sample_n_frames=97, fps=25)
        assert cfg.max_duration == pytest.approx(3.88)


# ---------------------------------------------------------------------------
# ProjectConfig integration
# ---------------------------------------------------------------------------


class TestProjectConfigEngine:
    def test_default_engine_is_humo(self):
        cfg = ProjectConfig()
        assert cfg.video_engine == VideoEngineType.HUMO

    def test_hunyuan_avatar_config_exists(self):
        cfg = ProjectConfig()
        assert isinstance(cfg.hunyuan_avatar, HunyuanAvatarConfig)

    def test_set_engine_type(self):
        cfg = ProjectConfig(video_engine=VideoEngineType.HUNYUAN_AVATAR)
        assert cfg.video_engine == VideoEngineType.HUNYUAN_AVATAR

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ProjectConfig(
            video_engine=VideoEngineType.HUNYUAN_AVATAR,
            hunyuan_avatar=HunyuanAvatarConfig(
                hva_repo_dir="/home/user/HVA",
                infer_steps=30,
            ),
        )
        path = tmp_path / "project.yaml"
        cfg.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.video_engine == VideoEngineType.HUNYUAN_AVATAR
        assert loaded.hunyuan_avatar.infer_steps == 30
        assert loaded.hunyuan_avatar.hva_repo_dir == "/home/user/HVA"


# ---------------------------------------------------------------------------
# Scene.video_engine field
# ---------------------------------------------------------------------------


class TestSceneVideoEngine:
    def test_default_is_none(self):
        scene = Scene(id="s1", order=1, time_start=0, time_end=3.0)
        assert scene.video_engine is None

    def test_set_engine(self):
        scene = Scene(
            id="s1", order=1, time_start=0, time_end=3.0,
            video_engine=VideoEngineType.HUNYUAN_AVATAR,
        )
        assert scene.video_engine == VideoEngineType.HUNYUAN_AVATAR

    def test_needs_sub_clips_humo(self):
        scene = Scene(id="s1", order=1, time_start=0, time_end=5.0)
        assert scene.needs_sub_clips  # 5.0 > 3.88

    def test_needs_sub_clips_for_hva(self):
        scene = Scene(id="s1", order=1, time_start=0, time_end=5.0)
        hva = HunyuanAvatarConfig()
        # 5.0 < 5.16, so no sub-clips needed for HVA
        assert not scene.needs_sub_clips_for_engine(VideoEngineType.HUNYUAN_AVATAR, hva)

    def test_needs_sub_clips_for_hva_long(self):
        scene = Scene(id="s1", order=1, time_start=0, time_end=10.0)
        hva = HunyuanAvatarConfig()
        # 10.0 > 5.16, so sub-clips needed
        assert scene.needs_sub_clips_for_engine(VideoEngineType.HUNYUAN_AVATAR, hva)


# ---------------------------------------------------------------------------
# Sub-clip suffixes (same logic as HuMo but local to HVA engine)
# ---------------------------------------------------------------------------


class TestHvaSubClipSuffixes:
    def test_single(self):
        assert _sub_clip_suffixes(1) == ["a"]

    def test_three(self):
        assert _sub_clip_suffixes(3) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_creates_hva_engine(self, hva_config):
        engine = create_video_engine(hva_config, engine_type=VideoEngineType.HUNYUAN_AVATAR)
        assert isinstance(engine, HunyuanAvatarEngine)

    def test_creates_humo_engine_default(self):
        """Default (HumoConfig) → HumoEngine, requires torch."""
        torch = pytest.importorskip("torch", reason="torch required")
        from musicvision.utils.gpu import DeviceMap
        from musicvision.video.humo_engine import HumoEngine

        cpu = torch.device("cpu")
        dm = DeviceMap(dit_device=cpu, encoder_device=cpu, vae_device=cpu, offload_device=cpu)
        engine = create_video_engine(HumoConfig(), device_map=dm)
        assert isinstance(engine, HumoEngine)

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError, match="HunyuanAvatarConfig"):
            create_video_engine(HumoConfig(), engine_type=VideoEngineType.HUNYUAN_AVATAR)

    def test_humo_without_device_map_raises(self):
        with pytest.raises(ValueError, match="device_map"):
            create_video_engine(HumoConfig(), engine_type=VideoEngineType.HUMO)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_not_loaded_initially(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        assert not engine.is_loaded

    def test_load_validates_paths(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        engine.load()
        assert engine.is_loaded

    def test_load_bad_repo_raises(self, bad_config):
        engine = HunyuanAvatarEngine(bad_config)
        with pytest.raises(FileNotFoundError, match="repo dir"):
            engine.load()

    def test_load_bad_python_raises(self, tmp_path):
        repo = tmp_path / "repo"
        repo.mkdir()
        cfg = HunyuanAvatarConfig(
            hva_repo_dir=str(repo),
            hva_venv_python="/nonexistent/python",
        )
        engine = HunyuanAvatarEngine(cfg)
        with pytest.raises(FileNotFoundError, match="venv python"):
            engine.load()

    def test_unload_resets_state(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        engine.load()
        engine.unload()
        assert not engine.is_loaded

    def test_unload_when_not_loaded_is_safe(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        engine.unload()  # should not raise

    def test_generate_before_load_raises(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        inp = VideoInput(
            text_prompt="test",
            reference_image=Path("/ref.png"),
            audio_segment=Path("/seg.wav"),
            output_path=Path("/out.mp4"),
        )
        with pytest.raises(RuntimeError, match="load"):
            engine.generate(inp)

    def test_generate_scene_before_load_raises(self, hva_config):
        engine = HunyuanAvatarEngine(hva_config)
        with pytest.raises(RuntimeError, match="load"):
            engine.generate_scene(
                text_prompt="test",
                reference_image=Path("/ref.png"),
                audio_segment=Path("/seg.wav"),
                output_dir=Path("/clips"),
                scene_id="scene_001",
                duration=3.0,
            )


# ---------------------------------------------------------------------------
# Generate (mocked subprocess)
# ---------------------------------------------------------------------------


class TestGenerate:
    def test_generate_calls_subprocess(self, hva_config, tmp_path):
        engine = HunyuanAvatarEngine(hva_config)
        engine.load()

        output_path = tmp_path / "output.mp4"
        inp = VideoInput(
            text_prompt="A woman singing",
            reference_image=tmp_path / "ref.png",
            audio_segment=tmp_path / "audio.wav",
            output_path=output_path,
        )

        mock_response = {
            "status": "success",
            "video_path": str(output_path),
            "frames_generated": 129,
            "duration": 5.16,
            "error": None,
        }

        def fake_run(cmd, **kwargs):
            # Write the response JSON that the wrapper would produce
            # Find --response arg
            resp_idx = cmd.index("--response") + 1
            resp_path = Path(cmd[resp_idx])
            resp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resp_path, "w") as f:
                json.dump(mock_response, f)

            mock = MagicMock()
            mock.returncode = 0
            return mock

        with patch("musicvision.video.hunyuan_avatar_engine.subprocess.run", side_effect=fake_run):
            result = engine.generate(inp)

        assert isinstance(result, VideoResult)
        assert result.frames_generated == 129
        assert result.duration_seconds == pytest.approx(5.16)
        assert result.metadata["engine"] == "hunyuan_avatar"

    def test_generate_error_raises(self, hva_config, tmp_path):
        engine = HunyuanAvatarEngine(hva_config)
        engine.load()

        inp = VideoInput(
            text_prompt="test",
            reference_image=tmp_path / "ref.png",
            audio_segment=tmp_path / "audio.wav",
            output_path=tmp_path / "out.mp4",
        )

        mock_response = {
            "status": "error",
            "video_path": None,
            "frames_generated": 0,
            "duration": 0.0,
            "error": "Checkpoint not found",
        }

        def fake_run(cmd, **kwargs):
            resp_idx = cmd.index("--response") + 1
            resp_path = Path(cmd[resp_idx])
            resp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(resp_path, "w") as f:
                json.dump(mock_response, f)
            mock = MagicMock()
            mock.returncode = 1
            return mock

        with patch("musicvision.video.hunyuan_avatar_engine.subprocess.run", side_effect=fake_run):
            with pytest.raises(RuntimeError, match="Checkpoint not found"):
                engine.generate(inp)


# ---------------------------------------------------------------------------
# Generate scene splitting
# ---------------------------------------------------------------------------


class TestGenerateScene:
    def test_short_scene_single_clip(self, hva_config, tmp_path):
        engine = HunyuanAvatarEngine(hva_config)
        engine._loaded = True

        mock_result = VideoResult(
            video_path=tmp_path / "scene_001.mp4",
            frames_generated=129,
            duration_seconds=4.0,
        )

        with patch.object(engine, "generate", return_value=mock_result) as mock_gen:
            results = engine.generate_scene(
                text_prompt="test",
                reference_image=tmp_path / "ref.png",
                audio_segment=tmp_path / "seg.wav",
                output_dir=tmp_path,
                scene_id="scene_001",
                duration=4.0,
            )

        assert len(results) == 1
        assert results[0] is mock_result
        mock_gen.assert_called_once()

    def test_long_scene_splits(self, hva_config, tmp_path):
        """12s scene with 5.16s max → 3 sub-clips."""
        engine = HunyuanAvatarEngine(hva_config)
        engine._loaded = True

        duration = 12.0
        max_dur = hva_config.max_duration  # 5.16
        expected_clips = math.ceil(duration / max_dur)  # 3

        # Create fake sub-clip audio files
        for i in range(expected_clips):
            (tmp_path / f"scene_001_sub_{i:02d}.wav").write_bytes(b"fake")

        call_count = 0
        def mock_generate(inp):
            nonlocal call_count
            call_count += 1
            return VideoResult(
                video_path=inp.output_path,
                frames_generated=129,
                duration_seconds=max_dur,
            )

        with patch.object(engine, "generate", side_effect=mock_generate):
            results = engine.generate_scene(
                text_prompt="test",
                reference_image=tmp_path / "ref.png",
                audio_segment=tmp_path / "seg.wav",
                output_dir=tmp_path,
                scene_id="scene_001",
                duration=duration,
            )

        assert len(results) == expected_clips
        assert call_count == expected_clips

        # Check output filenames have sub-clip suffixes
        assert "scene_001_a" in results[0].video_path.name
        assert "scene_001_b" in results[1].video_path.name
        assert "scene_001_c" in results[2].video_path.name

    def test_scene_at_boundary_no_split(self, hva_config, tmp_path):
        """Scene exactly at max_duration should not split."""
        engine = HunyuanAvatarEngine(hva_config)
        engine._loaded = True

        mock_result = VideoResult(
            video_path=tmp_path / "scene_001.mp4",
            frames_generated=129,
            duration_seconds=5.16,
        )

        with patch.object(engine, "generate", return_value=mock_result):
            results = engine.generate_scene(
                text_prompt="test",
                reference_image=tmp_path / "ref.png",
                audio_segment=tmp_path / "seg.wav",
                output_dir=tmp_path,
                scene_id="scene_001",
                duration=5.16,
            )

        assert len(results) == 1
