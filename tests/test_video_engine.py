"""Tests for the HuMo video engine."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for video engine tests")

from musicvision.models import (
    HumoConfig,
    HumoTier,
    Scene,
)
from musicvision.utils.gpu import DeviceMap
from musicvision.video.factory import create_video_engine
from musicvision.video.humo_engine import (
    FPS,
    MAX_DURATION,
    MAX_FRAMES,
    HumoEngine,
    HumoInput,
    HumoOutput,
    _save_frames_as_mp4_ffmpeg,
    _sub_clip_suffixes,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_device_map():
    cpu = torch.device("cpu")
    return DeviceMap(
        dit_device=cpu,
        encoder_device=cpu,
        vae_device=cpu,
        offload_device=cpu,
    )


@pytest.fixture
def default_config():
    return HumoConfig()


@pytest.fixture
def preview_config():
    return HumoConfig(
        tier=HumoTier.PREVIEW,
        resolution="480p",
        denoising_steps=30,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_max_frames(self):
        assert MAX_FRAMES == 97

    def test_fps(self):
        assert FPS == 25

    def test_max_duration(self):
        assert MAX_DURATION == pytest.approx(3.88)

    def test_scene_needs_sub_clips_matches(self):
        """Scene.needs_sub_clips threshold matches HuMo's MAX_DURATION."""
        short = Scene(id="s1", order=1, time_start=0, time_end=3.5)
        long = Scene(id="s2", order=2, time_start=0, time_end=5.0)
        assert not short.needs_sub_clips  # 3.5 < 3.88
        assert long.needs_sub_clips       # 5.0 > 3.88


# ---------------------------------------------------------------------------
# Sub-clip suffixes
# ---------------------------------------------------------------------------


class TestSubClipSuffixes:
    def test_single(self):
        assert _sub_clip_suffixes(1) == ["a"]

    def test_three(self):
        assert _sub_clip_suffixes(3) == ["a", "b", "c"]

    def test_26(self):
        result = _sub_clip_suffixes(26)
        assert result[0] == "a"
        assert result[25] == "z"

    def test_beyond_26(self):
        result = _sub_clip_suffixes(28)
        assert result[26] == "aa"
        assert result[27] == "ab"


# ---------------------------------------------------------------------------
# HumoInput / HumoOutput
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_humo_input(self):
        inp = HumoInput(
            text_prompt="A person singing",
            reference_image=Path("/images/ref.png"),
            audio_segment=Path("/segments/s1.wav"),
            output_path=Path("/clips/s1.mp4"),
        )
        assert inp.text_prompt == "A person singing"
        assert inp.output_path == Path("/clips/s1.mp4")
        assert inp.seed is None

    def test_humo_output(self):
        result = HumoOutput(
            video_path=Path("/clips/s1.mp4"),
            frames_generated=97,
            duration_seconds=3.88,
        )
        assert result.frames_generated == 97
        assert result.seed_used == 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_humo_engine(self, cpu_device_map, default_config):
        engine = create_video_engine(default_config, cpu_device_map)
        assert isinstance(engine, HumoEngine)

    def test_creates_with_preview_tier(self, cpu_device_map, preview_config):
        engine = create_video_engine(preview_config, cpu_device_map)
        assert isinstance(engine, HumoEngine)
        assert engine.config.tier == HumoTier.PREVIEW
        assert engine.config.model_size == "1.7B"


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_not_loaded_initially(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        assert engine._bundle is None

    def test_generate_before_load_raises(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        inp = HumoInput(
            text_prompt="test",
            reference_image=Path("/ref.png"),
            audio_segment=Path("/seg.wav"),
            output_path=Path("/out.mp4"),
        )
        with pytest.raises(RuntimeError, match="load"):
            engine.generate(inp)

    def test_generate_scene_before_load_raises(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        with pytest.raises(RuntimeError, match="load"):
            engine.generate_scene(
                text_prompt="test",
                reference_image=Path("/ref.png"),
                audio_segment=Path("/seg.wav"),
                output_dir=Path("/clips"),
                scene_id="scene_001",
                duration=3.0,
            )

    def test_unload_when_not_loaded_is_safe(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        engine.unload()  # should not raise


# ---------------------------------------------------------------------------
# HumoConfig
# ---------------------------------------------------------------------------


class TestHumoConfig:
    def test_defaults(self):
        cfg = HumoConfig()
        assert cfg.tier == HumoTier.FP8_SCALED
        assert cfg.model_size == "17B"
        assert cfg.resolution == "544p"
        assert cfg.height == 544
        assert cfg.width == 960
        assert cfg.scale_a == 5.5
        assert cfg.scale_t == 5.0
        assert cfg.denoising_steps == 50

    def test_720p_resolution(self):
        cfg = HumoConfig(resolution="720p")
        assert cfg.width == 1280
        assert cfg.height == 720

    def test_544p_resolution(self):
        cfg = HumoConfig(resolution="544p")
        assert cfg.width == 960
        assert cfg.height == 544

    def test_480p_resolution(self):
        cfg = HumoConfig(resolution="480p")
        assert cfg.width == 832
        assert cfg.height == 480

    def test_preview_tier(self):
        cfg = HumoConfig(tier=HumoTier.PREVIEW)
        assert cfg.model_size == "1.7B"


# ---------------------------------------------------------------------------
# Generate scene splitting
# ---------------------------------------------------------------------------


class TestGenerateScene:
    def test_short_scene_single_clip(self, tmp_path, cpu_device_map, default_config):
        """Short scene (<3.88s) → single clip, no splitting."""
        engine = HumoEngine(default_config, cpu_device_map)
        engine._bundle = MagicMock()  # pretend loaded

        mock_result = HumoOutput(
            video_path=tmp_path / "scene_001.mp4",
            frames_generated=50,
            duration_seconds=2.0,
        )

        with patch.object(engine, "generate", return_value=mock_result) as mock_gen:
            results = engine.generate_scene(
                text_prompt="test",
                reference_image=tmp_path / "ref.png",
                audio_segment=tmp_path / "seg.wav",
                output_dir=tmp_path,
                scene_id="scene_001",
                duration=2.0,
            )

        assert len(results) == 1
        assert results[0] is mock_result
        mock_gen.assert_called_once()

    def test_long_scene_splits(self, tmp_path, cpu_device_map, default_config):
        """Long scene (8s) → 3 sub-clips."""
        engine = HumoEngine(default_config, cpu_device_map)
        engine._bundle = MagicMock()  # pretend loaded

        duration = 8.0
        expected_clips = math.ceil(duration / MAX_DURATION)  # 3

        # The engine expects pre-sliced sub-clip audio files
        seg_dir = tmp_path
        for i in range(expected_clips):
            sub_audio = seg_dir / f"scene_001_sub_{i:02d}.wav"
            sub_audio.write_bytes(b"fake")

        call_count = 0
        def mock_generate(inp):
            nonlocal call_count
            call_count += 1
            return HumoOutput(
                video_path=inp.output_path,
                frames_generated=97,
                duration_seconds=MAX_DURATION,
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


# ---------------------------------------------------------------------------
# Save frames helper
# ---------------------------------------------------------------------------


class TestSaveFrames:
    @patch("subprocess.Popen")
    def test_save_frames_calls_ffmpeg(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_popen.return_value = mock_proc

        frames = torch.randint(0, 255, (10, 480, 832, 3), dtype=torch.uint8)
        output = tmp_path / "test.mp4"

        _save_frames_as_mp4_ffmpeg(frames, output, fps=25)

        mock_popen.assert_called_once()
        cmd = mock_popen.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert "-r" in cmd
        assert "25" in cmd
        assert str(output) in cmd

    @patch("subprocess.Popen")
    def test_save_frames_ffmpeg_failure_raises(self, mock_popen, tmp_path):
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.stdin = MagicMock()
        mock_proc.stderr.read.return_value = b"error message"
        mock_popen.return_value = mock_proc

        frames = torch.randint(0, 255, (10, 480, 832, 3), dtype=torch.uint8)
        output = tmp_path / "test.mp4"

        with pytest.raises(RuntimeError, match="ffmpeg failed"):
            _save_frames_as_mp4_ffmpeg(frames, output)
