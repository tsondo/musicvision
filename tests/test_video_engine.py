"""Tests for the HuMo video engine."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for video engine tests")

from musicvision.models import (
    HumoConfig,
    HumoModelSize,
    HumoResolution,
    Scene,
)
from musicvision.utils.gpu import DeviceMap
from musicvision.video.base import VideoEngine, VideoInput, VideoResult
from musicvision.video.factory import create_video_engine
from musicvision.video.humo_engine import (
    FPS,
    MAX_DURATION,
    MAX_FRAMES,
    HumoEngine,
    _save_frames_as_mp4,
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
def gpu_device_map():
    return DeviceMap(
        dit_device=torch.device("cuda:0"),
        encoder_device=torch.device("cuda:1"),
        vae_device=torch.device("cuda:1"),
        offload_device=torch.device("cpu"),
    )


@pytest.fixture
def default_config():
    return HumoConfig()


@pytest.fixture
def fast_config():
    return HumoConfig(
        model_size=HumoModelSize.SMALL,
        resolution=HumoResolution.SD,
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
# VideoInput / VideoResult
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_video_input(self):
        inp = VideoInput(
            text_prompt="A person singing",
            reference_image=Path("/images/ref.png"),
            audio_segment=Path("/segments/s1.wav"),
            output_path=Path("/clips/s1.mp4"),
        )
        assert inp.text_prompt == "A person singing"
        assert inp.output_path == Path("/clips/s1.mp4")

    def test_video_result(self):
        result = VideoResult(
            video_path=Path("/clips/s1.mp4"),
            frames_generated=97,
            duration_seconds=3.88,
        )
        assert result.frames_generated == 97
        assert result.metadata == {}

    def test_video_result_metadata(self):
        result = VideoResult(
            video_path=Path("/clips/s1.mp4"),
            frames_generated=50,
            duration_seconds=2.0,
            metadata={"scale_t": 7.5},
        )
        assert result.metadata["scale_t"] == 7.5


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFactory:
    def test_creates_humo_engine(self, cpu_device_map, default_config):
        engine = create_video_engine(default_config, cpu_device_map)
        assert isinstance(engine, HumoEngine)
        assert isinstance(engine, VideoEngine)

    def test_creates_with_small_model(self, cpu_device_map, fast_config):
        engine = create_video_engine(fast_config, cpu_device_map)
        assert isinstance(engine, HumoEngine)
        assert engine.config.model_size == HumoModelSize.SMALL


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_not_loaded_initially(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        assert not engine.is_loaded

    def test_generate_before_load_raises(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        inp = VideoInput(
            text_prompt="test",
            reference_image=Path("/ref.png"),
            audio_segment=Path("/seg.wav"),
            output_path=Path("/out.mp4"),
        )
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.generate(inp)

    def test_generate_scene_before_load_raises(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        with pytest.raises(RuntimeError, match="not loaded"):
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
        with patch("musicvision.video.humo_engine.clear_vram"):
            engine.unload()  # should not raise

    def test_unload_clears_components(self, cpu_device_map, default_config):
        engine = HumoEngine(default_config, cpu_device_map)
        engine._dit = MagicMock()
        engine._t5_encoder = MagicMock()
        engine._vae = MagicMock()
        engine._whisper_encoder = MagicMock()

        with patch("musicvision.video.humo_engine.clear_vram") as mock_clear:
            engine.unload()

        assert engine._dit is None
        assert engine._t5_encoder is None
        assert engine._vae is None
        assert engine._whisper_encoder is None
        mock_clear.assert_called_once()


# ---------------------------------------------------------------------------
# HumoConfig
# ---------------------------------------------------------------------------


class TestHumoConfig:
    def test_defaults(self):
        cfg = HumoConfig()
        assert cfg.model_size == HumoModelSize.LARGE
        assert cfg.resolution == HumoResolution.HD
        assert cfg.scale_a == 2.0
        assert cfg.scale_t == 7.5
        assert cfg.denoising_steps == 50

    def test_hd_resolution(self):
        cfg = HumoConfig(resolution=HumoResolution.HD)
        assert cfg.width == 1280
        assert cfg.height == 720

    def test_sd_resolution(self):
        cfg = HumoConfig(resolution=HumoResolution.SD)
        assert cfg.width == 832
        assert cfg.height == 480


# ---------------------------------------------------------------------------
# Mock generation
# ---------------------------------------------------------------------------


class TestMockGeneration:
    @patch("musicvision.video.humo_engine.clear_vram")
    @patch("musicvision.video.humo_engine._save_frames_as_mp4")
    def test_generate_single_clip(self, mock_save, mock_clear, tmp_path, cpu_device_map):
        """Test generate() with fully mocked model components."""
        config = HumoConfig(denoising_steps=2)  # minimal steps
        engine = HumoEngine(config, cpu_device_map)

        # Mock all model components
        dim = 64  # small dimensions for testing
        seq_len = 512

        # T5 tokenizer
        engine._t5_tokenizer = MagicMock()
        engine._t5_tokenizer.return_value = {
            "input_ids": torch.zeros(1, seq_len, dtype=torch.long),
            "attention_mask": torch.ones(1, seq_len, dtype=torch.long),
        }
        # Make tokenizer output moveable to device
        mock_encoded = MagicMock()
        mock_encoded.__getitem__ = lambda self, key: torch.zeros(1, seq_len, dtype=torch.long)
        mock_encoded.to = MagicMock(return_value=mock_encoded)
        engine._t5_tokenizer.return_value = mock_encoded

        # T5 encoder
        engine._t5_encoder = MagicMock()
        mock_t5_output = MagicMock()
        mock_t5_output.last_hidden_state = torch.randn(1, seq_len, dim)
        engine._t5_encoder.return_value = mock_t5_output

        # Image processor
        engine._image_processor = MagicMock()
        engine._image_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        # Whisper processor + encoder
        engine._whisper_processor = MagicMock()
        engine._whisper_processor.return_value = MagicMock(
            input_features=torch.randn(1, 80, 3000),
        )
        mock_whisper_output = MagicMock()
        mock_whisper_output.last_hidden_state = torch.randn(1, 100, dim)
        engine._whisper_encoder = MagicMock(return_value=mock_whisper_output)

        # Scheduler
        engine._scheduler = MagicMock()
        engine._scheduler.timesteps = torch.linspace(1.0, 0.0, 2)
        step_output = MagicMock()
        step_output.prev_sample = torch.randn(1, 4, 1, 90, 160)
        engine._scheduler.step.return_value = step_output

        # DiT
        mock_dit_config = MagicMock()
        mock_dit_config.in_channels = 4
        engine._dit = MagicMock()
        engine._dit.config = mock_dit_config
        mock_dit_output = MagicMock()
        mock_dit_output.sample = torch.randn(1, 4, 1, 90, 160)
        engine._dit.return_value = mock_dit_output

        # VAE
        mock_vae_output = MagicMock()
        mock_vae_output.sample = torch.randn(1, 3, 4, 720, 1280)  # (B, C, T, H, W)
        engine._vae = MagicMock()
        engine._vae.decode.return_value = mock_vae_output

        # Create test reference image
        ref_image = tmp_path / "ref.png"
        from PIL import Image
        Image.new("RGB", (512, 512), color="red").save(ref_image)

        # Create test audio (mock torchaudio.load)
        audio_path = tmp_path / "segment.wav"
        audio_path.write_bytes(b"fake")

        output_path = tmp_path / "output.mp4"

        mock_torchaudio = MagicMock()
        mock_torchaudio.load.return_value = (torch.randn(1, 32000), 16000)
        mock_torchaudio.functional.resample = MagicMock()

        with patch.dict("sys.modules", {"torchaudio": mock_torchaudio}):
            result = engine.generate(VideoInput(
                text_prompt="A person singing on stage",
                reference_image=ref_image,
                audio_segment=audio_path,
                output_path=output_path,
            ))

        assert isinstance(result, VideoResult)
        assert result.video_path == output_path
        assert result.frames_generated > 0
        assert result.duration_seconds > 0
        assert result.metadata["scale_t"] == 7.5
        mock_save.assert_called_once()

    @patch("musicvision.video.humo_engine.clear_vram")
    def test_generate_scene_short(self, mock_clear, tmp_path, cpu_device_map, default_config):
        """Short scene (<3.88s) → single clip, no splitting."""
        engine = HumoEngine(default_config, cpu_device_map)
        engine._dit = MagicMock()  # pretend loaded

        mock_result = VideoResult(
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

    @patch("musicvision.video.humo_engine.clear_vram")
    @patch("musicvision.video.humo_engine.slice_audio")
    def test_generate_scene_long_splits(self, mock_slice, mock_clear, tmp_path, cpu_device_map, default_config):
        """Long scene (8s) → 3 sub-clips."""
        engine = HumoEngine(default_config, cpu_device_map)
        engine._dit = MagicMock()  # pretend loaded

        duration = 8.0
        expected_clips = math.ceil(duration / MAX_DURATION)  # 3

        call_count = 0
        def mock_generate(inp):
            nonlocal call_count
            call_count += 1
            return VideoResult(
                video_path=inp.output_path,
                frames_generated=97,
                duration_seconds=MAX_DURATION,
            )

        # Make slice_audio create the file
        def mock_slice_impl(source, output, start, end, **kwargs):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"fake")
            return output
        mock_slice.side_effect = mock_slice_impl

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

        # Check audio slicing was called for each sub-clip
        assert mock_slice.call_count == expected_clips

        # Check output filenames
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

        _save_frames_as_mp4(frames, output, fps=25)

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
            _save_frames_as_mp4(frames, output)
