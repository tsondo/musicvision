"""Tests for the HuMo video engine."""

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch", reason="torch required for video engine tests")

from musicvision.models import (
    HumoConfig,
    HumoTier,
)
from musicvision.utils.gpu import DeviceMap
from musicvision.video.factory import create_video_engine
from musicvision.video.humo_engine import (
    MAX_DURATION,
    HumoEngine,
    HumoInput,
    HumoOutput,
    _save_frames_as_mp4_ffmpeg,
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




# ---------------------------------------------------------------------------
# Sub-clip suffixes
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# HumoInput / HumoOutput
# ---------------------------------------------------------------------------




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

    @pytest.mark.parametrize(("res", "wh"), [("720p", (1280, 720)), ("544p", (960, 544)), ("480p", (832, 480))])
    def test_resolution(self, res, wh):
        cfg = HumoConfig(resolution=res)
        assert (cfg.width, cfg.height) == wh



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


class TestFrameAlignment:
    """#30: the VAE decodes 4k+1 frames; clips must still come out exactly as planned."""

    @pytest.mark.parametrize(("n", "up"), [(1, 1), (49, 49), (70, 73), (71, 73), (88, 89), (96, 97), (97, 97)])
    def test_round_up_to_4k1(self, n, up):
        from musicvision.video.humo_engine import _round_up_to_4k1

        assert _round_up_to_4k1(n) == up

    @pytest.mark.parametrize(
        ("audio_seconds", "planned"),
        [(71 / 25, 71), (3.52, 88), (55 * 1920 / 48000, 55), (97 / 25, 97)],  # 55 frames: float gives 55.000...01
    )
    def test_generate_samples_4k1_and_trims_to_plan(self, monkeypatch, tmp_path, audio_seconds, planned):
        import numpy as np

        import musicvision.video.humo_engine as he

        engine = object.__new__(he.HumoEngine)
        engine._bundle = object()
        engine.config = HumoConfig(seed=1)
        seen: dict = {}

        monkeypatch.setattr(he, "_audio_duration", lambda p: audio_seconds)
        monkeypatch.setattr(he, "_mux_clip_audio", lambda *a: None)
        monkeypatch.setattr(he, "_save_mp4", lambda frames, path, fps: seen.setdefault("saved", len(frames)))
        for name in ("_offload", "_reload"):
            monkeypatch.setattr(engine, name, lambda *a: None, raising=False)
        monkeypatch.setattr(engine, "_encode_text", lambda p: None, raising=False)
        monkeypatch.setattr(engine, "_encode_image", lambda p, n: seen.setdefault("image_n", n), raising=False)
        monkeypatch.setattr(engine, "_encode_audio", lambda p, n: seen.setdefault("audio_n", n), raising=False)
        monkeypatch.setattr(engine, "_denoise", lambda n_frames, **kw: n_frames, raising=False)
        monkeypatch.setattr(engine, "_decode_latent", lambda n: np.zeros((n, 2, 2, 3), np.uint8), raising=False)

        out = engine.generate(he.HumoInput("p", tmp_path / "r.png", tmp_path / "a.wav", tmp_path / "o.mp4"))
        sampled = he._round_up_to_4k1(planned)
        assert seen["image_n"] == seen["audio_n"] == sampled  # conditioned on the 4k+1 length
        assert seen["saved"] == out.frames_generated == planned  # but saved at exactly the plan
