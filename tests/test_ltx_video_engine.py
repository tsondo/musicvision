"""Tests for the LTX-Video 2 engine integration."""

from __future__ import annotations

import pytest

from musicvision.engine_registry import get_constraints, snap_subclip_frames_ltx
from musicvision.models import (
    HumoConfig,
    LtxVideoConfig,
    ProjectConfig,
    VideoEngineType,
)
from musicvision.video.factory import create_video_engine

# ---------------------------------------------------------------------------
# VideoEngineType enum
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# LtxVideoConfig
# ---------------------------------------------------------------------------


class TestLtxVideoConfig:
    def test_audio_conditioning_off_by_default(self):
        # Disabled until mel spectrogram encoding is implemented.
        assert LtxVideoConfig().use_audio_conditioning is False



    def test_max_duration(self):
        cfg = LtxVideoConfig()
        assert cfg.max_duration == pytest.approx(257 / 24, rel=0.01)

    @pytest.mark.parametrize(
        ("frames", "snapped"),
        [(9, 9), (121, 121), (257, 257),    # already (N*8)+1
         (100, 97), (120, 121), (130, 129),  # round to nearest (N*8)+1
         (1, 9), (4, 9), (5, 9)],            # floor at the 9-frame minimum
    )
    def test_snap_frames(self, frames, snapped):
        assert LtxVideoConfig.snap_frames(frames) == snapped


# ---------------------------------------------------------------------------
# ProjectConfig integration
# ---------------------------------------------------------------------------


class TestProjectConfigLtxVideo:

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ProjectConfig(
            video_engine=VideoEngineType.LTX_VIDEO,
            ltx_video=LtxVideoConfig(width=480, height=320, num_inference_steps=20),
        )
        path = tmp_path / "project.yaml"
        cfg.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.video_engine == VideoEngineType.LTX_VIDEO
        assert loaded.ltx_video.width == 480
        assert loaded.ltx_video.height == 320
        assert loaded.ltx_video.num_inference_steps == 20


# ---------------------------------------------------------------------------
# Engine constraints
# ---------------------------------------------------------------------------


class TestEngineConstraints:
    def test_ltx_video_registered(self):
        c = get_constraints("ltx_video")
        assert (c.max_frames, c.min_frames, c.fps) == (257, 9, 24)
        assert c.max_seconds == pytest.approx(257 / 24) and c.min_seconds == pytest.approx(9 / 24)


# ---------------------------------------------------------------------------
# Frame snapping
# ---------------------------------------------------------------------------


class TestSnapSubclipFramesLtx:
    def test_single_clip_unchanged(self):
        assert snap_subclip_frames_ltx([200], 200) == [200]

    def test_two_clips_snapped(self):
        result = snap_subclip_frames_ltx([150, 150], 300)
        assert sum(result) == 300
        assert len(result) == 2
        # First clip should be snapped to (N*8)+1
        assert (result[0] - 1) % 8 == 0

    def test_preserves_total(self):
        result = snap_subclip_frames_ltx([100, 100, 100], 300)
        assert sum(result) == 300

    def test_merges_tiny_last_clip(self):
        # If snapping makes last clip < 9, it should merge
        result = snap_subclip_frames_ltx([250, 5], 255)
        assert sum(result) == 255
        assert all(c >= 9 or c == result[-1] for c in result)

    def test_three_clips(self):
        result = snap_subclip_frames_ltx([120, 120, 120], 360)
        assert sum(result) == 360
        assert len(result) == 3
        # All but last should be (N*8)+1
        for c in result[:-1]:
            assert (c - 1) % 8 == 0


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


class TestFactoryDispatch:
    def test_creates_ltx_engine(self):
        torch = pytest.importorskip("torch")
        from musicvision.utils.gpu import DeviceMap

        cpu = torch.device("cpu")
        dm = DeviceMap(dit_device=cpu, encoder_device=cpu, vae_device=cpu, offload_device=cpu)
        engine = create_video_engine(LtxVideoConfig(), device_map=dm, engine_type=VideoEngineType.LTX_VIDEO)
        from musicvision.video.ltx_video_engine import LtxVideoEngine

        assert isinstance(engine, LtxVideoEngine)

    def test_wrong_config_type_raises(self):
        with pytest.raises(TypeError, match="LtxVideoConfig"):
            create_video_engine(HumoConfig(), engine_type=VideoEngineType.LTX_VIDEO)

    def test_missing_device_map_raises(self):
        with pytest.raises(ValueError, match="device_map"):
            create_video_engine(LtxVideoConfig(), engine_type=VideoEngineType.LTX_VIDEO)


# ---------------------------------------------------------------------------
# Engine lifecycle (mocked — no GPU)
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    def test_not_loaded_initially(self):
        torch = pytest.importorskip("torch")
        from musicvision.utils.gpu import DeviceMap

        cpu = torch.device("cpu")
        dm = DeviceMap(dit_device=cpu, encoder_device=cpu, vae_device=cpu, offload_device=cpu)
        engine = create_video_engine(LtxVideoConfig(), device_map=dm, engine_type=VideoEngineType.LTX_VIDEO)
        assert not engine.is_loaded

    def test_generate_before_load_raises(self):
        torch = pytest.importorskip("torch")
        from musicvision.utils.gpu import DeviceMap
        from musicvision.video.base import VideoInput

        cpu = torch.device("cpu")
        dm = DeviceMap(dit_device=cpu, encoder_device=cpu, vae_device=cpu, offload_device=cpu)
        engine = create_video_engine(LtxVideoConfig(), device_map=dm, engine_type=VideoEngineType.LTX_VIDEO)

        with pytest.raises(RuntimeError, match="load"):
            engine.generate(VideoInput(
                text_prompt="test",
                reference_image=Path("/fake.png"),
                audio_segment=Path("/fake.wav"),
                output_path=Path("/fake.mp4"),
            ))

    def test_unload_when_not_loaded_is_safe(self):
        torch = pytest.importorskip("torch")
        from musicvision.utils.gpu import DeviceMap

        cpu = torch.device("cpu")
        dm = DeviceMap(dit_device=cpu, encoder_device=cpu, vae_device=cpu, offload_device=cpu)
        engine = create_video_engine(LtxVideoConfig(), device_map=dm, engine_type=VideoEngineType.LTX_VIDEO)
        engine.unload()  # should not raise
        assert not engine.is_loaded


# Need Path import for VideoInput
from pathlib import Path
