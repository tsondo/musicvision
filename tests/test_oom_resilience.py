"""Tests for OOM resilience utilities and behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from musicvision.utils.gpu import (
    _oom_suggestion,
    estimate_vram_gb,
    is_oom_error,
)


# ---------------------------------------------------------------------------
# is_oom_error
# ---------------------------------------------------------------------------


class TestIsOomError:
    """Detect CUDA OOM from various exception types."""

    def test_cuda_oom_message(self):
        """Standard CUDA OOM RuntimeError message."""
        exc = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        assert is_oom_error(exc) is True

    def test_torch_oom_class(self):
        """torch.cuda.OutOfMemoryError (subclass of RuntimeError)."""
        # We can't import the real class without CUDA, so test via message match
        exc = RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        assert is_oom_error(exc) is True

    def test_subprocess_stderr_oom(self):
        """OOM embedded in subprocess stderr output."""
        exc = RuntimeError(
            "HVA exited with code 1: torch.cuda.OutOfMemoryError: "
            "CUDA out of memory. Tried to allocate 30.00 GiB"
        )
        assert is_oom_error(exc) is True

    def test_outofmemoryerror_string(self):
        """Match on 'OutOfMemoryError' class name in message."""
        exc = RuntimeError("torch.cuda.OutOfMemoryError")
        assert is_oom_error(exc) is True

    def test_non_oom_error(self):
        """Regular errors should not match."""
        exc = RuntimeError("Connection refused")
        assert is_oom_error(exc) is False

    def test_case_insensitive(self):
        """Match should be case-insensitive."""
        exc = RuntimeError("CUDA OUT OF MEMORY")
        assert is_oom_error(exc) is True

    def test_value_error_not_oom(self):
        exc = ValueError("invalid shape")
        assert is_oom_error(exc) is False


# ---------------------------------------------------------------------------
# estimate_vram_gb
# ---------------------------------------------------------------------------


class TestEstimateVramGb:
    """Empirical VRAM estimation for HVA engine."""

    def test_known_320p(self):
        """320p data point should match known measurement."""
        result = estimate_vram_gb("hunyuan_avatar", image_size=320)
        assert result == pytest.approx(16.6, abs=0.1)

    def test_known_704p(self):
        """704p data point should match known measurement."""
        result = estimate_vram_gb("hunyuan_avatar", image_size=704)
        assert result == pytest.approx(31.9, abs=0.1)

    def test_interpolated_512p(self):
        """512p should interpolate between 320p and 704p data points."""
        result = estimate_vram_gb("hunyuan_avatar", image_size=512)
        assert 16.6 < result < 31.9

    def test_no_cpu_offload_adds_weight_overhead(self):
        """Without cpu_offload, model weights stay on GPU (~29 GB extra)."""
        with_offload = estimate_vram_gb("hunyuan_avatar", image_size=320, cpu_offload=True)
        without_offload = estimate_vram_gb("hunyuan_avatar", image_size=320, cpu_offload=False)
        assert without_offload > with_offload + 20  # at least 20 GB more

    def test_unknown_engine_returns_zero(self):
        """Unknown engines should return 0.0 (caller decides)."""
        result = estimate_vram_gb("some_future_engine", image_size=704)
        assert result == 0.0

    def test_unknown_image_size(self):
        """Image sizes not in the lookup should still produce a reasonable estimate."""
        result = estimate_vram_gb("hunyuan_avatar", image_size=1024)
        assert result > 31.9  # Larger than 704p


# ---------------------------------------------------------------------------
# _oom_suggestion
# ---------------------------------------------------------------------------


class TestOomSuggestion:
    def test_hva_default_suggestion(self):
        suggestion = _oom_suggestion("hunyuan_avatar")
        assert "image_size" in suggestion.lower() or "resolution" in suggestion.lower()

    def test_hva_already_minimum(self):
        config = MagicMock(image_size=256, cpu_offload=True)
        suggestion = _oom_suggestion("hunyuan_avatar", config)
        assert "minimum" in suggestion.lower() or "more vram" in suggestion.lower()

    def test_hva_cpu_offload_disabled(self):
        config = MagicMock(image_size=704, cpu_offload=False)
        suggestion = _oom_suggestion("hunyuan_avatar", config)
        assert "cpu_offload" in suggestion.lower() or "offload" in suggestion.lower()

    def test_unknown_engine_suggestion(self):
        suggestion = _oom_suggestion("unknown_engine")
        assert "resolution" in suggestion.lower() or "offload" in suggestion.lower()


# ---------------------------------------------------------------------------
# HVA engine: OOM skips wrapper fallback
# ---------------------------------------------------------------------------


class TestOomSkipWrapperFallback:
    """When server OOMs, wrapper should NOT be called (same params will OOM too)."""

    @patch("musicvision.video.hunyuan_avatar_engine._SERVER_SCRIPT")
    def test_server_oom_skips_wrapper(self, mock_script_path):
        """Server OOM → exception propagates, wrapper NOT called."""
        from musicvision.video.hunyuan_avatar_engine import HunyuanAvatarEngine
        from musicvision.video.base import VideoInput

        engine = HunyuanAvatarEngine.__new__(HunyuanAvatarEngine)
        engine._loaded = True
        engine._server_mode = True
        engine._server_proc = MagicMock()
        engine._server_proc.poll.return_value = None  # alive
        engine._stderr_thread = None
        engine.config = MagicMock()

        oom_exc = RuntimeError("CUDA out of memory. Tried to allocate 30.00 GiB")
        engine._generate_via_server = MagicMock(side_effect=oom_exc)
        engine._generate_via_wrapper = MagicMock()

        dummy_input = VideoInput(
            text_prompt="test",
            reference_image=Path("/fake/image.png"),
            audio_segment=Path("/fake/audio.wav"),
            output_path=Path("/fake/output.mp4"),
        )

        with pytest.raises(RuntimeError, match="out of memory"):
            engine.generate(dummy_input)

        engine._generate_via_wrapper.assert_not_called()

    @patch("musicvision.video.hunyuan_avatar_engine._SERVER_SCRIPT")
    def test_non_oom_falls_back_to_wrapper(self, mock_script_path):
        """Non-OOM server error → wrapper IS called as fallback."""
        from musicvision.video.hunyuan_avatar_engine import HunyuanAvatarEngine
        from musicvision.video.base import VideoInput, VideoResult

        engine = HunyuanAvatarEngine.__new__(HunyuanAvatarEngine)
        engine._loaded = True
        engine._server_mode = True
        engine._server_proc = MagicMock()
        engine._server_proc.poll.return_value = None  # alive
        engine._stderr_thread = None
        engine.config = MagicMock()

        engine._generate_via_server = MagicMock(side_effect=RuntimeError("Connection reset"))
        wrapper_result = VideoResult(
            video_path=Path("/fake/output.mp4"),
            frames_generated=129,
            duration_seconds=5.16,
        )
        engine._generate_via_wrapper = MagicMock(return_value=wrapper_result)

        dummy_input = VideoInput(
            text_prompt="test",
            reference_image=Path("/fake/image.png"),
            audio_segment=Path("/fake/audio.wav"),
            output_path=Path("/fake/output.mp4"),
        )

        result = engine.generate(dummy_input)
        engine._generate_via_wrapper.assert_called_once()
        assert result == wrapper_result


# ---------------------------------------------------------------------------
# Early abort after consecutive OOMs (API-level logic)
# ---------------------------------------------------------------------------


class TestEarlyAbort:
    """After 2 consecutive OOMs, remaining scenes should be skipped."""

    def test_api_early_abort(self):
        """Simulate 2 OOMs then verify 3rd scene is skipped with oom_skipped."""
        from musicvision.utils.gpu import is_oom_error

        # Simulate the logic from api/app.py's generate_videos endpoint
        scenes = ["scene_01", "scene_02", "scene_03", "scene_04"]
        consecutive_ooms = 0
        generated = []
        failed = []

        def mock_generate(scene_id):
            if scene_id in ("scene_01", "scene_02"):
                raise RuntimeError("CUDA out of memory. Tried to allocate 30.00 GiB")
            return "ok"

        for scene_id in scenes:
            if consecutive_ooms >= 2:
                failed.append({
                    "scene_id": scene_id,
                    "error": f"Skipped after {consecutive_ooms} consecutive OOMs",
                    "error_type": "oom_skipped",
                })
                continue

            try:
                mock_generate(scene_id)
                generated.append(scene_id)
                consecutive_ooms = 0
            except Exception as exc:
                if is_oom_error(exc):
                    consecutive_ooms += 1
                    failed.append({
                        "scene_id": scene_id,
                        "error": str(exc),
                        "error_type": "oom",
                    })
                else:
                    consecutive_ooms = 0
                    failed.append({"scene_id": scene_id, "error": str(exc)})

        assert generated == []
        assert len(failed) == 4
        assert failed[0]["error_type"] == "oom"
        assert failed[1]["error_type"] == "oom"
        assert failed[2]["error_type"] == "oom_skipped"
        assert failed[3]["error_type"] == "oom_skipped"

    def test_success_resets_counter(self):
        """A successful generation between OOMs resets the counter."""
        from musicvision.utils.gpu import is_oom_error

        scenes = ["s1", "s2", "s3", "s4"]
        consecutive_ooms = 0
        generated = []
        failed = []

        def mock_generate(scene_id):
            if scene_id == "s1":
                raise RuntimeError("CUDA out of memory")
            if scene_id == "s3":
                raise RuntimeError("CUDA out of memory")
            return "ok"

        for scene_id in scenes:
            if consecutive_ooms >= 2:
                failed.append({"scene_id": scene_id, "error_type": "oom_skipped"})
                continue
            try:
                mock_generate(scene_id)
                generated.append(scene_id)
                consecutive_ooms = 0
            except Exception as exc:
                if is_oom_error(exc):
                    consecutive_ooms += 1
                    failed.append({"scene_id": scene_id, "error_type": "oom"})
                else:
                    consecutive_ooms = 0
                    failed.append({"scene_id": scene_id, "error_type": "error"})

        # s1=oom, s2=ok (reset), s3=oom, s4=ok (no abort since only 1 consecutive)
        assert generated == ["s2", "s4"]
        assert len(failed) == 2
        assert all(f["error_type"] == "oom" for f in failed)
