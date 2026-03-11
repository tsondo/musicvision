"""Tests for OOM resilience utilities and behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

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
    """Empirical VRAM estimation."""

    def test_ltx_video_estimate(self):
        """LTX-Video 2 should return a non-zero estimate."""
        result = estimate_vram_gb("ltx_video", image_size=768)
        assert result > 0.0

    def test_unknown_engine_returns_zero(self):
        """Unknown engines should return 0.0 (caller decides)."""
        result = estimate_vram_gb("some_future_engine", image_size=704)
        assert result == 0.0


# ---------------------------------------------------------------------------
# _oom_suggestion
# ---------------------------------------------------------------------------


class TestOomSuggestion:
    def test_ltx_video_suggestion(self):
        suggestion = _oom_suggestion("ltx_video")
        assert "resolution" in suggestion.lower() or "offload" in suggestion.lower()

    def test_unknown_engine_suggestion(self):
        suggestion = _oom_suggestion("unknown_engine")
        assert "resolution" in suggestion.lower() or "offload" in suggestion.lower()


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
