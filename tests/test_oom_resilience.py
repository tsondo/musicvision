"""Tests for OOM resilience utilities and behavior."""

from __future__ import annotations

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

