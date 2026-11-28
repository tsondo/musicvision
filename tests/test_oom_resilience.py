"""Tests for OOM resilience utilities and behavior."""

from __future__ import annotations

import pytest

from musicvision.utils.gpu import (
    _oom_suggestion,
    estimate_vram_gb,
    is_oom_error,
)

# ---------------------------------------------------------------------------
# is_oom_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), True),
        (RuntimeError("exited with code 1: torch.cuda.OutOfMemoryError: CUDA out of memory."), True),
        (RuntimeError("torch.cuda.OutOfMemoryError"), True),
        (RuntimeError("CUDA OUT OF MEMORY"), True),  # case-insensitive
        (RuntimeError("Connection refused"), False),
        (ValueError("invalid shape"), False),
    ],
    ids=["cuda-message", "subprocess-stderr", "class-name", "case-insensitive", "other-runtime", "value-error"],
)
def test_is_oom_error(exc, expected):
    assert is_oom_error(exc) is expected


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

