"""Tests for the engine registry and frame-accurate math."""

from __future__ import annotations

import pytest

from musicvision.engine_registry import (
    compute_subclip_frames,
    compute_subclip_frames_at_silences,
    frames_to_seconds,
    get_constraints,
    scene_frames,
    sub_clip_suffixes,
)

# ---------------------------------------------------------------------------
# get_constraints
# ---------------------------------------------------------------------------


class TestGetConstraints:
    def test_humo(self):
        c = get_constraints("humo")
        assert (c.max_frames, c.min_frames, c.fps) == (97, 25, 25)
        assert c.max_seconds == pytest.approx(3.88) and c.min_seconds == pytest.approx(1.0)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_constraints("nonexistent")


# ---------------------------------------------------------------------------
# scene_frames / frames_to_seconds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("start", "end", "frames"),
    [(0.0, 4.0, 100), (0.0, 3.88, 97), (10.0, 18.0, 200), (5.0, 5.0, 0)],
    ids=["exact", "fractional-3.88s", "offset", "zero"],
)
def test_scene_frames(start, end, frames):
    assert scene_frames(start, end, 25) == frames


@pytest.mark.parametrize(("frames", "seconds"), [(97, 3.88), (0, 0.0), (1, 0.04)])
def test_frames_to_seconds(frames, seconds):
    assert frames_to_seconds(frames, 25) == pytest.approx(seconds)


# ---------------------------------------------------------------------------
# compute_subclip_frames
# ---------------------------------------------------------------------------


class TestComputeSubclipFrames:
    @pytest.mark.parametrize(
        ("total", "max_f", "min_f", "expected"),
        [
            (50, 97, 25, [50]),            # fits in one clip
            (97, 97, 25, [97]),            # exactly max
            (0, 97, 25, []),               # empty
            (200, 97, 25, [67, 67, 66]),   # spec example (HuMo): 200//3=66, remainder spread
            (150, 129, 33, [75, 75]),      # spec example (HVA)
            (100, 97, 25, [50, 50]),       # naive split leaves 3 < min: redistributed
        ],
        ids=["fits", "exactly-max", "empty", "humo-200", "hva-150", "remainder-redistribution"],
    )
    def test_split(self, total, max_f, min_f, expected):
        assert compute_subclip_frames(total, max_f, min_f) == expected

    @pytest.mark.parametrize("total", [50, 97, 98, 100, 150, 200, 300, 500, 1000])
    def test_sum_and_bounds_invariants(self, total):
        result = compute_subclip_frames(total, 97, 25)
        assert sum(result) == total
        if total > 97:
            assert all(25 <= c <= 97 for c in result)


# ---------------------------------------------------------------------------
# sub_clip_suffixes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "checks"),
    [(0, {}), (1, {0: "a"}), (3, {0: "a", 1: "b", 2: "c"}), (26, {0: "a", 25: "z"}), (28, {26: "aa", 27: "ab"})],
)
def test_sub_clip_suffixes(n, checks):
    result = sub_clip_suffixes(n)
    assert len(result) == n
    assert all(result[i] == v for i, v in checks.items())


# ---------------------------------------------------------------------------
# compute_subclip_frames_at_silences
# ---------------------------------------------------------------------------


class TestSilenceSplit:
    """Tests for silence-aware sub-clip splitting."""

    # HVA constraints: max=129, min=33, fps=25
    MAX = 129
    MIN = 33
    FPS = 25

    def test_single_clip_no_split(self):
        """Scene fits in one clip — no split regardless of silences."""
        silences = [(1.0, 1.2), (3.0, 3.3)]
        result = compute_subclip_frames_at_silences(100, self.MAX, self.MIN, self.FPS, silences)
        assert result == [100]

    def test_basic_silence_split(self):
        """300 frames (~12s) with silences — splits at silence midpoints."""
        # Silences at ~5s and ~9s
        silences = [(4.8, 5.2), (8.8, 9.2)]
        result = compute_subclip_frames_at_silences(300, self.MAX, self.MIN, self.FPS, silences)

        assert sum(result) == 300
        assert all(c >= self.MIN for c in result)
        assert all(c <= self.MAX for c in result)
        # Should split near silence midpoints (frame 125 for 5.0s, frame 225 for 9.0s)
        assert len(result) == 3

    def test_no_silences_hard_cuts(self):
        """No silences at all → hard cuts at max_frames."""
        result = compute_subclip_frames_at_silences(300, self.MAX, self.MIN, self.FPS, silences=[])
        assert sum(result) == 300
        assert all(c >= self.MIN for c in result)
        assert all(c <= self.MAX for c in result)
        # Greedy walk: 129 + 129 + 42 (hard cut at max, last chunk gets remainder)
        assert result == [129, 129, 42]

    def test_respects_min_frames(self):
        """Silence too close to start is skipped (would create chunk < min_frames)."""
        # 260 frames, silence at 0.5s (frame 12-13) — too close to start
        # Should skip this silence and use a later one or hard cut
        silences = [(0.4, 0.6), (5.0, 5.2)]
        result = compute_subclip_frames_at_silences(260, self.MAX, self.MIN, self.FPS, silences)

        assert sum(result) == 260
        assert all(c >= self.MIN for c in result)
        assert all(c <= self.MAX for c in result)
        # First chunk should NOT be ~13 frames (that's < min)
        assert result[0] >= self.MIN

    def test_respects_max_frames(self):
        """Long stretch with no silence in range → hard cut at max_frames."""
        # 300 frames, only silence at 11s (frame 275) — way beyond first clip's max window
        silences = [(10.9, 11.1)]
        result = compute_subclip_frames_at_silences(300, self.MAX, self.MIN, self.FPS, silences)

        assert sum(result) == 300
        assert all(c >= self.MIN for c in result)
        assert all(c <= self.MAX for c in result)
        # First clip should be hard-cut at 129 since no silence before it
        assert result[0] == self.MAX

    def test_sum_invariant(self):
        """sum(result) == total_frames for various inputs."""
        test_cases = [
            (200, [(3.5, 3.7)]),
            (300, [(2.0, 2.2), (5.0, 5.3), (8.0, 8.1)]),
            (500, [(4.0, 4.5), (9.0, 9.3), (14.0, 14.2), (18.0, 18.5)]),
        ]
        for total, silences in test_cases:
            result = compute_subclip_frames_at_silences(total, self.MAX, self.MIN, self.FPS, silences)
            assert sum(result) == total, f"Failed for total={total}: {result}"

    def test_empty_total(self):
        assert compute_subclip_frames_at_silences(0, self.MAX, self.MIN, self.FPS, []) == []

    def test_humo_constraints(self):
        """Works with HuMo's tighter constraints (max=97, min=25)."""
        silences = [(1.5, 1.7), (3.5, 3.7), (5.5, 5.7), (7.5, 7.7)]
        result = compute_subclip_frames_at_silences(250, 97, 25, 25, silences)

        assert sum(result) == 250
        assert all(c >= 25 for c in result)
        assert all(c <= 97 for c in result)

    def test_splits_at_silence_midpoint(self):
        """Verify the split happens at the silence midpoint, not start/end."""
        # Silence from 4.0-5.0s → midpoint = 4.5s → frame 112
        silences = [(4.0, 5.0)]
        result = compute_subclip_frames_at_silences(200, self.MAX, self.MIN, self.FPS, silences)

        assert sum(result) == 200
        # First chunk should be ~112 frames (midpoint of 4.0–5.0s silence)
        assert result[0] == round(4.5 * self.FPS)  # 112 or 113
