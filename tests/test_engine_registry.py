"""Tests for the engine registry and frame-accurate math."""

from __future__ import annotations

import pytest

from musicvision.engine_registry import (
    ENGINES,
    EngineConstraints,
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
        assert c.max_frames == 97
        assert c.min_frames == 25
        assert c.fps == 25

    def test_hunyuan_avatar(self):
        c = get_constraints("hunyuan_avatar")
        assert c.max_frames == 129
        assert c.min_frames == 33
        assert c.fps == 25

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown engine"):
            get_constraints("nonexistent")

    def test_max_seconds(self):
        c = get_constraints("humo")
        assert c.max_seconds == pytest.approx(3.88)

    def test_min_seconds(self):
        c = get_constraints("humo")
        assert c.min_seconds == pytest.approx(1.0)

    def test_hva_max_seconds(self):
        c = get_constraints("hunyuan_avatar")
        assert c.max_seconds == pytest.approx(5.16)


# ---------------------------------------------------------------------------
# scene_frames
# ---------------------------------------------------------------------------


class TestSceneFrames:
    def test_exact_seconds(self):
        assert scene_frames(0.0, 4.0, 25) == 100

    def test_fractional(self):
        # 3.88s * 25 = 97 frames
        assert scene_frames(0.0, 3.88, 25) == 97

    def test_offset(self):
        assert scene_frames(10.0, 18.0, 25) == 200

    def test_zero_duration(self):
        assert scene_frames(5.0, 5.0, 25) == 0


# ---------------------------------------------------------------------------
# frames_to_seconds
# ---------------------------------------------------------------------------


class TestFramesToSeconds:
    def test_basic(self):
        assert frames_to_seconds(97, 25) == pytest.approx(3.88)

    def test_zero(self):
        assert frames_to_seconds(0, 25) == 0.0

    def test_one_frame(self):
        assert frames_to_seconds(1, 25) == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# compute_subclip_frames
# ---------------------------------------------------------------------------


class TestComputeSubclipFrames:
    def test_fits_in_one_clip(self):
        assert compute_subclip_frames(50, 97, 25) == [50]

    def test_exactly_max(self):
        assert compute_subclip_frames(97, 97, 25) == [97]

    def test_humo_200_frames(self):
        """Spec example: 200 frames with HuMo (max=97, min=25)."""
        result = compute_subclip_frames(200, 97, 25)
        assert sum(result) == 200
        assert len(result) == 3
        # Equal distribution: 200//3=66, 200%3=2 → [67, 67, 66]
        assert result == [67, 67, 66]

    def test_hva_150_frames(self):
        """Spec example: 150 frames with HVA (max=129, min=33)."""
        result = compute_subclip_frames(150, 129, 33)
        assert sum(result) == 150
        assert len(result) == 2
        assert result == [75, 75]

    def test_sum_invariant(self):
        """sum(result) == total_frames always."""
        for total in [50, 97, 98, 100, 150, 200, 300, 500, 1000]:
            result = compute_subclip_frames(total, 97, 25)
            assert sum(result) == total, f"Failed for total={total}: {result}"

    def test_all_within_bounds(self):
        """Every sub-clip is within [min, max]."""
        for total in [98, 150, 200, 300, 500]:
            result = compute_subclip_frames(total, 97, 25)
            for c in result:
                assert c >= 25, f"Below min for total={total}: {c}"
                assert c <= 97, f"Above max for total={total}: {c}"

    def test_empty_for_zero(self):
        assert compute_subclip_frames(0, 97, 25) == []

    def test_small_total_below_max(self):
        assert compute_subclip_frames(30, 97, 25) == [30]

    def test_remainder_redistribution(self):
        """When naive split gives a too-small remainder, redistribution kicks in."""
        # 100 frames, max=97, min=25
        # Naive: ceil(100/97)=2, remainder=100-97=3 < 25 → n=1 → but 100>97
        # So stays at n=2: 100//2=50, 100%2=0 → [50, 50]
        result = compute_subclip_frames(100, 97, 25)
        assert result == [50, 50]
        assert sum(result) == 100


# ---------------------------------------------------------------------------
# sub_clip_suffixes
# ---------------------------------------------------------------------------


class TestSubClipSuffixes:
    def test_single(self):
        assert sub_clip_suffixes(1) == ["a"]

    def test_three(self):
        assert sub_clip_suffixes(3) == ["a", "b", "c"]

    def test_26(self):
        result = sub_clip_suffixes(26)
        assert result[0] == "a"
        assert result[25] == "z"

    def test_beyond_26(self):
        result = sub_clip_suffixes(28)
        assert result[26] == "aa"
        assert result[27] == "ab"

    def test_empty(self):
        assert sub_clip_suffixes(0) == []


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
