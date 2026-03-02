"""Tests for the intake pipeline components that don't require GPU or model weights."""

from musicvision.engine_registry import EngineConstraints, get_constraints
from musicvision.intake.segmentation import (
    _engine_constraint_prompt,
    _format_lyrics_for_llm,
    _validate_and_adjust_scenes,
    segment_scenes_simple,
)
from musicvision.intake.transcription import WordTimestamp, align_lyrics_with_timestamps, TranscriptionResult
from musicvision.models import Scene, SceneType


class TestSimpleSegmentation:
    """Test rule-based segmentation (no LLM needed)."""

    def _make_words(self, lines: list[tuple[str, float, float]]) -> list[WordTimestamp]:
        """Helper: create word timestamps from (text, start, end) tuples."""
        words = []
        for text, start, end in lines:
            for i, w in enumerate(text.split()):
                n = len(text.split())
                word_dur = (end - start) / n
                ws = start + i * word_dur
                we = ws + word_dur
                words.append(WordTimestamp(word=w, start=ws, end=we))
        return words

    def test_basic_segmentation(self):
        words = self._make_words([
            ("Standing in the rain tonight", 2.0, 4.5),
            ("Waiting for the morning light", 5.0, 7.5),
            ("Everything will be alright", 8.0, 10.5),
        ])

        result = segment_scenes_simple(words, song_duration=12.0)

        assert len(result.scenes) >= 2
        assert result.scenes[0].time_start == 0.0  # starts at beginning
        assert all(s.time_end > s.time_start for s in result.scenes)

    def test_instrumental_only(self):
        result = segment_scenes_simple([], song_duration=30.0)
        assert len(result.scenes) == 1
        assert result.scenes[0].type == SceneType.INSTRUMENTAL
        assert result.scenes[0].time_end == 30.0

    def test_trailing_instrumental(self):
        words = self._make_words([
            ("Hello world", 1.0, 3.0),
        ])
        result = segment_scenes_simple(words, song_duration=10.0)

        # Should have at least the vocal scene + trailing instrumental
        types = [s.type for s in result.scenes]
        assert SceneType.VOCAL in types
        assert result.scenes[-1].type == SceneType.INSTRUMENTAL
        assert result.scenes[-1].time_end == 10.0

    def test_max_scene_duration_split(self):
        # One long continuous block of words (no gaps)
        words = []
        for i in range(50):
            words.append(WordTimestamp(word=f"word{i}", start=i * 0.3, end=(i + 1) * 0.3))

        result = segment_scenes_simple(words, song_duration=15.5, max_scene_seconds=5.0)

        # All scenes should be ≤ max + small tolerance
        for s in result.scenes:
            assert s.duration <= 6.0, f"Scene {s.id} too long: {s.duration:.1f}s"


class TestLyricsFormatting:
    def test_format_with_gaps(self):
        words = [
            WordTimestamp("Hello", 1.0, 1.5),
            WordTimestamp("world", 1.5, 2.0),
            # gap
            WordTimestamp("Goodbye", 3.0, 3.5),
            WordTimestamp("moon", 3.5, 4.0),
        ]
        formatted = _format_lyrics_for_llm(words)
        assert "Hello world" in formatted
        assert "Goodbye moon" in formatted
        # Should be two lines due to the gap
        assert formatted.count("\n") >= 1


class TestLyricsAlignment:
    def test_exact_match(self):
        transcription = TranscriptionResult(
            text="hello world goodbye moon",
            words=[
                WordTimestamp("hello", 1.0, 1.5),
                WordTimestamp("world", 1.6, 2.0),
                WordTimestamp("goodbye", 3.0, 3.5),
                WordTimestamp("moon", 3.6, 4.0),
            ],
        )

        aligned = align_lyrics_with_timestamps("hello world goodbye moon", transcription)

        assert len(aligned) == 4
        assert aligned[0].word == "hello"
        assert aligned[0].start == 1.0
        assert aligned[2].word == "goodbye"
        assert aligned[2].start == 3.0

    def test_partial_match(self):
        """User lyrics differ slightly from Whisper output."""
        transcription = TranscriptionResult(
            text="hello world goodby moon",
            words=[
                WordTimestamp("hello", 1.0, 1.5),
                WordTimestamp("world", 1.6, 2.0),
                WordTimestamp("goodby", 3.0, 3.5),  # Whisper typo
                WordTimestamp("moon", 3.6, 4.0),
            ],
        )

        # User provides correct lyrics
        aligned = align_lyrics_with_timestamps("hello world goodbye moon", transcription)

        assert len(aligned) == 4
        # "goodbye" should still align to "goodby" timestamps
        assert aligned[2].start == 3.0

    def test_empty_transcription(self):
        transcription = TranscriptionResult(text="", words=[])
        aligned = align_lyrics_with_timestamps("some lyrics", transcription)
        assert aligned == []


# ---------------------------------------------------------------------------
# Engine constraint injection
# ---------------------------------------------------------------------------


class TestEngineConstraintPrompt:
    def test_contains_engine_name(self):
        c = get_constraints("humo")
        prompt = _engine_constraint_prompt(c)
        assert "HuMo" in prompt

    def test_contains_max_frames(self):
        c = get_constraints("humo")
        prompt = _engine_constraint_prompt(c)
        assert "97 frames" in prompt
        assert "3.88s" in prompt

    def test_contains_min_frames(self):
        c = get_constraints("humo")
        prompt = _engine_constraint_prompt(c)
        assert "25 frames" in prompt


# ---------------------------------------------------------------------------
# _validate_and_adjust_scenes
# ---------------------------------------------------------------------------


class TestValidateAndAdjustScenes:
    def test_populates_frame_fields(self):
        constraints = get_constraints("humo")
        scenes = [
            Scene(id="scene_001", order=1, time_start=0.0, time_end=3.88),
            Scene(id="scene_002", order=2, time_start=3.88, time_end=8.0),
        ]
        result = _validate_and_adjust_scenes(scenes, 8.0, constraints)
        assert result[0].frame_start == 0
        assert result[0].frame_end == 97
        assert result[0].total_frames == 97
        assert result[0].subclip_frame_counts == [97]
        assert result[1].total_frames == 103

    def test_empty_list(self):
        constraints = get_constraints("humo")
        assert _validate_and_adjust_scenes([], 10.0, constraints) == []

    def test_single_scene_covering_song(self):
        constraints = get_constraints("hunyuan_avatar")
        scenes = [Scene(id="scene_001", order=1, time_start=0.0, time_end=10.0)]
        result = _validate_and_adjust_scenes(scenes, 10.0, constraints)
        assert result[0].total_frames == 250
        # 250 frames with max=129, min=33 → 2 clips
        assert len(result[0].subclip_frame_counts) == 2
        assert sum(result[0].subclip_frame_counts) == 250

    def test_with_engine_constraints_simple_segmenter(self):
        """Rule-based segmenter passes engine constraints through."""
        words = [
            WordTimestamp("Hello", 1.0, 1.5),
            WordTimestamp("world", 1.5, 2.0),
            WordTimestamp("Goodbye", 5.0, 5.5),
            WordTimestamp("moon", 5.5, 6.0),
        ]
        constraints = get_constraints("humo")
        result = segment_scenes_simple(words, song_duration=8.0, engine_constraints=constraints)
        # All scenes should have frame fields populated
        for scene in result.scenes:
            assert scene.frame_start is not None
            assert scene.total_frames is not None
            assert scene.subclip_frame_counts is not None
