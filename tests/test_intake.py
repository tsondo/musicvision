"""Tests for the intake pipeline components that don't require GPU or model weights."""

from musicvision.engine_registry import EngineConstraints, get_constraints
from musicvision.intake.segmentation import (
    _engine_constraint_prompt,
    _format_lyrics_for_llm,
    _merge_short_scenes,
    _validate_and_adjust_scenes,
    segment_scenes_simple,
)
from musicvision.intake.pipeline import _lyrics_for_scene_bpm
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


class TestMergeShortScenes:
    """Test post-processing merge of too-short scenes."""

    def test_merges_short_into_neighbor(self):
        scenes = [
            Scene(id="s1", order=1, time_start=0.0, time_end=3.0, type=SceneType.VOCAL, lyrics="a"),
            Scene(id="s2", order=2, time_start=3.0, time_end=3.5, type=SceneType.VOCAL, lyrics="b"),
            Scene(id="s3", order=3, time_start=3.5, time_end=7.0, type=SceneType.VOCAL, lyrics="c"),
        ]
        result = _merge_short_scenes(scenes, min_duration=2.0)
        assert len(result) == 2
        # The 0.5s scene should have been absorbed
        assert all((s.time_end - s.time_start) >= 2.0 for s in result)

    def test_many_tiny_scenes_collapse(self):
        # Simulate LLM producing per-word scenes
        scenes = [
            Scene(id=f"s{i}", order=i, time_start=i * 0.5, time_end=(i + 1) * 0.5,
                  type=SceneType.VOCAL, lyrics=f"word{i}")
            for i in range(20)
        ]
        result = _merge_short_scenes(scenes, min_duration=2.0)
        assert len(result) < 10  # collapsed significantly
        assert all((s.time_end - s.time_start) >= 2.0 for s in result)

    def test_no_merge_needed(self):
        scenes = [
            Scene(id="s1", order=1, time_start=0.0, time_end=4.0, type=SceneType.VOCAL, lyrics="a"),
            Scene(id="s2", order=2, time_start=4.0, time_end=8.0, type=SceneType.VOCAL, lyrics="b"),
        ]
        result = _merge_short_scenes(scenes, min_duration=2.0)
        assert len(result) == 2

    def test_single_scene_not_merged(self):
        scenes = [
            Scene(id="s1", order=1, time_start=0.0, time_end=0.5, type=SceneType.VOCAL, lyrics="a"),
        ]
        result = _merge_short_scenes(scenes, min_duration=2.0)
        assert len(result) == 1

    def test_preserves_vocal_type(self):
        scenes = [
            Scene(id="s1", order=1, time_start=0.0, time_end=3.0, type=SceneType.INSTRUMENTAL, lyrics=""),
            Scene(id="s2", order=2, time_start=3.0, time_end=3.5, type=SceneType.VOCAL, lyrics="word"),
        ]
        result = _merge_short_scenes(scenes, min_duration=2.0)
        assert len(result) == 1
        assert result[0].type == SceneType.VOCAL


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
        constraints = get_constraints("humo")
        scenes = [Scene(id="scene_001", order=1, time_start=0.0, time_end=10.0)]
        result = _validate_and_adjust_scenes(scenes, 10.0, constraints)
        assert result[0].total_frames == 250
        # 250 frames with max=97, min=25 → 3 clips
        assert len(result[0].subclip_frame_counts) == 3
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


class TestAceStepSections:
    """Test AceStep section marker parsing."""

    def test_parse_sections_basic(self):
        from musicvision.intake.pipeline import parse_acestep_sections

        lyrics = "(Intro)\nLa la la\n(Verse 1)\nHello world\nGoodbye moon\n(Chorus)\nSing along"
        words = [
            WordTimestamp("La", 0.0, 0.5),
            WordTimestamp("la", 0.5, 1.0),
            WordTimestamp("la", 1.0, 1.5),
            WordTimestamp("Hello", 5.0, 5.5),
            WordTimestamp("world", 5.5, 6.0),
            WordTimestamp("Goodbye", 8.0, 8.5),
            WordTimestamp("moon", 8.5, 9.0),
            WordTimestamp("Sing", 15.0, 15.5),
            WordTimestamp("along", 15.5, 16.0),
        ]
        sections = parse_acestep_sections(lyrics, words, 20.0)

        assert len(sections) == 3
        assert sections[0].name == "Intro"
        assert sections[1].name == "Verse 1"
        assert sections[2].name == "Chorus"
        # Sections should have increasing times
        assert sections[0].time < sections[1].time < sections[2].time

    def test_parse_sections_no_markers(self):
        from musicvision.intake.pipeline import parse_acestep_sections

        lyrics = "Just some lyrics\nNo section markers"
        sections = parse_acestep_sections(lyrics, [], 10.0)
        assert sections == []

    def test_parse_sections_no_timestamps(self):
        from musicvision.intake.pipeline import parse_acestep_sections

        lyrics = "(Verse 1)\nHello\n(Chorus)\nWorld"
        sections = parse_acestep_sections(lyrics, [], 20.0)
        assert len(sections) == 2
        assert sections[0].name == "Verse 1"
        assert sections[0].time == 0.0
        assert sections[1].time == 10.0  # evenly distributed


class TestSongInfoFields:
    """Test new SongInfo fields survive roundtrip."""

    def test_beat_times_roundtrip(self, tmp_path):
        from musicvision.models import ProjectConfig, SongSection

        config = ProjectConfig(
            song={"audio_file": "test.wav", "beat_times": [0.5, 1.0, 1.5],
                  "sections": [{"name": "Verse 1", "time": 0.5}],
                  "analyzed": True},
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)

        assert loaded.song.beat_times == [0.5, 1.0, 1.5]
        assert loaded.song.analyzed is True
        assert len(loaded.song.sections) == 1
        assert loaded.song.sections[0].name == "Verse 1"

    def test_defaults_backward_compat(self):
        from musicvision.models import SongInfo

        song = SongInfo()
        assert song.beat_times == []
        assert song.sections == []
        assert song.sections_source == ""
        assert song.analyzed is False

    def test_sections_source_roundtrip(self, tmp_path):
        from musicvision.models import ProjectConfig

        config = ProjectConfig(
            song={"audio_file": "test.wav",
                  "sections": [{"name": "Verse", "time": 0.0}],
                  "sections_source": "auto"},
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert loaded.song.sections_source == "auto"
        assert len(loaded.song.sections) == 1


class TestBpmLyricsFallback:
    """Test BPM-based lyrics estimation for scenes."""

    LYRICS = """(Verse 1)
Standing in the rain tonight
Watching shadows dance in light
(Chorus)
We are the dreamers of the dawn
Singing until the night is gone
(Verse 2)
Walking down the empty street
Heartbeat echoing in the heat"""

    def test_early_scene_gets_verse(self):
        # Scene in the first third should get verse 1 lyrics
        text = _lyrics_for_scene_bpm(self.LYRICS, 5.0, 12.0, 180.0, bpm=120.0)
        assert text  # should have some lyrics
        # Should not be empty for a vocal section early in the song

    def test_late_scene_gets_later_lyrics(self):
        # Scene near the end should not get verse 1 lyrics
        early = _lyrics_for_scene_bpm(self.LYRICS, 5.0, 12.0, 180.0, bpm=120.0)
        late = _lyrics_for_scene_bpm(self.LYRICS, 60.0, 70.0, 180.0, bpm=120.0)
        assert early != late or (not early and not late)

    def test_skips_section_markers(self):
        text = _lyrics_for_scene_bpm(self.LYRICS, 0.0, 180.0, 180.0, bpm=120.0)
        assert "(Verse 1)" not in text
        assert "(Chorus)" not in text

    def test_empty_lyrics(self):
        text = _lyrics_for_scene_bpm("", 0.0, 10.0, 180.0, bpm=120.0)
        assert text == ""

    def test_no_bpm_uses_default(self):
        text = _lyrics_for_scene_bpm(self.LYRICS, 5.0, 15.0, 180.0, bpm=None)
        assert text  # should still work with default BPM


class TestDetectSections:
    """Test auto section detection from audio features."""

    def test_detect_sections_returns_list(self, tmp_path):
        """detect_sections returns labeled sections for a simple audio file."""
        import numpy as np
        import soundfile as sf

        from musicvision.intake.audio_analysis import detect_sections

        # Create a 30-second synthetic audio with energy variation
        sr = 22050
        duration = 30.0
        t = np.linspace(0, duration, int(sr * duration))
        # Low energy intro, high energy middle, low energy ending
        envelope = np.concatenate([
            np.linspace(0.1, 0.3, int(sr * 10)),
            np.linspace(0.8, 0.9, int(sr * 10)),
            np.linspace(0.3, 0.1, int(sr * 10)),
        ])
        audio = np.sin(2 * np.pi * 440 * t) * envelope

        audio_path = tmp_path / "test.wav"
        sf.write(str(audio_path), audio, sr)

        beats = [float(i) for i in range(1, 30)]
        sections = detect_sections(audio_path, beats, duration, min_section_seconds=5.0)

        assert len(sections) >= 1
        assert sections[0][1] == 0.0  # first section starts at 0
        for name, time in sections:
            assert isinstance(name, str)
            assert isinstance(time, float)
            assert 0.0 <= time < duration
