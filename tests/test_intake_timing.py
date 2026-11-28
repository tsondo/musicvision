"""Intake timing and lyrics fixes (#30): contiguous scenes, vocal-onset word trimming,
lyrics from full-song word timestamps."""

from __future__ import annotations

import pytest

from musicvision.engine_registry import get_constraints
from musicvision.intake.pipeline import _assign_lyrics_from_words, _trim_words_to_vocal_activity
from musicvision.intake.segmentation import LARGE_GAP_SECONDS, _make_contiguous, _validate_and_adjust_scenes
from musicvision.intake.transcription import WordTimestamp
from musicvision.models import Scene, SceneType

FPS = 25


def snap(t: float) -> float:
    return round(t * FPS) / FPS


def _scenes(*spans, kind=SceneType.VOCAL):
    return [
        Scene(id=f"x{i}", order=i, time_start=a, time_end=b, type=kind, lyrics=f"l{i}")
        for i, (a, b) in enumerate(spans)
    ]


def _assert_tiles(scenes, song):
    assert scenes[0].time_start == 0.0
    assert scenes[-1].time_end == pytest.approx(snap(song))
    for a, b in zip(scenes, scenes[1:]):
        assert a.time_end == b.time_start
    assert [s.id for s in scenes] == [f"scene_{i:03d}" for i in range(1, len(scenes) + 1)]
    assert [s.order for s in scenes] == list(range(1, len(scenes) + 1))


class TestMakeContiguous:
    def test_small_gaps_absorbed_by_previous_scene(self):
        out = _make_contiguous(_scenes((0, 4), (4.8, 8), (9.9, 12)), 12.0, FPS)
        _assert_tiles(out, 12.0)
        assert [(s.time_start, s.time_end) for s in out] == [(0, 4.8), (4.8, snap(9.9)), (snap(9.9), 12.0)]

    def test_long_intro_gap_and_outro_become_instrumental(self):
        # Karina: singing starts ~13.5 s; last word ends well before the song ends.
        out = _make_contiguous(_scenes((13.54, 20), (20, 110)), 122.4, FPS)
        _assert_tiles(out, 122.4)
        inst, voc = SceneType.INSTRUMENTAL, SceneType.VOCAL
        assert [s.type for s in out] == [inst, voc, voc, inst]
        assert out[0].lyrics == "" and out[0].time_end == snap(13.54)  # on the 25 fps grid

    def test_long_middle_gap_becomes_instrumental(self):
        out = _make_contiguous(_scenes((0, 5), (5 + LARGE_GAP_SECONDS + 1, 15)), 15.0, FPS)
        _assert_tiles(out, 15.0)
        assert [s.type for s in out] == [SceneType.VOCAL, SceneType.INSTRUMENTAL, SceneType.VOCAL]

    def test_short_leading_gap_extends_first_scene(self):
        out = _make_contiguous(_scenes((1.0, 6)), 6.0, FPS)
        assert len(out) == 1 and out[0].time_start == 0.0

    def test_overlap_and_swallowed_scene(self):
        out = _make_contiguous(_scenes((0, 6), (5, 10), (6.5, 9.5)), 10.0, FPS)
        _assert_tiles(out, 10.0)
        assert [(s.time_start, s.time_end) for s in out] == [(0, 6), (6, 10.0)]  # 6.5-9.5 lies inside 6-10

    def test_unordered_input_sorted(self):
        out = _make_contiguous(_scenes((5, 10), (0, 5)), 10.0, FPS)
        assert [s.lyrics for s in out] == ["l1", "l0"]


def test_validated_frames_are_contiguous():
    """Frame math on snapped boundaries: no frame lost or doubled between scenes."""
    c = get_constraints("humo")
    out = _validate_and_adjust_scenes(_scenes((0, 4.33), (4.71, 9.02), (9.9, 15.17)), 15.17, c)
    for a, b in zip(out, out[1:]):
        assert a.frame_end == b.frame_start
    assert out[-1].frame_end == round(15.17 * c.fps)
    assert sum(s.total_frames for s in out) == round(15.17 * c.fps)


class TestTrimWords:
    def test_phrase_initial_word_moves_to_vocal_onset(self):
        words = [WordTimestamp("Karina,", 0.0, 14.38), WordTimestamp("your", 14.38, 15.22)]
        active = [(13.54, 14.74), (15.22, 18.27)]
        out = _trim_words_to_vocal_activity(words, active)
        assert (out[0].start, out[0].end) == (13.54, 14.38)
        assert out[1].start == 14.38  # already inside an active region: unchanged

    def test_no_onset_inside_word_leaves_it(self):
        words = [WordTimestamp("hm", 2.0, 3.0)]
        assert _trim_words_to_vocal_activity(words, [(5.0, 6.0)])[0].start == 2.0

    def test_no_activity_data_is_noop(self):
        words = [WordTimestamp("a", 0.0, 1.0)]
        assert _trim_words_to_vocal_activity(words, []) == words


class TestAssignLyrics:
    def test_boundary_word_lands_in_exactly_one_scene(self):
        scenes = _scenes((0, 2), (2, 4))
        words = [WordTimestamp("one", 0.5, 1.0), WordTimestamp("edge", 1.5, 2.5), WordTimestamp("three", 3.0, 4.0)]
        _assign_lyrics_from_words(scenes, words, set_type=True)
        assert [s.lyrics for s in scenes] == ["one", "edge three"]

    def test_type_follows_words_and_instrumental_gets_no_text(self):
        scenes = _scenes((0, 13.56), (13.56, 20), kind=SceneType.VOCAL)
        scenes[0].lyrics = "Thank you."  # e.g. stale LLM/Whisper text on an intro
        _assign_lyrics_from_words(scenes, [WordTimestamp("Karina,", 13.54, 14.38)], set_type=True)
        assert (scenes[0].type, scenes[0].lyrics) == (SceneType.INSTRUMENTAL, "")
        assert (scenes[1].type, scenes[1].lyrics) == (SceneType.VOCAL, "Karina,")

    def test_last_scene_includes_song_end(self):
        scenes = _scenes((0, 2), (2, 4))
        _assign_lyrics_from_words(scenes, [WordTimestamp("end", 4.0, 4.0)], set_type=False)
        assert scenes[1].lyrics == "end"

    def test_set_type_false_keeps_user_types(self):
        scenes = _scenes((0, 2), kind=SceneType.VOCAL)
        _assign_lyrics_from_words(scenes, [], set_type=False)
        assert scenes[0].type == SceneType.VOCAL and scenes[0].lyrics == ""


class TestAlignProvidedLyrics:
    @staticmethod
    def _tx(*spec):
        from musicvision.intake.transcription import TranscriptionResult

        return TranscriptionResult(text="", words=[WordTimestamp(w, a, b) for w, a, b in spec])

    def test_section_tags_dropped_and_do_not_shift_timings(self):
        from musicvision.intake.transcription import align_lyrics_with_timestamps

        tx = self._tx(("hello", 1.0, 1.5), ("world", 1.6, 2.0), ("again", 5.0, 5.5))
        out = align_lyrics_with_timestamps("[Verse 1]\nhello world\n[Chorus]\nagain", tx)
        assert [(w.word, w.start) for w in out] == [("hello", 1.0), ("world", 1.6), ("again", 5.0)]

    def test_uneven_mishearing_spread_over_heard_span(self):
        from musicvision.intake.transcription import align_lyrics_with_timestamps

        # Karina: "rush past and miss" heard as "rush past endless"
        tx = self._tx(("rush", 1.0, 1.4), ("past", 1.5, 2.0), ("endless", 2.0, 3.0), ("Sharp", 6.0, 6.5))
        out = align_lyrics_with_timestamps("rush past and miss Sharp", tx)
        assert [w.word for w in out] == ["rush", "past", "and", "miss", "Sharp"]
        assert (out[2].start, out[3].end) == (2.0, 3.0)
        assert out[4].start == 6.0  # later words keep their own timing

    def test_word_whisper_missed_is_interpolated(self):
        from musicvision.intake.transcription import align_lyrics_with_timestamps

        tx = self._tx(("a", 1.0, 2.0), ("c", 4.0, 5.0))
        out = align_lyrics_with_timestamps("a b c", tx)
        assert out[1].start == 2.0 and out[1].end == 4.0 and out[2].start == 4.0

    def test_curly_apostrophes_match(self):
        from musicvision.intake.transcription import align_lyrics_with_timestamps

        out = align_lyrics_with_timestamps("doesn’t", self._tx(("doesn't", 3.0, 3.4)))
        assert out[0].start == 3.0


def test_sliver_scene_merged_after_tiling():
    """A one-word 0.84 s scene (below HuMo's 1 s minimum clip) is merged into a neighbour."""
    c = get_constraints("humo")
    scenes = _scenes((0, 13.56), (13.56, 14.40), (14.40, 17.36), (17.36, 24.0))
    out = _validate_and_adjust_scenes(scenes, 24.0, c)
    assert all(s.time_end - s.time_start >= 2.0 for s in out)
    _assert_tiles(out, 24.0)
