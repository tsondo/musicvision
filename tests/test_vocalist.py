"""Lead-vocalist profile: pitch estimate, user override, and prompt context."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from musicvision.imaging import prompt_generator as image_pg
from musicvision.intake.audio_analysis import estimate_vocalist
from musicvision.intake.pipeline import _update_vocalist
from musicvision.models import AssetDef, AssetType, ProjectConfig, Scene, SongInfo, VocalistProfile
from musicvision.style_context import template_subject, vocalist_context_line
from musicvision.video import prompt_generator as video_pg

SR = 16000


@pytest.fixture(autouse=True)
def _short_minimum(monkeypatch):
    """Lower the 5 s sung-audio minimum so the pitch tests can use 2 s tones (pyin cost ~ length)."""
    monkeypatch.setattr("musicvision.intake.audio_analysis._MIN_VOICED_SECONDS", 1.0)


def _sung_tone(path, f0, seconds=2.0):
    """A vibrato tone with harmonics: close enough to a sung note for pyin."""
    t = np.arange(int(SR * seconds)) / SR
    freq = f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    y = sum(np.sin(k * phase) / k for k in range(1, 5)) * 0.2
    sf.write(path, y.astype(np.float32), SR)
    return path


@pytest.mark.parametrize(("f0", "voice"), [(140, "male"), (260, "female"), (190, "unknown")])
def test_estimate_from_pitch(tmp_path, f0, voice):
    prof = estimate_vocalist(_sung_tone(tmp_path / "v.wav", f0))
    assert prof.voice == voice
    assert prof.source == "detected"
    assert abs(prof.median_f0_hz - f0) < f0 * 0.05


def test_silence_is_unknown(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR * 2, dtype=np.float32), SR)
    assert estimate_vocalist(path).voice == "unknown"


class TestUpdate:
    def test_detected_value_is_refreshed(self, tmp_path, monkeypatch):
        import musicvision.intake.pipeline as pipeline

        monkeypatch.setattr(pipeline, "estimate_vocalist", lambda p: VocalistProfile(voice="male", median_f0_hz=148))
        song = SongInfo(vocalist=VocalistProfile(voice="female", source="detected"))
        _update_vocalist(song, tmp_path / "v.wav")
        assert song.vocalist.voice == "male"

    def test_user_value_is_never_overwritten(self, tmp_path, monkeypatch):
        import musicvision.intake.pipeline as pipeline

        monkeypatch.setattr(pipeline, "estimate_vocalist", lambda p: VocalistProfile(voice="male"))
        song = SongInfo(vocalist=VocalistProfile(voice="female", source="user"))
        _update_vocalist(song, tmp_path / "v.wav")
        assert song.vocalist.voice == "female"

    def test_estimate_failure_does_not_raise(self, tmp_path, monkeypatch):
        import musicvision.intake.pipeline as pipeline

        def boom(p):
            raise RuntimeError("pyin exploded")

        monkeypatch.setattr(pipeline, "estimate_vocalist", boom)
        song = SongInfo()
        _update_vocalist(song, tmp_path / "v.wav")
        assert song.vocalist is None

    def test_old_project_yaml_without_vocalist_loads(self):
        assert SongInfo.model_validate({"audio_file": "input/song.wav"}).vocalist is None


def _config(voice=None, character=None) -> ProjectConfig:
    cfg = ProjectConfig(name="t")
    if voice:
        cfg.song.vocalist = VocalistProfile(voice=voice)
    if character:
        cfg.style_sheet.assets = [
            AssetDef(id="singer", name="s", asset_type=AssetType.CHARACTER, description=character),
        ]
    return cfg


SCENE = Scene(id="scene_001", order=1, time_start=0.0, time_end=4.0, lyrics="la la")


class TestPromptContext:
    @pytest.mark.parametrize("pg", [image_pg, video_pg], ids=["image", "video"])
    def test_known_voice_reaches_context(self, pg):
        ctx = pg._build_style_context(_config(voice="male"), SCENE)
        assert "Lead vocalist: male voice" in ctx and "is a man" in ctx

    def test_unknown_voice_adds_nothing(self):
        assert vocalist_context_line(_config(voice="unknown")) is None
        assert vocalist_context_line(_config()) is None

    def test_character_asset_wins(self):
        cfg = _config(voice="male", character="a woman with a silver bob")
        assert vocalist_context_line(cfg) is None
        assert template_subject(cfg, "a performer on stage") == "a woman with a silver bob"

    def test_template_subject_uses_voice(self):
        assert template_subject(_config(voice="male"), "a performer on stage") == "a male singer on stage"
        assert template_subject(_config(), "a performer on stage") == "a performer on stage"


def test_too_little_singing_is_unknown(tmp_path, monkeypatch):
    """The real 5 s minimum: a clear 2 s male tone is still 'unknown'."""
    monkeypatch.setattr("musicvision.intake.audio_analysis._MIN_VOICED_SECONDS", 5.0)
    assert estimate_vocalist(_sung_tone(tmp_path / "short.wav", 140)).voice == "unknown"
