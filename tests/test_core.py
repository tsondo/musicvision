"""Tests for the core data models and project service."""

import json
from pathlib import Path

import pytest

from musicvision.models import (
    ApprovalStatus,
    HumoConfig,
    ProjectConfig,
    Scene,
    SceneList,
    SceneType,
    StyleSheet,
    CharacterDef,
)
from musicvision.project import ProjectService
from musicvision.assembly.timecode import seconds_to_filename_stamp, scene_clip_filename


class TestModels:
    def test_project_config_roundtrip(self, tmp_path):
        config = ProjectConfig(
            name="Test Video",
            style_sheet=StyleSheet(
                visual_style="cinematic, moody",
                characters=[CharacterDef(id="singer", description="Woman with short hair")],
            ),
            humo=HumoConfig(model_size="17B", scale_a=2.5),
        )

        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)

        assert loaded.name == "Test Video"
        assert loaded.style_sheet.visual_style == "cinematic, moody"
        assert len(loaded.style_sheet.characters) == 1
        assert loaded.humo.scale_a == 2.5

    def test_scene_list_roundtrip(self, tmp_path):
        scenes = SceneList(scenes=[
            Scene(
                id="scene_001", order=1,
                time_start=0.0, time_end=3.88,
                type=SceneType.VOCAL,
                lyrics="Standing in the rain",
            ),
            Scene(
                id="scene_002", order=2,
                time_start=3.88, time_end=8.0,
                type=SceneType.VOCAL,
                lyrics="Waiting for the sun",
            ),
        ])

        path = tmp_path / "scenes.json"
        scenes.save(path)
        loaded = SceneList.load(path)

        assert len(loaded.scenes) == 2
        assert loaded.scenes[0].lyrics == "Standing in the rain"
        assert loaded.get_scene("scene_002").time_end == 8.0

    def test_scene_needs_sub_clips(self):
        short = Scene(id="s1", order=1, time_start=0, time_end=3.5)
        long = Scene(id="s2", order=2, time_start=0, time_end=8.0)

        assert not short.needs_sub_clips
        assert long.needs_sub_clips

    def test_scene_effective_prompt_override(self):
        scene = Scene(
            id="s1", order=1, time_start=0, time_end=3.0,
            image_prompt="auto generated prompt",
            image_prompt_user_override="my custom prompt",
        )
        assert scene.effective_image_prompt == "my custom prompt"

    def test_humo_config_resolution(self):
        hd = HumoConfig(resolution="720p")
        sd = HumoConfig(resolution="480p")
        assert hd.width == 1280 and hd.height == 720
        assert sd.width == 832 and sd.height == 480


class TestProjectService:
    def test_create_and_open(self, tmp_path):
        project_dir = tmp_path / "my_video"
        svc = ProjectService.create(project_dir, name="My Video")

        assert svc.config.name == "My Video"
        assert (project_dir / "project.yaml").exists()
        assert (project_dir / "scenes.json").exists()
        assert (project_dir / "input").is_dir()
        assert (project_dir / "clips" / "sub").is_dir()

        # Re-open
        svc2 = ProjectService.open(project_dir)
        assert svc2.config.name == "My Video"

    def test_open_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProjectService.open(tmp_path / "nope")

    def test_acestep_auto_import(self, tmp_path):
        """AceStep JSON detected alongside audio file."""
        project_dir = tmp_path / "proj"
        svc = ProjectService.create(project_dir, name="AceStep Test")

        # Create fake audio + companion JSON
        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"fake audio")

        acestep_json = tmp_path / "song.json"
        acestep_json.write_text(json.dumps({
            "bpm": 92,
            "keyscale": "A minor",
            "duration": 240,
            "instrumental": False,
            "caption": "trap-influenced hip-hop, heavy 808 bass",
            "lyrics": "(Intro)\nAyy, it's G-Money\n(Verse 1)\nNow I've been high",
            "seed": 12345,
        }))

        svc.import_audio(audio_file)

        # BPM, duration, key should be populated from AceStep
        assert svc.config.song.bpm == 92
        assert svc.config.song.duration_seconds == 240
        assert svc.config.song.keyscale == "A minor"
        assert svc.config.song.acestep is not None
        assert svc.config.song.acestep.caption.startswith("trap")
        assert svc.config.song.acestep.instrumental is False
        assert svc.config.song.acestep.seed == 12345

        # Lyrics should have been extracted
        assert svc.config.song.lyrics_file == "input/lyrics_acestep.txt"
        lyrics_path = project_dir / svc.config.song.lyrics_file
        assert lyrics_path.exists()
        assert "G-Money" in lyrics_path.read_text()

    def test_acestep_no_json(self, tmp_path):
        """No companion JSON — song fields remain empty."""
        project_dir = tmp_path / "proj"
        svc = ProjectService.create(project_dir, name="Plain Import")

        audio_file = tmp_path / "song.wav"
        audio_file.write_bytes(b"fake audio")

        svc.import_audio(audio_file)

        assert svc.config.song.acestep is None
        assert svc.config.song.bpm is None


class TestTimecode:
    def test_filename_stamp(self):
        assert seconds_to_filename_stamp(0.0) == "00m00s000"
        assert seconds_to_filename_stamp(3.88) == "00m03s880"
        assert seconds_to_filename_stamp(63.88) == "01m03s880"

    def test_scene_clip_filename(self):
        name = scene_clip_filename("scene_001", 0.0, 3.88)
        assert name == "scene_001_00m00s000_00m03s880.mp4"
