"""Vision performer-gender check on generated images (best effort, never fatal)."""

from __future__ import annotations

import pytest
from PIL import Image

from musicvision.imaging.performer_check import NOTE_PREFIX, check_performer, run_performer_check
from musicvision.llm import _anthropic_content, _encode_image, _openai_content
from musicvision.models import (
    AssetDef,
    AssetType,
    ProjectConfig,
    Scene,
    SceneTreatment,
    SceneType,
    VocalistProfile,
)


class FakeClient:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def chat(self, system, user, images=None):
        self.calls.append(images)
        if self.exc:
            raise self.exc
        return self.reply


@pytest.fixture
def image(tmp_path):
    p = tmp_path / "scene_001.png"
    Image.new("RGB", (1280, 720), "gray").save(p)
    return p


def _config(voice="male", character=None) -> ProjectConfig:
    cfg = ProjectConfig(name="t")
    cfg.song.vocalist = VocalistProfile(voice=voice)
    if character:
        cfg.style_sheet.assets = [AssetDef(id="s", name="s", asset_type=AssetType.CHARACTER, description=character)]
    return cfg


def _scene(**kw) -> Scene:
    return Scene(id="scene_001", order=1, time_start=0, time_end=4, type=SceneType.VOCAL, **kw)


@pytest.mark.parametrize(
    ("reply", "status"),
    [
        ('{"main_performer": "man"}', "match"),
        ('```json\n{"main_performer": "woman"}\n```', "mismatch"),
        ('{"main_performer": "none"}', "no_person"),
        ('{"main_performer": "unclear"}', "unclear"),
        ("I think it's a woman", "unclear"),
    ],
)
def test_check_parses_reply(image, reply, status):
    client = FakeClient(reply)
    assert check_performer(image, "man", client).status == status
    assert client.calls == [[image]]


def test_llm_error_is_unavailable_not_raised(image):
    err = RuntimeError("At most 0 image(s) may be provided in one prompt")
    assert check_performer(image, "man", FakeClient(exc=err)).status == "unavailable"


def _seen(who):
    """Patch target: check_performer answering as if the model saw *who*."""
    return lambda p, e: check_performer(p, e, FakeClient(f'{{"main_performer": "{who}"}}'))


class TestRunOnScene:
    def test_mismatch_notes_scene_and_returns_warning(self, image, monkeypatch):
        import musicvision.imaging.performer_check as pc

        monkeypatch.setattr(pc, "check_performer", _seen("woman"))
        scene = _scene(notes="keep this")
        warning = run_performer_check(scene, _config(), image)
        assert warning["code"] == "performer_mismatch" and warning["scene_id"] == "scene_001"
        assert scene.notes.splitlines()[0] == "keep this"
        assert scene.notes.splitlines()[1].startswith(NOTE_PREFIX)

        # Regenerating with a correct image clears the auto note but keeps the user's.
        monkeypatch.setattr(pc, "check_performer", _seen("man"))
        assert run_performer_check(scene, _config(), image) is None
        assert scene.notes == "keep this"

    @pytest.mark.parametrize(
        ("config", "scene"),
        [
            (_config(voice="unknown"), _scene()),
            (_config(character="a woman with a silver bob"), _scene()),
            (_config(), _scene(treatment=SceneTreatment.NARRATIVE)),
        ],
        ids=["unknown-voice", "character-asset", "narrative-scene"],
    )
    def test_skipped_without_llm_call(self, image, monkeypatch, config, scene):
        import musicvision.imaging.performer_check as pc

        def must_not_call(*a, **k):
            raise AssertionError("check should not run")

        monkeypatch.setattr(pc, "check_performer", must_not_call)
        assert run_performer_check(scene, config, image) is None


class TestImagePayload:
    def test_encode_downscales_to_jpeg(self, image):
        import base64
        import io

        data = base64.b64decode(_encode_image(image))
        with Image.open(io.BytesIO(data)) as img:
            assert img.format == "JPEG" and max(img.size) == 768

    def test_openai_and_anthropic_parts(self):
        assert _openai_content("hi", []) == "hi"
        parts = _openai_content("hi", ["AAA"])
        assert parts[0]["image_url"]["url"] == "data:image/jpeg;base64,AAA"
        assert parts[-1] == {"type": "text", "text": "hi"}
        parts = _anthropic_content("hi", ["AAA"])
        assert parts[0]["source"]["data"] == "AAA" and parts[-1]["text"] == "hi"
