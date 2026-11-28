"""Tests for musicvision.video.render_mode (job-level preview/final switch)."""

from __future__ import annotations

import pytest

from musicvision.models import HumoConfig, HumoQuality, HumoTier, ProjectConfig, VideoEngineType
from musicvision.video.render_mode import config_for_render_mode, missing_render_weights

LORA = "lightx2v_i2v_480p"


def _project(**humo) -> ProjectConfig:
    cfg = ProjectConfig(name="t")
    cfg.humo = HumoConfig(**{"tier": HumoTier.GGUF_Q6, "resolution": "720p", "denoising_steps": 50, **humo})
    return cfg


class TestHumo:
    def test_preview_applies_preset_but_keeps_project_tier(self):
        proj = _project()
        cfg = config_for_render_mode(VideoEngineType.HUMO, proj, "preview")
        preset = HumoConfig.from_quality(HumoQuality.PREVIEW)
        assert cfg.resolution == preset.resolution == "480p"
        assert cfg.denoising_steps == preset.denoising_steps
        assert (cfg.shift, cfg.scale_t, cfg.scale_a) == (preset.shift, preset.scale_t, preset.scale_a)
        assert cfg.lora == preset.lora
        assert cfg.tier == HumoTier.GGUF_Q6  # what fits / what is downloaded stays the user's choice

    def test_final_is_project_settings(self):
        proj = _project()
        cfg = config_for_render_mode(VideoEngineType.HUMO, proj, "final")
        assert cfg == proj.humo
        assert cfg is not proj.humo

    def test_never_mutates_project_config(self):
        proj = _project()
        before = proj.model_dump()
        cfg = config_for_render_mode(VideoEngineType.HUMO, proj, "preview")
        cfg.seed = 123
        cfg.shift = 3.0
        assert proj.model_dump() == before


class TestLtx:
    @pytest.mark.parametrize(("mode", "expected"), [("preview", (480, 320, 20)), ("final", (768, 512, 40))])
    def test_resolution_and_steps(self, mode, expected):
        proj = _project()
        cfg = config_for_render_mode(VideoEngineType.LTX_VIDEO, proj, mode)
        assert (cfg.width, cfg.height, cfg.num_inference_steps) == expected
        assert cfg is not proj.ltx_video


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match="render_mode"):
        config_for_render_mode(VideoEngineType.HUMO, _project(), "draft")


class TestMissingWeights:
    def test_no_lora_needs_nothing(self):
        assert missing_render_weights(HumoConfig(lora=None)) is None

    def test_missing_lora_is_described(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSICVISION_WEIGHTS_DIR", str(tmp_path))
        missing = missing_render_weights(HumoConfig(lora=LORA))
        assert missing["missing"] == "lora" and missing["lora"] == LORA
        assert "download-weights --lora" in missing["hint"]

    def test_present_lora_is_fine(self, tmp_path, monkeypatch):
        from musicvision.video.weight_registry import LORA_SPECS

        monkeypatch.setenv("MUSICVISION_WEIGHTS_DIR", str(tmp_path))
        spec = LORA_SPECS[LORA]
        path = tmp_path / "humo" / spec.local_subdir / spec.filename
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
        assert missing_render_weights(HumoConfig(lora=LORA)) is None

    def test_unknown_lora_key_is_missing(self):
        assert missing_render_weights(HumoConfig(lora="no_such_lora"))["lora"] == "no_such_lora"


@pytest.mark.parametrize(
    ("quality", "shift"),
    [("preview", 1.5), ("fast", 1.5), ("draft", 5.0), ("final", 5.0)],
)
def test_preset_shift_matches_guidance(quality, shift):
    """Distilled (CFG-off) presets use a low shift; guided presets use 5.0 (identity hold)."""
    cfg = HumoConfig.from_quality(quality)
    assert cfg.shift == shift
    assert (cfg.scale_t == 1.0) == (shift == 1.5)  # low shift only where CFG is off


def test_default_humo_config_is_guided_with_shift_5():
    cfg = HumoConfig()
    assert cfg.scale_t > 1.0 and cfg.shift == 5.0


def test_sub_clip_chaining_off_by_default():
    """#30: chaining compounds artifacts and does not smooth boundaries."""
    assert HumoConfig().sub_clip_continuity is False
    for q in ("preview", "fast", "draft", "final"):
        assert HumoConfig.from_quality(q).sub_clip_continuity is False
