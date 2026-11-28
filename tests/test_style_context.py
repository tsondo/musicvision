"""Style-sheet assets reach the LLM prompt context (image + video generators)."""

from __future__ import annotations

import pytest

from musicvision.imaging import prompt_generator as image_pg
from musicvision.models import AssetDef, AssetType, CharacterDef, ProjectConfig, Scene, StyleSheet
from musicvision.video import prompt_generator as video_pg

SINGER = "a man in his thirties with a shaved head and a denim jacket"


def _config(**ss) -> ProjectConfig:
    cfg = ProjectConfig(name="t")
    cfg.style_sheet = StyleSheet(**ss)
    return cfg


SCENE = Scene(id="scene_001", order=1, time_start=0.0, time_end=4.0, lyrics="la la")


@pytest.mark.parametrize("pg", [image_pg, video_pg], ids=["image", "video"])
class TestAssetsInContext:
    def test_asset_character_reaches_context(self, pg):
        cfg = _config(assets=[
            AssetDef(id="singer", name="Lead singer", asset_type=AssetType.CHARACTER, description=SINGER),
            AssetDef(id="rooftop", name="rooftop", asset_type=AssetType.LOCATION, description="a rainy rooftop"),
        ])
        ctx = pg._build_style_context(cfg, SCENE)
        assert f"Character — singer (Lead singer): {SINGER}" in ctx
        assert "Setting — rooftop: a rainy rooftop" in ctx

    def test_legacy_character_appears_once(self, pg):
        cfg = _config(characters=[CharacterDef(id="singer", description=SINGER)])
        ctx = pg._build_style_context(cfg, SCENE)
        assert ctx.count(SINGER) == 1

    def test_undescribed_assets_are_skipped(self, pg):
        cfg = _config(assets=[AssetDef(id="x", name="x", asset_type=AssetType.PROP)])
        assert "Prop" not in pg._build_style_context(cfg, SCENE)


def test_templates_use_character_asset():
    cfg = _config(assets=[AssetDef(id="singer", name="s", asset_type=AssetType.CHARACTER, description=SINGER)])
    assert image_pg._auto_template_image(SCENE, cfg).lower().startswith(SINGER[:20])
    assert SINGER in video_pg._auto_template_video(SCENE, cfg).lower()
