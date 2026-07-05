"""
Phase 1 asset-library data-model tests.

Covers AssetDef/AssetImage helpers and the StyleSheet legacy migration
validator (characters/props/settings → assets). No torch, no GPU.
"""

from __future__ import annotations

from musicvision.models import (
    AssetDef,
    AssetImage,
    AssetType,
    ConsistencyMethod,
    ProjectConfig,
    StyleSheet,
)


class TestAssetDef:
    def test_primary_image_explicit(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[
                AssetImage(filename="a.png"),
                AssetImage(filename="b.png", is_primary=True),
            ],
        )
        assert asset.primary_image.filename == "b.png"

    def test_primary_image_fallback_to_first(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[AssetImage(filename="a.png")],
        )
        assert asset.primary_image.filename == "a.png"

    def test_primary_image_none_when_empty(self):
        asset = AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER)
        assert asset.primary_image is None

    def test_has_lora(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            lora_path="loras/singer.safetensors", consistency=ConsistencyMethod.LORA,
        )
        assert asset.has_lora
        assert not asset.has_ip_adapter

    def test_has_lora_requires_lora_consistency(self):
        """A lora_path with consistency=none is not an active LoRA."""
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            lora_path="loras/singer.safetensors", consistency=ConsistencyMethod.NONE,
        )
        assert not asset.has_lora

    def test_has_ip_adapter(self):
        asset = AssetDef(
            id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
            consistency=ConsistencyMethod.IP_ADAPTER,
            images=[AssetImage(filename="ref.png", is_primary=True)],
        )
        assert asset.has_ip_adapter
        assert not asset.has_lora

    def test_has_ip_adapter_requires_images(self):
        asset = AssetDef(
            id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
            consistency=ConsistencyMethod.IP_ADAPTER,
        )
        assert not asset.has_ip_adapter

    def test_has_both(self):
        asset = AssetDef(
            id="hero", name="Hero", asset_type=AssetType.CHARACTER,
            consistency=ConsistencyMethod.BOTH,
            lora_path="loras/hero.safetensors",
            images=[AssetImage(filename="ref.png", is_primary=True)],
        )
        assert asset.has_lora
        assert asset.has_ip_adapter

    def test_training_images_filter(self):
        asset = AssetDef(
            id="singer", name="Singer", asset_type=AssetType.CHARACTER,
            images=[
                AssetImage(filename="ref.png", role="reference"),
                AssetImage(filename="train_01.png", role="training", caption="Singer on stage"),
                AssetImage(filename="train_02.png", role="training", caption="Singer closeup"),
            ],
        )
        assert len(asset.training_images) == 2
        assert len(asset.reference_images) == 1


class TestStyleSheetLookups:
    def test_get_asset(self):
        ss = StyleSheet(assets=[
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER),
        ])
        assert ss.get_asset("singer").name == "Singer"
        assert ss.get_asset("missing") is None

    def test_assets_by_type(self):
        ss = StyleSheet(assets=[
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER),
            AssetDef(id="guitar", name="Guitar", asset_type=AssetType.PROP),
            AssetDef(id="dancer", name="Dancer", asset_type=AssetType.CHARACTER),
        ])
        chars = ss.assets_by_type(AssetType.CHARACTER)
        assert {a.id for a in chars} == {"singer", "dancer"}
        assert len(ss.assets_by_type(AssetType.LOCATION)) == 0


class TestStyleSheetMigration:
    def test_legacy_characters_migrate(self):
        data = {
            "visual_style": "cinematic",
            "characters": [
                {"id": "singer", "description": "Woman with red hair", "lora_path": "loras/singer.safetensors"}
            ],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].id == "singer"
        assert ss.assets[0].asset_type == AssetType.CHARACTER
        assert ss.assets[0].lora_path == "loras/singer.safetensors"
        assert ss.assets[0].consistency == ConsistencyMethod.LORA

    def test_legacy_character_reference_image_becomes_primary(self):
        data = {
            "characters": [
                {"id": "singer", "description": "Woman", "reference_image": "characters/singer.png"}
            ],
        }
        ss = StyleSheet.model_validate(data)
        img = ss.assets[0].primary_image
        assert img is not None
        assert img.filename == "characters/singer.png"
        assert img.is_primary
        # No lora_path → consistency stays none
        assert ss.assets[0].consistency == ConsistencyMethod.NONE

    def test_legacy_props_migrate(self):
        data = {
            "props": [{"id": "guitar", "description": "Red electric guitar", "reference_image": "props/guitar.png"}],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].asset_type == AssetType.PROP
        assert ss.assets[0].images[0].filename == "props/guitar.png"
        assert ss.assets[0].images[0].is_primary

    def test_legacy_settings_migrate_to_locations(self):
        data = {
            "settings": [{"id": "rooftop", "description": "City rooftop at sunset"}],
        }
        ss = StyleSheet.model_validate(data)
        assert ss.assets[0].asset_type == AssetType.LOCATION

    def test_no_duplicate_migration(self):
        """If asset already exists in assets list, don't re-migrate from legacy (id de-dup)."""
        data = {
            "assets": [{"id": "singer", "name": "Singer", "asset_type": "character", "description": "Updated"}],
            "characters": [{"id": "singer", "description": "Old description"}],
        }
        ss = StyleSheet.model_validate(data)
        assert len(ss.assets) == 1
        assert ss.assets[0].description == "Updated"

    def test_all_three_legacy_types_together(self):
        data = {
            "characters": [{"id": "singer", "description": "Singer"}],
            "props": [{"id": "guitar", "description": "Guitar"}],
            "settings": [{"id": "rooftop", "description": "Rooftop"}],
        }
        ss = StyleSheet.model_validate(data)
        by_type = {a.asset_type for a in ss.assets}
        assert by_type == {AssetType.CHARACTER, AssetType.PROP, AssetType.LOCATION}
        assert len(ss.assets) == 3

    def test_existing_projects_load_unchanged(self):
        """Projects with no legacy fields and no assets load cleanly."""
        data = {"visual_style": "cinematic", "color_palette": "warm tones"}
        ss = StyleSheet.model_validate(data)
        assert ss.assets == []

    def test_roundtrip_with_assets(self, tmp_path):
        config = ProjectConfig(
            name="Test",
            style_sheet=StyleSheet(
                visual_style="cinematic",
                assets=[
                    AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                             description="Woman with red hair", consistency=ConsistencyMethod.LORA,
                             lora_path="loras/singer.safetensors"),
                ],
            ),
        )
        path = tmp_path / "project.yaml"
        config.save(path)
        loaded = ProjectConfig.load(path)
        assert len(loaded.style_sheet.assets) == 1
        assert loaded.style_sheet.assets[0].has_lora
