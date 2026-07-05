"""Phase 4 consistency-resolver tests. Pure models + dataclass — no torch, no GPU."""

from __future__ import annotations

from musicvision.assets.consistency import (
    SceneConditioning,
    conditioning_sort_key,
    resolve_scene_conditioning,
    should_enable_ip_adapter,
)
from musicvision.models import (
    AssetDef,
    AssetImage,
    AssetType,
    ConsistencyMethod,
    ImageModel,
    Scene,
    StyleSheet,
)


def _make_scene(characters=None, props=None, settings=None):
    return Scene(
        id="s1", order=1, time_start=0, time_end=3.0,
        characters=characters or [], props=props or [], settings=settings or [],
    )


def _make_stylesheet(assets):
    return StyleSheet(assets=assets)


class TestResolveSceneConditioning:
    def test_prompt_only(self):
        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Woman with red hair"),
        ])
        scene = _make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.prompt_fragments == ["Woman with red hair"]
        assert not cond.has_lora
        assert not cond.has_ip_adapter

    def test_lora_selected(self):
        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Singer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/singer.safetensors", lora_weight=0.9),
        ])
        scene = _make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.lora_path == "loras/singer.safetensors"
        assert cond.lora_weight == 0.9

    def test_first_lora_wins(self):
        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     description="Singer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/singer.safetensors"),
            AssetDef(id="dancer", name="Dancer", asset_type=AssetType.CHARACTER,
                     description="Dancer", consistency=ConsistencyMethod.LORA,
                     lora_path="loras/dancer.safetensors"),
        ])
        scene = _make_scene(characters=["singer", "dancer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.lora_path == "loras/singer.safetensors"

    def test_character_lora_beats_prop_lora(self):
        """Characters are checked before props/settings, regardless of asset order."""
        ss = _make_stylesheet([
            AssetDef(id="guitar", name="Guitar", asset_type=AssetType.PROP,
                     consistency=ConsistencyMethod.LORA, lora_path="loras/guitar.safetensors"),
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.LORA, lora_path="loras/singer.safetensors"),
        ])
        scene = _make_scene(characters=["singer"], props=["guitar"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.lora_path == "loras/singer.safetensors"

    def test_ip_adapter_collected(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     description="Sidekick", consistency=ConsistencyMethod.IP_ADAPTER,
                     ip_adapter_scale=0.5,
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = _make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert len(cond.ip_adapter_images) == 1
        assert cond.ip_adapter_scales == [0.5]
        assert cond.has_ip_adapter

    def test_ip_adapter_relative_path_resolved_against_project_root(self, tmp_path):
        (tmp_path / "assets").mkdir()
        (tmp_path / "assets" / "ref.png").write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     images=[AssetImage(filename="assets/ref.png", is_primary=True)]),
        ])
        scene = _make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV, project_root=tmp_path)

        assert cond.ip_adapter_images == [tmp_path / "assets" / "ref.png"]

    def test_ip_adapter_missing_file_skipped(self, tmp_path):
        ss = _make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     images=[AssetImage(filename="assets/ghost.png", is_primary=True)]),
        ])
        scene = _make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV, project_root=tmp_path)

        assert not cond.has_ip_adapter

    def test_ip_adapter_embedding_preferred_over_image(self, tmp_path):
        (tmp_path / "ip_cache").mkdir()
        (tmp_path / "ip_cache" / "singer.ipadpt").write_bytes(b"emb")
        (tmp_path / "ref.png").write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER, ip_adapter_scale=0.4,
                     ip_adapter_embedding_path="ip_cache/singer.ipadpt",
                     images=[AssetImage(filename=str(tmp_path / "ref.png"), is_primary=True)]),
        ])
        scene = _make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV, project_root=tmp_path)

        assert cond.ip_adapter_embeddings == [tmp_path / "ip_cache" / "singer.ipadpt"]
        assert cond.ip_adapter_images == []
        assert cond.ip_adapter_scales == [0.4]

    def test_ip_adapter_stale_embedding_falls_back_to_image(self, tmp_path):
        """Embedding path set but file missing → fall back to the primary image."""
        (tmp_path / "ref.png").write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     ip_adapter_embedding_path="ip_cache/gone.ipadpt",
                     images=[AssetImage(filename=str(tmp_path / "ref.png"), is_primary=True)]),
        ])
        scene = _make_scene(characters=["singer"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV, project_root=tmp_path)

        assert cond.ip_adapter_embeddings == []
        assert len(cond.ip_adapter_images) == 1

    def test_ip_adapter_skipped_for_zimage(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="sidekick", name="Sidekick", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = _make_scene(characters=["sidekick"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.ZIMAGE)

        assert not cond.has_ip_adapter  # Z-Image doesn't support IPA

    def test_zimage_still_gets_lora_and_fragments(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="hero", name="Hero", asset_type=AssetType.CHARACTER,
                     description="Hero in a red coat", consistency=ConsistencyMethod.BOTH,
                     lora_path="loras/hero.safetensors",
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = _make_scene(characters=["hero"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.ZIMAGE_TURBO)

        assert cond.has_lora
        assert cond.prompt_fragments == ["Hero in a red coat"]
        assert not cond.has_ip_adapter

    def test_both_lora_and_ip_adapter(self, tmp_path):
        ref_img = tmp_path / "ref.png"
        ref_img.write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="hero", name="Hero", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.BOTH,
                     lora_path="loras/hero.safetensors",
                     ip_adapter_scale=0.4,
                     images=[AssetImage(filename=str(ref_img), is_primary=True)]),
        ])
        scene = _make_scene(characters=["hero"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.has_lora
        assert cond.has_ip_adapter

    def test_multiple_ipa_assets_all_collected(self, tmp_path):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        a.write_bytes(b"fake")
        b.write_bytes(b"fake")

        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER, ip_adapter_scale=0.6,
                     images=[AssetImage(filename=str(a), is_primary=True)]),
            AssetDef(id="guitar", name="Guitar", asset_type=AssetType.PROP,
                     consistency=ConsistencyMethod.IP_ADAPTER, ip_adapter_scale=0.3,
                     images=[AssetImage(filename=str(b), is_primary=True)]),
        ])
        scene = _make_scene(characters=["singer"], props=["guitar"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert len(cond.ip_adapter_images) == 2
        assert cond.ip_adapter_scales == [0.6, 0.3]

    def test_missing_asset_ignored(self):
        ss = _make_stylesheet([])
        scene = _make_scene(characters=["nonexistent"])
        cond = resolve_scene_conditioning(scene, ss, ImageModel.FLUX_DEV)

        assert cond.prompt_fragments == []
        assert not cond.has_lora
        assert not cond.has_ip_adapter


class TestApplyToPrompt:
    def test_fragments_prepended(self):
        cond = SceneConditioning(prompt_fragments=["Woman with red hair", "Red guitar"])
        assert cond.apply_to_prompt("Singing on a rooftop") == (
            "Woman with red hair Red guitar. Singing on a rooftop"
        )

    def test_no_fragments_prompt_unchanged(self):
        cond = SceneConditioning()
        assert cond.apply_to_prompt("A rooftop") == "A rooftop"


class TestConditioningSortKey:
    def test_groups_by_lora_then_ipa_refs(self, tmp_path):
        a = SceneConditioning(lora_path="loras/x.safetensors")
        b = SceneConditioning(lora_path="loras/x.safetensors")
        c = SceneConditioning()  # no conditioning sorts first (empty lora key)
        d = SceneConditioning(ip_adapter_images=[tmp_path / "r.png"])

        assert conditioning_sort_key(a) == conditioning_sort_key(b)
        keys = sorted([conditioning_sort_key(x) for x in (a, c, d)])
        assert keys[0] == conditioning_sort_key(c) or keys[0] == conditioning_sort_key(d)
        # Same-LoRA scenes are adjacent after sorting
        ordered = sorted([a, c, b], key=conditioning_sort_key)
        lora_positions = [i for i, x in enumerate(ordered) if x.lora_path]
        assert lora_positions == [1, 2]

    def test_key_is_hashable_and_stable(self, tmp_path):
        cond = SceneConditioning(
            lora_path="loras/x.safetensors",
            ip_adapter_images=[tmp_path / "r.png"],
            ip_adapter_embeddings=[tmp_path / "e.ipadpt"],
        )
        k1, k2 = conditioning_sort_key(cond), conditioning_sort_key(cond)
        assert k1 == k2
        hash(k1)  # tuple of strings — usable as a dict key


class TestShouldEnableIPAdapter:
    def _ipa_sheet(self, tmp_path):
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"fake")
        return _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.IP_ADAPTER,
                     images=[AssetImage(filename=str(ref), is_primary=True)]),
        ])

    def test_enabled_for_flux_with_ipa_asset(self, tmp_path):
        assert should_enable_ip_adapter(self._ipa_sheet(tmp_path), ImageModel.FLUX_DEV)
        assert should_enable_ip_adapter(self._ipa_sheet(tmp_path), ImageModel.FLUX_SCHNELL)

    def test_disabled_for_zimage(self, tmp_path):
        assert not should_enable_ip_adapter(self._ipa_sheet(tmp_path), ImageModel.ZIMAGE)
        assert not should_enable_ip_adapter(self._ipa_sheet(tmp_path), ImageModel.ZIMAGE_TURBO)

    def test_disabled_without_ipa_assets(self):
        ss = _make_stylesheet([
            AssetDef(id="singer", name="Singer", asset_type=AssetType.CHARACTER,
                     consistency=ConsistencyMethod.LORA, lora_path="loras/s.safetensors"),
        ])
        assert not should_enable_ip_adapter(ss, ImageModel.FLUX_DEV)
