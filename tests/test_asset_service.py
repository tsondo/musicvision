"""
Phase 2 AssetService tests — storage + CRUD.

Exercises create/update/delete, image add/remove/set-primary edge cases, and
directory creation/cleanup against a real tmp project. No torch, no GPU.
"""

from __future__ import annotations

import pytest

from musicvision.assets import AssetService
from musicvision.models import AssetType, ConsistencyMethod
from musicvision.project import ProjectService


@pytest.fixture
def project(tmp_path):
    return ProjectService.create(tmp_path / "proj", name="Test")


@pytest.fixture
def svc(project):
    return AssetService(project)


def _make_source_image(tmp_path, name="ref.png", data=b"fakepng"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# --- ProjectPaths ---

class TestProjectPaths:
    def test_asset_dir_by_type(self, project):
        paths = project.paths
        assert paths.asset_dir(AssetType.CHARACTER, "singer") == paths.assets_dir / "characters" / "singer"
        assert paths.asset_dir(AssetType.PROP, "guitar") == paths.assets_dir / "props" / "guitar"
        assert paths.asset_dir(AssetType.LOCATION, "rooftop") == paths.assets_dir / "locations" / "rooftop"

    def test_training_dir(self, project):
        paths = project.paths
        assert paths.asset_training_dir(AssetType.CHARACTER, "singer") == (
            paths.assets_dir / "characters" / "singer" / "training"
        )

    def test_loras_and_ip_cache_at_root(self, project):
        assert project.paths.loras_dir == project.paths.root / "loras"
        assert project.paths.ip_cache_dir == project.paths.root / "ip_cache"

    def test_scaffold_creates_ip_cache(self, project):
        assert project.paths.ip_cache_dir.exists()
        assert project.paths.loras_dir.exists()


# --- Create / read / update / delete ---

class TestAssetCrud:
    def test_create_asset_makes_dir_and_persists(self, svc, project):
        asset = svc.create_asset("singer", "Singer", AssetType.CHARACTER, "Red hair")
        assert asset.id == "singer"
        assert project.paths.asset_dir(AssetType.CHARACTER, "singer").exists()
        # Reload from disk to confirm persistence
        reopened = ProjectService.open(project.paths.root)
        assert reopened.config.style_sheet.get_asset("singer").description == "Red hair"

    def test_create_duplicate_raises(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        with pytest.raises(ValueError, match="already exists"):
            svc.create_asset("singer", "Other", AssetType.CHARACTER)

    def test_list_assets_and_type_filter(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.create_asset("guitar", "Guitar", AssetType.PROP)
        assert len(svc.list_assets()) == 2
        chars = svc.list_assets(AssetType.CHARACTER)
        assert [a.id for a in chars] == ["singer"]

    def test_get_asset(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        assert svc.get_asset("singer").name == "Singer"
        assert svc.get_asset("missing") is None

    def test_update_asset_fields(self, svc, project):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.update_asset(
            "singer", name="Lead Singer", consistency=ConsistencyMethod.LORA,
            lora_path="loras/singer.safetensors", lora_weight=0.9,
        )
        reopened = ProjectService.open(project.paths.root).config.style_sheet.get_asset("singer")
        assert reopened.name == "Lead Singer"
        assert reopened.consistency == ConsistencyMethod.LORA
        assert reopened.lora_weight == 0.9

    def test_update_ignores_unknown_keys(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        # Should not raise and should not create a stray attribute
        asset = svc.update_asset("singer", bogus_field="x", name="New")
        assert asset.name == "New"
        assert not hasattr(asset, "bogus_field")

    def test_update_missing_raises(self, svc):
        with pytest.raises(ValueError, match="not found"):
            svc.update_asset("nope", name="x")

    def test_delete_asset_removes_dir_and_config(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path))
        asset_dir = project.paths.asset_dir(AssetType.CHARACTER, "singer")
        assert asset_dir.exists()

        svc.delete_asset("singer")
        assert svc.get_asset("singer") is None
        assert not asset_dir.exists()
        assert ProjectService.open(project.paths.root).config.style_sheet.get_asset("singer") is None

    def test_delete_removes_cached_embedding(self, svc, project):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        cache = project.paths.ip_cache_dir / "singer.ipadpt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"emb")
        svc.update_asset("singer", ip_adapter_embedding_path="ip_cache/singer.ipadpt")

        svc.delete_asset("singer")
        assert not cache.exists()

    def test_delete_missing_raises(self, svc):
        with pytest.raises(ValueError, match="not found"):
            svc.delete_asset("nope")


# --- Images ---

class TestAssetImages:
    def test_add_first_image_is_primary_and_copied(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        img = svc.add_image("singer", _make_source_image(tmp_path))
        assert img.is_primary
        # Relative to project root, copied under the asset dir
        assert img.filename == "assets/characters/singer/ref.png"
        assert (project.paths.root / img.filename).exists()

    def test_add_second_image_not_primary(self, svc, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path, "a.png"))
        second = svc.add_image("singer", _make_source_image(tmp_path, "b.png"))
        assert not second.is_primary
        assert svc.get_asset("singer").primary_image.filename.endswith("a.png")

    def test_add_explicit_primary_demotes_others(self, svc, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path, "a.png"))
        svc.add_image("singer", _make_source_image(tmp_path, "b.png"), is_primary=True)
        asset = svc.get_asset("singer")
        primaries = [i for i in asset.images if i.is_primary]
        assert len(primaries) == 1
        assert primaries[0].filename.endswith("b.png")

    def test_add_collision_auto_suffix(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path, "ref.png", b"one"))
        svc.add_image("singer", _make_source_image(tmp_path, "ref.png", b"two"))
        names = sorted(p.name for p in project.paths.asset_dir(AssetType.CHARACTER, "singer").glob("*.png"))
        assert names == ["ref.png", "ref_01.png"]
        # Registered filenames are distinct
        assert len({i.filename for i in svc.get_asset("singer").images}) == 2

    def test_add_training_image_goes_to_training_dir_with_caption(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        img = svc.add_image(
            "singer", _make_source_image(tmp_path, "001.png"),
            role="training", caption="Singer on stage",
        )
        assert img.role == "training"
        training_dir = project.paths.asset_training_dir(AssetType.CHARACTER, "singer")
        assert (training_dir / "001.png").exists()
        assert (training_dir / "001.txt").read_text(encoding="utf-8") == "Singer on stage"

    def test_add_image_missing_asset_raises(self, svc, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            svc.add_image("nope", _make_source_image(tmp_path))

    def test_remove_image_deletes_file(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        img = svc.add_image("singer", _make_source_image(tmp_path))
        full = project.paths.root / img.filename
        assert full.exists()
        svc.remove_image("singer", img.filename)
        assert not full.exists()
        assert svc.get_asset("singer").images == []

    def test_remove_primary_promotes_first_remaining(self, svc, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        first = svc.add_image("singer", _make_source_image(tmp_path, "a.png"))
        svc.add_image("singer", _make_source_image(tmp_path, "b.png"))
        assert first.is_primary
        svc.remove_image("singer", first.filename)
        remaining = svc.get_asset("singer").images
        assert len(remaining) == 1
        assert remaining[0].is_primary

    def test_remove_training_deletes_caption(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        img = svc.add_image(
            "singer", _make_source_image(tmp_path, "001.png"), role="training", caption="cap",
        )
        caption = (project.paths.root / img.filename).with_suffix(".txt")
        assert caption.exists()
        svc.remove_image("singer", img.filename)
        assert not caption.exists()

    def test_remove_missing_image_raises(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        with pytest.raises(ValueError, match="not found"):
            svc.remove_image("singer", "assets/characters/singer/ghost.png")

    def test_set_primary_image(self, svc, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        a = svc.add_image("singer", _make_source_image(tmp_path, "a.png"))
        b = svc.add_image("singer", _make_source_image(tmp_path, "b.png"))
        svc.set_primary_image("singer", b.filename)
        asset = svc.get_asset("singer")
        primaries = [i.filename for i in asset.images if i.is_primary]
        assert primaries == [b.filename]
        assert a.filename not in primaries

    def test_set_primary_missing_raises(self, svc, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path, "a.png"))
        with pytest.raises(ValueError, match="not found"):
            svc.set_primary_image("singer", "assets/characters/singer/ghost.png")

    def test_add_image_persists_across_reopen(self, svc, project, tmp_path):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        svc.add_image("singer", _make_source_image(tmp_path))
        reopened = ProjectService.open(project.paths.root)
        assert len(reopened.config.style_sheet.get_asset("singer").images) == 1


class TestInvalidateEmbeddingCache:
    def test_invalidate_removes_file_and_clears_field(self, svc, project):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        cache = project.paths.ip_cache_dir / "singer.ipadpt"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(b"emb")
        svc.update_asset("singer", ip_adapter_embedding_path="ip_cache/singer.ipadpt")

        svc.invalidate_embedding_cache("singer")
        assert not cache.exists()
        assert svc.get_asset("singer").ip_adapter_embedding_path is None

    def test_invalidate_noop_when_no_embedding(self, svc):
        svc.create_asset("singer", "Singer", AssetType.CHARACTER)
        # Should not raise
        svc.invalidate_embedding_cache("singer")
