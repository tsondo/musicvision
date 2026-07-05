"""
AssetService — CRUD for reusable project assets and their images.

This is pure data management on top of ``ProjectService``: it mutates
``config.style_sheet.assets`` and the asset directory tree, then persists via
``ProjectService.save_config()``. It performs no image generation and imports no
inference engines (per ASSET_LIBRARY_SPEC.md — generation goes through the
API/CLI → engine path).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from musicvision.models import AssetDef, AssetImage, AssetType

if TYPE_CHECKING:
    from musicvision.project import ProjectService

log = logging.getLogger(__name__)


class AssetService:
    """Manages asset lifecycle: create, update, delete, and image management."""

    def __init__(self, project: ProjectService):
        self.project = project

    # --- Read ---

    def list_assets(self, asset_type: Optional[AssetType] = None) -> list[AssetDef]:
        """List all assets, optionally filtered by type."""
        assets = self.project.config.style_sheet.assets
        if asset_type is not None:
            return [a for a in assets if a.asset_type == asset_type]
        return list(assets)

    def get_asset(self, asset_id: str) -> Optional[AssetDef]:
        """Return the asset with this id, or None if it does not exist."""
        return self.project.config.style_sheet.get_asset(asset_id)

    # --- Create / update / delete ---

    def create_asset(
        self, asset_id: str, name: str, asset_type: AssetType, description: str = "",
    ) -> AssetDef:
        """Create a new asset, scaffold its directory, and persist the config."""
        if self.get_asset(asset_id):
            raise ValueError(f"Asset '{asset_id}' already exists")

        asset = AssetDef(id=asset_id, name=name, asset_type=asset_type, description=description)
        self.project.config.style_sheet.assets.append(asset)

        self.project.paths.asset_dir(asset_type, asset_id).mkdir(parents=True, exist_ok=True)

        self.project.save_config()
        log.info("Created asset '%s' (%s)", asset_id, asset_type.value)
        return asset

    def update_asset(self, asset_id: str, **updates) -> AssetDef:
        """Update scalar asset fields (name, description, consistency, lora_*, ...).

        Only known model fields are applied; unknown keys are ignored. Persists
        the config.
        """
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        for key, value in updates.items():
            if key in asset.__class__.model_fields:
                setattr(asset, key, value)

        self.project.save_config()
        return asset

    def delete_asset(self, asset_id: str) -> None:
        """Delete an asset, its directory tree, and any cached embedding. Persists."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        self.project.config.style_sheet.assets = [
            a for a in self.project.config.style_sheet.assets if a.id != asset_id
        ]

        asset_dir = self.project.paths.asset_dir(asset.asset_type, asset_id)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)

        if asset.ip_adapter_embedding_path:
            cache_path = self.project.resolve_path(asset.ip_adapter_embedding_path)
            if cache_path.exists():
                cache_path.unlink()

        self.project.save_config()
        log.info("Deleted asset '%s'", asset_id)

    # --- Images ---

    def add_image(
        self,
        asset_id: str,
        source_path: Path,
        role: str = "reference",
        caption: str = "",
        is_primary: bool = False,
    ) -> AssetImage:
        """Copy an image into the asset's directory and register it.

        ``role="training"`` images go into the asset's ``training/`` subdir and,
        if a caption is supplied, get a sibling ``.txt`` caption file. Filename
        collisions are auto-suffixed (``name_01.png``) rather than overwritten.
        The first image added to an asset becomes primary automatically.
        """
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        if role == "training":
            dest_dir = self.project.paths.asset_training_dir(asset.asset_type, asset_id)
        else:
            dest_dir = self.project.paths.asset_dir(asset.asset_type, asset_id)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Auto-suffix on collision so we never overwrite an existing file.
        dest = dest_dir / source_path.name
        if dest.exists():
            stem, suffix = source_path.stem, source_path.suffix
            counter = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{counter:02d}{suffix}"
                counter += 1
        shutil.copy2(source_path, dest)

        rel_path = str(dest.relative_to(self.project.paths.root))

        # First image is always primary; an explicit primary demotes the rest.
        if not asset.images:
            is_primary = True
        elif is_primary:
            for img in asset.images:
                img.is_primary = False

        img = AssetImage(filename=rel_path, role=role, caption=caption, is_primary=is_primary)
        asset.images.append(img)

        if role == "training" and caption:
            dest.with_suffix(".txt").write_text(caption, encoding="utf-8")

        self.project.save_config()
        return img

    def remove_image(self, asset_id: str, filename: str) -> None:
        """Remove an image (and its caption file) from an asset. Persists.

        If the removed image was primary, the first remaining image is promoted.
        """
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        img = next((i for i in asset.images if i.filename == filename), None)
        if not img:
            raise ValueError(f"Image '{filename}' not found on asset '{asset_id}'")

        full_path = self.project.resolve_path(filename)
        if full_path.exists():
            full_path.unlink()
            caption_path = full_path.with_suffix(".txt")
            if caption_path.exists():
                caption_path.unlink()

        asset.images = [i for i in asset.images if i.filename != filename]

        if img.is_primary and asset.images:
            asset.images[0].is_primary = True

        self.project.save_config()

    def set_primary_image(self, asset_id: str, filename: str) -> None:
        """Mark one image as the primary reference (unsets all others). Persists."""
        asset = self.get_asset(asset_id)
        if not asset:
            raise ValueError(f"Asset '{asset_id}' not found")

        if not any(i.filename == filename for i in asset.images):
            raise ValueError(f"Image '{filename}' not found on asset '{asset_id}'")

        for img in asset.images:
            img.is_primary = img.filename == filename

        self.project.save_config()

    def invalidate_embedding_cache(self, asset_id: str) -> None:
        """Delete a cached IP-Adapter embedding when reference images change.

        Wiring this into add/remove/set-primary is deferred to Phase 7
        (embedding precomputation). Provided here as the data-management hook.
        """
        asset = self.get_asset(asset_id)
        if asset and asset.ip_adapter_embedding_path:
            cache_path = self.project.resolve_path(asset.ip_adapter_embedding_path)
            if cache_path.exists():
                cache_path.unlink()
            asset.ip_adapter_embedding_path = None
            self.project.save_config()
