"""
Asset library — reusable project assets (characters, props, locations).

Pure data management: create/update/delete assets and their reference/training
images on disk. No image generation and no engine imports live here — image
generation goes through the API/CLI → engine path (see ASSET_LIBRARY_SPEC.md).
"""

from __future__ import annotations

from musicvision.assets.service import AssetService

__all__ = ["AssetService"]
