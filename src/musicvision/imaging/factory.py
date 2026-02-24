"""
Factory for creating the appropriate image generation engine.

Reads the model from ImageGenConfig and returns the correct engine subclass.
"""

from __future__ import annotations

from musicvision.imaging.base import ImageEngine
from musicvision.models import ImageGenConfig, ImageModel
from musicvision.utils.gpu import DeviceMap


def create_engine(config: ImageGenConfig, device_map: DeviceMap) -> ImageEngine:
    """Create an image engine based on the configured model."""
    if config.model in (ImageModel.FLUX_DEV, ImageModel.FLUX_SCHNELL):
        from musicvision.imaging.flux_engine import FluxEngine

        return FluxEngine(config, device_map)

    if config.model in (ImageModel.ZIMAGE, ImageModel.ZIMAGE_TURBO):
        from musicvision.imaging.zimage_engine import ZImageEngine

        return ZImageEngine(config, device_map)

    raise ValueError(f"Unknown image model: {config.model}")
