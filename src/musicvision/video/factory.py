"""
Factory for creating video generation engines.

Currently only HuMo is supported, but the factory pattern
makes it easy to add new backends.
"""

from __future__ import annotations

from musicvision.models import HumoConfig
from musicvision.utils.gpu import DeviceMap
from musicvision.video.base import VideoEngine


def create_video_engine(config: HumoConfig, device_map: DeviceMap) -> VideoEngine:
    """Create a video engine based on the configured model."""
    from musicvision.video.humo_engine import HumoEngine

    return HumoEngine(config, device_map)
