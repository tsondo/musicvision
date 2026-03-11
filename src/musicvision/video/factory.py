"""
Factory for creating video generation engines.

Supports HuMo (GPU-local) and LTX-Video 2 (GPU-local via diffusers).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from musicvision.models import (
    HumoConfig,
    LtxVideoConfig,
    VideoEngineType,
)
from musicvision.video.base import VideoEngine

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap


def create_video_engine(
    config: HumoConfig | LtxVideoConfig,
    device_map: DeviceMap | None = None,
    engine_type: VideoEngineType | None = None,
) -> VideoEngine:
    """Create a video engine based on config type or explicit engine_type.

    Args:
        config: Engine-specific configuration.
        device_map: GPU device map (required for HuMo and LTX-Video 2).
        engine_type: Explicit engine type override. If None, inferred from config type.

    Returns:
        A VideoEngine instance.
    """
    # Infer engine type from config if not explicitly given
    if engine_type is None:
        if isinstance(config, LtxVideoConfig):
            engine_type = VideoEngineType.LTX_VIDEO
        else:
            engine_type = VideoEngineType.HUMO

    if engine_type == VideoEngineType.LTX_VIDEO:
        from musicvision.video.ltx_video_engine import LtxVideoEngine

        if not isinstance(config, LtxVideoConfig):
            raise TypeError(f"Expected LtxVideoConfig, got {type(config).__name__}")
        if device_map is None:
            raise ValueError("device_map is required for LtxVideoEngine")
        return LtxVideoEngine(config, device_map)

    # Default: HuMo
    from musicvision.video.humo_engine import HumoEngine

    if not isinstance(config, HumoConfig):
        raise TypeError(f"Expected HumoConfig, got {type(config).__name__}")
    if device_map is None:
        raise ValueError("device_map is required for HumoEngine")
    return HumoEngine(config, device_map)
