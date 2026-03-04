"""
Factory for creating upscale engines.

Dispatches on UpscalerType to the appropriate engine class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from musicvision.models import UpscalerConfig, UpscalerType
from musicvision.upscaling.base import UpscaleEngine

if TYPE_CHECKING:
    from musicvision.utils.gpu import DeviceMap


def create_upscale_engine(
    upscaler_type: UpscalerType,
    config: UpscalerConfig,
    device_map: DeviceMap | None = None,
) -> UpscaleEngine:
    """Create an upscale engine for the given type.

    Args:
        upscaler_type: Which upscaler to instantiate.
        config: Project-level upscaler configuration.
        device_map: GPU device map (required for LTX Spatial).

    Returns:
        An UpscaleEngine instance.

    Raises:
        ValueError: If upscaler_type is NONE or unknown.
    """
    if upscaler_type == UpscalerType.REAL_ESRGAN:
        from musicvision.upscaling.realesrgan_engine import RealEsrganEngine

        return RealEsrganEngine(model_name=config.realesrgan_model)

    if upscaler_type == UpscalerType.SEEDVR2:
        from musicvision.upscaling.seedvr2_engine import SeedVR2Engine

        return SeedVR2Engine(
            repo_dir=config.seedvr2_repo_dir,
            venv_python=config.seedvr2_venv_python,
            model_id=config.seedvr2_model_id,
            use_fp8=config.seedvr2_use_fp8,
        )

    if upscaler_type == UpscalerType.LTX_SPATIAL:
        from musicvision.upscaling.ltx_spatial_engine import LtxSpatialEngine

        return LtxSpatialEngine(
            model_id=config.ltx_spatial_model_id,
            num_inference_steps=config.ltx_spatial_steps,
            device_map=device_map,
        )

    raise ValueError(f"Cannot create engine for upscaler type: {upscaler_type}")
