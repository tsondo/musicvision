"""
Video upscaling engines.

Provides ABC, factory, and engine implementations for post-generation
resolution enhancement and normalization.
"""

from __future__ import annotations

from musicvision.upscaling.base import UpscaleEngine, UpscaleInput, UpscaleResult
from musicvision.upscaling.factory import create_upscale_engine

__all__ = ["UpscaleEngine", "UpscaleInput", "UpscaleResult", "create_upscale_engine"]
