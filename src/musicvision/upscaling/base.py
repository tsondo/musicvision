"""
Abstract base class for video upscaling engines.

All upscaler backends implement this interface so the pipeline
orchestrator never cares which upscaler is running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class UpscaleInput:
    """Input for a single upscale call."""

    video_path: Path        # source clip to upscale
    output_path: Path       # where to write upscaled clip
    target_width: int       # target output width
    target_height: int      # target output height


@dataclass
class UpscaleResult:
    """Result of a single upscale call."""

    video_path: Path            # path to upscaled clip
    source_resolution: tuple[int, int]   # (w, h) of input
    output_resolution: tuple[int, int]   # (w, h) of output
    metadata: dict = field(default_factory=dict)


class UpscaleEngine(ABC):
    """Base class for all video upscaling backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into VRAM. Must be called before upscale()."""

    @abstractmethod
    def upscale(self, input: UpscaleInput) -> UpscaleResult:
        """Upscale a single video clip. Returns an UpscaleResult with the saved path."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
