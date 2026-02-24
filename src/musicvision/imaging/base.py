"""
Abstract base class for image generation engines.

All backends (FLUX, Z-Image, etc.) implement this interface so the pipeline
code never cares which model is actually running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ImageResult:
    """Result of a single image generation call."""

    path: Path
    seed: int
    prompt: str
    width: int
    height: int
    metadata: dict = field(default_factory=dict)


class ImageEngine(ABC):
    """Base class for all image generation backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into VRAM. Must be called before generate()."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        width: int = 1280,
        height: int = 720,
        seed: int | None = None,
        lora_path: str | None = None,
        lora_weight: float = 0.8,
        output_path: Path | None = None,
    ) -> ImageResult:
        """Generate a single image. Returns an ImageResult with the saved path."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
