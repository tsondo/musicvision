"""
Abstract base class for video generation engines.

All video backends implement this interface so the pipeline
code never cares which model is actually running.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoInput:
    """Input for a single video generation call."""

    text_prompt: str
    reference_image: Path       # PNG, clear face visible
    audio_segment: Path         # WAV, exact duration for the clip
    output_path: Path           # where to save the generated MP4


@dataclass
class VideoResult:
    """Result of a single video generation call."""

    video_path: Path
    frames_generated: int
    duration_seconds: float
    metadata: dict = field(default_factory=dict)


class VideoEngine(ABC):
    """Base class for all video generation backends."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights into VRAM. Must be called before generate()."""

    @abstractmethod
    def generate(self, input: VideoInput) -> VideoResult:
        """Generate a single video clip. Returns a VideoResult with the saved path."""

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free VRAM."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded in memory."""
