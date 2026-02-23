"""
FLUX inference wrapper for reference image generation.

Loads FLUX-dev or FLUX-schnell via diffusers, with optional LoRA.
DiT runs on primary GPU, text encoder on secondary GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

from musicvision.models import FluxConfig
from musicvision.utils.gpu import DeviceMap

log = logging.getLogger(__name__)


class FluxEngine:
    """
    FLUX image generation engine.

    Lifecycle:
        engine = FluxEngine(config, device_map)
        engine.load()           # load model weights
        image = engine.generate(prompt, ...)
        engine.unload()         # free VRAM before HuMo stage
    """

    def __init__(self, config: FluxConfig, device_map: DeviceMap):
        self.config = config
        self.device_map = device_map
        self._pipe = None

    def load(self) -> None:
        """Load FLUX pipeline with multi-GPU device map."""
        # TODO: Implement
        # - Load via diffusers FluxPipeline
        # - Place transformer on device_map.dit_device
        # - Place text_encoder on device_map.encoder_device
        # - Load LoRA if specified
        raise NotImplementedError("FLUX loading not yet implemented")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1280,
        height: int = 720,
        seed: int | None = None,
        lora_path: str | None = None,
        lora_weight: float = 0.8,
    ) -> Path:
        """Generate a single image. Returns path to saved PNG."""
        raise NotImplementedError("FLUX generation not yet implemented")

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        from musicvision.utils.gpu import clear_vram
        clear_vram()
        log.info("FLUX engine unloaded")
