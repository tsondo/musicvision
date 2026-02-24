"""
Z-Image inference wrapper for reference image generation.

Loads Tongyi-MAI/Z-Image or Z-Image-Turbo via diffusers, with optional LoRA.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from musicvision.imaging.base import ImageEngine, ImageResult
from musicvision.models import ImageGenConfig, ImageModel
from musicvision.utils.gpu import DeviceMap, clear_vram

log = logging.getLogger(__name__)

MODEL_IDS: dict[ImageModel, str] = {
    ImageModel.ZIMAGE: "Tongyi-MAI/Z-Image",
    ImageModel.ZIMAGE_TURBO: "Tongyi-MAI/Z-Image-Turbo",
}


class ZImageEngine(ImageEngine):
    """
    Z-Image generation engine (Alibaba/Tongyi-MAI, 6B params).

    Lifecycle:
        engine = ZImageEngine(config, device_map)
        engine.load()
        result = engine.generate(prompt, ...)
        engine.unload()
    """

    def __init__(self, config: ImageGenConfig, device_map: DeviceMap):
        self.config = config
        self.device_map = device_map
        self._pipe = None
        self._current_lora: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Load Z-Image pipeline with CPU offload."""
        from diffusers import FluxPipeline

        model_id = MODEL_IDS[self.config.model]
        log.info("Loading Z-Image pipeline: %s", model_id)

        self._pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )

        gpu_index = self.device_map.dit_device.index
        if gpu_index is not None:
            self._pipe.enable_model_cpu_offload(gpu_id=gpu_index)
        else:
            log.warning("No GPU available — Z-Image running on CPU (very slow)")

        self._current_lora = None
        log.info("Z-Image pipeline loaded")

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
        if not self.is_loaded:
            raise RuntimeError("ZImageEngine not loaded. Call load() first.")

        self._apply_lora(lora_path)

        # Turbo uses 8 steps with low guidance
        is_turbo = self.config.model == ImageModel.ZIMAGE_TURBO
        steps = min(self.config.steps, 8) if is_turbo else self.config.steps
        guidance = min(self.config.guidance_scale, 1.0) if is_turbo else self.config.guidance_scale

        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        actual_seed = seed if seed is not None else torch.seed()

        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
        }
        if lora_path and self._current_lora:
            kwargs["joint_attention_kwargs"] = {"scale": lora_weight}

        image = self._pipe(**kwargs).images[0]

        if output_path is None:
            raise ValueError("output_path is required")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        log.info("Saved image: %s", output_path)

        return ImageResult(
            path=output_path,
            seed=actual_seed,
            prompt=prompt,
            width=width,
            height=height,
            metadata={"steps": steps, "guidance_scale": guidance},
        )

    def _apply_lora(self, lora_path: str | None) -> None:
        """Load or swap LoRA weights. Skips if already loaded."""
        if lora_path == self._current_lora:
            return

        if self._current_lora is not None:
            self._pipe.unload_lora_weights()
            log.info("Unloaded LoRA: %s", self._current_lora)

        if lora_path is not None:
            self._pipe.load_lora_weights(lora_path)
            log.info("Loaded LoRA: %s", lora_path)

        self._current_lora = lora_path

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._pipe is not None:
            if self._current_lora is not None:
                self._pipe.unload_lora_weights()
            del self._pipe
            self._pipe = None
        self._current_lora = None
        clear_vram()
        log.info("Z-Image engine unloaded")
