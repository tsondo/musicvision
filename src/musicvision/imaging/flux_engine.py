"""
FLUX inference wrapper for reference image generation.

Supports FLUX.1-dev and FLUX.1-schnell with automatic VRAM-tiered loading:

  Tier A — bf16, no offload       (≥28 GB free, single GPU or split multi-GPU)
  Tier B — bf16 + CPU offload     (14–28 GB free, T5 moves to CPU between calls)
  Tier C — quantized + CPU offload(8–14 GB free, transformer in fp8 or int8)
  Tier D — quantized + seq offload(<8 GB free,  everything moves layer-by-layer)

Tier is chosen from available VRAM at load time unless ImageGenConfig.quant is
set explicitly (BF16/FP8/INT8 force the quantization; AUTO = let engine decide).

FLUX.1-dev is a gated model — set HUGGINGFACE_TOKEN in .env or run
`huggingface-cli login` once before using it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from musicvision.imaging.base import ImageEngine, ImageResult
from musicvision.models import FluxQuant, ImageGenConfig, ImageModel
from musicvision.utils.gpu import DeviceMap, clear_vram

log = logging.getLogger(__name__)

# HuggingFace model IDs — keyed by canonical ImageModel values
MODEL_IDS: dict[ImageModel, str] = {
    ImageModel.FLUX_DEV:     "black-forest-labs/FLUX.1-dev",
    ImageModel.FLUX_SCHNELL: "black-forest-labs/FLUX.1-schnell",
}

# Backward-compat alias used by _select_strategy helpers
_HF_IDS = MODEL_IDS

# VRAM thresholds for tier selection (GB, free VRAM on the primary device)
_TIER_A_GB = 28.0   # bf16, no offload
_TIER_B_GB = 14.0   # bf16 + model cpu offload
_TIER_C_GB = 8.0    # quantized + model cpu offload
# below _TIER_C_GB → Tier D: quantized + sequential cpu offload


class FluxEngine(ImageEngine):
    """
    FLUX image generation engine.

    Lifecycle:
        engine = FluxEngine(config, device_map)
        engine.load()           # load model weights
        result = engine.generate(prompt, output_path, ...)
        engine.unload()         # free VRAM before HuMo stage
    """

    def __init__(
        self,
        config: ImageGenConfig,
        device_map: DeviceMap,
        project_root: Optional[Path] = None,
    ):
        self.config = config
        self.device_map = device_map
        self.project_root = project_root
        self._pipe = None
        self._loaded_lora: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Load FLUX pipeline. Chooses quantization/offload strategy from VRAM."""
        hf_token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")

        # Resolve the model ID — fall back to FLUX_DEV for alias values
        model_key = self.config.model
        model_id = MODEL_IDS.get(model_key)
        if model_id is None:
            # Try by value (handles DEV/SCHNELL aliases)
            for k, v in MODEL_IDS.items():
                if k.value == model_key.value:
                    model_id = v
                    break
        if model_id is None:
            model_id = MODEL_IDS[ImageModel.FLUX_DEV]

        primary = self.device_map.primary

        free_gb = _free_vram_gb(primary)
        strategy = _select_strategy(free_gb, self.config)

        log.info(
            "Loading FLUX (%s) — free VRAM %.1f GB → strategy: %s",
            self.config.model.value, free_gb, strategy,
        )

        if strategy == "bf16_split" and self.device_map.dit_device != self.device_map.encoder_device:
            self._pipe = self._load_split(model_id, hf_token)
        elif strategy in ("bf16_split", "bf16_offload"):
            self._pipe = self._load_bf16_offload(model_id, hf_token)
        else:
            # quantized (fp8 or int8)
            quant_type = _pick_quant_type(primary, self.config.quant)
            self._pipe = self._load_quantized(model_id, hf_token, quant_type, strategy)

        # Load project-level LoRA if configured
        if self.config.lora_path:
            self._apply_lora(self.config.lora_path, self.config.lora_weight)

        log.info("FLUX engine ready (%s, %s)", self.config.model.value, strategy)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        width: int = 1280,
        height: int = 720,
        seed: Optional[int] = None,
        lora_path: Optional[str] = None,
        lora_weight: float = 0.8,
    ) -> ImageResult:
        """
        Generate a single image and save it as PNG.

        Args:
            prompt: Text prompt (FLUX does not use negative prompts).
            output_path: Where to save the PNG.
            width/height: Output resolution.
            seed: Optional RNG seed for reproducibility.
            lora_path: Scene-level LoRA to apply on top of project LoRA.
            lora_weight: LoRA fusion scale (0.0–1.0).

        Returns:
            ImageResult with the saved path and metadata.
        """
        if self._pipe is None:
            raise RuntimeError("Call load() before generate()")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Scene-level LoRA (character consistency)
        scene_lora_active = False
        if lora_path:
            self._apply_lora(lora_path, lora_weight)
            scene_lora_active = True

        import torch

        generator = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else None
        )
        actual_seed = seed if seed is not None else torch.seed()

        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.config.effective_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
        ).images[0]

        result.save(str(output_path))
        log.info("Saved image → %s (%dx%d)", output_path.name, width, height)

        if scene_lora_active:
            self._remove_scene_lora()

        return ImageResult(
            path=output_path,
            seed=actual_seed,
            prompt=prompt,
            width=width,
            height=height,
            metadata={"steps": self.config.effective_steps, "guidance_scale": self.config.guidance_scale},
        )

    def unload(self) -> None:
        """Unload pipeline and free VRAM."""
        if self._pipe is not None:
            if self._loaded_lora is not None:
                self._pipe.unload_lora_weights()
            del self._pipe
            self._pipe = None
        self._loaded_lora = None
        clear_vram()
        log.info("FLUX engine unloaded")

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _load_split(self, model_id: str, token: Optional[str]):
        """Tier A (multi-GPU): bf16, transformer on GPU0, encoders on GPU1."""
        import torch
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        pipe.transformer.to(self.device_map.dit_device)
        pipe.text_encoder.to(self.device_map.encoder_device)    # CLIP-L
        pipe.text_encoder_2.to(self.device_map.encoder_device)  # T5-XXL
        pipe.vae.to(self.device_map.vae_device)
        return pipe

    def _load_bf16_offload(self, model_id: str, token: Optional[str]):
        """Tier B (single GPU, ≥14 GB): bf16, model cpu offload for T5."""
        import torch
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        pipe.enable_model_cpu_offload()
        return pipe

    def _load_quantized(
        self,
        model_id: str,
        token: Optional[str],
        quant_type,           # quanto qfloat8 or qint8 object
        strategy: str,
    ):
        """Tier C/D: quantized transformer + cpu offload."""
        import torch
        from diffusers import FluxPipeline
        from optimum.quanto import freeze, quantize

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            token=token,
        )
        log.info("Quantizing transformer (%s)…", _quant_name(quant_type))
        quantize(pipe.transformer, weights=quant_type)
        freeze(pipe.transformer)

        if strategy == "quantized_sequential":
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.enable_model_cpu_offload()
        return pipe

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def _resolve_lora(self, lora_path: str) -> str:
        """Resolve lora_path relative to project root if not absolute."""
        p = Path(lora_path)
        if not p.is_absolute() and self.project_root:
            p = self.project_root / p
        return str(p)

    def _apply_lora(self, lora_path: str, weight: float) -> None:
        resolved = self._resolve_lora(lora_path)
        if not Path(resolved).exists():
            log.warning("LoRA not found, skipping: %s", resolved)
            return
        self._pipe.load_lora_weights(resolved)
        self._pipe.fuse_lora(lora_scale=weight)
        self._loaded_lora = resolved
        log.info("LoRA applied: %s (weight=%.2f)", Path(resolved).name, weight)

    def _remove_scene_lora(self) -> None:
        """Unfuse the scene-level LoRA, leaving project-level LoRA in place."""
        self._pipe.unfuse_lora()
        self._pipe.unload_lora_weights()
        # Re-apply project-level LoRA if it was set
        if self.config.lora_path:
            self._apply_lora(self.config.lora_path, self.config.lora_weight)


# ------------------------------------------------------------------
# Strategy selection helpers
# ------------------------------------------------------------------

def _free_vram_gb(device) -> float:
    """Return free VRAM in GB on the given device. Returns 0 for CPU."""
    try:
        import torch
        if device.type == "cpu":
            return 0.0
        free_bytes, _ = torch.cuda.mem_get_info(device)
        return free_bytes / 1024**3
    except Exception:
        return 0.0


def _select_strategy(free_gb: float, config: ImageGenConfig) -> str:
    """
    Choose loading strategy from available VRAM and config.quant.

    Returns one of:
      "bf16_split"          — full precision, multi-GPU (Tier A with 2 GPUs)
      "bf16_offload"        — full precision, model cpu offload (Tier A/B, 1 GPU)
      "quantized_offload"   — quantized transformer, model cpu offload (Tier C)
      "quantized_sequential"— quantized, sequential cpu offload (Tier D)
    """
    # Explicit quant overrides tier selection
    if config.quant == FluxQuant.BF16:
        return "bf16_offload"   # caller promotes to split if 2 GPUs
    if config.quant in (FluxQuant.FP8, FluxQuant.INT8):
        return "quantized_offload"

    # AUTO: pick from VRAM
    if free_gb >= _TIER_A_GB:
        return "bf16_split"
    if free_gb >= _TIER_B_GB:
        return "bf16_offload"
    if free_gb >= _TIER_C_GB:
        return "quantized_offload"
    return "quantized_sequential"


def _supports_fp8(device) -> bool:
    """FP8 requires compute capability ≥ 8.9 (Ada Lovelace / Hopper, RTX 40xx+)."""
    try:
        import torch
        if device.type == "cpu":
            return False
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor) >= (8, 9)
    except Exception:
        return False


def _pick_quant_type(device, quant: FluxQuant):
    """Return the quanto quantization type appropriate for the device and config."""
    from optimum.quanto import qfloat8, qint8

    if quant == FluxQuant.FP8:
        if not _supports_fp8(device):
            log.warning(
                "FP8 requested but GPU compute capability < 8.9 — falling back to INT8. "
                "FP8 hardware acceleration requires RTX 40xx / RTX 50xx or newer."
            )
            return qint8
        return qfloat8

    if quant == FluxQuant.INT8:
        return qint8

    # AUTO: prefer fp8 on Ada/Hopper, int8 otherwise
    return qfloat8 if _supports_fp8(device) else qint8


def _quant_name(quant_type) -> str:
    return getattr(quant_type, "__name__", str(quant_type))
