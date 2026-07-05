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
        self._ip_adapter_loaded = False
        self._offload_active = False  # True once cpu-offload hooks are installed (see load-order audit)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        """Load FLUX pipeline. Chooses quantization/offload strategy from VRAM.

        Load-order audit (2026-07-05) — this sequence is binding:

          1. Build      — from_pretrained (+ optional quanto quantization).
                          All weights CPU-resident; no placement, no offload
                          hooks.
          2. IP-Adapter — load_ip_adapter() + CLIP-L image encoder. MUST run
                          before any enable_model_cpu_offload() /
                          enable_sequential_cpu_offload(): diffusers installs
                          offload hooks over the pipeline's component set at
                          call time, so an image encoder loaded after offload
                          sits outside the offload graph and errors at
                          inference. (The original spec said the opposite;
                          post-spec research corrected it.)
          3. Place      — device placement or offload:
                            * split (two-GPU): transformer → DiT device
                              (5090); text encoders, VAE, and the IP-Adapter
                              image encoder → encoder device (4080). CLIP-L
                              vision lives with the other encoders.
                            * single-GPU tiers: .to(device), or cpu-offload —
                              the image encoder is inside the offload graph
                              because step 2 ran first.
          4. LoRA       — project LoRA load+fuse (works under offload;
                          preserves the pre-existing order).
        """
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
            "Loading FLUX (%s) — free VRAM/RAM %.1f GB → strategy: %s",
            self.config.model.value, free_gb, strategy,
        )

        # Map tier strategy → placement plan
        if primary.type == "mps":
            # MPS: optimum-quanto is CUDA-only; use bf16 (fp16 fallback on M1/M2), no offload.
            log.info("MPS device: using bf16 no-offload (optimum-quanto not supported on MPS)")
            placement = "single_device"
        elif strategy == "bf16_split" and self.device_map.dit_device != self.device_map.encoder_device:
            # Multi-GPU: split transformer + encoders across devices
            placement = "split"
        elif strategy == "bf16_split":
            # Single GPU with ≥28 GB free (e.g. A100 80GB, H100): no offload needed
            placement = "single_device"
        elif strategy == "quantized_sequential":
            placement = "sequential_offload"
        else:
            # bf16_offload / quantized_offload
            placement = "model_offload"

        # Stage 1 — build (CPU-resident; quantization applied for Tier C/D)
        quant_type = None
        if strategy in ("quantized_offload", "quantized_sequential") and primary.type != "mps":
            quant_type = _pick_quant_type(primary, self.config.quant)
        self._pipe = self._build_pipeline(model_id, hf_token, primary, quant_type)

        # Stage 2 — IP-Adapter, strictly before placement/offload (see audit above)
        if self.config.ip_adapter.enabled:
            self._load_ip_adapter()

        # Stage 3 — placement / offload
        self._place_pipeline(placement, primary)

        # Stage 4 — project-level LoRA
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
        ip_adapter_images: Optional[list[Path]] = None,
        ip_adapter_scales: Optional[list[float]] = None,
        ip_adapter_embeddings: Optional[list[Path]] = None,
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
            ip_adapter_images: Reference image paths for IP-Adapter conditioning
                (one per conditioning asset; multi-IPA supported on FLUX).
            ip_adapter_scales: Per-image influence scale (0.0–1.0).
            ip_adapter_embeddings: Pre-computed embedding paths (.ipadpt files).
                If provided, used instead of ip_adapter_images (skips the image
                encoder — faster for repeated generations with the same reference).

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

        # IP-Adapter conditioning — no-op unless inputs were passed and the
        # adapter is (or can safely be) loaded.
        ipa_kwargs: dict = {}
        if (ip_adapter_images or ip_adapter_embeddings) and self._ensure_ip_adapter():
            ipa_kwargs = self._prepare_ip_adapter(
                ip_adapter_images, ip_adapter_scales, ip_adapter_embeddings,
            )

        import torch

        generator = (
            torch.Generator().manual_seed(seed)
            if seed is not None
            else None
        )
        actual_seed = seed if seed is not None else torch.seed()

        # NOTE: never batch positive/negative prompts into a single forward pass
        # while the IP-Adapter is active — XLabs adapters are known to produce
        # garbage under batched CFG. diffusers' FluxPipeline runs true-CFG
        # passes separately by design; do not "optimize" this into a
        # concatenated batch.
        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=self.config.effective_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            **ipa_kwargs,
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
        """Unload pipeline and free VRAM.

        The IP-Adapter (and its image encoder) is torn down with the pipeline —
        symmetric with the lazy/eager load in load()/generate().
        """
        if self._pipe is not None:
            if self._loaded_lora is not None:
                self._pipe.unload_lora_weights()
            del self._pipe
            self._pipe = None
        self._loaded_lora = None
        self._ip_adapter_loaded = False
        self._offload_active = False
        clear_vram()
        log.info("FLUX engine unloaded")

    # ------------------------------------------------------------------
    # IP-Adapter runtime helpers
    # ------------------------------------------------------------------

    def _ensure_ip_adapter(self) -> bool:
        """Lazy-load the IP-Adapter on first use if load() didn't load it.

        Only safe before cpu-offload: offload hooks installed by
        enable_model_cpu_offload()/enable_sequential_cpu_offload() do not cover
        components added afterwards, so a late-loaded image encoder would sit
        outside the offload graph and error. In that case we warn and skip
        IP-Adapter conditioning for this call — set config.ip_adapter.enabled
        before engine.load() to get the correct ordering (Phase 5 wiring does
        this automatically).
        """
        if self._ip_adapter_loaded:
            return True
        if self._offload_active:
            log.warning(
                "IP-Adapter inputs provided but the adapter is not loaded and cpu-offload "
                "is active — cannot load post-offload; skipping IP-Adapter conditioning "
                "for this generation. Set image_gen.ip_adapter.enabled before load()."
            )
            return False
        log.info("Lazy-loading IP-Adapter on first use")
        self._load_ip_adapter()
        # Place the freshly loaded image encoder with the other encoders.
        if (
            getattr(self._pipe, "image_encoder", None) is not None
            and self.device_map.encoder_device.type != "cpu"
        ):
            self._pipe.image_encoder.to(self.device_map.encoder_device)
        return True

    def _prepare_ip_adapter(
        self,
        images: Optional[list[Path]],
        scales: Optional[list[float]],
        embeddings: Optional[list[Path]],
    ) -> dict:
        """Build IP-Adapter kwargs for the pipeline call.

        Pre-computed embeddings take precedence over raw images — they skip
        the image encoder entirely (faster for repeated generations with the
        same reference). Returns a dict with ip_adapter_image or
        ip_adapter_image_embeds; also sets the adapter scale(s).
        """
        import torch

        kwargs: dict = {}

        if embeddings:
            loaded = [
                torch.load(p, map_location="cpu", weights_only=True) for p in embeddings
            ]
            kwargs["ip_adapter_image_embeds"] = loaded
        elif images:
            from PIL import Image

            loaded_images = [Image.open(p).convert("RGB") for p in images]
            kwargs["ip_adapter_image"] = (
                loaded_images if len(loaded_images) > 1 else loaded_images[0]
            )
        else:
            return {}

        if scales:
            # Single adapter loaded once — set_ip_adapter_scale takes float or list
            self._pipe.set_ip_adapter_scale(scales[0] if len(scales) == 1 else list(scales))
        else:
            self._pipe.set_ip_adapter_scale(self.config.ip_adapter.default_scale)

        return kwargs

    # ------------------------------------------------------------------
    # Internal loading helpers (build → IP-Adapter → place; see load())
    # ------------------------------------------------------------------

    def _build_pipeline(self, model_id: str, token: Optional[str], primary, quant_type=None):
        """Stage 1: from_pretrained, CPU-resident. Optional quanto quantization.

        No device placement and no offload hooks here — the IP-Adapter must be
        able to load between build and placement (see load() audit).
        """
        import torch
        from diffusers import FluxPipeline

        # MPS on M1/M2 does not support bfloat16; probe at runtime.
        dtype = torch.bfloat16
        if primary.type == "mps":
            try:
                _probe = torch.zeros(1, dtype=torch.bfloat16, device=primary)
                del _probe
            except (RuntimeError, TypeError):
                dtype = torch.float16
                log.info("MPS: bfloat16 not supported on this chip — using float16 for FLUX")

        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=token,
        )

        if quant_type is not None:
            from optimum.quanto import freeze, quantize

            log.info("Quantizing transformer (%s)…", _quant_name(quant_type))
            quantize(pipe.transformer, weights=quant_type)
            freeze(pipe.transformer)

        return pipe

    def _load_ip_adapter(self) -> None:
        """Stage 2: load IP-Adapter weights + CLIP vision encoder into the pipeline.

        Stock diffusers path only — load_ip_adapter() / set_ip_adapter_scale()
        with ip_adapter_image / ip_adapter_image_embeds call kwargs. No vendored
        pipeline code, no custom attention processors. Swapping XLabs v2 → v1
        (or any loader-compatible adapter) is a pure config change via
        IPAdapterConfig.model_repo / weight_name.
        """
        ipa = self.config.ip_adapter
        log.info("Loading IP-Adapter: %s (%s)", ipa.model_repo, ipa.weight_name)
        self._pipe.load_ip_adapter(
            ipa.model_repo,
            weight_name=ipa.weight_name,
            image_encoder_pretrained_model_name_or_path=ipa.image_encoder_repo,
        )
        self._ip_adapter_loaded = True
        log.info("IP-Adapter loaded (image encoder: %s)", ipa.image_encoder_repo)

    def _place_pipeline(self, placement: str, primary) -> None:
        """Stage 3: device placement or cpu-offload. MUST run after _load_ip_adapter()."""
        pipe = self._pipe
        if placement == "split":
            # Two-GPU: DiT on the primary (5090), all encoders on the secondary (4080).
            pipe.transformer.to(self.device_map.dit_device)
            pipe.text_encoder.to(self.device_map.encoder_device)    # CLIP-L (text)
            pipe.text_encoder_2.to(self.device_map.encoder_device)  # T5-XXL
            pipe.vae.to(self.device_map.vae_device)
            # IP-Adapter image encoder (CLIP-L vision) belongs with the other
            # encoders on the secondary GPU — same budget pool as CLIP/T5/VAE.
            if getattr(pipe, "image_encoder", None) is not None:
                pipe.image_encoder.to(self.device_map.encoder_device)
        elif placement == "single_device":
            pipe.to(primary)
        elif placement == "sequential_offload":
            pipe.enable_sequential_cpu_offload()
            self._offload_active = True
        else:  # "model_offload"
            pipe.enable_model_cpu_offload()
            self._offload_active = True

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
    """Return free VRAM/RAM in GB on the given device. Returns 0 for CPU."""
    try:
        import torch
        if device.type == "cpu":
            return 0.0
        if device.type == "mps":
            # MPS uses unified system RAM; report available system memory.
            import psutil
            return psutil.virtual_memory().available / 1024**3
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
    """FP8 requires compute capability ≥ 8.9 (Ada Lovelace / Hopper / Blackwell).

    The >= (8, 9) check covers RTX 40xx (CC 8.9), H100 (CC 9.0), and
    Blackwell RTX 50xx (CC 12.0). torch.float8_e4m3fn works on all three.
    """
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
    # optimum-quanto is CUDA-only as of 0.2.x; MPS must use bf16/fp16 path.
    if device.type == "mps":
        return None

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
