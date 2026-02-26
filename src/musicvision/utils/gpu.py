"""
Multi-GPU management for MusicVision.

Strategy:
  Primary GPU (highest VRAM) — DiT/UNet for FLUX and HuMo
  Secondary GPU — text encoders, VAE, Whisper, audio separator

GPU assignment is automatic: the GPU with the most VRAM becomes primary.
If VRAM is equal, the higher-end GPU (by name) is preferred.
This module provides device mapping so engine wrappers don't hardcode device indices.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


@dataclass
class DeviceMap:
    """Device assignments for the two-GPU split."""

    dit_device: "torch.device"          # primary: DiT / UNet
    encoder_device: "torch.device"      # secondary: T5, CLIP, Whisper
    vae_device: "torch.device"          # secondary: VAE encode/decode
    offload_device: "torch.device"      # CPU for idle models

    @property
    def primary(self) -> "torch.device":
        return self.dit_device

    @property
    def secondary(self) -> "torch.device":
        return self.encoder_device


def _gpu_sort_key(index: int) -> tuple:
    """
    Return a sort key for ranking GPUs: (vram_bytes, model_number).

    Higher VRAM wins. If VRAM is equal, the higher GPU model number wins
    (e.g. 4080 > 3080). This ensures the beefiest GPU becomes primary
    regardless of PCIe slot / CUDA index.
    """
    import torch

    props = torch.cuda.get_device_properties(index)
    vram = props.total_memory

    # Extract the leading numeric part of the GPU model name for tiebreaking.
    # E.g. "NVIDIA GeForce RTX 5080" → 5080, "NVIDIA RTX A6000" → 6000
    name = props.name
    numbers = re.findall(r"\d+", name)
    # Use the largest number found (handles names like "RTX 3080 Ti 12GB")
    model_number = max((int(n) for n in numbers), default=0)

    return (vram, model_number)


def detect_devices() -> DeviceMap:
    """
    Auto-detect GPU configuration and return a DeviceMap.

    Rules:
      - Apple Silicon MPS → single MPS device for all components
      - 2+ CUDA GPUs → highest-VRAM GPU = DiT (primary), next = encoders/VAE
      - 1 CUDA GPU   → everything on that GPU, offload to CPU
      - 0 GPUs       → CPU-only (for testing / assembly-only workflows)
    """
    import torch

    # Apple Silicon MPS — single unified memory device
    if torch.backends.mps.is_available():
        mps = torch.device("mps")
        log.info("Apple Silicon MPS detected — single-device mode (all components on MPS)")
        return DeviceMap(
            dit_device=mps,
            encoder_device=mps,
            vae_device=mps,
            offload_device=torch.device("cpu"),
        )

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        # Rank GPUs: highest VRAM first, tiebreak by model number
        ranked = sorted(range(n_gpus), key=_gpu_sort_key, reverse=True)
        primary_idx = ranked[0]
        secondary_idx = ranked[1]

        primary = torch.device(f"cuda:{primary_idx}")
        secondary = torch.device(f"cuda:{secondary_idx}")
        log.info(
            "Multi-GPU detected: cuda:%d %s [%.1f GB] (DiT) + cuda:%d %s [%.1f GB] (encoders/VAE)",
            primary_idx,
            torch.cuda.get_device_name(primary_idx),
            torch.cuda.get_device_properties(primary_idx).total_memory / 1024**3,
            secondary_idx,
            torch.cuda.get_device_name(secondary_idx),
            torch.cuda.get_device_properties(secondary_idx).total_memory / 1024**3,
        )
        return DeviceMap(
            dit_device=primary,
            encoder_device=secondary,
            vae_device=secondary,
            offload_device=torch.device("cpu"),
        )

    if n_gpus == 1:
        gpu = torch.device("cuda:0")
        log.info("Single GPU detected: %s — all models share device", torch.cuda.get_device_name(0))
        return DeviceMap(
            dit_device=gpu,
            encoder_device=gpu,
            vae_device=gpu,
            offload_device=torch.device("cpu"),
        )

    log.warning("No CUDA GPUs detected — running in CPU-only mode (assembly/config only)")
    cpu = torch.device("cpu")
    return DeviceMap(
        dit_device=cpu,
        encoder_device=cpu,
        vae_device=cpu,
        offload_device=cpu,
    )


def log_vram_usage() -> None:
    """Log current VRAM/RAM usage for available accelerators."""
    import torch

    if torch.backends.mps.is_available():
        allocated = torch.mps.current_allocated_memory() / 1024**3
        try:
            import psutil
            total = psutil.virtual_memory().total / 1024**3
            available = psutil.virtual_memory().available / 1024**3
        except ImportError:
            total = available = float("nan")
        log.info(
            "MPS: %.2f GB allocated by PyTorch, %.1f GB system RAM available (%.1f GB total)",
            allocated, available, total,
        )
        return

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        log.info(
            "GPU %d (%s): %.1f GB allocated, %.1f GB reserved, %.1f GB total",
            i, torch.cuda.get_device_name(i), allocated, reserved, total,
        )


def clear_vram() -> None:
    """Aggressively free VRAM/RAM. Call between pipeline stages (FLUX → HuMo swap)."""
    import gc
    import torch

    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
    log.info("VRAM cleared")


def recommend_tier(device_map: DeviceMap) -> "HumoTier":
    """
    Suggest the best HumoTier for the detected hardware.

    Conservative: selects the highest-quality tier that fits comfortably,
    leaving headroom for text encoder, VAE, and Whisper on the secondary device.

    Platform notes:
      - Apple Silicon MPS: preview tier only (initial release; FP8 not supported on MPS)
      - Single GPU ≥48 GB (A100 80GB, H100, H200): qualifies for FP16 tier
      - Dual GPU ≥24 GB primary (RTX 5090 + 4080): FP16 tier
    """
    import torch
    from musicvision.models import HumoTier

    # MPS: preview tier only (no FP8, limited quantization support)
    if device_map.dit_device.type == "mps":
        log.info("Apple Silicon MPS detected — recommending preview tier")
        return HumoTier.PREVIEW

    try:
        primary_gb = torch.cuda.get_device_properties(device_map.dit_device).total_memory / 1024**3
        n_gpus = torch.cuda.device_count()
    except Exception:
        log.warning("CUDA not available — recommending preview tier (CPU-only not supported)")
        return HumoTier.PREVIEW

    if n_gpus >= 2 and primary_gb >= 24:
        return HumoTier.FP16
    if n_gpus == 1 and primary_gb >= 48:   # A100 80GB / H100 / H200 single-GPU
        return HumoTier.FP16
    if primary_gb >= 20:
        return HumoTier.FP8_SCALED
    if primary_gb >= 16:
        return HumoTier.GGUF_Q6
    if primary_gb >= 12:
        return HumoTier.GGUF_Q4
    return HumoTier.PREVIEW


def vram_info() -> list[dict]:
    """Return VRAM/RAM info for available accelerators as a list of dicts (for CLI/API output)."""
    import torch

    if torch.backends.mps.is_available():
        allocated_gb = torch.mps.current_allocated_memory() / 1024**3
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / 1024**3
            free_gb = vm.available / 1024**3
        except ImportError:
            total_gb = free_gb = float("nan")
        return [{
            "index": 0,
            "name": "Apple Silicon MPS (unified memory)",
            "total_gb": round(total_gb, 1),
            "allocated_gb": round(allocated_gb, 2),
            "free_gb": round(free_gb, 1),
            "compute_capability": "mps",
        }]

    result = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        allocated_gb = torch.cuda.memory_allocated(i) / 1024**3
        free_gb = total_gb - allocated_gb
        result.append({
            "index": i,
            "name": props.name,
            "total_gb": round(total_gb, 1),
            "allocated_gb": round(allocated_gb, 1),
            "free_gb": round(free_gb, 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return result
