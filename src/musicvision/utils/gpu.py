"""
Multi-GPU management for MusicVision.

Strategy (from PIPELINE_SPEC):
  GPU 0 (RTX 5090, 32GB) — DiT/UNet for FLUX and HuMo
  GPU 1 (RTX 4080, 16GB) — text encoders, VAE, Whisper, audio separator

This module provides device mapping so engine wrappers don't hardcode device indices.
"""

from __future__ import annotations

import logging
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


def detect_devices() -> DeviceMap:
    """
    Auto-detect GPU configuration and return a DeviceMap.

    Rules:
      - 2+ GPUs → GPU0 = DiT, GPU1 = encoders/VAE
      - 1 GPU   → everything on GPU0, offload to CPU
      - 0 GPUs  → CPU-only (for testing / assembly-only workflows)
    """
    import torch

    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        gpu0 = torch.device("cuda:0")
        gpu1 = torch.device("cuda:1")
        log.info(
            "Multi-GPU detected: %s (DiT) + %s (encoders/VAE)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_name(1),
        )
        return DeviceMap(
            dit_device=gpu0,
            encoder_device=gpu1,
            vae_device=gpu1,
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
    """Log current VRAM usage for all GPUs."""
    import torch

    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = torch.cuda.get_device_properties(i).total_mem / 1024**3
        log.info(
            "GPU %d (%s): %.1f GB allocated, %.1f GB reserved, %.1f GB total",
            i, torch.cuda.get_device_name(i), allocated, reserved, total,
        )


def clear_vram() -> None:
    """Aggressively free VRAM. Call between pipeline stages (FLUX → HuMo swap)."""
    import gc
    import torch

    gc.collect()
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    log.info("VRAM cleared")


def recommend_tier(device_map: DeviceMap) -> "HumoTier":
    """
    Suggest the best HumoTier for the detected hardware.

    Conservative: selects the highest-quality tier that fits comfortably,
    leaving headroom for text encoder, VAE, and Whisper on the secondary device.
    """
    import torch
    from musicvision.models import HumoTier

    try:
        primary_gb = torch.cuda.get_device_properties(device_map.dit_device).total_memory / 1024**3
        n_gpus = torch.cuda.device_count()
    except Exception:
        log.warning("CUDA not available — recommending preview tier (CPU-only not supported)")
        return HumoTier.PREVIEW

    if n_gpus >= 2 and primary_gb >= 24:
        return HumoTier.FP16
    if primary_gb >= 20:
        return HumoTier.FP8_SCALED
    if primary_gb >= 16:
        return HumoTier.GGUF_Q6
    if primary_gb >= 12:
        return HumoTier.GGUF_Q4
    return HumoTier.PREVIEW


def vram_info() -> list[dict]:
    """Return VRAM info for all GPUs as a list of dicts (for CLI/API output)."""
    import torch

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
