"""
Multi-GPU management for MusicVision.

Strategy (from PIPELINE_SPEC):
  GPU 0 (RTX 5080, 32GB) — DiT/UNet for FLUX and HuMo
  GPU 1 (RTX 3080 Ti, 12GB) — text encoders, VAE, Whisper, audio separator

This module provides device mapping so engine wrappers don't hardcode device indices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

log = logging.getLogger(__name__)


@dataclass
class DeviceMap:
    """Device assignments for the two-GPU split."""

    dit_device: torch.device          # primary: DiT / UNet
    encoder_device: torch.device      # secondary: T5, CLIP, Whisper
    vae_device: torch.device          # secondary: VAE encode/decode
    offload_device: torch.device      # CPU for idle models

    @property
    def primary(self) -> torch.device:
        return self.dit_device

    @property
    def secondary(self) -> torch.device:
        return self.encoder_device


def detect_devices() -> DeviceMap:
    """
    Auto-detect GPU configuration and return a DeviceMap.

    Rules:
      - 2+ GPUs → GPU0 = DiT, GPU1 = encoders/VAE
      - 1 GPU   → everything on GPU0, offload to CPU
      - 0 GPUs  → CPU-only (for testing / assembly-only workflows)
    """
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

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log.info("VRAM cleared")
