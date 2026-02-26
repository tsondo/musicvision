"""
Block-swap manager for HuMo 17B DiT inference.

The HuMo-17B DiT has ~40 transformer blocks.  Block swap keeps only a subset
of blocks resident on GPU at any time, offloading the rest to CPU RAM.  Since
denoising steps execute blocks sequentially, only the currently active block
(plus an optional prefetch) needs to live on GPU.

This is the same technique used in kijai's ComfyUI-WanVideoWrapper
(WanVideoBlockSwap node) but adapted for our DeviceMap-based approach.

Usage
-----
    swap = BlockSwapManager(
        blocks=dit.transformer_blocks,   # nn.ModuleList
        num_gpu_blocks=20,               # keep 20 of 40 on GPU
        gpu_device=device_map.primary,
    )
    swap.prepare()       # initial placement

    for block_idx, block in enumerate(dit.transformer_blocks):
        hidden = swap.execute_block(block_idx, hidden, *args, **kwargs)

Block count recommendations
---------------------------
    block_swap_count = 0    → all blocks on GPU (fastest, max VRAM)
    block_swap_count = 10   → moderate swap (moderate speed penalty ~15%)
    block_swap_count = 20   → significant swap (~35% slower)
    block_swap_count = 30+  → aggressive swap, fits on 8-12 GB GPU
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class BlockSwapManager:
    """
    Manages CPU↔GPU block migration for the HuMo DiT transformer.

    Parameters
    ----------
    blocks:
        The nn.ModuleList of transformer blocks from the loaded DiT.
    num_gpu_blocks:
        How many blocks to keep on GPU at a time.
        0 is treated as len(blocks) (all on GPU — no swapping).
    gpu_device:
        Primary GPU device for the DiT.
    """

    def __init__(
        self,
        blocks: Any,               # nn.ModuleList — lazy-typed to avoid top-level torch import
        num_gpu_blocks: int,
        gpu_device: Any,           # torch.device
        cpu_device: Any = None,    # torch.device, defaults to cpu
    ) -> None:
        self.blocks = blocks
        self.total = len(blocks)
        self.num_gpu_blocks = num_gpu_blocks if num_gpu_blocks > 0 else self.total
        self.gpu_device = gpu_device
        self.cpu_device = cpu_device
        self._swap_enabled = self.num_gpu_blocks < self.total

    def prepare(self) -> None:
        """
        Place blocks in their initial positions:
          blocks[0 … num_gpu_blocks-1]  → GPU
          blocks[num_gpu_blocks …]       → CPU
        Called once after model load, before the first denoising step.
        """
        import torch

        cpu = self.cpu_device or torch.device("cpu")

        if not self._swap_enabled:
            log.debug("BlockSwap: all %d blocks on GPU (swap disabled)", self.total)
            for block in self.blocks:
                block.to(self.gpu_device)
            return

        log.info(
            "BlockSwap: %d blocks on GPU, %d on CPU",
            self.num_gpu_blocks,
            self.total - self.num_gpu_blocks,
        )
        for i, block in enumerate(self.blocks):
            target = self.gpu_device if i < self.num_gpu_blocks else cpu
            block.to(target)

    def execute_block(self, block_idx: int, *args: Any, **kwargs: Any) -> Any:
        """
        Run blocks[block_idx].forward(*args, **kwargs) with automatic swap.

        If swap is disabled this is equivalent to a direct call.
        If swap is enabled:
          1. Move the block to GPU (no-op if already there)
          2. Run forward
          3. Move back to CPU (freeing GPU slot for the next block)
        """
        import torch

        block = self.blocks[block_idx]

        if not self._swap_enabled:
            return block(*args, **kwargs)

        cpu = self.cpu_device or torch.device("cpu")

        # Move to GPU
        block.to(self.gpu_device)
        try:
            result = block(*args, **kwargs)
        finally:
            # Always move back to CPU, even on exception, to avoid VRAM leak
            block.to(cpu)

        return result

    def teardown(self) -> None:
        """
        Move all blocks back to CPU and clear CUDA cache.
        Call after the denoising loop finishes (before VAE decode).
        """
        import gc
        import torch

        cpu = self.cpu_device or torch.device("cpu")
        for block in self.blocks:
            block.to(cpu)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        log.debug("BlockSwap: all blocks moved to CPU, cache cleared")

    @classmethod
    def from_config(
        cls,
        blocks: Any,
        block_swap_count: int,
        gpu_device: Any,
    ) -> "BlockSwapManager":
        """
        Factory from HumoConfig.block_swap_count.

        block_swap_count = 0  → no swap (num_gpu_blocks = total)
        block_swap_count = N  → keep (total - N) blocks on GPU
        """
        total = len(blocks)
        num_gpu = total - block_swap_count if block_swap_count > 0 else total
        num_gpu = max(1, min(num_gpu, total))  # clamp
        return cls(blocks=blocks, num_gpu_blocks=num_gpu, gpu_device=gpu_device)
