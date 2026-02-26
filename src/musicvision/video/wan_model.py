"""
WanModel — HuMo DiT wrapper with block-swap-compatible interface.

Imports the vendored architecture from vendor/wan_dit_arch.py (which matches
upstream Phantom-video/HuMo state_dict keys exactly) and adds:
  - CONFIG_14B / CONFIG_1_7B presets
  - from_config() factory (including meta-device support)
  - pre_blocks() / post_blocks() split for BlockSwapManager integration
  - Batch-friendly forward() that accepts pre-concatenated [B, 36, F, H, W] tensors
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.amp  # noqa: F401 — used inline as torch.amp.autocast
import torch.nn as nn

from musicvision.video.vendor.wan_dit_arch import (
    WanModel as _WanModel,
    AudioProjModel,
    AudioCrossAttentionWrapper,
    WanAttentionBlock,
    WanRMSNorm,
    WanLayerNorm,
    Head,
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
)

__all__ = [
    "WanModel",
    "AudioProjModel",
    "AudioCrossAttentionWrapper",
    "WanAttentionBlock",
    "CONFIG_14B",
    "CONFIG_1_7B",
]


class WanModel(_WanModel):
    """
    Extended WanModel with block-swap support and batch-friendly API.

    This wraps the vendored HuMo DiT architecture with:
      - pre_blocks() / post_blocks() for BlockSwapManager integration
      - Batch-tensor forward() that handles pre-concatenated x (already has
        image conditioning channels), pre-padded text embeds, and batched
        audio features
      - Named config presets for 14B and 1.7B models

    For the upstream list-of-tensors interface, use the parent class directly.
    """

    # Architecture presets (in_dim=36 for HuMo I2V: 16 noise + 4 mask + 16 ref)
    CONFIG_14B = dict(
        model_type='i2v', in_dim=36, dim=5120, num_heads=40,
        num_layers=40, text_dim=4096, ffn_dim=13824,
    )
    CONFIG_1_7B = dict(
        model_type='i2v', in_dim=36, dim=1536, num_heads=12,
        num_layers=30, text_dim=4096, ffn_dim=8960,
    )

    @classmethod
    def from_config(cls, config_name: str, device: str = "meta", **kwargs) -> "WanModel":
        """
        Construct a WanModel from a named preset.

        Uses device="meta" by default so no actual memory is allocated
        until weights are loaded.
        """
        presets = {"14B": cls.CONFIG_14B, "1_7B": cls.CONFIG_1_7B}
        if config_name not in presets:
            raise ValueError(f"Unknown config '{config_name}'. Available: {list(presets.keys())}")
        cfg = {**presets[config_name], **kwargs}
        with torch.device(device):
            return cls(**cfg)

    # ------------------------------------------------------------------
    # Block-swap-compatible interface
    # ------------------------------------------------------------------

    def pre_blocks(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ):
        """
        Execute everything before the transformer block loop.

        Accepts batch tensors (our engine convention):
            x:              [B, 36, F, H, W]   pre-concatenated noise + image cond
            timestep:       [B]                 flow-matching sigmas
            text_embeds:    [B, text_len, 4096] pre-padded T5 embeddings
            audio_features: [B, F, 8, 5, 1280]  windowed Whisper features, or None

        Returns:
            (x_seq, block_kwargs, time_emb_raw, F_frames, h, w)

            x_seq:       [B, L, D]     patchified video tokens
            block_kwargs: dict          keyword args for each block's forward()
            time_emb_raw: [B, D]       raw time embedding (for post_blocks)
            F_frames:     int           number of frames
            h, w:         int           spatial dims after patch embed
        """
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        B = x.shape[0]

        # 1. Patch embed: [B, 36, F, H, W] → [B, D, F, h, w]
        x_patch = self.patch_embedding(x)
        _, _, F_frames, h, w = x_patch.shape
        grid_sizes = torch.tensor([[F_frames, h, w]], dtype=torch.long).expand(B, 3)

        # Flatten to sequence: [B, D, F, h, w] → [B, F*h*w, D]
        x_seq = x_patch.flatten(2).transpose(1, 2)
        seq_len = x_seq.size(1)
        seq_lens = torch.tensor([seq_len] * B, dtype=torch.long, device=device)

        # 2. Time embedding + projection
        with torch.amp.autocast('cuda', dtype=torch.float32):
            time_emb_raw = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, timestep).float()
            ).float()
            e0 = self.time_projection(time_emb_raw).unflatten(1, (6, self.dim)).float()

        # 3. Text embedding: [B, text_len, 4096] → [B, text_len, D]
        text_ctx = self.text_embedding(text_embeds)

        # 4. Audio projection
        audio_flat = None
        audio_seq_len = None
        if self.insert_audio and audio_features is not None and self.audio_proj is not None:
            # audio_features: [B, F_audio, 8, 5, 1280]
            audio_proj_out = self.audio_proj(audio_features)  # [B, F_audio, 16, 1536]
            audio_proj_out = audio_proj_out.permute(0, 3, 1, 2)  # [B, 1536, F_audio, 16]
            audio_seq_len = torch.tensor(
                audio_proj_out.shape[2] * audio_proj_out.shape[3], device=device)
            audio_flat = audio_proj_out.flatten(2).transpose(1, 2)  # [B, F_audio*16, 1536]

        # 5. Build block kwargs
        block_kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=text_ctx,
            context_lens=None,
            audio=audio_flat,
            audio_seq_len=audio_seq_len,
        )

        return x_seq, block_kwargs, time_emb_raw, F_frames, h, w

    def post_blocks(
        self,
        x: torch.Tensor,
        time_emb_raw: torch.Tensor,
        F_frames: int,
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Apply head and unpatchify after the block loop.

        Args:
            x:            [B, F*h*w, D]  output of final transformer block
            time_emb_raw: [B, D]         raw time embedding (from pre_blocks)
            F_frames:     int
            h, w:         int

        Returns:
            [B, out_dim, F, H, W]  predicted velocity field
        """
        B = x.shape[0]

        # Head (AdaLN + projection)
        x = self.head(x, time_emb_raw)

        # Unpatchify: [B, F*h*w, out_dim*prod(patch_size)] → [B, out_dim, F, H, W]
        grid_sizes = torch.tensor([[F_frames, h, w]], dtype=torch.long).expand(B, 3)
        results = self.unpatchify(x, grid_sizes)

        # Stack batch (unpatchify returns list)
        return torch.stack(results)

    # ------------------------------------------------------------------
    # Batch-friendly forward (used by humo_engine when no block swap)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch-tensor forward pass for direct inference (no block swap).

        Args:
            x:              [B, 36, F, H, W]    pre-concatenated noise + image cond
            timestep:       [B]                  flow-matching sigmas
            text_embeds:    [B, text_len, 4096]  pre-padded T5 embeddings
            audio_features: [B, F, 8, 5, 1280]   windowed Whisper features, or None

        Returns:
            [B, out_dim, F, H, W]  predicted velocity field
        """
        x_seq, block_kwargs, time_emb_raw, F_frames, h, w = self.pre_blocks(
            x, timestep, text_embeds, audio_features
        )

        for block in self.blocks:
            x_seq = block(x_seq, **block_kwargs)

        return self.post_blocks(x_seq, time_emb_raw, F_frames, h, w)


# Module-level config aliases
CONFIG_14B = WanModel.CONFIG_14B
CONFIG_1_7B = WanModel.CONFIG_1_7B
