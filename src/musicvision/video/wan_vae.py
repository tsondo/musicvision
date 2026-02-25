"""
WanVideoVAE — 3D causal Video VAE for HuMo / Wan2.1 video generation.

Uses the vendored Wan-AI VAE architecture (vendor/wan_vae_arch.py) which
matches the Wan2.1_VAE.pth checkpoint key format exactly:
  encoder.conv1, encoder.downsamples.N.residual.M.gamma, etc.

The Wan-AI VAE uses RMS_norm (not GroupNorm), CausalConv3d (nn.Conv3d
subclass), and per-channel latent mean/std normalization.

Compression factors
-------------------
  Temporal:  ×4 reduction   (97 frames → 25 latent frames)
  Spatial:   ×8 reduction   (720px → 90 latent px)
  Latent channels: 16
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


class WanVideoVAE:
    """
    Wan 2.1 Video VAE — 3D causal convolutional encoder-decoder.

    Wraps the vendored WanVAE class which builds the correct architecture
    and loads checkpoints with no key remapping needed.

    Compression::

        Encoder: [C, T, H, W] → [16, T//4, H//8, W//8]
        Decoder: [16, T, H, W] → [C, T*4, H*8, W*8]  (clamped [-1,1])

    Usage::

        vae = WanVideoVAE(device=torch.device("cuda:0"))
        vae.load(Path("~/.cache/musicvision/weights/shared/vae/Wan2.1_VAE.pth"))
        latents = vae.encode(pixel_frames)   # list of [16, T//4, H//8, W//8]
        frames  = vae.decode(latents)        # list of [3, T*4, H*8, W*8] in [-1, 1]
    """

    def __init__(self, device, dtype=None) -> None:
        import torch

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype if dtype is not None else torch.float16
        self._vae = None  # WanVAE instance

    def load(self, weights_path: Path) -> None:
        """
        Load VAE weights from a Wan2.1_VAE.pth checkpoint.

        The vendored WanVAE handles model construction, weight loading (strict
        via assign=True on meta-device init), and latent normalization scale.
        """
        from musicvision.video.vendor.wan_vae_arch import WanVAE

        log.info("Loading WanVideoVAE from %s onto %s (dtype=%s) …",
                 weights_path.name, self.device, self.dtype)

        self._vae = WanVAE(
            z_dim=16,
            vae_pth=str(weights_path),
            dtype=self.dtype,
            device=str(self.device),
        )

        log.info("WanVideoVAE ready.")

    def encode(self, pixels: "torch.Tensor") -> "torch.Tensor":
        """
        Encode pixel video tensor into VAE latents.

        Args:
            pixels: [B, 3, T, H, W] float tensor in [0, 1].

        Returns:
            Latent tensor [B, 16, T//4, H//8, W//8].
        """
        import torch

        if self._vae is None:
            raise RuntimeError("WanVideoVAE.load() must be called before encode().")

        # Normalize [0,1] → [-1, 1] to match VAE training
        x = pixels.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0

        # WanVAE.encode expects list of [C, T, H, W] tensors (unbatched)
        video_list = [x[i] for i in range(x.shape[0])]
        latent_list = self._vae.encode(video_list)

        # Stack back into batch: list of [16, T', H', W'] → [B, 16, T', H', W']
        return torch.stack(latent_list, dim=0)

    def encode_image(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Encode a single reference image into a one-frame latent.

        Args:
            image: [B, 3, H, W] or [B, 3, 1, H, W] float tensor in [0, 1].

        Returns:
            Latent tensor [B, 16, 1, H//8, W//8].
        """
        if image.ndim == 4:
            image = image.unsqueeze(2)   # → [B, 3, 1, H, W]
        return self.encode(image)

    def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
        """
        Decode VAE latents to pixel frames.

        Args:
            latent: [B, 16, T, H, W] latent tensor.

        Returns:
            Pixel tensor [B, 3, T*4, H*8, W*8] clamped to [0, 1].
        """
        import torch

        if self._vae is None:
            raise RuntimeError("WanVideoVAE.load() must be called before decode().")

        z = latent.to(device=self.device, dtype=self.dtype)

        # WanVAE.decode expects list of [16, T, H, W] tensors
        latent_list = [z[i] for i in range(z.shape[0])]
        decoded_list = self._vae.decode(latent_list)

        # Stack and map [-1, 1] → [0, 1]
        out = torch.stack(decoded_list, dim=0)
        return torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)

    def unload(self) -> None:
        """Release model from VRAM."""
        import gc
        import torch

        self._vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("WanVideoVAE unloaded.")
