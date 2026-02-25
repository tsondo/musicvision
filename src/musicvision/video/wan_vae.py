"""
WanVideoVAE — 3D convolutional Video VAE for HuMo.

Architecture is inferred from the Wan2.1_VAE.pth state dict structure.
Uses causal temporal convolutions (pad left only) matching the training setup.

Compression factors
-------------------
  Temporal:  stride = (1, 2, 2) → ×4 reduction   (97 frames → 25 latent frames)
  Spatial:   stride = (2, 2, 2) → ×8 reduction   (720px → 90 latent px)
  Latent channels: 16

Channel progression (encoder)
------------------------------
  3 → 128 → 256 → 512 → 512 → (mid) → 32 (projected to 16 mean + 16 logvar)
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Architecture constants — match Wan2.1 VAE
_BASE_CHANNELS   = 128
_CHANNEL_MULT    = (1, 2, 4, 4)   # 128, 256, 512, 512
_NUM_RES_BLOCKS  = 2
_LATENT_CHANNELS = 16
# Per-stage temporal strides (stage 0 does no temporal downsampling)
_TEMPORAL_STRIDES = (1, 2, 2)
# Per-stage spatial strides (all stages downsample spatially)
_SPATIAL_STRIDES  = (2, 2, 2)


# ---------------------------------------------------------------------------
# CausalConv3d
# ---------------------------------------------------------------------------

class CausalConv3d:
    """
    3D convolution with causal (left-only) temporal padding.

    Standard Conv3d pads symmetrically, which would let the model "see into
    the future" along the time axis.  CausalConv3d pads only on the left of
    the temporal dimension so that output frame t depends only on input
    frames ≤ t.  Spatial dimensions use standard symmetric padding.

    This class is defined as a plain Python class and constructed as an
    nn.Module inside _build_causal_conv3d() so that it can be used without
    importing torch at module import time.
    """


def _build_causal_conv3d(in_ch: int, out_ch: int, kernel_size, stride=1):
    """
    Factory — returns an nn.Module implementing CausalConv3d.

    Args:
        in_ch:       Input channel count.
        out_ch:      Output channel count.
        kernel_size: int or (kt, kh, kw).
        stride:      int or (st, sh, sw).

    Returns:
        nn.Module with causal temporal padding applied in forward().
    """
    import torch.nn as nn
    import torch.nn.functional as F

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)

    kt, kh, kw = kernel_size
    st, sh, sw = stride
    # Causal temporal pad: cover all preceding frames consumed by the kernel.
    # When stride=1 we pad kt-1 frames; for stride=2 we pad kt-stride[0].
    time_pad = kt - 1
    # Symmetric spatial padding so H and W stay "same" (before striding)
    sp = kh // 2

    inner_conv = nn.Conv3d(
        in_ch, out_ch,
        kernel_size=kernel_size,
        stride=stride,
        padding=(0, sp, sp),   # temporal handled manually in forward
    )

    class _CausalConv3d(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_pad = time_pad
            self.conv = inner_conv

        def forward(self, x):
            # x: [B, C, T, H, W]
            if self.time_pad > 0:
                x = F.pad(x, (0, 0, 0, 0, self.time_pad, 0))
            return self.conv(x)

    return _CausalConv3d()


# ---------------------------------------------------------------------------
# ResBlock3D
# ---------------------------------------------------------------------------

def _build_res_block(channels: int):
    """
    Residual block with GroupNorm + SiLU + two CausalConv3d layers.

    Layout::

        x → norm1 → silu → conv1 → norm2 → silu → conv2 → (+x) → out

    Args:
        channels: Number of input and output channels (in-place residual).

    Returns:
        nn.Module.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    norm1     = nn.GroupNorm(32, channels)
    norm2     = nn.GroupNorm(32, channels)
    conv1     = _build_causal_conv3d(channels, channels, 3, stride=1)
    conv2     = _build_causal_conv3d(channels, channels, 3, stride=1)

    class _ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = norm1
            self.norm2 = norm2
            self.conv1 = conv1
            self.conv2 = conv2

        def forward(self, x):
            h = self.conv1(F.silu(self.norm1(x)))
            h = self.conv2(F.silu(self.norm2(h)))
            return x + h

    return _ResBlock()


# ---------------------------------------------------------------------------
# MidBlock (self-attention + ResBlocks)
# ---------------------------------------------------------------------------

def _build_mid_block(channels: int):
    """
    Mid-block: ResBlock → (optional single-head attention) → ResBlock.

    The Wan2.1 VAE mid-block follows the common LDM pattern.  We include a
    lightweight single-head attention at the bottleneck.

    Args:
        channels: Bottleneck channel count (512 for Wan2.1).

    Returns:
        nn.Module.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    res1   = _build_res_block(channels)
    norm   = nn.GroupNorm(32, channels)
    # Single-head self-attention (1×1×1 spatial, full temporal)
    attn_q = nn.Conv3d(channels, channels, 1)
    attn_k = nn.Conv3d(channels, channels, 1)
    attn_v = nn.Conv3d(channels, channels, 1)
    attn_o = nn.Conv3d(channels, channels, 1)
    res2   = _build_res_block(channels)

    class _MidBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.res1   = res1
            self.norm   = norm
            self.attn_q = attn_q
            self.attn_k = attn_k
            self.attn_v = attn_v
            self.attn_o = attn_o
            self.res2   = res2

        def forward(self, x):
            # x: [B, C, T, H, W]
            x = self.res1(x)
            # Flatten spatial+temporal for self-attention
            B, C, T, H, W = x.shape
            h = self.norm(x)
            q = self.attn_q(h).view(B, C, -1)          # [B, C, THW]
            k = self.attn_k(h).view(B, C, -1)
            v = self.attn_v(h).view(B, C, -1)
            scale = C ** -0.5
            attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * scale, dim=-1)
            # attn: [B, THW, THW]
            out = torch.bmm(attn, v.transpose(1, 2))    # [B, THW, C]
            out = out.transpose(1, 2).view(B, C, T, H, W)
            out = self.attn_o(out)
            x = x + out
            x = self.res2(x)
            return x

    return _MidBlock()


# ---------------------------------------------------------------------------
# Encoder stages
# ---------------------------------------------------------------------------

def _build_encoder_stage(
    in_ch: int,
    out_ch: int,
    num_res: int,
    t_stride: int,
    s_stride: int,
):
    """
    Single encoder stage: num_res ResBlocks followed by a strided downsample.

    Args:
        in_ch:     Input channels.
        out_ch:    Output channels after channel-change conv.
        num_res:   Number of residual blocks.
        t_stride:  Temporal stride for downsampling conv.
        s_stride:  Spatial stride for downsampling conv.

    Returns:
        nn.Module.
    """
    import torch.nn as nn

    # Channel-change conv (1×1×1, no striding)
    chan_conv = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    res_blocks = nn.ModuleList([_build_res_block(out_ch) for _ in range(num_res)])
    # Strided downsample
    downsample = _build_causal_conv3d(
        out_ch, out_ch,
        kernel_size=(3, 3, 3),
        stride=(t_stride, s_stride, s_stride),
    )
    do_downsample = (t_stride > 1) or (s_stride > 1)

    class _EncoderStage(nn.Module):
        def __init__(self):
            super().__init__()
            self.chan_conv    = chan_conv
            self.res_blocks   = res_blocks
            self.downsample   = downsample
            self.do_downsample = do_downsample

        def forward(self, x):
            x = self.chan_conv(x)
            for blk in self.res_blocks:
                x = blk(x)
            if self.do_downsample:
                x = self.downsample(x)
            return x

    return _EncoderStage()


# ---------------------------------------------------------------------------
# Decoder stages
# ---------------------------------------------------------------------------

def _build_decoder_stage(
    in_ch: int,
    out_ch: int,
    num_res: int,
    t_stride: int,
    s_stride: int,
):
    """
    Single decoder stage: optional upsample → ResBlocks → channel-change conv.

    Upsampling is done with nn.Upsample (trilinear) followed by a
    CausalConv3d to mix features.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels after channel-change conv.
        num_res:  Number of residual blocks.
        t_stride: Temporal upsampling factor.
        s_stride: Spatial upsampling factor.

    Returns:
        nn.Module.
    """
    import torch.nn as nn

    do_upsample = (t_stride > 1) or (s_stride > 1)
    # After upsample, mix features with a conv before residual blocks
    upsample_conv = _build_causal_conv3d(in_ch, in_ch, 3, stride=1) if do_upsample else None

    res_blocks = nn.ModuleList([_build_res_block(in_ch) for _ in range(num_res)])
    chan_conv   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    class _DecoderStage(nn.Module):
        def __init__(self):
            super().__init__()
            if do_upsample:
                self.upsample      = nn.Upsample(
                    scale_factor=(t_stride, s_stride, s_stride),
                    mode="trilinear",
                    align_corners=False,
                )
                self.upsample_conv = upsample_conv
            else:
                self.upsample      = None
                self.upsample_conv = None
            self.res_blocks = res_blocks
            self.chan_conv   = chan_conv

        def forward(self, x):
            if self.upsample is not None:
                x = self.upsample(x)
                x = self.upsample_conv(x)
            for blk in self.res_blocks:
                x = blk(x)
            x = self.chan_conv(x)
            return x

    return _DecoderStage()


# ---------------------------------------------------------------------------
# WanVideoVAE
# ---------------------------------------------------------------------------

class WanVideoVAE:
    """
    Wan 2.1 Video VAE — 3D causal convolutional encoder-decoder.

    Compression::

        Encoder: [B, 3, T, H, W] → [B, 16, T//4, H//8, W//8]
        Decoder: [B, 16, T, H, W] → [B, 3, T*4, H*8, W*8]  (clamped [0,1])

    Architecture::

        Encoder
          conv_in          :  3 → 128   (CausalConv3d 3×3×3, stride 1)
          stage_0          :  128 → 128, 2 ResBlocks, spatial ×2 down
          stage_1          :  128 → 256, 2 ResBlocks, temporal ×2 + spatial ×2 down
          stage_2          :  256 → 512, 2 ResBlocks, temporal ×2 + spatial ×2 down
          bottleneck_conv  :  512 → 512  (CausalConv3d, stride 1)
          mid_block        :  512 (ResBlock + Attn + ResBlock)
          norm_out         :  GroupNorm(32, 512)
          conv_out         :  Conv3d(512 → 32, 1×1×1)
          quant_conv       :  Conv3d(32  → 32, 1×1×1)
          → take first 16 channels as latent mean

        Decoder (reversed)
          post_quant_conv  :  Conv3d(16 → 16, 1×1×1)
          dec_conv_in      :  Conv3d(16 → 512, 1×1×1)
          mid_block        :  512 (same structure as encoder mid)
          stage_2          :  512 → 256, temporal ×2 + spatial ×2 up
          stage_1          :  256 → 128, temporal ×2 + spatial ×2 up
          stage_0          :  128 → 128, spatial ×2 up
          norm_out         :  GroupNorm(32, 128)
          conv_out         :  CausalConv3d(128 → 3, 3×3×3)

    Usage::

        vae = WanVideoVAE(device=torch.device("cuda:0"))
        vae.load(Path("~/.cache/musicvision/weights/humo/shared/vae/Wan2.1_VAE.pth"))
        latents = vae.encode(pixel_frames)   # [B, 16, T//4, H//8, W//8]
        frames  = vae.decode(latents)        # [B, 3, T, H, W] in [0, 1]
    """

    def __init__(self, device, dtype=None) -> None:
        """
        Args:
            device: Target torch.device (or str).
            dtype:  Weight dtype.  Defaults to torch.float16.
        """
        import torch

        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype  = dtype if dtype is not None else torch.float16
        self.encoder: "torch.nn.Module | None" = None
        self.decoder: "torch.nn.Module | None" = None
        self._built  = False

    # ------------------------------------------------------------------
    # Architecture construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Construct encoder and decoder nn.Module trees."""
        import torch.nn as nn
        import torch.nn.functional as F

        ch  = _BASE_CHANNELS                                   # 128
        cms = [ch * m for m in _CHANNEL_MULT]                  # [128,256,512,512]

        # ---- Encoder ----
        enc_conv_in  = _build_causal_conv3d(3, cms[0], 3, stride=1)
        enc_stage_0  = _build_encoder_stage(cms[0], cms[1], _NUM_RES_BLOCKS,
                                             _TEMPORAL_STRIDES[0], _SPATIAL_STRIDES[0])
        enc_stage_1  = _build_encoder_stage(cms[1], cms[2], _NUM_RES_BLOCKS,
                                             _TEMPORAL_STRIDES[1], _SPATIAL_STRIDES[1])
        enc_stage_2  = _build_encoder_stage(cms[2], cms[3], _NUM_RES_BLOCKS,
                                             _TEMPORAL_STRIDES[2], _SPATIAL_STRIDES[2])
        enc_mid      = _build_mid_block(cms[3])
        enc_norm_out = nn.GroupNorm(32, cms[3])
        enc_conv_out = nn.Conv3d(cms[3], 32, 1)   # 32 = 16 mean + 16 logvar
        quant_conv   = nn.Conv3d(32, 32, 1)

        enc_stages = nn.ModuleList([enc_stage_0, enc_stage_1, enc_stage_2])

        class _Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_in   = enc_conv_in
                self.stages    = enc_stages
                self.mid_block = enc_mid
                self.norm_out  = enc_norm_out
                self.conv_out  = enc_conv_out
                self.quant_conv = quant_conv

            def forward(self, x):
                x = self.conv_in(x)
                for stage in self.stages:
                    x = stage(x)
                x = self.mid_block(x)
                x = F.silu(self.norm_out(x))
                x = self.conv_out(x)
                x = self.quant_conv(x)
                return x  # [B, 32, T//4, H//8, W//8]

        # ---- Decoder ----
        post_quant = nn.Conv3d(_LATENT_CHANNELS, _LATENT_CHANNELS, 1)
        dec_conv_in = nn.Conv3d(_LATENT_CHANNELS, cms[3], 1)
        dec_mid     = _build_mid_block(cms[3])
        dec_stage_2 = _build_decoder_stage(cms[3], cms[2], _NUM_RES_BLOCKS,
                                            _TEMPORAL_STRIDES[2], _SPATIAL_STRIDES[2])
        dec_stage_1 = _build_decoder_stage(cms[2], cms[1], _NUM_RES_BLOCKS,
                                            _TEMPORAL_STRIDES[1], _SPATIAL_STRIDES[1])
        dec_stage_0 = _build_decoder_stage(cms[1], cms[0], _NUM_RES_BLOCKS,
                                            _TEMPORAL_STRIDES[0], _SPATIAL_STRIDES[0])
        dec_norm_out = nn.GroupNorm(32, cms[0])
        dec_conv_out = _build_causal_conv3d(cms[0], 3, 3, stride=1)

        dec_stages = nn.ModuleList([dec_stage_2, dec_stage_1, dec_stage_0])

        class _Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.post_quant = post_quant
                self.conv_in    = dec_conv_in
                self.mid_block  = dec_mid
                self.stages     = dec_stages
                self.norm_out   = dec_norm_out
                self.conv_out   = dec_conv_out

            def forward(self, z):
                z = self.post_quant(z)
                z = self.conv_in(z)
                z = self.mid_block(z)
                for stage in self.stages:
                    z = stage(z)
                z = F.silu(self.norm_out(z))
                z = self.conv_out(z)
                return z  # [B, 3, T*4, H*8, W*8]

        self.encoder = _Encoder()
        self.decoder = _Decoder()
        self._built  = True
        log.debug("WanVideoVAE architecture built.")

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, weights_path: Path) -> None:
        """
        Load VAE weights from a Wan2.1_VAE.pth checkpoint.

        Uses strict=False to tolerate minor key mismatches between the
        reference architecture and the actual checkpoint.

        Args:
            weights_path: Path to the .pth VAE weight file.
        """
        import torch

        if not self._built:
            self._build()

        log.info("Loading WanVideoVAE weights from %s …", weights_path)
        state = torch.load(weights_path, map_location="cpu", weights_only=False)

        if isinstance(state, dict):
            for wrapper in ("state_dict", "model_state_dict"):
                if wrapper in state:
                    state = state[wrapper]
                    log.debug("Unwrapped VAE checkpoint key: %s", wrapper)
                    break

        # Split into encoder / decoder sub-dicts
        enc_sd: dict = {}
        dec_sd: dict = {}
        for k, v in state.items():
            if k.startswith("encoder.") or k.startswith("quant_conv."):
                enc_sd[k.removeprefix("encoder.")] = v
            elif k.startswith("decoder.") or k.startswith("post_quant_conv."):
                dec_sd[k.removeprefix("decoder.")] = v
            else:
                # Unknown prefix — try loading into both with strict=False
                enc_sd[k] = v
                dec_sd[k] = v

        enc_missing, enc_unexp = self.encoder.load_state_dict(enc_sd, strict=False)
        dec_missing, dec_unexp = self.decoder.load_state_dict(dec_sd, strict=False)

        if enc_missing:
            log.warning("VAE encoder — %d missing keys (first 5: %s)", len(enc_missing), enc_missing[:5])
        if dec_missing:
            log.warning("VAE decoder — %d missing keys (first 5: %s)", len(dec_missing), dec_missing[:5])
        if enc_unexp:
            log.debug("VAE encoder — %d unexpected keys", len(enc_unexp))
        if dec_unexp:
            log.debug("VAE decoder — %d unexpected keys", len(dec_unexp))

        log.info("Moving WanVideoVAE to %s (dtype=%s) …", self.device, self.dtype)
        self.encoder = self.encoder.to(device=self.device, dtype=self.dtype).eval()
        self.decoder = self.decoder.to(device=self.device, dtype=self.dtype).eval()
        log.info("WanVideoVAE ready.")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, pixels: "torch.Tensor") -> "torch.Tensor":
        """
        Encode a pixel video tensor into VAE latents.

        Args:
            pixels: [B, 3, T, H, W] float tensor in [0, 1].

        Returns:
            Latent tensor [B, 16, T//4, H//8, W//8] on self.device.
        """
        import torch

        if self.encoder is None:
            raise RuntimeError("WanVideoVAE.load() must be called before encode().")

        # Normalize [0,1] → [-1, 1] to match VAE training
        x = pixels.to(device=self.device, dtype=self.dtype) * 2.0 - 1.0

        with torch.no_grad():
            out = self.encoder(x)   # [B, 32, T//4, H//8, W//8]

        # Take the first 16 channels as the latent mean (discard logvar)
        return out[:, :_LATENT_CHANNELS]

    def encode_image(self, image: "torch.Tensor") -> "torch.Tensor":
        """
        Encode a single reference image into a one-frame latent.

        Args:
            image: [B, 3, H, W] or [B, 3, 1, H, W] float tensor in [0, 1].

        Returns:
            Latent tensor [B, 16, 1, H//8, W//8].
        """
        import torch

        if image.ndim == 4:
            image = image.unsqueeze(2)   # → [B, 3, 1, H, W]
        return self.encode(image)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, latent: "torch.Tensor") -> "torch.Tensor":
        """
        Decode VAE latents to pixel frames.

        Args:
            latent: [B, 16, T, H, W] latent tensor.

        Returns:
            Pixel tensor [B, 3, T*4, H*8, W*8] clamped to [0, 1].
        """
        import torch

        if self.decoder is None:
            raise RuntimeError("WanVideoVAE.load() must be called before decode().")

        z = latent.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            out = self.decoder(z)

        # Map from [-1, 1] back to [0, 1] and clamp to valid range
        return torch.clamp((out + 1.0) / 2.0, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Release encoder and decoder from VRAM."""
        import gc
        import torch

        self.encoder = None
        self.decoder = None
        self._built  = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("WanVideoVAE unloaded.")
