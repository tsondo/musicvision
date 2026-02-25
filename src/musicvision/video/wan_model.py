"""
WanModel — self-contained DiT architecture for HuMo TIA video generation.

Ported from kijai/ComfyUI-WanVideoWrapper (proven on consumer GPUs with
block swap + FP8), cross-referenced with Wan-AI/Wan2.1 source.

No ComfyUI dependencies.  Pure PyTorch.

Architecture overview
---------------------
WanModel is a 3D video DiT (Diffusion Transformer) that conditions jointly on:
  - text embeddings    (T5-XXL, 4096-dim)
  - image latent       (reference frame for I2V / TIA mode)
  - audio embeddings   (Whisper encoder output, projected by AudioProjModel)
  - timestep           (flow-matching sigma, encoded as sinusoidal + MLP)

Key design decisions (following Wan-AI/kijai):
  - 3D factorized RoPE: separate temporal / H / W frequency bands
  - Per-block AdaLN-Zero modulation: each block projects time_emb → 6*dim
    (shift/scale/gate for self-attn + FFN) via its own adaLN_modulation layer
  - Gated cross-attention for text (WanI2VCrossAttention, gate init=0)
  - Gated per-frame audio cross-attention (AudioCrossAttentionWrapper, gate init=0)
  - Gated FFN (SwiGLU-like with a third projection)
  - Patch embed: Conv3d with spatial stride=2 (temporal stride=1)

Dimension conventions
---------------------
  B  = batch size
  F  = number of video frames (temporal)
  H,W = spatial height / width (at model resolution, e.g. 480 or 720 px)
  h,w = H//2, W//2 after patch embed (spatial stride=2)
  L  = sequence length = F * h * w
  D  = model hidden dim (5120 for 14B, 1536 for 1.7B)
  Nh = number of heads
  Hd = head dim = D // Nh
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int = 256) -> torch.Tensor:
    """
    Standard diffusion sinusoidal timestep embedding.

    Args:
        timesteps: [B] float tensor of timestep values (flow-matching sigmas).
        dim:       Embedding dimension.  Must be even.

    Returns:
        [B, dim] float32 tensor.
    """
    assert dim % 2 == 0, f"sinusoidal_embedding requires even dim, got {dim}"
    device = timesteps.device

    half = dim // 2
    # Geometric progression of frequencies: omega_i = 1 / 10000^(i / (half-1))
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=device) / (half - 1)
    )                                              # [half]
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


# ---------------------------------------------------------------------------
# 3D Factorized Rotary Position Embedding
# ---------------------------------------------------------------------------

class WanRoPE:
    """
    3D factorized rotary position embeddings for video DiTs.

    The head dimension is split into three bands:
      - Temporal  (25%  of Hd): encodes frame index
      - Height    (37.5% of Hd): encodes vertical position
      - Width     (37.5% of Hd): encodes horizontal position

    All methods are static — WanRoPE is used as a namespace, not instantiated.

    References
    ----------
    - Wan-AI/Wan2.1: wan/modules/model.py — rope_apply / rope_params
    - kijai/ComfyUI-WanVideoWrapper: wan_video_nodes.py — get_3d_rope_freqs
    """

    @staticmethod
    def precompute_freqs(
        dim: int,
        max_positions: int = 65536,
        base: float = 10000.0,
    ) -> torch.Tensor:
        """
        Compute 1D RoPE frequencies for a single axis.

        Args:
            dim:           Full dimension for this axis (uses dim//2 frequency pairs).
            max_positions: Maximum sequence length to support.
            base:          RoPE base frequency.

        Returns:
            [max_positions, dim//2] float32 angles.
        """
        half = dim // 2
        freqs = 1.0 / (
            base ** (torch.arange(0, half * 2, 2, dtype=torch.float32) / (half * 2))
        )                                                  # [half]
        positions = torch.arange(max_positions, dtype=torch.float32)  # [T]
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)           # [T, half]
        return angles                                                   # [T, half]

    @staticmethod
    def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary position embeddings to queries or keys.

        Args:
            x:     [B, Nh, L, Hd] — the tensor to rotate.
            freqs: [L, Hd//2]     — precomputed angles for the sequence.

        Returns:
            Rotated tensor with same shape as x.
        """
        xshape = x.shape
        # Reshape last dim into pairs: [..., Hd] -> [..., Hd//2, 2]
        x_ = x.float().reshape(*xshape[:-1], -1, 2)   # [B, Nh, L, Hd//2, 2]

        cos_v = torch.cos(freqs)   # [L, Hd//2]
        sin_v = torch.sin(freqs)   # [L, Hd//2]

        # Broadcast to [1, 1, L, Hd//2] for [B, Nh, L, Hd//2, 2] inputs
        cos_v = cos_v.unsqueeze(0).unsqueeze(0)   # [1, 1, L, Hd//2]
        sin_v = sin_v.unsqueeze(0).unsqueeze(0)   # [1, 1, L, Hd//2]

        x0 = x_[..., 0]   # [B, Nh, L, Hd//2]
        x1 = x_[..., 1]   # [B, Nh, L, Hd//2]

        rotated = torch.stack(
            [x0 * cos_v - x1 * sin_v,
             x0 * sin_v + x1 * cos_v],
            dim=-1,
        )  # [B, Nh, L, Hd//2, 2]

        return rotated.reshape(xshape).to(x.dtype)

    @staticmethod
    def get_3d_freqs(
        head_dim: int = None,
        F: int = None,
        H: int = None,
        W: int = None,
        base: float = 10000.0,
        *,
        dim: int = None,
    ) -> torch.Tensor:
        """
        Build 3D factorized RoPE frequency tensor for a video sequence.

        The head dimension is partitioned into three bands:
          - n_t = head_dim // 4 // 2          (temporal pairs)
          - n_h = head_dim * 3 // 8 // 2      (height pairs)
          - n_w = head_dim // 2 - n_t - n_h   (width pairs, remainder)

        Each band uses its own set of frequencies evaluated at position indices.

        Args:
            head_dim: Per-head dimension (D // num_heads). Must be divisible by 2.
            F:        Number of frames.
            H:        Spatial height (after patch embed).
            W:        Spatial width  (after patch embed).
            base:     RoPE base.

        Returns:
            [F*H*W, head_dim//2] float32 tensor of concatenated 3D angles.
        """
        # Allow 'dim' as an alias for 'head_dim' (test convenience)
        if head_dim is None and dim is not None:
            head_dim = dim
        if head_dim is None:
            raise TypeError("get_3d_freqs() requires 'head_dim' (or 'dim') argument")
        half_hd = head_dim // 2

        # Proportion of freq pairs per axis: 25% temporal, 37.5% H, 37.5% W
        n_t = max(1, head_dim // 4 // 2)
        n_h = max(1, head_dim * 3 // 8 // 2)
        n_w = half_hd - n_t - n_h   # remainder (may be 0 for small head_dims)
        if n_w < 1:
            # Fallback: distribute evenly across three axes
            n_t = half_hd // 3
            n_h = half_hd // 3
            n_w = half_hd - n_t - n_h

        # Compute 1D angles for each axis using full dim (2*n_X) to get n_X pairs
        freqs_t = WanRoPE.precompute_freqs(n_t * 2, max_positions=max(F, 1), base=base)
        # freqs_t: [F, n_t]
        freqs_h = WanRoPE.precompute_freqs(n_h * 2, max_positions=max(H, 1), base=base)
        # freqs_h: [H, n_h]
        freqs_w = WanRoPE.precompute_freqs(n_w * 2, max_positions=max(W, 1), base=base)
        # freqs_w: [W, n_w]

        # Broadcast to [F, H, W, n_X] and concatenate
        f_idx = freqs_t[:, None, None, :].expand(F, H, W, n_t)   # [F, H, W, n_t]
        h_idx = freqs_h[None, :, None, :].expand(F, H, W, n_h)   # [F, H, W, n_h]
        w_idx = freqs_w[None, None, :, :].expand(F, H, W, n_w)   # [F, H, W, n_w]

        freqs_3d = torch.cat(
            [f_idx, h_idx, w_idx], dim=-1
        ).reshape(F * H * W, half_hd)                             # [F*H*W, head_dim//2]

        return freqs_3d


# ---------------------------------------------------------------------------
# Self-Attention with 3D RoPE
# ---------------------------------------------------------------------------

class WanSelfAttention(nn.Module):
    """
    Multi-head self-attention with 3D factorized rotary embeddings.

    RoPE is applied separately to queries and keys (not to values).
    Uses PyTorch's F.scaled_dot_product_attention for efficient FlashAttention
    when available.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj   = nn.Linear(dim, dim, bias=True)
        self.k_proj   = nn.Linear(dim, dim, bias=True)
        self.v_proj   = nn.Linear(dim, dim, bias=True)
        self.out_proj  = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     [B, L, D] hidden states.
            freqs: [L, head_dim//2] 3D RoPE angles.

        Returns:
            [B, L, D] after self-attention + output projection.
        """
        B, L, D = x.shape
        Nh, Hd = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(B, L, Nh, Hd).transpose(1, 2)   # [B, Nh, L, Hd]
        k = self.k_proj(x).reshape(B, L, Nh, Hd).transpose(1, 2)   # [B, Nh, L, Hd]
        v = self.v_proj(x).reshape(B, L, Nh, Hd).transpose(1, 2)   # [B, Nh, L, Hd]

        # Apply 3D RoPE to queries and keys
        q = WanRoPE.apply_rotary(q, freqs)   # [B, Nh, L, Hd]
        k = WanRoPE.apply_rotary(k, freqs)   # [B, Nh, L, Hd]

        # Scaled dot-product attention (uses FlashAttention kernel when available)
        attn_out = F.scaled_dot_product_attention(q, k, v)   # [B, Nh, L, Hd]

        # Merge heads and project
        attn_out = attn_out.transpose(1, 2).reshape(B, L, D)   # [B, L, D]
        return self.out_proj(attn_out)


# ---------------------------------------------------------------------------
# Gated Cross-Attention for Text Conditioning (I2V / TIA)
# ---------------------------------------------------------------------------

class WanI2VCrossAttention(nn.Module):
    """
    Gated cross-attention that injects text conditioning into video hidden states.

    The gate parameter is initialised to zero so that, at the start of training,
    the cross-attention has no effect (residual stream is unchanged).  The gate
    is learned and gradually opens as training progresses.

    This is the standard Flamingo-style gated cross-attention mechanism,
    adapted here for the Wan video DiT.
    """

    def __init__(self, dim: int, text_dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.q_proj   = nn.Linear(dim,      dim, bias=True)   # from video hidden states
        self.k_proj   = nn.Linear(text_dim, dim, bias=True)   # from text context
        self.v_proj   = nn.Linear(text_dim, dim, bias=True)   # from text context
        self.out_proj  = nn.Linear(dim,      dim, bias=True)

        # Gating: tanh(gate) in (-1, 1); starts at 0 (tanh(0) = 0)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       [B, L_x, D]         video hidden states (queries).
            context: [B, L_ctx, text_dim] text embeddings (keys + values).

        Returns:
            [B, L_x, D] — x + tanh(gate) * cross_attn_output.
        """
        B, L_x, D   = x.shape
        _, L_c, _   = context.shape
        Nh, Hd      = self.num_heads, self.head_dim

        q = self.q_proj(x      ).reshape(B, L_x, Nh, Hd).transpose(1, 2)  # [B, Nh, L_x, Hd]
        k = self.k_proj(context).reshape(B, L_c, Nh, Hd).transpose(1, 2)  # [B, Nh, L_c, Hd]
        v = self.v_proj(context).reshape(B, L_c, Nh, Hd).transpose(1, 2)  # [B, Nh, L_c, Hd]

        attn_out = F.scaled_dot_product_attention(q, k, v)    # [B, Nh, L_x, Hd]
        attn_out = attn_out.transpose(1, 2).reshape(B, L_x, D)
        attn_out = self.out_proj(attn_out)

        return x + torch.tanh(self.gate) * attn_out


# ---------------------------------------------------------------------------
# Per-Frame Audio Cross-Attention
# ---------------------------------------------------------------------------

class AudioCrossAttentionWrapper(nn.Module):
    """
    Per-frame cross-attention that injects audio conditioning into video tokens.

    Unlike text cross-attention (which operates globally over the full sequence),
    audio cross-attention is applied independently per frame: each frame's spatial
    tokens attend to the audio context tokens for that specific frame.

    This per-frame design matches the TIA architecture from HuMo, where each
    video frame has its own 16-token audio context derived from the Whisper
    encoder output projected through AudioProjModel.

    Architecture:
        queries = spatial tokens of frame f       [B*F, H*W, D]
        keys/values = audio tokens for frame f    [B*F, 16, D]

    Gate initialised to zero (same as text cross-attention).
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads

        self.norm    = nn.LayerNorm(dim)
        self.q_proj  = nn.Linear(dim, dim, bias=True)
        self.k_proj  = nn.Linear(dim, dim, bias=True)
        self.v_proj  = nn.Linear(dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x: torch.Tensor,
        audio_proj_out: torch.Tensor,
        F_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, F*H*W, D] full video sequence.
            audio_proj_out: [B, F, 16, D] projected audio context tokens.
            F_frames:       Number of frames F (needed to reshape).

        Returns:
            [B, F*H*W, D] — x + tanh(gate) * per-frame audio attention output.
        """
        original_x = x
        B, L, D = x.shape
        HW = L // F_frames   # tokens per frame

        Nh, Hd = self.num_heads, self.head_dim

        # Reshape video tokens to per-frame: [B*F, HW, D]
        x_frame = x.reshape(B * F_frames, HW, D)

        # Reshape audio tokens to per-frame: [B*F, 16, D]
        audio_frame = audio_proj_out.reshape(B * F_frames, 16, D)

        # Apply LayerNorm to queries
        x_normed = self.norm(x_frame)   # [B*F, HW, D]

        q = self.q_proj(x_normed   ).reshape(B * F_frames, HW, Nh, Hd).transpose(1, 2)
        k = self.k_proj(audio_frame).reshape(B * F_frames, 16, Nh, Hd).transpose(1, 2)
        v = self.v_proj(audio_frame).reshape(B * F_frames, 16, Nh, Hd).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)   # [B*F, Nh, HW, Hd]
        attn_out = attn_out.transpose(1, 2).reshape(B * F_frames, HW, D)
        attn_out = self.out_proj(attn_out)

        # Reshape back to full sequence: [B, F*HW, D]
        attn_out = attn_out.reshape(B, L, D)

        return original_x + torch.tanh(self.gate) * attn_out


# ---------------------------------------------------------------------------
# Gated Feed-Forward Network (SwiGLU variant)
# ---------------------------------------------------------------------------

class WanFFN(nn.Module):
    """
    Gated FFN with SwiGLU-style activation.

    Forward:
        out = fc3(silu(fc1(x)) * fc2(x))

    fc1 and fc2 are both dim -> ffn_dim (gate and value projections).
    fc3 is ffn_dim -> dim (output projection).

    This is equivalent to SwiGLU but with an explicit third matrix (fc3)
    rather than using fc2 as the output.  The Wan codebase uses this 3-matrix
    form consistently.
    """

    def __init__(self, dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim,     ffn_dim, bias=True)   # gate projection
        self.fc2 = nn.Linear(dim,     ffn_dim, bias=True)   # value projection
        self.fc3 = nn.Linear(ffn_dim, dim,     bias=True)   # output projection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))


# ---------------------------------------------------------------------------
# Transformer Block with AdaLN-Zero Modulation
# ---------------------------------------------------------------------------

class WanAttentionBlock(nn.Module):
    """
    Single transformer block for the WanModel DiT.

    Conditioning mechanism: AdaLN-Zero
    ----------------------------------
    Each block projects time_emb [B, D] into 6 modulation parameters via its own
    adaLN_modulation layer (nn.Linear(dim, 6*dim)):
        [shift1, scale1, gate1, shift2, scale2, gate2]
    Applied as:
        self-attn   : norm1(x) * (1 + scale1) + shift1
        FFN         : norm3(x) * (1 + scale2) + shift2
        gates       : x += gate1 * self_attn(...)
                      x += gate2 * ffn(...)

    Cross-attentions (text, audio) use their own gating and are inserted
    between self-attn and FFN.

    All LayerNorms use elementwise_affine=False because the affine transform
    is replaced by the AdaLN scale/shift from the timestep conditioning.
    """

    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_heads: int,
        ffn_dim: int,
        humo_audio: bool = True,
    ) -> None:
        super().__init__()
        self.humo_audio = humo_audio

        # Pre-norms (no learnable affine — replaced by AdaLN modulation)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # Attention sub-modules
        self.self_attn  = WanSelfAttention(dim, num_heads)
        self.cross_attn = WanI2VCrossAttention(dim, text_dim, num_heads)
        self.audio_attn: Optional[AudioCrossAttentionWrapper] = (
            AudioCrossAttentionWrapper(dim, num_heads) if humo_audio else None
        )

        # FFN
        self.ffn = WanFFN(dim, ffn_dim)

        # AdaLN modulation: time_emb [B, D] -> 6 modulation scalars [B, 6*D]
        # broadcast over sequence length inside forward()
        self.adaLN_modulation = nn.Linear(dim, 6 * dim, bias=True)
        # AdaLN-Zero: init output to zero so gates start at 0
        nn.init.zeros_(self.adaLN_modulation.weight)
        nn.init.zeros_(self.adaLN_modulation.bias)

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_proj_out: Optional[torch.Tensor],
        freqs: torch.Tensor,
        F_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, L, D]         video hidden states.
            time_emb:       [B, D]            timestep embedding (raw, from WanModel.time_embed).
            text_embeds:    [B, L_ctx, D]     projected text context.
            audio_proj_out: [B, F, 16, D]     projected audio context, or None.
            freqs:          [L, head_dim//2]  3D RoPE frequencies.
            F_frames:       int               number of frames (for per-frame audio attn).

        Returns:
            [B, L, D] updated hidden states.
        """
        # AdaLN modulation from timestep embedding: [B, D] -> [B, 6*D]
        mod = self.adaLN_modulation(time_emb)
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Unsqueeze for broadcast over sequence length L: [B, D] -> [B, 1, D]
        shift1  = shift1.unsqueeze(1)
        scale1  = scale1.unsqueeze(1)
        gate1   = gate1.unsqueeze(1)
        shift2  = shift2.unsqueeze(1)
        scale2  = scale2.unsqueeze(1)
        gate2   = gate2.unsqueeze(1)

        # --- Self-attention with AdaLN-Zero pre-norm ---
        x_sa = self.norm1(x) * (1.0 + scale1) + shift1
        x = x + gate1 * self.self_attn(x_sa, freqs)

        # --- Text cross-attention (gated, with its own pre-norm via norm2) ---
        # cross_attn already applies the residual internally (x + gate * attn_out)
        x = self.cross_attn(self.norm2(x), text_embeds)

        # --- Audio cross-attention (per-frame, gated) ---
        if self.humo_audio and self.audio_attn is not None and audio_proj_out is not None:
            # audio_attn applies its own norm and residual internally
            x = self.audio_attn(x, audio_proj_out, F_frames)

        # --- FFN with AdaLN-Zero pre-norm ---
        x_ffn = self.norm3(x) * (1.0 + scale2) + shift2
        x = x + gate2 * self.ffn(x_ffn)

        return x


# ---------------------------------------------------------------------------
# Audio Projection Model
# ---------------------------------------------------------------------------

class AudioProjModel(nn.Module):
    """
    Projects Whisper encoder features to the DiT audio conditioning tokens.

    Input:  [B, F, seq_len, bands, channels]  — per-frame Whisper features
    Output: [B, F, context_tokens, output_dim] — audio context for cross-attention

    The input is first flattened across seq_len x bands x channels dimensions,
    then processed through a 3-layer MLP with ReLU activations, and finally
    reshaped to produce context_tokens separate tokens per frame.

    Default dimensions match HuMo's Whisper integration:
        seq_len       = 8      (temporal window within a frame)
        bands         = 5      (log-mel bands grouping)
        channels      = 1280   (Whisper-large hidden size)
        flat input    = 8*5*1280 = 51200
        intermediate  = 512
        output_dim    = 5120   (matches 14B DiT hidden dim)
        context_tokens = 16   (audio tokens per frame)
    """

    def __init__(
        self,
        seq_len: int = 8,
        bands: int = 5,
        channels: int = 1280,
        intermediate_dim: int = 512,
        output_dim: int = 5120,
        context_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.seq_len       = seq_len
        self.bands         = bands
        self.channels      = channels
        self.context_tokens = context_tokens
        self.output_dim    = output_dim

        flat_in = seq_len * bands * channels   # 8 * 5 * 1280 = 51200

        self.fc1  = nn.Linear(flat_in,          intermediate_dim)
        self.fc2  = nn.Linear(intermediate_dim, intermediate_dim)
        self.fc3  = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_features: [B, F, seq_len, bands, channels]

        Returns:
            [B, F, context_tokens, output_dim] — normalised audio context tokens.
        """
        B, Frm, S, Ba, C = audio_features.shape

        # Flatten and project
        x = audio_features.reshape(B * Frm, S * Ba * C)   # [B*F, flat_in]
        x = F.relu(self.fc1(x))                             # [B*F, intermediate_dim]
        x = F.relu(self.fc2(x))                             # [B*F, intermediate_dim]
        x = self.fc3(x)                                     # [B*F, context_tokens * output_dim]

        # Reshape to token sequence
        x = x.reshape(B, Frm, self.context_tokens, self.output_dim)  # [B, F, 16, output_dim]

        return self.norm(x)


# ---------------------------------------------------------------------------
# WanModel — Main DiT
# ---------------------------------------------------------------------------

class WanModel(nn.Module):
    """
    WanModel: 3D video Diffusion Transformer for HuMo TIA generation.

    Conditions jointly on timestep, text, and (optionally) audio.
    Supports both 14B and 1.7B architecture configs via class-level presets.

    Patch embed  : Conv3d(in_dim, dim, kernel=(1,2,2), stride=(1,2,2))
                   spatial downsampling x2, temporal stride=1
    Block count  : num_layers WanAttentionBlock, each with its own adaLN_modulation
    Head         : AdaLN (from head_modulation) + linear + unpatchify

    The output latent has the same spatial resolution as the input (H x W)
    because unpatchify reverses the x2 spatial downsampling from the patch embed.
    """

    # Architecture presets ------------------------------------------------
    CONFIG_14B = dict(
        dim=5120, num_heads=40, num_layers=40, text_dim=4096, ffn_dim=13824
    )
    CONFIG_1_7B = dict(
        dim=1536, num_heads=12, num_layers=30, text_dim=4096, ffn_dim=8960
    )
    # ---------------------------------------------------------------------

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        text_dim: int,
        ffn_dim: int,
        in_dim: int = 36,
        out_dim: int = 16,
        humo_audio: bool = True,
    ) -> None:
        super().__init__()
        self.dim        = dim
        self.num_heads  = num_heads
        self.num_layers = num_layers
        self.out_dim    = out_dim

        # Patch embedding: [B, in_dim, F, H, W] -> [B, dim, F, H//2, W//2]
        self.patch_embed = nn.Conv3d(
            in_dim, dim,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2),
        )

        # Text conditioning MLP: [B, L_ctx, text_dim] -> [B, L_ctx, dim]
        self.text_embed = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Timestep embedding: sinusoidal(256) -> [B, dim]
        self.time_embed = nn.Sequential(
            nn.Linear(256, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        # Transformer blocks — each block has its own adaLN_modulation(dim -> 6*dim)
        # that projects the raw time_emb [B, D] to per-block modulation scalars.
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                dim=dim,
                text_dim=dim,       # text is projected to dim before passing to blocks
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                humo_audio=humo_audio,
            )
            for _ in range(num_layers)
        ])

        # Audio projection (projects Whisper features -> audio context tokens)
        self.audio_proj: Optional[AudioProjModel] = (
            AudioProjModel(output_dim=dim) if humo_audio else None
        )

        # Output head
        patch_out_dim = out_dim * 1 * 2 * 2   # patch_size product = 1*2*2 = 4
        self.head_norm       = nn.LayerNorm(dim, eps=1e-6)
        self.head_proj       = nn.Linear(dim, patch_out_dim, bias=True)

        # Head AdaLN: time_emb [B, D] -> scale + shift [B, 2*D]
        self.head_modulation = nn.Linear(dim, 2 * dim, bias=True)
        # AdaLN-Zero init
        nn.init.zeros_(self.head_modulation.weight)
        nn.init.zeros_(self.head_modulation.bias)

    # ------------------------------------------------------------------
    # Classmethod factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_name: str, device: str = "meta", **kwargs) -> "WanModel":
        """
        Construct a WanModel from a named architecture preset.

        By default uses device="meta" so that no actual tensor memory is
        allocated.  This lets callers inspect architecture attributes (e.g.
        model.dim) without loading the full weights.  Pass
        device="cpu" or device="cuda" to materialise a real model.

        Args:
            config_name: "14B" or "1_7B"
            device:      PyTorch device string.  Defaults to "meta" (no alloc).
            **kwargs:    Override any field from the preset.

        Returns:
            WanModel instance (on meta device unless overridden).
        """
        presets = {
            "14B":  cls.CONFIG_14B,
            "1_7B": cls.CONFIG_1_7B,
        }
        if config_name not in presets:
            raise ValueError(
                f"Unknown config '{config_name}'. Available: {list(presets.keys())}"
            )
        cfg = {**presets[config_name], **kwargs}
        with torch.device(device):
            return cls(**cfg)

    # ------------------------------------------------------------------
    # Pre/Post block helpers (used by BlockSwapManager integration)
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

        Returns a tuple that is consumed by each block (for block swap) and
        then by post_blocks().

        Returns:
            (x_seq, time_emb, text_ctx, audio_proj_out, freqs, F, h, w)

            x_seq:          [B, F*h*w, D]  patchified + flattened video tokens.
            time_emb:       [B, D]         timestep embedding; passed raw to each block.
                                           Each block projects it to 6*D internally via
                                           its own adaLN_modulation layer.
            text_ctx:       [B, L_ctx, D]  projected text context.
            audio_proj_out: [B, F, 16, D]  projected audio, or None.
            freqs:          [F*h*w, Hd//2] 3D RoPE frequencies.
            F:              int  number of frames.
            h, w:           int  spatial dims after patch embed (H//2, W//2).
        """
        # 1. Timestep embedding: [B] -> [B, D]
        time_emb = self.time_embed(
            sinusoidal_embedding(timestep, dim=256).to(x.dtype)
        )   # [B, D]

        # 2. Project text embeddings to model dim: [B, L_ctx, text_dim] -> [B, L_ctx, D]
        text_ctx = self.text_embed(text_embeds)

        # 3. Project audio features if present: [B, F, 8, 5, 1280] -> [B, F, 16, D]
        audio_proj_out: Optional[torch.Tensor] = None
        if audio_features is not None and self.audio_proj is not None:
            audio_proj_out = self.audio_proj(audio_features)

        # 4. Patch embed: [B, in_dim, F, H, W] -> [B, D, F, h, w]
        x_patch = self.patch_embed(x)
        _, _, F, h, w = x_patch.shape

        # 5. Flatten spatial + temporal into sequence: [B, D, F, h, w] -> [B, F*h*w, D]
        x_seq = x_patch.flatten(2).transpose(1, 2)

        # 6. Compute 3D RoPE frequencies: [F*h*w, head_dim//2]
        head_dim = self.dim // self.num_heads
        freqs = WanRoPE.get_3d_freqs(head_dim, F, h, w).to(x.device)

        return x_seq, time_emb, text_ctx, audio_proj_out, freqs, F, h, w

    def post_blocks(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        F: int,
        h: int,
        w: int,
        out_dim: int = 16,
    ) -> torch.Tensor:
        """
        Apply head AdaLN, head projection, and unpatchify.

        Args:
            x:        [B, F*h*w, D]  output of the final transformer block.
            time_emb: [B, D]         raw timestep embedding (from pre_blocks).
            F, h, w:  spatial/temporal dimensions (from pre_blocks).
            out_dim:  output latent channels (default 16).

        Returns:
            [B, out_dim, F, H, W]  video latent at input spatial resolution (H=h*2, W=w*2).
        """
        # Head AdaLN: scale + shift from raw timestep embedding
        head_mod = self.head_modulation(time_emb)          # [B, 2*D]
        scale, shift = head_mod.chunk(2, dim=-1)           # each [B, D]
        scale = scale.unsqueeze(1)                          # [B, 1, D]
        shift = shift.unsqueeze(1)                          # [B, 1, D]

        # Apply AdaLN and head projection
        x = self.head_norm(x) * (1.0 + scale) + shift     # [B, L, D]
        x = self.head_proj(x)                              # [B, L, out_dim*4]

        # Unpatchify: recover original spatial resolution
        return self._unpatchify(x, F, h, w, out_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full forward pass through the WanModel DiT.

        Args:
            x:              [B, in_dim, F, H, W]      noisy video latent.
            timestep:       [B]                        float32 timesteps (flow sigma).
            text_embeds:    [B, seq_len, text_dim]     T5 text embeddings.
            audio_features: [B, F, 8, 5, 1280] or None Whisper encoder features.

        Returns:
            [B, out_dim, F, H, W]  predicted velocity field (v-prediction).
        """
        x_seq, time_emb, text_ctx, audio_proj_out, freqs, F, h, w = (
            self.pre_blocks(x, timestep, text_embeds, audio_features)
        )

        # Run transformer blocks — each block projects time_emb to 6*D internally
        for block in self.blocks:
            x_seq = block(
                x=x_seq,
                time_emb=time_emb,        # [B, D] — each block has its own adaLN_modulation
                text_embeds=text_ctx,
                audio_proj_out=audio_proj_out,
                freqs=freqs,
                F_frames=F,
            )

        return self.post_blocks(x_seq, time_emb, F, h, w, out_dim=self.out_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unpatchify(
        self,
        x: torch.Tensor,
        F: int,
        h: int,
        w: int,
        out_dim: int,
    ) -> torch.Tensor:
        """
        Reverse the patch embed operation.

        The patch embed had kernel/stride (1, 2, 2), so each output token
        corresponds to a 1x2x2 patch.  Unpatchify reconstructs the original
        spatial resolution by rearranging the patch sub-pixels back into place.

        Args:
            x:       [B, F*h*w, out_dim*4]  (patch_size product = 1*2*2 = 4)
            F:       number of frames
            h, w:    spatial dims after patch embed (H//2, W//2)
            out_dim: output channels (16 for VAE latent space)

        Returns:
            [B, out_dim, F, H, W]  where H = h*2, W = w*2.
        """
        B = x.shape[0]

        # Reshape to separate the patch sub-pixels
        # [B, F*h*w, out_dim*4] -> [B, F, h, w, out_dim, 2, 2]
        # (temporal patch size is 1, so we do not include it explicitly)
        x = x.reshape(B, F, h, w, out_dim, 2, 2)

        # Rearrange axes: channel first, then interleave spatial patch dims
        # [B, F, h, w, out_dim, 2, 2] -> [B, out_dim, F, h, 2, w, 2]
        x = x.permute(0, 4, 1, 2, 5, 3, 6).contiguous()

        # Merge (h, 2) -> H and (w, 2) -> W
        # [B, out_dim, F, h, 2, w, 2] -> [B, out_dim, F, h*2, w*2]
        x = x.reshape(B, out_dim, F, h * 2, w * 2)

        return x

    @staticmethod
    def _get_hw_from_x(x_patch: torch.Tensor):
        """
        Extract (F, h, w) from a patchified tensor.

        Args:
            x_patch: [B, dim, F, h, w] output of patch_embed.

        Returns:
            (F, h, w) as ints.
        """
        _, _, F, h, w = x_patch.shape
        return F, h, w

# ---------------------------------------------------------------------------
# Module-level config aliases (mirrors WanModel.CONFIG_* class attributes)
# ---------------------------------------------------------------------------

CONFIG_14B  = WanModel.CONFIG_14B
CONFIG_1_7B = WanModel.CONFIG_1_7B
