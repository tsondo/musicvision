"""
Vendored HuMo DiT architecture from Phantom-video/HuMo.

Sources:
  - humo/models/wan_modules/model_humo.py (DiT model + blocks)
  - humo/models/wan_modules/attention.py  (flash attention / SDPA fallback)
  - humo/models/audio/audio_proj.py       (AudioProjModel + DummyAdapterLayer)

Adaptations for self-contained use:
  - Flash attention optional; falls back to F.scaled_dot_product_attention
  - No einops dependency (manual reshape/permute)
  - No diffusers, common.distributed, or Ulysses SP dependencies
  - All state_dict keys match upstream exactly for checkpoint compatibility
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.amp as _amp

# Compat shim: upstream uses torch.cuda.amp.autocast; we use torch.amp equivalents
class _AmpCompat:
    """Namespace matching upstream's torch.cuda.amp usage."""
    @staticmethod
    def autocast(enabled=True, dtype=None):
        if dtype is not None:
            return torch.amp.autocast('cuda', enabled=enabled, dtype=dtype)
        return torch.amp.autocast('cuda', enabled=enabled)

amp = _AmpCompat
import torch.nn as nn
import torch.nn.functional as Fn

# ---------------------------------------------------------------------------
# Optional flash attention
# ---------------------------------------------------------------------------

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False


def flash_attention(
    q, k, v,
    q_lens=None, k_lens=None,
    dropout_p=0., softmax_scale=None, q_scale=None,
    causal=False, window_size=(-1, -1),
    deterministic=False, dtype=torch.bfloat16, version=None,
):
    """
    Attention with flash-attn when available, SDPA fallback otherwise.

    q: [B, Lq, Nq, C1], k: [B, Lk, Nk, C1], v: [B, Lk, Nk, C2].
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.size(-1) <= 256

    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # ---------- Flash Attention path ----------
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        # preprocess query
        if q_lens is None:
            q_fa = half(q.flatten(0, 1))
            q_lens_fa = torch.tensor(
                [lq] * b, dtype=torch.int32).to(device=q.device, non_blocking=True)
        else:
            q_fa = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
            q_lens_fa = q_lens

        # preprocess key, value
        if k_lens is None:
            k_fa = half(k.flatten(0, 1))
            v_fa = half(v.flatten(0, 1))
            k_lens_fa = torch.tensor(
                [lk] * b, dtype=torch.int32).to(device=k.device, non_blocking=True)
        else:
            k_fa = half(torch.cat([u[:vl] for u, vl in zip(k, k_lens)]))
            v_fa = half(torch.cat([u[:vl] for u, vl in zip(v, k_lens)]))
            k_lens_fa = k_lens

        q_fa = q_fa.to(v_fa.dtype)
        k_fa = k_fa.to(v_fa.dtype)
        if q_scale is not None:
            q_fa = q_fa * q_scale

        if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
            warnings.warn('Flash attention 3 not available, using FA2.')

        if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
            x = flash_attn_interface.flash_attn_varlen_func(
                q=q_fa, k=k_fa, v=v_fa,
                cu_seqlens_q=torch.cat([q_lens_fa.new_zeros([1]), q_lens_fa]).cumsum(
                    0, dtype=torch.int32).to(q_fa.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens_fa.new_zeros([1]), k_lens_fa]).cumsum(
                    0, dtype=torch.int32).to(q_fa.device, non_blocking=True),
                seqused_q=None, seqused_k=None,
                max_seqlen_q=lq, max_seqlen_k=lk,
                softmax_scale=softmax_scale,
                causal=causal,
                deterministic=deterministic)[0].unflatten(0, (b, lq))
        else:
            assert FLASH_ATTN_2_AVAILABLE
            x = flash_attn.flash_attn_varlen_func(
                q=q_fa, k=k_fa, v=v_fa,
                cu_seqlens_q=torch.cat([q_lens_fa.new_zeros([1]), q_lens_fa]).cumsum(
                    0, dtype=torch.int32).to(q_fa.device, non_blocking=True),
                cu_seqlens_k=torch.cat([k_lens_fa.new_zeros([1]), k_lens_fa]).cumsum(
                    0, dtype=torch.int32).to(q_fa.device, non_blocking=True),
                max_seqlen_q=lq, max_seqlen_k=lk,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                deterministic=deterministic).unflatten(0, (b, lq))

        return x.type(out_dtype)

    # ---------- SDPA fallback ----------
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask disabled in SDPA fallback. May affect performance.')

    q_s = q.transpose(1, 2).to(dtype)   # [B, N, Lq, C]
    k_s = k.transpose(1, 2).to(dtype)
    v_s = v.transpose(1, 2).to(dtype)
    if q_scale is not None:
        q_s = q_s * q_scale
    out = Fn.scaled_dot_product_attention(
        q_s, k_s, v_s, attn_mask=None, is_causal=causal, dropout_p=dropout_p)
    return out.transpose(1, 2).contiguous().type(out_dtype)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding (upstream convention)
# ---------------------------------------------------------------------------

def sinusoidal_embedding_1d(dim, position):
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


# ---------------------------------------------------------------------------
# 3D Factorized RoPE (complex-number implementation)
# ---------------------------------------------------------------------------

@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast(enabled=False)
def rope_apply(x, grid_sizes, freqs):
    """Apply 3D factorized RoPE via complex multiplication."""
    n, c = x.size(2), x.size(3) // 2
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).float()


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------

class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


# ---------------------------------------------------------------------------
# DummyAdapterLayer (wraps nn.Linear/LayerNorm for AudioProjModel)
# ---------------------------------------------------------------------------

class DummyAdapterLayer(nn.Module):
    """Trivial wrapper that delegates forward to self.layer.
    Adds a '.layer.' prefix to state_dict keys — required for checkpoint compat."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, *args, **kwargs):
        return self.layer(*args, **kwargs)


# ---------------------------------------------------------------------------
# AudioProjModel — projects Whisper features to audio context tokens
# ---------------------------------------------------------------------------

class AudioProjModel(nn.Module):
    """
    Projects windowed Whisper features to audio conditioning tokens.

    Input:  [B, F, seq_len, blocks, channels]
    Output: [B, F, context_tokens, output_dim]

    State dict keys (via DummyAdapterLayer):
        audio_proj_glob_1.layer.weight/bias
        audio_proj_glob_2.layer.weight/bias
        audio_proj_glob_3.layer.weight/bias
        audio_proj_glob_norm.layer.weight/bias
    """

    def __init__(
        self,
        seq_len=5,
        blocks=13,
        channels=768,
        intermediate_dim=512,
        output_dim=1536,
        context_tokens=16,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.audio_proj_glob_1 = DummyAdapterLayer(nn.Linear(self.input_dim, intermediate_dim))
        self.audio_proj_glob_2 = DummyAdapterLayer(nn.Linear(intermediate_dim, intermediate_dim))
        self.audio_proj_glob_3 = DummyAdapterLayer(nn.Linear(intermediate_dim, context_tokens * output_dim))
        self.audio_proj_glob_norm = DummyAdapterLayer(nn.LayerNorm(output_dim))

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

    def forward(self, audio_embeds):
        """audio_embeds: [B, F, seq_len, blocks, channels]"""
        bz = audio_embeds.shape[0]
        video_length = audio_embeds.shape[1]
        # rearrange "bz f w b c -> (bz f) w b c"
        audio_embeds = audio_embeds.reshape(bz * video_length, *audio_embeds.shape[2:])
        batch_size, window_size, blk, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blk * channels)

        audio_embeds = torch.relu(self.audio_proj_glob_1(audio_embeds))
        audio_embeds = torch.relu(self.audio_proj_glob_2(audio_embeds))

        context_tokens = self.audio_proj_glob_3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim)
        context_tokens = self.audio_proj_glob_norm(
            context_tokens.to(self.audio_proj_glob_norm.layer.weight.dtype)
        ).to(audio_embeds.dtype)

        # rearrange "(bz f) m c -> bz f m c"
        context_tokens = context_tokens.reshape(bz, video_length, self.context_tokens, self.output_dim)
        return context_tokens


# ---------------------------------------------------------------------------
# Self-Attention with QK-norm and RoPE
# ---------------------------------------------------------------------------

class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        x = x.flatten(2)
        x = self.o(x)
        return x


class WanSelfAttentionSepKVDim(nn.Module):
    """Self-attention variant with separate KV input dimension (for audio cross-attn)."""
    def __init__(self, kv_dim, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)
        x = x.flatten(2)
        x = self.o(x)
        return x


# ---------------------------------------------------------------------------
# Cross-Attention variants
# ---------------------------------------------------------------------------

class WanT2VCrossAttention(WanSelfAttention):
    """Text-to-video cross-attention (no RoPE on context)."""
    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttentionGather(WanSelfAttentionSepKVDim):
    """Per-frame audio cross-attention: reshapes to [F, HW] queries × [F, 16] keys."""
    def forward(self, x, context, context_lens, grid_sizes, freqs, audio_seq_len):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        hlen_wlen = int(grid_sizes[0][1] * grid_sizes[0][2])
        q = q.reshape(-1, hlen_wlen, n, d)   # [F, H*W, N, D]
        k = k.reshape(-1, 16, n, d)          # [F, 16, N, D]
        v = v.reshape(-1, 16, n, d)

        x = flash_attention(q, k, v, k_lens=None)
        x = x.view(b, -1, n, d).flatten(2)
        x = self.o(x)
        return x


class WanI2VCrossAttention(WanSelfAttention):
    """HuMo I2V cross-attention (simplified — no CLIP image path, same as T2V)."""
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        super().__init__(dim, num_heads, window_size, qk_norm, eps)

    def forward(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)
        x = flash_attention(q, k, v, k_lens=context_lens)
        x = x.flatten(2)
        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {
    't2v_cross_attn': WanT2VCrossAttention,
    'i2v_cross_attn': WanI2VCrossAttention,
}


# ---------------------------------------------------------------------------
# Audio Cross-Attention Wrapper
# ---------------------------------------------------------------------------

class AudioCrossAttentionWrapper(nn.Module):
    """Wraps per-frame audio cross-attention with pre-norm and residual."""
    def __init__(self, dim, kv_dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.audio_cross_attn = WanT2VCrossAttentionGather(
            kv_dim, dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm1_audio = WanLayerNorm(dim, eps, elementwise_affine=True)

    def forward(self, x, audio, seq_lens, grid_sizes, freqs, audio_seq_len):
        x = x + self.audio_cross_attn(
            self.norm1_audio(x), audio, seq_lens, grid_sizes, freqs, audio_seq_len)
        return x


# ---------------------------------------------------------------------------
# Transformer Block with additive modulation
# ---------------------------------------------------------------------------

class WanAttentionBlock(nn.Module):
    """
    Single DiT block with:
      - WanSelfAttention + 3D RoPE
      - Cross-attention (text)
      - AudioCrossAttentionWrapper (optional, per-frame)
      - 2-layer GELU FFN
      - Additive modulation parameter [1, 6, dim] (NOT per-block linear projection)
    """

    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        use_audio=True,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # Additive modulation parameter (NOT a linear projection)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

        self.use_audio = use_audio
        if use_audio:
            self.audio_cross_attn_wrapper = AudioCrossAttentionWrapper(
                dim, 1536, num_heads, qk_norm, eps)

    def forward(
        self, x, e, seq_lens, grid_sizes, freqs, context, context_lens,
        audio=None, audio_seq_len=None, ref_num_list=None,
    ):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e).chunk(6, dim=1)
        assert e[0].dtype == torch.float32

        # Self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1]) + e[0],
            seq_lens, grid_sizes, freqs)
        with amp.autocast(dtype=torch.float32):
            x = x + y * e[2]

        # Cross-attention (text) + audio + FFN
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            if self.use_audio and audio is not None:
                x = self.audio_cross_attn_wrapper(
                    x, audio, seq_lens, grid_sizes, freqs, audio_seq_len)
            y = self.ffn(self.norm2(x).float() * (1 + e[4]) + e[3])
            with amp.autocast(dtype=torch.float32):
                x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


# ---------------------------------------------------------------------------
# Output Head
# ---------------------------------------------------------------------------

class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        out_dim_flat = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim_flat)

        # Additive modulation for head (scale + shift)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        assert e.dtype == torch.float32
        with amp.autocast(dtype=torch.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + e[1]) + e[0]))
        return x


# ---------------------------------------------------------------------------
# MLPProj (for base I2V CLIP conditioning — kept for completeness)
# ---------------------------------------------------------------------------

class MLPProj(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, in_dim),
            nn.GELU(), nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        return self.proj(image_embeds)


# ---------------------------------------------------------------------------
# WanModel — Main DiT (HuMo variant with audio)
# ---------------------------------------------------------------------------

class WanModel(nn.Module):
    """
    HuMo DiT: 3D video diffusion transformer with text + image + audio conditioning.

    Checkpoint-compatible with Phantom-video/HuMo and kijai/ComfyUI-WanVideoWrapper.

    Key architectural features vs base Wan-AI model:
      - AudioProjModel projects Whisper features → 1536-dim tokens
      - Per-block AudioCrossAttentionWrapper (per-frame, kv_dim=1536)
      - Simplified I2V cross-attention (no CLIP image path)
      - Image conditioning via latent channel concatenation (in_dim=36)
    """

    _no_split_modules = ['WanAttentionBlock']

    def __init__(
        self,
        model_type='i2v',
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=5120,
        ffn_dim=13824,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=40,
        num_layers=40,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        audio_token_num=16,
        insert_audio=True,
    ):
        super().__init__()
        assert model_type in ('t2v', 'i2v')
        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.insert_audio = insert_audio

        # Patch embedding
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)

        # Text conditioning MLP
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        # Global time projection → [B, 6*dim] for block modulation
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # Transformer blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            WanAttentionBlock(
                cross_attn_type, dim, ffn_dim, num_heads,
                window_size, qk_norm, cross_attn_norm, eps,
                use_audio=insert_audio)
            for _ in range(num_layers)
        ])

        # Output head
        self.head = Head(dim, out_dim, patch_size, eps)

        # Audio projection
        if insert_audio:
            self.audio_proj = AudioProjModel(
                seq_len=8, blocks=5, channels=1280,
                intermediate_dim=512, output_dim=1536,
                context_tokens=audio_token_num)

        # RoPE frequencies (not registered as buffer to preserve dtype)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1)

        # Initialize weights
        self.init_weights()

    def forward(self, x, t, context, seq_len, audio=None, y=None):
        """
        Upstream-compatible forward pass.

        Args:
            x:       List of [C, F, H, W] tensors (one per batch element).
            t:       [B] timesteps.
            context: List of [L, text_dim] text embeddings.
            seq_len: Max sequence length for positional encoding.
            audio:   List of [F, 8, 5, 1280] windowed audio features, or None.
            y:       List of [C_cond, F, H, W] image conditioning, or None.
        """
        if self.model_type == 'i2v':
            assert y is not None

        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # Patch embed
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # Time embedding + projection
        with amp.autocast(dtype=torch.float32):
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, t).float()).float()
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).float()
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # Text embedding
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # Audio projection
        if self.insert_audio and audio is not None:
            audio = [self.audio_proj(au.unsqueeze(0)).permute(0, 3, 1, 2) for au in audio]
            audio_seq_len = torch.tensor(
                max([au.shape[2] for au in audio]) * audio[0].shape[3], device=device)
            audio = [au.flatten(2).transpose(1, 2) for au in audio]
            audio = torch.cat([
                torch.cat([au, au.new_zeros(1, audio_seq_len - au.size(1), au.size(2))], dim=1)
                for au in audio
            ])
        else:
            audio = None
            audio_seq_len = None

        # Block loop
        kwargs = dict(
            e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes, freqs=self.freqs,
            context=context, context_lens=context_lens,
            audio=audio, audio_seq_len=audio_seq_len)
        for block in self.blocks:
            x = block(x, **kwargs)

        # Head
        x = self.head(x, e)

        # Unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        """Reconstruct video tensors from patch embeddings."""
        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        nn.init.zeros_(self.head.head.weight)
